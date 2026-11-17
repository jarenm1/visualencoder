"""GPU-resident recurrent PPO (continuous Gaussian actions) with finite-horizon terminal GAE."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.profiler import record_function

from .curriculum import MODES as CURRICULUM_MODES, Curriculum
from .model import saliency_loss, squashed_entropy, squashed_log_prob
from .native import HOVER_ACTIONS, PRIVILEGED_WIDTH, PROPRIO_WIDTH

# finish_rollout layout: rollout totals, PPO losses, bucket totals, used/next weights, EMAs.
ROLLOUT_WIDTH, LOSS_WIDTH, BUCKET_WIDTH = 5, 8, 12
RESULT_WIDTH = ROLLOUT_WIDTH + LOSS_WIDTH + BUCKET_WIDTH + 3 + 3 + 3
LOSS_NAMES = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
              "aux_saliency_loss", "aux_patch_accuracy")


@contextmanager
def gpu_range(name):
    with record_function(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


@dataclass
class Config:
    total_steps: int = 1_000_000
    rollout_steps: int = 64
    sequence_length: int = 16
    epochs: int = 4
    minibatch_sequences: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    curriculum: str = "adaptive"
    aux_coefficient: float = 0.5
    no_gps_fraction: float = 0.0
    appearance: float = 1.0

    def validate(self):
        if min(self.total_steps, self.rollout_steps, self.sequence_length,
               self.epochs, self.minibatch_sequences) < 1:
            raise ValueError("PPO sizes and budget must be positive")
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if self.curriculum not in CURRICULUM_MODES:
            raise ValueError(f"curriculum must be one of {CURRICULUM_MODES}")
        if self.aux_coefficient < 0:
            raise ValueError("aux_coefficient must be non-negative")
        if not 0.0 <= self.no_gps_fraction <= 1.0:
            raise ValueError("no_gps_fraction must be in [0, 1]")
        if not 0.0 <= self.appearance <= 1.0:
            raise ValueError("appearance must be in [0, 1]")


def terminal_gae(rewards, values, done, gamma, lam, out=None):
    """Every done, including the finite episode clock, suppresses bootstrap/recursion.

    values has T+1 entries; the final value may describe an autoreset observation,
    but multiplication by (1-done) makes it irrelevant to terminal transitions.
    """
    result = torch.empty_like(rewards) if out is None else out
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        live = 1 - done[t]
        delta = rewards[t] + gamma * values[t + 1] * live - values[t]
        carry = delta + gamma * lam * live * carry
        result[t].copy_(carry)
    return result


class Trainer:
    """Rollout storage is preallocated; collection and update never read values back to the host."""

    def __init__(self, env, features, policy, config):
        config.validate()
        if policy.action_dim != env.action_dim:
            raise ValueError("policy action dimension differs from native city-drone task")
        if policy.mode == "visual" and not features.visual:
            raise ValueError("visual policy requires a visual encoder")
        if policy.mode != "visual" and features.visual:
            raise ValueError("non-visual policy must not own a visual encoder")
        self.env, self.features, self.policy, self.config = env, features, policy, config
        self.hover = torch.tensor(HOVER_ACTIONS[env.control_mode], device=env.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, fused=True, capturable=True)
        self.steps = 0
        self.curriculum = Curriculum(config.curriculum, env.device, config.no_gps_fraction, config.appearance)
        self.curriculum.write(env)
        t, n, h = config.rollout_steps, env.n, policy.hidden_size
        allocate = lambda *shape: torch.empty(shape, device=env.device)
        self.proprio = allocate(t, n, PROPRIO_WIDTH)
        self.privileged = allocate(t, n, PRIVILEGED_WIDTH)
        self.catalog = torch.zeros((t, n), device=env.device, dtype=torch.long)
        self.starts = allocate(t, n)
        self.hidden = allocate(t, n, h)
        self.actions = allocate(t, n, env.action_dim)
        self.logprobs, self.rewards, self.done = allocate(t, n), allocate(t, n), allocate(t, n)
        self.success, self.collision, self.out_of_bounds, self.ended = (allocate(t, n) for _ in range(4))
        self.values, self.advantages, self.returns = allocate(t + 1, n), allocate(t, n), allocate(t, n)
        self.completed_returns = allocate(t, n)
        self.applied = allocate(n, env.action_dim)
        self.bucket_ids = torch.arange(3, device=env.device, dtype=torch.float32)
        self.memory = torch.zeros((n, h), device=env.device)
        self.next_start = torch.ones(n, device=env.device)
        self.episode_return = torch.zeros(n, device=env.device)
        if features.visual:
            patches = env.height // 16 * (env.width // 16)
            self.goals = features.catalog_goals(env, config.appearance)
            self.cls = allocate(t, n, features.dim)
            self.tokens = torch.empty((t, n, patches, features.dim), device=env.device, dtype=torch.bfloat16)
        else:
            self.goals = self.cls = self.tokens = None
        self.current_cls = self.current_tokens = None
        self._sequence_offsets = torch.arange(config.sequence_length, device=env.device).unsqueeze(1)
        self._observe()

    def _observe(self):
        if self.features.visual:
            self.current_cls, self.current_tokens = self.features.observe(self.env)

    def _inputs(self, cls, tokens, catalog, proprio, privileged):
        """Returns (actor inputs, saliency logits or None); the goal is the catalog entry's crop embedding."""
        goal = self.goals[catalog].float() if self.features.visual else None
        return self.policy.actor_inputs_and_features(cls, tokens, goal, proprio, privileged)

    @torch.no_grad()
    def warmup(self, seed, iterations=3):
        """Initialize kernels/allocator, then discard warmup physics episodes."""
        hover = self.hover.expand(self.env.n, -1).contiguous()
        for _ in range(iterations):
            inputs, _ = self._inputs(self.current_cls, self.current_tokens, self.env.catalog_index.long(),
                                     self.env.proprio, self.env.privileged)
            self.policy.step(inputs, self.memory, self.next_start)
            self.env.step(hover)
            self._observe()
        self.env.reset(seed)
        self._observe()
        self.memory.zero_()
        self.next_start.fill_(1)
        self.episode_return.zero_()
        self.env.stream.synchronize()  # Explicit initialization boundary, not collection.

    @torch.no_grad()
    def collect(self, length=None):
        length = self.config.rollout_steps if length is None else length
        if length < 1 or length > self.config.rollout_steps or length % self.config.sequence_length:
            raise ValueError("rollout must fit preallocation and contain complete recurrent sequences")
        log_std = self.policy.log_std
        with gpu_range("sim.collect"):
            for t in range(length):
                self.proprio[t].copy_(self.env.proprio)
                self.privileged[t].copy_(self.env.privileged)
                self.catalog[t].copy_(self.env.catalog_index)
                self.starts[t].copy_(self.next_start)
                self.hidden[t].copy_(self.memory)
                if self.features.visual:
                    self.cls[t].copy_(self.current_cls)
                    self.tokens[t].copy_(self.current_tokens)
                inputs, _ = self._inputs(self.cls[t] if self.features.visual else None,
                                         self.tokens[t] if self.features.visual else None,
                                         self.catalog[t], self.proprio[t], self.privileged[t])
                mean, memory = self.policy.step(inputs, self.memory, self.next_start)
                raw = mean + log_std.exp() * torch.randn_like(mean)
                self.actions[t].copy_(raw)
                self.logprobs[t].copy_(squashed_log_prob(mean, log_std, raw))
                self.values[t].copy_(self.policy.value(self.privileged[t], self.proprio[t]))
                self.memory.copy_(memory)
                # The simulator receives the squashed command; rollouts keep the pre-squash
                # sample, whose log-probability includes the tanh Jacobian.
                torch.tanh(raw, out=self.applied)
                self.env.step(self.applied)
                self.rewards[t].copy_(self.env.reward)
                self.done[t].copy_(self.env.done)
                self.success[t].copy_(self.env.success)
                self.collision[t].copy_(self.env.collision)
                self.out_of_bounds[t].copy_(self.env.out_of_bounds)
                self.ended[t].copy_(self.env.ended_bucket)
                self.episode_return.add_(self.env.reward)
                self.completed_returns[t].copy_(self.episode_return * self.env.done)
                self.episode_return.mul_(1 - self.env.done)
                self.next_start.copy_(self.env.done)
                # Native returns the reset observation on completion; the next
                # tick sees it with a reset GRU while completion flags remain old.
                self._observe()
            self.values[length].copy_(self.policy.value(self.env.privileged, self.env.proprio))
        self.steps += length * self.env.n
        return length

    def update(self, length):
        cfg, n = self.config, self.env.n
        with gpu_range("sim.ppo_update"):
            with torch.no_grad():
                advantage = terminal_gae(self.rewards[:length], self.values[:length + 1],
                                         self.done[:length], cfg.gamma, cfg.gae_lambda,
                                         self.advantages[:length])
                self.returns[:length].copy_(advantage + self.values[:length])
                advantage.sub_(advantage.mean()).div_(advantage.std(unbiased=False) + 1e-8)
            sequences = length // cfg.sequence_length * n
            losses = torch.zeros(LOSS_WIDTH, device=self.env.device)
            batches = 0
            for _ in range(cfg.epochs):
                permutation = torch.randperm(sequences, device=self.env.device)
                for begin in range(0, sequences, cfg.minibatch_sequences):
                    ids = permutation[begin:begin + cfg.minibatch_sequences]
                    env_ids = ids.remainder(n)
                    first = ids.div(n, rounding_mode="floor") * cfg.sequence_length
                    times = first.unsqueeze(0) + self._sequence_offsets
                    proprio = self.proprio[times, env_ids]
                    privileged = self.privileged[times, env_ids]
                    length_b, batch = times.shape
                    if self.features.visual:
                        inputs, attended = self._inputs(self.cls[times, env_ids].flatten(0, 1),
                                                        self.tokens[times, env_ids].flatten(0, 1),
                                                        self.catalog[times, env_ids].flatten(),
                                                        proprio.flatten(0, 1), privileged.flatten(0, 1))
                    else:
                        inputs, attended = self._inputs(None, None, None, proprio.flatten(0, 1),
                                                        privileged.flatten(0, 1))
                    means = self.policy.sequence(inputs.view(length_b, batch, -1),
                                                 self.hidden[first, env_ids], self.starts[times, env_ids])
                    logprobs = squashed_log_prob(means, self.policy.log_std, self.actions[times, env_ids])
                    entropy = squashed_entropy(means, self.policy.log_std)
                    values = self.policy.value(privileged, proprio)
                    logratio = logprobs - self.logprobs[times, env_ids]
                    ratio = logratio.exp()
                    adv = self.advantages[times, env_ids]
                    policy_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    old_value = self.values[times, env_ids]
                    clipped_value = old_value + (values - old_value).clamp(-cfg.clip, cfg.clip)
                    target = self.returns[times, env_ids]
                    value_loss = 0.5 * torch.maximum((values - target).square(), (clipped_value - target).square()).mean()
                    if attended is not None:
                        aux_saliency, aux_accuracy = saliency_loss(
                            attended, privileged.flatten(0, 1), self.policy.attention.grid_h,
                            self.policy.attention.grid_w)
                    else:
                        aux_saliency = aux_accuracy = torch.zeros((), device=self.env.device)
                    loss = (policy_loss + cfg.value_coefficient * value_loss - cfg.entropy * entropy
                            + cfg.aux_coefficient * aux_saliency)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm, foreach=True)
                    self.optimizer.step()
                    with torch.no_grad():
                        losses.add_(torch.stack((policy_loss, value_loss, entropy,
                                                 ((ratio - 1) - logratio).mean(),
                                                 ((ratio - 1).abs() > cfg.clip).float().mean(), grad_norm,
                                                 aux_saliency.detach(), aux_accuracy)))
                    batches += 1
            losses.div_(batches)
        return losses

    @torch.no_grad()
    def bucket_totals(self, length):
        """float32 [4, 3]: completed episodes, successes, collisions, out-of-bounds per visibility bucket."""
        done = self.done[:length].reshape(-1)
        onehot = (self.ended[:length].reshape(-1, 1) == self.bucket_ids).float()
        flags = torch.stack((done, done * self.success[:length].reshape(-1),
                             done * self.collision[:length].reshape(-1),
                             done * self.out_of_bounds[:length].reshape(-1)))
        return flags @ onehot

    @torch.no_grad()
    def finish_rollout(self, length, losses):
        """GPU-only end-of-rollout bookkeeping: curriculum update for the next rollout and metrics.

        Returns float32 [RESULT_WIDTH] (34): rollout totals (5), PPO losses (8, see LOSS_NAMES),
        bucket totals (12, row-major [4,3]), weights used this rollout (3), weights for the
        next rollout (3), success EMAs (3).
        """
        totals = self.bucket_totals(length)
        used = self.curriculum.weights.clone()
        self.curriculum.observe(totals[0], totals[1])
        self.curriculum.advance()
        self.curriculum.write(self.env)
        rollout = torch.stack((self.rewards[:length].mean(), self.done[:length].sum(),
                               self.success[:length].sum(), self.collision[:length].sum(),
                               self.completed_returns[:length].sum()))
        return torch.cat((rollout, losses, totals.flatten(), used,
                          self.curriculum.weights, self.curriculum.ema))
