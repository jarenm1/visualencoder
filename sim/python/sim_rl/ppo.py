"""GPU-resident recurrent PPO (continuous Gaussian actions) with finite-horizon terminal GAE.

Look-alike confusion: a goal step where the policy believes the target is in view but its peak
patch is another object of the target's class is an encoder/matcher error, not a flight error.
Those steps are counted (telemetry, per catalog entry for failure mining) and, with
``mask_confusion``, get zero weight in the policy and value losses; the supervised saliency loss
on the true target box still trains the matcher to tell them apart.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.profiler import record_function

from .category import category_references
from .curriculum import MODES as CURRICULUM_MODES, Curriculum
from .features import RefreshedGoals
from .levels import HardEpisodes
from .memory import CHANNELS, EGO, SpatialMemory, observe as observe_memory
from .model import EXPLORE_COMMAND, GOAL_COMMAND, saliency_loss, squashed_entropy, squashed_log_prob
from .native import HOVER_ACTIONS, PRIVILEGED_WIDTH, PROPRIO_WIDTH

# finish_rollout layout: rollout totals, PPO losses, bucket totals, used/next weights, EMAs.
ROLLOUT_WIDTH, LOSS_WIDTH, BUCKET_WIDTH = 9, 9, 12
RESULT_WIDTH = ROLLOUT_WIDTH + LOSS_WIDTH + BUCKET_WIDTH + 3 + 3 + 3
CONFUSION_IN_VIEW = 0.5          # saliency P(target in view) above which the peak patch is a claim
# Explore-command reward: newly observed surface tiles (native coverage, 128x128 city grid), a step
# cost, and the same failure penalty as goal episodes. No timeout penalty: exploring has no deadline.
EXPLORE_TILE_REWARD, EXPLORE_STEP_COST, EXPLORE_FAILURE = 0.02, 0.01, 10.0
LOSS_NAMES = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
              "aux_saliency_loss", "aux_patch_accuracy", "aux_memory_loss")


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
    coarse_hint_fraction: float = 0.0
    wrong_hint_fraction: float = 0.0
    appearance: float = 1.0
    goal_refresh_steps: int = 0
    explore_fraction: float = 0.0
    mask_confusion: bool = False
    replay_fraction: float = 0.0
    category_goal_fraction: float = 0.0

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
        if self.goal_refresh_steps < 0:
            raise ValueError("goal_refresh_steps must be non-negative")
        if not 0.0 <= self.explore_fraction < 1.0:
            raise ValueError("explore_fraction must be in [0, 1)")
        if not isinstance(self.mask_confusion, bool):
            raise ValueError("mask_confusion must be a bool")
        if not 0.0 <= self.replay_fraction <= 1.0:
            raise ValueError("replay_fraction must be in [0, 1]")
        if not 0.0 <= self.category_goal_fraction <= 1.0:
            raise ValueError("category_goal_fraction must be in [0, 1]")


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

    def __init__(self, env, features, policy, config, solvable=None):
        """``solvable`` (bool [catalog], verifier.py) enables hard-episode replay when
        ``config.replay_fraction`` > 0 (levels.py)."""
        config.validate()
        if policy.action_dim != env.action_dim:
            raise ValueError("policy action dimension differs from native city-drone task")
        if policy.mode == "visual" and not features.visual:
            raise ValueError("visual policy requires a visual encoder")
        if policy.mode != "visual" and features.visual:
            raise ValueError("non-visual policy must not own a visual encoder")
        if (config.explore_fraction > 0) != (policy.commands > 1):
            raise ValueError("explore episodes need a command-conditioned policy, and only they")
        self.env, self.features, self.policy, self.config = env, features, policy, config
        self.hover = torch.tensor(HOVER_ACTIONS[env.control_mode], device=env.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, fused=True, capturable=True)
        self.steps = 0
        self.curriculum = Curriculum(config.curriculum, env.device, config.no_gps_fraction, config.appearance,
                                     config.coarse_hint_fraction, config.wrong_hint_fraction)
        self.curriculum.write(env)
        t, n, h = config.rollout_steps, env.n, policy.hidden_size
        allocate = lambda *shape: torch.empty(shape, device=env.device)
        self.proprio = allocate(t, n, PROPRIO_WIDTH)
        self.privileged = allocate(t, n, PRIVILEGED_WIDTH)
        self.starts = allocate(t, n)
        self.hidden = allocate(t, n, h)
        self.actions = allocate(t, n, env.action_dim)
        self.logprobs, self.rewards, self.done = allocate(t, n), allocate(t, n), allocate(t, n)
        self.success, self.collision, self.out_of_bounds, self.ended = (allocate(t, n) for _ in range(4))
        self.values, self.advantages, self.returns = allocate(t + 1, n), allocate(t, n), allocate(t, n)
        self.completed_returns = allocate(t, n)
        self.commands = torch.zeros((t, n), dtype=torch.long, device=env.device)
        self.command = torch.zeros(n, dtype=torch.long, device=env.device)
        self.explore_tiles = allocate(t, n)
        self.tiles = torch.zeros(n, device=env.device)
        self.applied = allocate(n, env.action_dim)
        self.bucket_ids = torch.arange(3, device=env.device, dtype=torch.float32)
        self.memory = torch.zeros((n, h), device=env.device)
        self.next_start = torch.ones(n, device=env.device)
        self.episode_return = torch.zeros(n, device=env.device)
        if features.visual:
            patches = env.height // 16 * (env.width // 16)
            self.goals = features.catalog_goals(env, config.appearance)
            category = category_references(env) if config.category_goal_fraction > 0 else None
            self.goal_source = RefreshedGoals(env, features, self.goals, config.goal_refresh_steps, category,
                                              config.category_goal_fraction)
            self.rollout_goals = torch.empty((t, n, features.goal_dim), device=env.device, dtype=torch.bfloat16)
            self.cls = allocate(t, n, features.dim)
            self.tokens = torch.empty((t, n, patches, features.dim), device=env.device, dtype=torch.bfloat16)
        else:
            self.goals = self.cls = self.tokens = None
        # Look-alike confusion telemetry (visual arms): per step, and per catalog entry.
        self.loss_weights = torch.ones((t, n), device=env.device)
        self.confused = torch.zeros((t, n), device=env.device)
        self.claimed = torch.zeros((t, n), device=env.device)
        self.confusion_by_catalog = torch.zeros(env.catalog_size, device=env.device)
        if features.visual:
            layout = json.loads((env.city_path / "layout.json").read_text())["objects"]
            self.object_class = torch.tensor([env.layout_classes.index(o["class"]) for o in layout],
                                             device=env.device, dtype=torch.long)
        if policy.spatial:
            self.spatial = SpatialMemory(n, env.height // 16, env.width // 16, env.metadata["camera"], env.device)
            self.memory_ego = torch.empty((t, n, CHANNELS, EGO, EGO), device=env.device, dtype=torch.bfloat16)
            self.memory_vector = allocate(t, n, policy.memory_vector_width)
        else:
            self.spatial = None
        self.current_cls = self.current_tokens = None
        self._sequence_offsets = torch.arange(config.sequence_length, device=env.device).unsqueeze(1)
        self._observe()
        if config.replay_fraction > 0 and solvable is None:
            raise ValueError("hard-episode replay needs the verifier's solvable labels")
        self.levels = HardEpisodes(env, solvable, config.replay_fraction) if config.replay_fraction > 0 else None

    def _observe(self):
        if self.features.visual:
            self.current_cls, self.current_tokens = self.features.observe(self.env)

    def _confusion(self, t, logits, explore):
        """Mark goal steps whose saliency peak claims a same-class look-alike of the target
        (label: native per-patch object); see the module docstring."""
        probs = logits.softmax(-1)
        claimed = (probs[:, -1] < 1 - CONFUSION_IN_VIEW) & ~explore
        objects = self.env.patch_object.long()
        peak = objects.gather(1, probs[:, :-1].argmax(-1, keepdim=True)).squeeze(1)
        target = self.env.diagnostics[:, 27].long()
        same_class = self.object_class[peak.clamp_min(0)] == self.env.diagnostics[:, 14].long()
        # A type-level reference makes any same-type object a fair approach; only instance goals count.
        confused = claimed & (peak >= 0) & (peak != target) & same_class & self.goal_source.instance
        self.claimed[t].copy_(claimed.float())
        self.confused[t].copy_(confused.float())
        self.loss_weights[t].copy_(1.0 - confused.float() if self.config.mask_confusion else torch.ones_like(self.confused[t]))
        self.confusion_by_catalog.index_add_(0, self.env.catalog_index.long(), confused.float())

    def _inputs(self, cls, tokens, goal, proprio, privileged, memory=None, command=None):
        """Use the exact goal and memory seen at collection, including any in-episode refresh."""
        return self.policy.actor_inputs_and_features(cls, tokens, goal, proprio, privileged, memory, command)

    def _memory(self, goal):
        """Spatial-memory tick for the current observation (None without spatial memory). Explore
        episodes have no goal, so their frames add looked-at cells but no goal evidence."""
        if self.spatial is None:
            return None
        return observe_memory(self.policy, self.spatial, self.current_cls, self.current_tokens, goal,
                              self.env.proprio, self.next_start, (self.command == GOAL_COMMAND).float())

    def _sample_commands(self):
        """New episodes explore with probability explore_fraction (GPU draw, no host decision)."""
        if self.config.explore_fraction > 0:
            fresh = (torch.rand(self.env.n, device=self.env.device) < self.config.explore_fraction).long()
            self.command.copy_(torch.where(self.next_start > 0.5, fresh, self.command))

    def _explore_reward(self, explore):
        """Explore episodes: EXPLORE_TILE_REWARD per newly observed surface tile (native coverage
        counter, terminal value on the done step), EXPLORE_STEP_COST per step, EXPLORE_FAILURE on a
        crash or leaving the area; no goal terms. Updates the per-env tile baseline."""
        diagnostics = self.env.episode_diagnostics
        done = self.env.done > 0.5
        gain = torch.where(done, diagnostics[:, 1], diagnostics[:, 0]) - self.tiles
        failed = (self.env.collision > 0.5) | (self.env.out_of_bounds > 0.5)
        reward = EXPLORE_TILE_REWARD * gain - EXPLORE_STEP_COST - EXPLORE_FAILURE * failed.float()
        self.explore_tiles_now = torch.where(done & explore, diagnostics[:, 1], torch.zeros_like(gain))
        self.tiles.copy_(diagnostics[:, 0])
        return reward

    @torch.no_grad()
    def warmup(self, seed, iterations=3):
        """Initialize kernels/allocator, then discard warmup physics episodes."""
        hover = self.hover.expand(self.env.n, -1).contiguous()
        for _ in range(iterations):
            goal = self.goal_source.step(self.next_start) if self.features.visual else None
            inputs, _ = self._inputs(self.current_cls, self.current_tokens, goal,
                                     self.env.proprio, self.env.privileged, self._memory(goal), self.command)
            self.policy.step(inputs, self.memory, self.next_start)
            self.env.step(hover)
            self._observe()
        self.env.reset(seed)
        if self.levels is not None:
            self.levels.episode.copy_(self.env.catalog_index.long())
        if self.features.visual:
            self.goal_source.ticks = 0
            self.goal_source.updates.zero_()
        self._observe()
        self.memory.zero_()
        self.next_start.fill_(1)
        self.episode_return.zero_()
        self.command.zero_()
        self.tiles.copy_(self.env.episode_diagnostics[:, 0])
        self.env.stream.synchronize()  # Explicit initialization boundary, not collection.

    @torch.no_grad()
    def collect(self, length=None):
        length = self.config.rollout_steps if length is None else length
        if length < 1 or length > self.config.rollout_steps or length % self.config.sequence_length:
            raise ValueError("rollout must fit preallocation and contain complete recurrent sequences")
        log_std = self.policy.log_std
        if self.levels is not None:
            self.levels.propose()
        with gpu_range("sim.collect"):
            for t in range(length):
                self._sample_commands()
                self.commands[t].copy_(self.command)
                explore = self.command == EXPLORE_COMMAND
                self.proprio[t].copy_(self.env.proprio)
                self.privileged[t].copy_(self.env.privileged)
                self.starts[t].copy_(self.next_start)
                self.hidden[t].copy_(self.memory)
                if self.features.visual:
                    # Explore episodes carry no goal: the goal input is zero.
                    goal = self.goal_source.step(self.next_start) * (~explore).unsqueeze(-1)
                    self.rollout_goals[t].copy_(goal)
                    self.cls[t].copy_(self.current_cls)
                    self.tokens[t].copy_(self.current_tokens)
                memory = self._memory(self.rollout_goals[t].float() if self.features.visual else None)
                if memory is not None:
                    self.memory_ego[t].copy_(memory[0])
                    self.memory_vector[t].copy_(memory[1])
                inputs, logits = self._inputs(self.cls[t] if self.features.visual else None,
                                              self.tokens[t] if self.features.visual else None,
                                              self.rollout_goals[t].float() if self.features.visual else None,
                                              self.proprio[t], self.privileged[t], memory, self.command)
                if logits is not None:
                    self._confusion(t, logits, explore)
                mean, hidden = self.policy.step(inputs, self.memory, self.next_start)
                raw = mean + log_std.exp() * torch.randn_like(mean)
                self.actions[t].copy_(raw)
                self.logprobs[t].copy_(squashed_log_prob(mean, log_std, raw))
                self.values[t].copy_(self.policy.value(self.privileged[t], self.proprio[t], self.command))
                self.memory.copy_(hidden)
                # The simulator receives the squashed command; rollouts keep the pre-squash
                # sample, whose log-probability includes the tanh Jacobian.
                torch.tanh(raw, out=self.applied)
                self.env.step(self.applied)
                if self.levels is not None:
                    self.levels.observe(self.env.done, self.env.success, ~explore)
                if self.config.explore_fraction > 0:
                    reward = torch.where(explore, self._explore_reward(explore), self.env.reward)
                    self.explore_tiles[t].copy_(self.explore_tiles_now)
                else:
                    reward = self.env.reward
                self.rewards[t].copy_(reward)
                self.done[t].copy_(self.env.done)
                # Goal success and the visibility-bucket curriculum describe goal episodes only.
                self.success[t].copy_(self.env.success * (~explore))
                self.collision[t].copy_(self.env.collision)
                self.out_of_bounds[t].copy_(self.env.out_of_bounds)
                self.ended[t].copy_(torch.where(explore, torch.full_like(self.env.ended_bucket, -1), self.env.ended_bucket))
                self.episode_return.add_(reward)
                self.completed_returns[t].copy_(self.episode_return * self.env.done)
                self.episode_return.mul_(1 - self.env.done)
                self.next_start.copy_(self.env.done)
                # Native returns the reset observation on completion; the next
                # tick sees it with a reset GRU while completion flags remain old.
                self._observe()
            self.values[length].copy_(self.policy.value(self.env.privileged, self.env.proprio, self.command))
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
                    command = self.commands[times, env_ids]
                    goal_rows = command.flatten(0, 1) == GOAL_COMMAND
                    if self.features.visual:
                        memory = ((self.memory_ego[times, env_ids].flatten(0, 1),
                                   self.memory_vector[times, env_ids].flatten(0, 1))
                                  if self.spatial is not None else None)
                        inputs, attended = self._inputs(self.cls[times, env_ids].flatten(0, 1),
                                                        self.tokens[times, env_ids].flatten(0, 1),
                                                        self.rollout_goals[times, env_ids].flatten(0, 1).float(),
                                                        proprio.flatten(0, 1), privileged.flatten(0, 1), memory,
                                                        command.flatten(0, 1))
                    else:
                        inputs, attended = self._inputs(None, None, None, proprio.flatten(0, 1),
                                                        privileged.flatten(0, 1), None, command.flatten(0, 1))
                    means = self.policy.sequence(inputs.view(length_b, batch, -1),
                                                 self.hidden[first, env_ids], self.starts[times, env_ids])
                    logprobs = squashed_log_prob(means, self.policy.log_std, self.actions[times, env_ids])
                    entropy = squashed_entropy(means, self.policy.log_std)
                    values = self.policy.value(privileged, proprio, command)
                    logratio = logprobs - self.logprobs[times, env_ids]
                    ratio = logratio.exp()
                    adv = self.advantages[times, env_ids]
                    # Confusion-masked steps (mask_confusion) carry zero weight; otherwise all ones.
                    weight = self.loss_weights[times, env_ids]
                    norm = weight.sum().clamp_min(1.0)
                    policy_loss = (torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))
                                   * weight).sum() / norm
                    old_value = self.values[times, env_ids]
                    clipped_value = old_value + (values - old_value).clamp(-cfg.clip, cfg.clip)
                    target = self.returns[times, env_ids]
                    value_loss = 0.5 * (torch.maximum((values - target).square(), (clipped_value - target).square())
                                        * weight).sum() / norm
                    if attended is not None:
                        aux_saliency, aux_accuracy = saliency_loss(
                            attended, privileged.flatten(0, 1), self.policy.attention.grid_h,
                            self.policy.attention.grid_w, goal_rows)
                    else:
                        aux_saliency = aux_accuracy = torch.zeros((), device=self.env.device)
                    aux_memory = (self.policy.memory_loss(inputs, privileged.flatten(0, 1), goal_rows.float())
                                  if self.spatial is not None else torch.zeros((), device=self.env.device))
                    loss = (policy_loss + cfg.value_coefficient * value_loss - cfg.entropy * entropy
                            + cfg.aux_coefficient * (aux_saliency + aux_memory))
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm, foreach=True)
                    self.optimizer.step()
                    with torch.no_grad():
                        losses.add_(torch.stack((policy_loss, value_loss, entropy,
                                                 ((ratio - 1) - logratio).mean(),
                                                 ((ratio - 1).abs() > cfg.clip).float().mean(), grad_norm,
                                                 aux_saliency.detach(), aux_accuracy, aux_memory.detach())))
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

        Returns float32 [RESULT_WIDTH] (39): rollout totals (9: mean reward, completed episodes,
        goal successes, collisions, completed returns, completed explore episodes, their terminal
        observed tiles, goal steps claiming the target in view, of those confused with a look-alike),
        PPO losses (9, see LOSS_NAMES), bucket totals (12, row-major [4,3]), weights used this
        rollout (3), weights for the next rollout (3), success EMAs (3).
        """
        totals = self.bucket_totals(length)
        used = self.curriculum.weights.clone()
        self.curriculum.observe(totals[0], totals[1])
        self.curriculum.advance()
        self.curriculum.write(self.env)
        explored = self.done[:length] * (self.commands[:length] == EXPLORE_COMMAND).float()
        rollout = torch.stack((self.rewards[:length].mean(), self.done[:length].sum(),
                               self.success[:length].sum(), self.collision[:length].sum(),
                               self.completed_returns[:length].sum(), explored.sum(),
                               self.explore_tiles[:length].sum(), self.claimed[:length].sum(),
                               self.confused[:length].sum()))
        return torch.cat((rollout, losses, totals.flatten(), used,
                          self.curriculum.weights, self.curriculum.ema))
