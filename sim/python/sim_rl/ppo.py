"""GPU-resident recurrent PPO with finite-horizon terminal (not truncation) GAE."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.profiler import record_function


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
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    def validate(self):
        if min(self.total_steps, self.rollout_steps, self.sequence_length,
               self.epochs, self.minibatch_sequences, self.hidden_size) < 1:
            raise ValueError("PPO sizes and budget must be positive")
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")


PROPRIO_SCALE = (32.0, 6.0, 32.0, 3.0, 3.0, 3.0, 1.0, 1.0)


class Policy(nn.Module):
    def __init__(self, feature_dim, action_count, hidden_size=128):
        super().__init__()
        self.feature_dim, self.action_count, self.hidden_size = feature_dim, action_count, hidden_size
        self.register_buffer("proprio_scale", torch.tensor(PROPRIO_SCALE))
        self.input = nn.Sequential(nn.Linear(feature_dim + 8, hidden_size),
                                   nn.LayerNorm(hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, features, proprio, hidden, starts):
        hidden = hidden * (1 - starts).unsqueeze(-1)
        hidden = self.gru(self.input(torch.cat((features, proprio / self.proprio_scale), -1)), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden

    def sequence(self, features, proprio, hidden, starts):
        logits, values = [], []
        for t in range(features.shape[0]):
            logit, value, hidden = self(features[t], proprio[t], hidden, starts[t])
            logits.append(logit)
            values.append(value)
        return torch.stack(logits), torch.stack(values)


def categorical(logits, actions=None):
    """Gumbel-max sampling avoids Categorical validation and multinomial syncs."""
    log_probs = logits.log_softmax(-1)
    if actions is None:
        uniform = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
        actions = (logits - (-uniform.log()).log()).argmax(-1)
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return actions, selected, entropy


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
    def __init__(self, env, features, policy, config):
        config.validate()
        if policy.action_count != env.action_count:
            raise ValueError("policy action dimension differs from native city task")
        self.env, self.features, self.policy, self.config = env, features, policy, config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, fused=True, capturable=True)
        self.steps = 0
        t, n, d, h = config.rollout_steps, env.n, features.dim, config.hidden_size
        allocate = lambda *shape: torch.empty(shape, device=env.device)
        self.visual = allocate(t + 1, n, d)
        self.proprio = allocate(t + 1, n, 8)
        self.starts = allocate(t + 1, n)
        self.hidden = allocate(t, n, h)
        self.actions = torch.empty((t, n), device=env.device, dtype=torch.int64)
        self.logprobs, self.rewards, self.done = (allocate(t, n) for _ in range(3))
        self.success, self.collision = allocate(t, n), allocate(t, n)
        self.values, self.advantages, self.returns = allocate(t + 1, n), allocate(t, n), allocate(t, n)
        self.current = allocate(n, d)
        self.memory = torch.zeros((n, h), device=env.device)
        self.next_start = torch.ones(n, device=env.device)
        self.episode_return = torch.zeros(n, device=env.device)
        self.completed_returns = allocate(t, n)
        self.current.copy_(features(env))
        self._sequence_offsets = torch.arange(config.sequence_length, device=env.device).unsqueeze(1)

    @torch.no_grad()
    def warmup(self, seed, iterations=3):
        """Initialize kernels/allocator, then discard warmup physics episodes."""
        action = torch.full((self.env.n,), self.env.neutral_action, dtype=torch.int64, device=self.env.device)
        for _ in range(iterations):
            self.policy(self.features(self.env), self.env.proprio, self.memory, self.next_start)
            self.env.step(action)
        self.env.reset(seed)
        self.current.copy_(self.features(self.env))
        self.memory.zero_()
        self.next_start.fill_(1)
        self.episode_return.zero_()
        self.env.stream.synchronize()  # Explicit initialization boundary, not collection.

    @torch.no_grad()
    def collect(self, length=None):
        length = self.config.rollout_steps if length is None else length
        if length < 1 or length > self.config.rollout_steps or length % self.config.sequence_length:
            raise ValueError("rollout must fit preallocation and contain complete recurrent sequences")
        self.policy.eval()
        with gpu_range("sim.collect"):
            for t in range(length):
                self.visual[t].copy_(self.current)
                self.proprio[t].copy_(self.env.proprio)
                self.starts[t].copy_(self.next_start)
                self.hidden[t].copy_(self.memory)
                logits, value, memory = self.policy(self.visual[t], self.proprio[t], self.memory, self.next_start)
                action, logprob, _ = categorical(logits)
                self.actions[t].copy_(action)
                self.logprobs[t].copy_(logprob)
                self.values[t].copy_(value)
                self.memory.copy_(memory)
                self.env.step(self.actions[t])
                self.rewards[t].copy_(self.env.reward)
                self.done[t].copy_(self.env.done)
                self.success[t].copy_(self.env.success)
                self.collision[t].copy_(self.env.collision)
                self.episode_return.add_(self.env.reward)
                self.completed_returns[t].copy_(self.episode_return * self.env.done)
                self.episode_return.mul_(1 - self.env.done)
                self.next_start.copy_(self.env.done)
                # Native returns the reset observation on completion. The next
                # action sees it with a reset GRU; completion flags remain old.
                self.current.copy_(self.features(self.env))
            self.visual[length].copy_(self.current)
            self.proprio[length].copy_(self.env.proprio)
            self.starts[length].copy_(self.next_start)
            _, bootstrap, _ = self.policy(self.current, self.env.proprio, self.memory, self.next_start)
            self.values[length].copy_(bootstrap)
        self.steps += length * self.env.n
        return length

    def update(self, length):
        cfg, n = self.config, self.env.n
        self.policy.train()
        with gpu_range("sim.ppo_update"):
            with torch.no_grad():
                advantage = terminal_gae(self.rewards[:length], self.values[:length + 1],
                                         self.done[:length], cfg.gamma, cfg.gae_lambda,
                                         self.advantages[:length])
                self.returns[:length].copy_(advantage + self.values[:length])
                advantage.sub_(advantage.mean()).div_(advantage.std(unbiased=False) + 1e-8)
            sequences = length // cfg.sequence_length * n
            losses = torch.zeros(6, device=self.env.device)
            batches = 0
            for _ in range(cfg.epochs):
                permutation = torch.randperm(sequences, device=self.env.device)
                for begin in range(0, sequences, cfg.minibatch_sequences):
                    ids = permutation[begin:begin + cfg.minibatch_sequences]
                    env_ids = ids.remainder(n)
                    first = ids.div(n, rounding_mode="floor") * cfg.sequence_length
                    times = first.unsqueeze(0) + self._sequence_offsets
                    logits, values = self.policy.sequence(
                        self.visual[times, env_ids], self.proprio[times, env_ids],
                        self.hidden[first, env_ids], self.starts[times, env_ids])
                    _, logprobs, entropy = categorical(logits, self.actions[times, env_ids])
                    logratio = logprobs - self.logprobs[times, env_ids]
                    ratio = logratio.exp()
                    adv = self.advantages[times, env_ids]
                    policy_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    old_value = self.values[times, env_ids]
                    clipped_value = old_value + (values - old_value).clamp(-cfg.clip, cfg.clip)
                    target = self.returns[times, env_ids]
                    value_loss = 0.5 * torch.maximum((values - target).square(), (clipped_value - target).square()).mean()
                    mean_entropy = entropy.mean()
                    loss = policy_loss + cfg.value_coefficient * value_loss - cfg.entropy * mean_entropy
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm, foreach=True)
                    self.optimizer.step()
                    with torch.no_grad():
                        losses.add_(torch.stack((policy_loss, value_loss, mean_entropy,
                                                 ((ratio - 1) - logratio).mean(),
                                                 ((ratio - 1).abs() > cfg.clip).float().mean(), grad_norm)))
                    batches += 1
            losses.div_(batches)
        return losses

    def metrics(self, length, losses):
        """Return GPU aggregates; the CLI reads them only outside hot ranges."""
        totals = torch.stack((self.rewards[:length].mean(), self.done[:length].sum(),
                              self.success[:length].sum(), self.collision[:length].sum(),
                              self.completed_returns[:length].sum()))
        return torch.cat((totals, losses))


@torch.no_grad()
def evaluate(env, features, policy=None, heuristic=None, seed=0):
    """One first completed episode per environment over the native finite horizon."""
    if (policy is None) == (heuristic is None):
        raise ValueError("provide exactly one trained policy or blind heuristic")
    n = env.n
    active = torch.ones(n, device=env.device)
    totals = torch.zeros((5, n), device=env.device)  # return, success, collision, length, completions
    hidden_size = policy.hidden_size if policy is not None else 1
    memory = torch.zeros((n, hidden_size), device=env.device)
    starts = torch.ones(n, device=env.device)
    actions = torch.empty(n, dtype=torch.int64, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    if policy is not None:
        policy.eval()
    with gpu_range("sim.evaluate"):
        for _ in range(env.horizon):
            if policy is not None:
                logits, _, memory = policy(features(env), env.proprio, memory, starts)
                actions.copy_(logits.argmax(-1))
            elif heuristic == "hover":
                actions.fill_(env.neutral_action)
            elif heuristic == "random":
                torch.randint(env.action_count, (n,), device=env.device, generator=generator, out=actions)
            elif heuristic == "goal_centering":
                # The city camera auto-yaws at the final goal: forward is the
                # honest blind goal-centering control, not a waypoint oracle.
                actions.fill_(1)
            else:
                raise ValueError(f"unknown blind heuristic {heuristic}")
            env.step(actions)
            totals[0].add_(active * env.reward)
            totals[1].add_(active * env.success)
            totals[2].add_(active * env.collision)
            totals[3].add_(active)
            totals[4].add_(active * env.done)
            active.mul_(1 - env.done)
            starts.copy_(env.done)
    # A single aggregate transfer after the finite horizon, never a scalar read in
    # the loop. Include squares for return standard errors and successful
    # lengths separately: collision duration is not time-to-goal.
    aggregate = torch.cat((totals.sum(1), totals[0].square().sum().reshape(1),
                           active.sum().reshape(1), (totals[3] * totals[1]).sum().reshape(1)))
    env.stream.synchronize()  # Explicit evaluation end-batch boundary.
    values = aggregate.cpu().tolist()
    if values[6] != 0 or values[4] != n:
        raise RuntimeError(f"native finite horizon violated: {values[6]} unfinished of {n}")
    mean_return = values[0] / n
    mean_success_steps = values[7] / values[1] if values[1] else None
    return {"episodes": n, "success": values[1] / n, "collision": values[2] / n,
            "mean_return": mean_return, "mean_steps": values[3] / n,
            "mean_success_steps": mean_success_steps,
            "return_standard_error": (max(0, values[5] / n - mean_return ** 2) / n) ** 0.5,
            "policy": heuristic or "checkpoint", "horizon_terminal": env.horizon}
