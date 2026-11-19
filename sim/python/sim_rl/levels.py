"""Hard-episode replay: prioritized level replay (Jiang et al., 2021) over solvable catalog entries.

Each rollout, a ``fraction`` of the environments get their next episode from catalog entries
already visited, sampled by learning potential; the rest keep the curriculum, which also brings
new entries in. The score of an entry is its **positive value loss**: the mean positive GAE
advantage of its steps, an exponential average over the rollouts that visited it. It is high
where outcomes still surprise the critic (the pilot sometimes does better than expected) and
near zero both where the pilot always succeeds and where it always fails the same way. The first
version prioritized by failure rate; it replayed episodes the pilot failed identically every
time, whose advantages carry no learning signal, and the pilot got worse.

Sampling mixes rank prioritization (P ∝ rank^(-1/TEMPERATURE)) with staleness (P ∝ rollouts
since last replayed, weight STALENESS) so old scores get refreshed. Only verifier-solvable entries
are replayed. All bookkeeping stays on the GPU.
"""
from __future__ import annotations

import torch

TEMPERATURE = 0.1
STALENESS = 0.3
SCORE_EMA = 0.5                  # weight of a rollout's new score for an entry it visited


class HardEpisodes:
    def __init__(self, env, solvable, fraction, generator=None):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("replay fraction must be in (0, 1]")
        if solvable.shape != (env.catalog_size,):
            raise ValueError(f"verifier labels cover {solvable.shape[0]} entries, catalogue has {env.catalog_size}")
        self.env, self.fraction, self.generator = env, fraction, generator
        size, device = env.catalog_size, env.device
        self.solvable = solvable.to(device, torch.bool)
        self.score = torch.zeros(size, device=device)
        self.scored = torch.zeros(size, dtype=torch.bool, device=device)
        self.last_replayed = torch.zeros(size, device=device)
        self.rollouts = 0
        self.attempts = torch.zeros(size, device=device)
        self.successes = torch.zeros(size, device=device)
        self.episode = env.catalog_index.long().clone()   # catalog entry of each env's current episode
        self.count = max(1, round(env.n * fraction))

    def priorities(self):
        """Sampling weights over the catalogue (zero for unscored or unsolvable entries)."""
        eligible = self.scored & self.solvable
        weights = torch.zeros_like(self.score)
        if not bool(eligible.any()):
            return weights
        scores = torch.where(eligible, self.score, torch.full_like(self.score, -1.0))
        rank = torch.empty_like(scores)
        rank[scores.argsort(descending=True)] = torch.arange(1, scores.numel() + 1, device=scores.device,
                                                              dtype=scores.dtype)
        by_score = torch.where(eligible, rank.pow(-1.0 / TEMPERATURE), weights)
        stale = torch.where(eligible, (self.rollouts - self.last_replayed).clamp_min(0), weights)
        by_score = by_score / by_score.sum()
        by_staleness = stale / stale.sum() if bool(stale.sum() > 0) else by_score
        return (1 - STALENESS) * by_score + STALENESS * by_staleness

    def propose(self):
        """Choose the next catalog entry of ``count`` random environments (others sample as usual).
        Until any entry is scored, nothing is replayed."""
        self.rollouts += 1
        weights = self.priorities()
        if not bool(weights.sum() > 0):
            return
        draws = torch.multinomial(weights, self.count, replacement=True, generator=self.generator)
        self.last_replayed[draws] = float(self.rollouts)
        envs = torch.randperm(self.env.n, device=self.env.device, generator=self.generator)[:self.count]
        wanted = torch.full((self.env.n,), -1, dtype=torch.int32, device=self.env.device)
        wanted[envs] = draws.to(torch.int32)
        self.env.next_catalog.copy_(wanted)

    def observe(self, done, success, goal):
        """After a step: credit ended goal episodes to their entry, then track the new episodes."""
        ended = (done > 0.5) & goal
        self.attempts.index_add_(0, self.episode, ended.float())
        self.successes.index_add_(0, self.episode, (ended & (success > 0.5)).float())
        self.episode.copy_(torch.where(done > 0.5, self.env.catalog_index.long(), self.episode))

    def update_scores(self, advantages, entries, goal):
        """advantages, entries [T, n] (unnormalized GAE and each step's catalog entry), goal [T, n]
        bool: fold this rollout's positive value loss into the entries it visited."""
        weight = goal.float()
        total = torch.zeros_like(self.score).index_add_(0, entries.flatten(), (advantages.clamp_min(0) * weight).flatten())
        steps = torch.zeros_like(self.score).index_add_(0, entries.flatten(), weight.flatten())
        visited = steps > 0
        fresh = total / steps.clamp_min(1)
        self.score.copy_(torch.where(visited & self.scored, (1 - SCORE_EMA) * self.score + SCORE_EMA * fresh,
                                     torch.where(visited, fresh, self.score)))
        self.scored |= visited

    def state(self):
        return {"score": self.score.cpu(), "scored": self.scored.cpu(), "attempts": self.attempts.cpu(),
                "successes": self.successes.cpu(), "solvable": self.solvable.cpu()}
