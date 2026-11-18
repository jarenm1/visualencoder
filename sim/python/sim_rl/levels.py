"""Hard-episode replay: post-training on the pilot's own failures (prioritized level replay).

Online failure mining: every completed goal episode adds one attempt (and its success) to its
catalog entry. Each rollout, a ``fraction`` of the environments get their next episode drawn
from the solvable entries (verifier.py) with probability proportional to the failure rate plus a
floor; entries never tried count as failures, so the sampler also explores the catalog. The
draw is written to the native ``next_catalog`` field and takes effect at each environment's next
reset; the rest keep sampling the curriculum as usual, which keeps the pilot from forgetting.
All bookkeeping stays on the GPU.
"""
from __future__ import annotations

import torch

FLOOR = 0.05                     # priority of an entry that always succeeds (relative to 1.0 for failing)


class HardEpisodes:
    def __init__(self, env, solvable, fraction, generator=None):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("replay fraction must be in (0, 1]")
        if solvable.shape != (env.catalog_size,):
            raise ValueError(f"verifier labels cover {solvable.shape[0]} entries, catalogue has {env.catalog_size}")
        self.env, self.fraction, self.generator = env, fraction, generator
        self.solvable = solvable.to(env.device, torch.float32)
        self.attempts = torch.zeros(env.catalog_size, device=env.device)
        self.successes = torch.zeros(env.catalog_size, device=env.device)
        self.episode = env.catalog_index.long().clone()   # catalog entry of each env's current episode
        self.count = max(1, round(env.n * fraction))

    def priorities(self):
        failure = torch.where(self.attempts > 0, 1 - self.successes / self.attempts.clamp_min(1),
                              torch.ones_like(self.attempts))
        return (failure + FLOOR) * self.solvable

    def propose(self):
        """Choose the next catalog entry of ``count`` random environments (others sample as usual)."""
        draws = torch.multinomial(self.priorities(), self.count, replacement=True, generator=self.generator)
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

    def state(self):
        return {"attempts": self.attempts.cpu(), "successes": self.successes.cpu(), "solvable": self.solvable.cpu()}
