"""Visibility-bucket curriculum: GPU-resident lanes written into native field 11.

Lanes are [w_visible, w_partial, w_hidden, p_no_gps, appearance, p_coarse, p_wrong]. The bucket
weights adapt (adaptive mode); the rest are fixed: p_no_gps is the probability of an episode
without a target hint, appearance the randomized-appearance strength (0 = plain render), and
p_coarse / p_wrong the share of hinted episodes whose hint is coarse or wrong (precise otherwise).
"""
from __future__ import annotations

import torch

BUCKETS = ("visible", "partial", "hidden")
START_WEIGHTS = (0.8, 0.15, 0.05)
FLOORS = (0.10, 0.10, 0.05)
UNIFORM_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
PROMOTE, DEMOTE = 0.70, 0.25
STEP_UP, STEP_BACK = 0.05, 0.025
EMA_ALPHA = 0.05
MODES = ("adaptive", "uniform")


def fixed_lanes(device, weights=UNIFORM_WEIGHTS, no_gps=0.0, appearance=1.0, coarse=0.0, wrong=0.0):
    """Lanes for evaluation and data collection: fixed bucket weights and hint mix."""
    return torch.tensor((*weights, no_gps, appearance, coarse, wrong), device=device, dtype=torch.float32)


def update_weights(weights, ema, floors):
    """Pure tensor transition; weights, ema, floors are float32 [3] tensors on one device.

    Promotion: bucket b hands 0.05 to b+1 when its EMA success is >= 0.70 and
    the floor allows it. Demotion: bucket b+1 returns up to 0.025 to b when its
    EMA falls below 0.25 and it is above its floor. Written with torch.where so
    no host decision or sync occurs.
    """
    result = weights.clone()
    for lower in (0, 1):
        upper = lower + 1
        can_promote = (ema[lower] >= PROMOTE) & (result[lower] - STEP_UP >= floors[lower])
        move = torch.where(can_promote, torch.full_like(result[lower], STEP_UP),
                           torch.zeros_like(result[lower]))
        result[lower] = result[lower] - move
        result[upper] = result[upper] + move
        can_demote = (ema[upper] < DEMOTE) & (result[upper] > floors[upper])
        back = torch.where(can_demote,
                           torch.minimum(torch.full_like(result[upper], STEP_BACK),
                                         result[upper] - floors[upper]),
                           torch.zeros_like(result[upper]))
        result[upper] = result[upper] - back
        result[lower] = result[lower] + back
    return result / result.sum()


class Curriculum:
    """Tracks per-bucket success EMAs and weights; ``uniform`` keeps weights fixed."""

    def __init__(self, mode, device, no_gps_fraction, appearance=1.0, coarse_hint_fraction=0.0,
                 wrong_hint_fraction=0.0):
        if mode not in MODES:
            raise ValueError(f"curriculum must be one of {MODES}")
        for name, value in (("no_gps_fraction", no_gps_fraction), ("appearance", appearance),
                            ("coarse_hint_fraction", coarse_hint_fraction), ("wrong_hint_fraction", wrong_hint_fraction)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if coarse_hint_fraction + wrong_hint_fraction > 1.0:
            raise ValueError("coarse and wrong hint fractions must sum to at most 1")
        self.mode = mode
        start = START_WEIGHTS if mode == "adaptive" else UNIFORM_WEIGHTS
        self.weights = torch.tensor(start, device=device, dtype=torch.float32)
        self.floors = torch.tensor(FLOORS, device=device, dtype=torch.float32)
        self.fixed = torch.tensor([no_gps_fraction, appearance, coarse_hint_fraction, wrong_hint_fraction],
                                  device=device, dtype=torch.float32)
        self.ema = torch.zeros(3, device=device, dtype=torch.float32)
        self.seen = torch.zeros(3, device=device, dtype=torch.bool)

    def lanes(self):
        """float32[7] device tensor: bucket weights then the fixed lanes (device cat, no host transfer)."""
        return torch.cat((self.weights, self.fixed))

    def observe(self, counts, successes):
        """counts, successes: float32 [3] tensors of completed episodes per bucket this rollout."""
        observed = counts > 0
        rate = 1 - (1 - EMA_ALPHA) ** counts
        mean = successes / counts.clamp_min(1)
        updated = torch.where(self.seen, self.ema + rate * (mean - self.ema), mean)
        self.ema = torch.where(observed, updated, self.ema)
        self.seen = self.seen | observed

    def advance(self):
        if self.mode == "adaptive":
            self.weights = update_weights(self.weights, self.ema, self.floors)

    def write(self, env):
        env.write_curriculum(self.lanes())
