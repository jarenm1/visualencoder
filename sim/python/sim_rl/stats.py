"""Binomial confidence intervals for reported episode rates."""
from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(successes, trials, z=Z95):
    """Wilson score interval [low, high] for a binomial proportion; None when trials == 0."""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]
