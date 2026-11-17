"""Binomial confidence intervals and rank metrics for reported episode rates and probes."""
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


def auroc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney rank statistic (ties get average ranks).

    scores: sequence of floats; labels: sequence of 0/1. None when a class is absent.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average = (start + end) / 2 + 1  # 1-based average rank of the tie group
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    positive_ranks = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores, labels):
    """Mean precision at each positive when ranked by descending score; None without positives."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    positives = sum(1 for label in labels if label)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    hits, total = 0, 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives
