"""Held-out hard set: the pilot on the test district's difficult, verified-solvable episodes.

Cohorts are drawn from the test catalogue entries the verifier (verifier.py) marks solvable, so a
failure is the pilot's. Each cohort fixes every environment's first episode through the native
catalogue override (a fixed sample of ``eval_envs`` entries, the same for every checkpoint) and
sets the hint mix:

  standard      all entries, visibility buckets in thirds no hint / precise hint
  hidden        target not visible from the start         no hint
  far           start >= FAR_M from the target box        no hint
  hidden_far    both                                      no hint
  coarse_hint   all entries                              coarse hint (10-40 m radius)
  wrong_hint    all entries                              wrong hint (30-80 m off, claims 1-5 m)
  look_alike    a same-class object within LOOK_ALIKE_M   no hint
  category      all entries, goal starts as another same-type object's reference   no hint

    train.py eval-hard --checkpoint runs/drone13-v5-seed0/best.pt --verifier runs/verifier/test.pt --out ...
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .category import CategoryReferences
from .evaluation import evaluate

FAR_M = 40.0
LOOK_ALIKE_M = 15.0
HIDDEN = 2
# name -> (entry filter, evaluate() hint keywords)
COHORTS = {
    "standard_no_hint": ("all", {"no_gps": 1.0}),
    "standard_precise": ("all", {"no_gps": 0.0}),
    "hidden": ("hidden", {"no_gps": 1.0}),
    "far": ("far", {"no_gps": 1.0}),
    "hidden_far": ("hidden_far", {"no_gps": 1.0}),
    "coarse_hint": ("all", {"no_gps": 0.0, "coarse": 1.0}),
    "wrong_hint": ("all", {"no_gps": 0.0, "wrong": 1.0}),
    "look_alike": ("look_alike", {"no_gps": 1.0}),
    "category": ("all", {"no_gps": 1.0, "category": True}),
}


def cohort_entries(labels, layout, count, seed):
    """name -> int32 [count] catalogue indices (sampled with replacement when a cohort is smaller)."""
    solvable = labels["solvable"].numpy().astype(bool)
    start, center, half = (labels[key].numpy() for key in ("start", "center", "half"))
    bucket, objects = labels["meta"][:, 1].numpy().astype(int), labels["meta"][:, 0].numpy().astype(int)
    distance = np.linalg.norm(np.maximum(np.abs(start - center) - half, 0), axis=-1)
    lo = np.array([o["min"] for o in layout]); hi = np.array([o["max"] for o in layout])
    classes = np.array([o["class"] for o in layout])
    look_alike = np.zeros(len(solvable), bool)
    for entry in np.flatnonzero(solvable):
        target = objects[entry]
        gap = np.linalg.norm(np.maximum(np.maximum(lo - hi[target], lo[target] - hi), 0), axis=-1)
        same = (classes == classes[target]) & (gap <= LOOK_ALIKE_M)
        same[target] = False
        look_alike[entry] = same.any()
    filters = {"all": solvable, "hidden": solvable & (bucket == HIDDEN), "far": solvable & (distance >= FAR_M),
               "hidden_far": solvable & (bucket == HIDDEN) & (distance >= FAR_M), "look_alike": solvable & look_alike}
    rng = np.random.default_rng(seed)
    chosen = {}
    for name, (kind, _) in COHORTS.items():
        if kind == "all":
            # Visibility buckets in equal thirds, as the native uniform curriculum samples them (88%
            # of test entries are hidden starts).
            pools = [np.flatnonzero(solvable & (bucket == b)) for b in range(3)]
            shares = np.diff(np.linspace(0, count, 4).round().astype(int))
            picks = [rng.choice(pool, share, replace=len(pool) < share) for pool, share in zip(pools, shares)]
            chosen[name] = rng.permutation(np.concatenate(picks)).astype(np.int32)
        else:
            pool = np.flatnonzero(filters[kind])
            chosen[name] = rng.choice(pool, count, replace=len(pool) < count).astype(np.int32)
    return chosen, {kind: int(mask.sum()) for kind, mask in filters.items()}


def evaluate_hard(env, features, policy, labels, seed, goal_refresh_steps=0, heuristic=None):
    layout = json.loads((env.city_path / "layout.json").read_text())["objects"]
    chosen, sizes = cohort_entries(labels, layout, env.n, seed)
    references = CategoryReferences(labels["meta"][:, 0].numpy().astype(np.int64), layout, env.device)
    results = {"cohort_sizes": sizes}
    for name, (_, hints) in COHORTS.items():
        entries = torch.as_tensor(chosen[name], device=env.device)
        hints = dict(hints)
        category = references if hints.pop("category", False) else None
        results[name] = evaluate(env, features, policy=policy, heuristic=heuristic, seed=seed,
                                 goal_refresh_steps=goal_refresh_steps, entries=entries, category=category, **hints)
    return results
