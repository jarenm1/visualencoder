"""Training-iteration stage profile: where one PPO iteration (rollout + update) spends its time.

Takes the same flags as `train.py train` (the policy and task configuration to profile) plus
--envs/--iterations/--out. For each env count it measures
  * plain throughput: collect + update with no added synchronization, and
  * a stage breakdown: every stage below is wrapped in CUDA synchronization, so stage times are
    serialized GPU+host time (their sum exceeds the plain iteration time where stages overlap).

    python sim/python/train_profile.py --encoder mean --goal patch --memory-heads runs/perception-v4/mean.pt \
        --depth-view --explore-fraction 0.3 --envs 64 256 --out reports/train-profile.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from sim_rl.cli import TRAIN_STYLE, build_policy, commands_of, make_config, open_env, parser
from sim_rl.features import FrozenFeatures
from sim_rl.ppo import Trainer


class Stages:
    """Synchronized timers. `phase` wrappers (collect/update totals) time the whole call; stage
    wrappers time only calls made inside the rollout, so update-time reuse is not double counted."""

    def __init__(self):
        self.seconds, self.calls, self.enabled, self.phase = defaultdict(float), defaultdict(int), False, None

    def wrap(self, owner, attribute, name, phase=False):
        function = getattr(owner, attribute)

        def timed(*args, **kwargs):
            if not self.enabled or (not phase and self.phase != "collect_total"):
                return function(*args, **kwargs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if phase:
                self.phase = name
            try:
                result = function(*args, **kwargs)
                torch.cuda.synchronize()
            finally:
                if phase:
                    self.phase = None
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1
            return result

        setattr(owner, attribute, timed)


def profile(args, n, iterations):
    args.num_envs = n
    features = FrozenFeatures(args.encoder, torch.device("cuda:0"), args.encoder_precision, args.goal)
    with open_env(args, n, args.seed, TRAIN_STYLE) as env, env.execution():
        policy = build_policy(env, features, memory_heads=args.memory_heads, commands=commands_of(args),
                              depth_view=args.depth_view)
        trainer = Trainer(env, features, policy, make_config(args))
        trainer.warmup(args.seed)
        stages = Stages()
        # Rollout stages (Trainer.collect) and the update.
        stages.wrap(env, "step", "sim_step")
        stages.wrap(features, "observe", "encoder")
        stages.wrap(trainer, "_memory", "spatial_memory")
        stages.wrap(trainer, "_inputs", "actor_inputs (saliency)")
        stages.wrap(trainer, "_confusion", "confusion_telemetry")
        stages.wrap(policy, "step", "actor_gru")
        stages.wrap(policy, "value", "critic")
        if getattr(trainer, "goal_source", None) is not None:
            stages.wrap(trainer.goal_source, "step", "goal_source")
        stages.wrap(trainer, "collect", "collect_total", phase=True)
        stages.wrap(trainer, "update", "update_total", phase=True)
        for _ in range(2):                                   # compile / allocator warm-up
            trainer.update(trainer.collect())
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            trainer.update(trainer.collect())
        torch.cuda.synchronize()
        plain = (time.perf_counter() - started) / iterations
        stages.enabled = True
        for _ in range(iterations):
            trainer.update(trainer.collect())
        steps = trainer.config.rollout_steps * n
        sequences = steps // trainer.config.sequence_length
        optimizer_steps = trainer.config.epochs * -(-sequences // trainer.config.minibatch_sequences)
        per_iteration = {name: seconds / iterations * 1e3 for name, seconds in stages.seconds.items()}
        rollout_named = sum(value for name, value in per_iteration.items()
                            if name not in ("collect_total", "update_total"))
        per_iteration["collect_other (python, buffers)"] = per_iteration["collect_total"] - rollout_named
        return {"envs": n, "rollout_steps": trainer.config.rollout_steps, "steps_per_iteration": steps,
                "optimizer_steps_per_iteration": optimizer_steps, "plain_iteration_ms": plain * 1e3,
                "plain_steps_per_second": steps / plain, "stage_ms_per_iteration": per_iteration,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--envs", type=int, nargs="+", default=[64, 256])
    own.add_argument("--iterations", type=int, default=3)
    own.add_argument("--out", type=Path, required=True)
    mine, rest = own.parse_known_args()
    args = parser().parse_args(["train", *rest, "--out", "/tmp/train-profile-unused"])
    rows = []
    for n in mine.envs:
        torch.cuda.reset_peak_memory_stats()
        row = profile(args, n, mine.iterations)
        rows.append(row)
        stages = ", ".join(f"{name} {value:.0f}" for name, value in
                           sorted(row["stage_ms_per_iteration"].items(), key=lambda item: -item[1]))
        print(f"n={n}: {row['plain_steps_per_second']:.0f} steps/s ({row['plain_iteration_ms']:.0f} ms per "
              f"{row['steps_per_iteration']} steps, {row['optimizer_steps_per_iteration']} optimizer steps, peak "
              f"{row['peak_memory_gib']:.1f} GiB) | ms: {stages}", flush=True)
        torch.cuda.empty_cache()
    mine.out.parent.mkdir(parents=True, exist_ok=True)
    mine.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "flags": rest, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
