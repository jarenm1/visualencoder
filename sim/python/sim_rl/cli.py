"""Command-line entry points: train, eval, probe, benchmark, profile, memory-check, verify-trace."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from .curriculum import MODES as CURRICULUM_MODES
from .evaluation import evaluate, evaluate_explore, heuristics_for
from .features import ARMS, GOALS, PRECISIONS, FrozenFeatures, checkpoint_features
from .model import ActorCritic
from .native import ABI_VERSION, CONTROL_MODES, DEFAULT_CITY, HOVER_ACTIONS, TASK, CudaEnv
from .perception import load_heads
from .ppo import BUCKET_WIDTH, LOSS_NAMES, LOSS_WIDTH, RESULT_WIDTH, ROLLOUT_WIDTH, Config, Trainer
from .probe import collect_frames, protocol_of, run_encoder, write_markdown
from .profiling import capture, verify_trace

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRARY = ROOT / ".sim-target/training/libsim_env.so"
CHECKPOINT_FORMAT = "sim-drone-ppo-v3"
TRAIN_STYLE, VALIDATION_STYLE, TEST_STYLE = 0, 1, 2
VALIDATION_SEED, TEST_SEED = 10_001, 20_002
PROBE_TRAIN_SEED = 30_003
MIN_EPISODES = 256
DEFAULTS = Config()
DEFAULT_ENCODERS = "pretrained,baseline,mean"
ACTOR_INPUTS = {
    "visual": "target-saliency(frozen CLS + patch tokens, goal-crop embedding: global or patch) + proprio(19)",
    "state": "privileged(29) + proprio(19)",
    "hint": "proprio(19)",
}
CRITIC_INPUTS = "privileged(29) + proprio(19)"
# Recorded and compared on load; the library hash is provenance only (rebuilds keep the contract).
CHECKED_KEYS = ("task", "abi_version", "action_dim", "horizon", "control_hz", "control_mode",
                "width", "height", "policy_mode", "encoder", "feature_dim", "encoder_fingerprint",
                "preprocessing", "goal", "native_task_contract")
# Values of keys added after earlier checkpoints were written.
UNRECORDED = {"goal": "global"}
LOSS_AT = ROLLOUT_WIDTH
BUCKET_AT = LOSS_AT + LOSS_WIDTH
USED_AT = BUCKET_AT + BUCKET_WIDTH
NEXT_AT = USED_AT + 3
EMA_AT = NEXT_AT + 3


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mode_of(features):
    return "visual" if features.visual else features.spec


def build_policy(env, features, saved=None, memory_heads=None, commands=1):
    """Visual arms attend over the frozen patch grid; the grid is height//16 x width//16.
    ``saved`` loads a checkpoint (its recorded spatial memory and commands included); ``memory_heads``
    starts a new spatial-memory policy from frozen perception heads trained on this exact encoder;
    ``commands`` > 1 makes a new policy command-conditioned (goal, explore)."""
    memory = memory_heads is not None or bool(saved and saved["metadata"].get("memory"))
    commands = saved["metadata"].get("commands", 1) if saved is not None else commands
    policy = ActorCritic(mode_of(features), features.dim, features.goal_dim, env.height // 16, env.width // 16,
                         env.action_dim, features.goal, memory, commands).to(env.device)
    if memory_heads is not None:
        policy.perception.load_state_dict(load_heads(memory_heads, features, env.device).state_dict())
    if saved is not None:
        policy.load_state_dict(saved["model"])
    return policy


def native_provenance(env, features):
    mode = mode_of(features)
    return {
        "format": CHECKPOINT_FORMAT, "task": TASK, "abi_version": ABI_VERSION,
        "action_dim": env.action_dim, "horizon": env.horizon, "control_hz": env.control_hz,
        "control_mode": env.control_mode, "catalog_size": env.catalog_size,
        "width": env.width, "height": env.height,
        "policy_mode": mode, "encoder": features.spec, "feature_dim": features.dim,
        "encoder_fingerprint": features.fingerprint, "preprocessing": features.preprocessing, "goal": features.goal,
        "actor_inputs": ACTOR_INPUTS[mode], "critic_inputs": CRITIC_INPUTS,
        "native_task_contract": env.task_contract, "native_library_sha256": sha256(env.library),
    }


def provenance(env, features, training):
    return {**native_provenance(env, features), **training}


def training_settings(args):
    from .memory import CELL, EGO, EGO_RADIUS, GRID  # noqa: PLC0415 (keeps the CLI import graph flat)
    memory = None if args.memory_heads is None else {
        "heads": str(args.memory_heads), "heads_sha256": sha256(args.memory_heads),
        "grid": GRID, "cell_m": CELL, "ego": EGO, "ego_radius_m": EGO_RADIUS,
        "pose": "dead reckoning from noisy gyro and body velocity", "depth": "frozen monocular patch heads"}
    return {"curriculum": args.curriculum, "no_gps_fraction": args.no_gps_fraction,
            "aux_coefficient": args.aux_coefficient, "appearance": args.appearance,
            "goal_refresh_steps": args.goal_refresh_steps,
            "goal_box_source": "simulator" if args.goal_refresh_steps else None,
            "start_distance": args.start_distance, "memory": memory,
            "explore_fraction": args.explore_fraction, "commands": commands_of(args)}


def commands_of(args):
    return 2 if args.explore_fraction > 0 else 1


def checkpoint(policy, metadata, steps):
    return {"format": CHECKPOINT_FORMAT, "metadata": {**metadata, "steps": steps},
            "model": policy.state_dict()}


# The previous task, which explicit visual-policy evaluation may transfer from. v4 changed the
# horizon (15 -> 40 s), ended episodes at the first sustained hold, and added camera effects and
# label fields; flight dynamics, sensors and encoders are unchanged.
TRANSFER_SOURCE = ("city-drone-goal-v3", 6)
TRANSFER_CHANGED = {"task", "abi_version", "success", "reward", "horizon", "camera_effects", "patch_labels"}


def load_checkpoint(path, *, for_task_transfer=False):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint; earlier checkpoints "
                         f"cannot be used for {TASK} ABI {ABI_VERSION}")
    metadata = saved["metadata"]
    source = (metadata.get("task"), metadata.get("abi_version"))
    if source != (TASK, ABI_VERSION) and not (for_task_transfer and source == TRANSFER_SOURCE):
        raise ValueError(f"checkpoint targets an incompatible native ABI or task; retrain on sim_env ABI v{ABI_VERSION}")
    return saved


def validate_checkpoint(saved, env, features, *, for_task_transfer=False):
    recorded = saved["metadata"]
    expected = native_provenance(env, features)
    transfer = (recorded.get("task"), recorded.get("abi_version")) != (TASK, ABI_VERSION)
    if transfer:
        if not (for_task_transfer and features.visual and
                (recorded.get("task"), recorded.get("abi_version")) == TRANSFER_SOURCE):
            raise ValueError(f"only explicit visual-policy evaluation may transfer a {TRANSFER_SOURCE} checkpoint")
        # Visual actors never read privileged hold progress; controls, sensors and encoders must match.
        source_contract = {k: v for k, v in recorded["native_task_contract"].items() if k not in TRANSFER_CHANGED}
        target_contract = {k: v for k, v in expected["native_task_contract"].items() if k not in TRANSFER_CHANGED}
        if source_contract != target_contract:
            raise ValueError("task-transfer checkpoint differs in dynamics, sensors or observation contract")
    for key in CHECKED_KEYS:
        if transfer and key in {"task", "abi_version", "horizon", "native_task_contract"}:
            continue
        if recorded.get(key, UNRECORDED.get(key)) != expected[key]:
            raise ValueError(f"checkpoint {key} differs from the native/feature configuration: "
                             f"recorded {recorded.get(key, UNRECORDED.get(key))!r}, current {expected[key]!r}")


def make_config(args):
    return Config(total_steps=args.total_steps, rollout_steps=args.rollout_steps,
                  sequence_length=args.sequence_length, epochs=args.epochs,
                  minibatch_sequences=args.minibatch_sequences, learning_rate=args.learning_rate,
                  gamma=args.gamma, gae_lambda=args.gae_lambda, clip=args.clip, entropy=args.entropy,
                  value_coefficient=args.value_coefficient, max_grad_norm=args.max_grad_norm,
                  curriculum=args.curriculum, aux_coefficient=args.aux_coefficient,
                  no_gps_fraction=args.no_gps_fraction, appearance=args.appearance,
                  goal_refresh_steps=args.goal_refresh_steps, explore_fraction=args.explore_fraction)


def open_env(args, n, seed, style):
    start = getattr(args, "start_distance", None)
    return CudaEnv(args.library, n, seed, args.width, args.height, style, args.control, args.city,
                   start_distance=tuple(start) if start else None)


def open_env_checkpoint(args, metadata, seed, style):
    return CudaEnv(args.library, args.eval_envs, seed, metadata["width"], metadata["height"], style,
                   metadata["control_mode"], args.city)


def check_episodes(result):
    if result["episodes"] < MIN_EPISODES:
        raise RuntimeError(f"evaluation produced {result['episodes']} episodes; at least {MIN_EPISODES} required")
    return result


def gps_modes(env, features, policy, seed, heuristics=True, goal_refresh_steps=0):
    """Policy (and optionally blind heuristics) in both GPS modes: GPS available (p_no_gps=0) and GPS denied (p_no_gps=1)."""
    results = {}
    for label, no_gps in (("gps", 0.0), ("no_gps", 1.0)):
        results[label] = {"policy": check_episodes(evaluate(
            env, features, policy=policy, seed=seed, no_gps=no_gps, goal_refresh_steps=goal_refresh_steps))}
        if heuristics:
            results[label]["heuristics"] = {
                name: check_episodes(evaluate(env, features, heuristic=name, seed=seed, no_gps=no_gps))
                for name in heuristics_for(env.control_mode)}
    return results


def selection_success(result, config):
    """Checkpoint selection: GPS success alone when no GPS episodes are trained, else the mean of both modes."""
    if config.no_gps_fraction == 0:
        return result["gps"]["policy"]["success"]
    return (result["gps"]["policy"]["success"] + result["no_gps"]["policy"]["success"]) / 2


def emit(payload, path=None):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def timed(fn, iterations):
    """Mean GPU seconds per call on the current (env) stream; one host sync at the end."""
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000 / iterations


def training_record(steps, elapsed, values):
    """Decodes the fixed finish_rollout layout (see ppo.RESULT_WIDTH)."""
    completed = values[1]
    return {
        "steps": steps, "elapsed_seconds": elapsed,
        "reward_mean": values[0], "completed_episodes": completed,
        "success": values[2], "collision": values[3],
        "episode_return": values[4] / completed if completed else None,
        "explore_completed": values[5],
        "explore_tiles_mean": values[6] / values[5] if values[5] else None,
        **dict(zip(LOSS_NAMES, values[LOSS_AT:BUCKET_AT])),
        "bucket_completed": values[BUCKET_AT:BUCKET_AT + 3],
        "bucket_success": values[BUCKET_AT + 3:BUCKET_AT + 6],
        "bucket_collision": values[BUCKET_AT + 6:BUCKET_AT + 9],
        "bucket_out_of_bounds": values[BUCKET_AT + 9:BUCKET_AT + 12],
        "weights_used": values[USED_AT:NEXT_AT], "weights_next": values[NEXT_AT:EMA_AT],
        "success_ema": values[EMA_AT:],
    }


def train_command(args, started):
    seed_all(args.seed)
    config = make_config(args)
    run = args.out
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    features = FrozenFeatures(args.encoder, device, args.encoder_precision, args.goal)
    budget_end = started + args.wall_seconds if args.wall_seconds else float("inf")
    curves = (run / "learning-curves.jsonl").open("w")
    validations = []
    best_success, best_path, last_path = -1.0, run / "best.pt", run / "last.pt"
    with open_env(args, args.num_envs, args.seed, TRAIN_STYLE) as env, \
            open_env(args, args.eval_envs, VALIDATION_SEED, VALIDATION_STYLE) as validation_env:
        with env.execution():
            policy = build_policy(env, features, memory_heads=args.memory_heads, commands=commands_of(args))
            trainer = Trainer(env, features, policy, config)
            trainer.warmup(args.seed)
            metadata = provenance(env, features, training_settings(args))

        def validate():
            with validation_env.execution():
                policy.eval()
                result = gps_modes(validation_env, features, policy, VALIDATION_SEED, heuristics=False,
                                   goal_refresh_steps=config.goal_refresh_steps)
                if policy.commands > 1:
                    result["explore"] = evaluate_explore(validation_env, features, policy, VALIDATION_SEED)
                policy.train()
            result["selection_success"] = selection_success(result, config)
            return result

        started_eval = time.monotonic()
        validate()  # Initial calibration: the evaluation duration sets the final reserve.
        eval_seconds = time.monotonic() - started_eval
        reserve = max(args.final_reserve_seconds, eval_seconds * 3.8 + 5)
        next_validation = args.validation_every
        stale = 0
        while trainer.steps < config.total_steps and time.monotonic() < budget_end - reserve:
            with env.execution():
                length = trainer.collect(config.rollout_steps)
                losses = trainer.update(length)
                values = trainer.finish_rollout(length, losses).cpu().tolist()
            if len(values) != RESULT_WIDTH:
                raise RuntimeError(f"finish_rollout returned {len(values)} values; expected {RESULT_WIDTH}")
            curves.write(json.dumps(training_record(trainer.steps, time.monotonic() - started, values)) + "\n")
            curves.flush()
            if trainer.steps >= next_validation:
                next_validation += args.validation_every
                result = validate()
                validations.append({"steps": trainer.steps, "result": result})
                emit({"validations": validations}, run / "validation.json")
                print(json.dumps({"validation_steps": trainer.steps, "result": result}), flush=True)
                if result["selection_success"] > best_success:
                    best_success, stale = result["selection_success"], 0
                    torch.save(checkpoint(policy, metadata, trainer.steps), best_path)
                else:
                    stale += 1
                torch.save(checkpoint(policy, metadata, trainer.steps), last_path)
                if args.patience and stale >= args.patience:
                    print(json.dumps({"stopped": "validation stalled", "steps": trainer.steps,
                                      "validations_without_improvement": stale}), flush=True)
                    break
        curves.close()
        final = validate()
        validations.append({"steps": trainer.steps, "result": final, "final": True})
        if final["selection_success"] > best_success:
            best_success = final["selection_success"]
            torch.save(checkpoint(policy, metadata, trainer.steps), best_path)
        torch.save(checkpoint(policy, metadata, trainer.steps), last_path)
    emit({"validations": validations}, run / "validation.json")

    saved = load_checkpoint(best_path)
    with open_env(args, args.eval_envs, TEST_SEED, TEST_STYLE) as test_env:
        with test_env.execution():
            policy = build_policy(test_env, features, saved)
            test = gps_modes(test_env, features, policy, TEST_SEED,
                             goal_refresh_steps=config.goal_refresh_steps)
            fixed_goal_test = (gps_modes(test_env, features, policy, TEST_SEED, heuristics=False)
                               if config.goal_refresh_steps else None)
    elapsed = time.monotonic() - started
    final_test = {"checkpoint": "best.pt", "checkpoint_sha256": sha256(best_path),
                  "control_mode": metadata["control_mode"], **test, "fixed_goal": fixed_goal_test}
    emit(final_test, run / "final-test.json")
    summary = {
        "task": TASK, "abi_version": ABI_VERSION, "control_mode": metadata["control_mode"],
        "policy_mode": metadata["policy_mode"], "encoder": metadata["encoder"], "seed": args.seed,
        "width": metadata["width"], "height": metadata["height"], "control_hz": metadata["control_hz"],
        "city_fingerprint": metadata["native_task_contract"]["city_fingerprint"],
        "steps": trainer.steps, "no_gps_fraction": args.no_gps_fraction, "aux_coefficient": args.aux_coefficient,
        "goal_refresh_steps": config.goal_refresh_steps,
        "goal_box_source": metadata["goal_box_source"],
        "fixed_goal_test_success": ({label: fixed_goal_test[label]["policy"]["success"]
                                     for label in ("gps", "no_gps")} if fixed_goal_test else None),
        "best_validation_selection_success": best_success,
        "test_success": {label: test[label]["policy"]["success"] for label in ("gps", "no_gps")},
        "test_success_wilson95": {label: test[label]["policy"]["success_wilson95"] for label in ("gps", "no_gps")},
        "elapsed_seconds_including_initialization_training_evaluation": elapsed,
        "wall_budget_seconds": args.wall_seconds or None,
        "within_wall_budget": elapsed <= args.wall_seconds if args.wall_seconds else None,
        "compile_excluded": True, "convergence_claim": False,
    }
    emit(summary, run / "summary.json")
    write_report(run, summary, final_test)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_report(run, summary, test):
    def row(name, result):
        low, high = result["success_wilson95"] or (None, None)
        return (f"| {name} | {result['episodes']} | {result['success']:.3f} "
                f"[{low:.3f}, {high:.3f}] | {result['collision']:.3f} | "
                f"{result['out_of_bounds']:.3f} | {result['timeout']:.3f} |")

    lines = ["# City drone goal test", "", f"Task: `{summary['task']}`, ABI {summary['abi_version']}, "
             f"control `{summary['control_mode']}`, policy mode `{summary['policy_mode']}`, "
             f"steps {summary['steps']}, no-GPS fraction {summary['no_gps_fraction']}."]
    if summary["goal_refresh_steps"]:
        lines += ["", f"Goal refresh: every {summary['goal_refresh_steps']} steps, using simulator boxes. "
                  "This is oracle perception, not measured LLM grounding."]
    for label, title in (("gps", "GPS available"), ("no_gps", "GPS denied")):
        section = test[label]
        lines += ["", f"## {title}", "",
                  "| arm | episodes | success [Wilson 95%] | collision | out of bounds | timeout |",
                  "|---|---|---|---|---|---|", row(summary["policy_mode"], section["policy"])]
        lines += [row(name, result) for name, result in section["heuristics"].items()]
        if test.get("fixed_goal") is not None:
            lines.append(row("policy, fixed initial goal", test["fixed_goal"][label]["policy"]))
        lines += ["", "Per visibility bucket (policy):", ""]
        lines += [f"- {name}: episodes {result['episodes']}, success {result.get('success')}"
                  for name, result in section["policy"]["buckets"].items()]
        lines += ["", "Per start (policy):", ""]
        lines += [f"- {name}: episodes {result['episodes']}, success {result.get('success')}"
                  for name, result in section["policy"]["start_landed"].items()]
    (run / "report.md").write_text("\n".join(lines) + "\n")


def eval_command(args, started):
    saved = load_checkpoint(args.checkpoint)
    metadata = saved["metadata"]
    device = torch.device("cuda:0")
    features = checkpoint_features(metadata, device)
    seed, style = (VALIDATION_SEED, VALIDATION_STYLE) if args.split == "validation" else (TEST_SEED, TEST_STYLE)
    with open_env_checkpoint(args, metadata, seed, style) as env:
        with env.execution():
            validate_checkpoint(saved, env, features)
            policy = build_policy(env, features, saved)
            results = gps_modes(env, features, policy, seed,
                                goal_refresh_steps=metadata.get("goal_refresh_steps", 0))
    payload = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
               "split": args.split, "control_mode": metadata["control_mode"], **results}
    emit(payload, args.out)


def cohorts_command(args, started):
    from .cohorts import run
    run(args)


def mine_command(args, started):
    from .mine import run
    run(args)


def post_train_command(args, started):
    from .posttrain import post_train
    post_train(args)


def confusion_probe_command(args, started):
    from .posttrain import confusion_probe
    confusion_probe(args)


def count_eval_command(args, started):
    from .counting import count_eval_command as run
    run(args, started)


def perception_command(args, started):
    from .perception import perception_command as run
    run(args, started)


def probe_command(args, started):
    if not args.encoders:
        raise SystemExit("--encoders must name at least one visual arm")
    device = torch.device("cuda:0")
    arms = {}
    for spec in args.encoders:
        arm = FrozenFeatures(spec, device, args.encoder_precision)
        if not arm.visual:
            raise SystemExit(f"probe needs a visual encoder; {spec} has none")
        arms[spec] = arm
    # Catalog goal tables render through the env, so each is built before that env's first reset.
    with open_env(args, args.probe_envs, PROBE_TRAIN_SEED, TRAIN_STYLE) as env:
        with env.execution():
            train_tables = {spec: arm.catalog_goals(env, args.appearance) for spec, arm in arms.items()}
            train_frames = collect_frames(env, args.train_frames, PROBE_TRAIN_SEED, args.seed, args.appearance)
            class_index = {name: env.layout_classes.index(name) for name in env.target_classes}
    with open_env(args, args.probe_envs, TEST_SEED, TEST_STYLE) as env:
        with env.execution():
            test_tables = {spec: arm.catalog_goals(env, args.appearance) for spec, arm in arms.items()}
            test_frames = collect_frames(env, args.test_frames, TEST_SEED, args.seed + 1, args.appearance)
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate}
    encoders = {}
    for spec in args.encoders:
        seed_all(args.seed)
        encoders[spec] = run_encoder(arms.pop(spec), train_frames, train_tables.pop(spec), test_frames,
                                     test_tables.pop(spec), args.height // 16, args.width // 16, config,
                                     args.seed, class_index)
    train_count, test_count = train_frames.images.shape[0], test_frames.images.shape[0]
    payload = {"protocol": protocol_of(train_count, test_count), "train_frames": train_count,
               "test_frames": test_count, "train_seed": PROBE_TRAIN_SEED, "test_seed": TEST_SEED,
               "control_mode": args.control, "width": args.width, "height": args.height,
               "seed": args.seed, "config": config, "appearance": args.appearance,
               "target_classes": list(class_index), "encoders": encoders}
    args.out.mkdir(parents=True, exist_ok=True)
    emit(payload, args.out / "probe.json")
    (args.out / "probe.md").write_text(write_markdown(payload))


def benchmark_command(args, started):
    rows = []
    for n in args.batch_sizes:
        seed_all(args.seed)
        features = FrozenFeatures(args.encoder, torch.device("cuda:0"), args.encoder_precision, args.goal)
        config = make_config(args)
        with open_env(args, n, args.seed, TRAIN_STYLE) as env:
            with env.execution():
                policy = build_policy(env, features, memory_heads=args.memory_heads, commands=commands_of(args))
                trainer = Trainer(env, features, policy, config)
                trainer.warmup(args.seed)
                hover = torch.tensor(HOVER_ACTIONS[env.control_mode], device=env.device).expand(n, -1).contiguous()
                native_step = timed(lambda: env.step(hover), args.iterations)
                encoder = timed(lambda: features.observe(env), args.iterations) if features.visual else None
                collect = timed(lambda: trainer.collect(config.rollout_steps), 1)
                update = timed(lambda: trainer.update(config.rollout_steps), 1)
        transitions = config.rollout_steps * n
        rows.append({"num_envs": n, "native_step_seconds": native_step, "encoder_seconds": encoder,
                     "collect_seconds": collect, "update_seconds": update,
                     "transitions_per_second": transitions / (collect + update)})
    emit({"arm": args.encoder, "policy_mode": mode_of(features), "control_mode": args.control,
          "rows": rows}, args.out)


def profile_command(args, started):
    seed_all(args.seed)
    features = FrozenFeatures(args.encoder, torch.device("cuda:0"), args.encoder_precision, args.goal)
    config = make_config(args)
    with open_env(args, args.num_envs, args.seed, TRAIN_STYLE) as env:
        with env.execution():
            policy = build_policy(env, features, memory_heads=args.memory_heads, commands=commands_of(args))
            trainer = Trainer(env, features, policy, config)
            trainer.warmup(args.seed)
            result = capture(trainer, args.out, args.iterations)
    emit(result)


def memory_check_command(args, started):
    seed_all(args.seed)
    features = FrozenFeatures(args.encoder, torch.device("cuda:0"), args.encoder_precision, args.goal)
    config = make_config(args)
    allocated = []
    with open_env(args, args.num_envs, args.seed, TRAIN_STYLE) as env:
        with env.execution():
            policy = build_policy(env, features, memory_heads=args.memory_heads, commands=commands_of(args))
            trainer = Trainer(env, features, policy, config)
            trainer.warmup(args.seed)
            for _ in range(args.repeats):
                length = trainer.collect(config.rollout_steps)
                trainer.update(length)
                env.stream.synchronize()
                allocated.append(torch.cuda.memory_allocated(env.device))
    growth = allocated[-1] - allocated[0]
    result = {"allocated_bytes": allocated, "growth_bytes": growth,
              "growth_tolerance_bytes": args.growth_tolerance_bytes}
    emit(result, args.out)
    if growth > args.growth_tolerance_bytes:
        raise RuntimeError(f"device memory grew by {growth} bytes across repeated iterations")


def verify_trace_command(args, started):
    emit({scope: verify_trace(args.trace, scope) for scope in ("sim.collect", "sim.ppo_update")})


def add_native(parser):
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)


def add_control(parser):
    parser.add_argument("--control", choices=tuple(CONTROL_MODES), default="velocity",
                        help="action interface: velocity (default, PX4 offboard style) or rates")


def add_precision(parser):
    parser.add_argument("--encoder-precision", choices=PRECISIONS, default="fp16",
                        help="frozen-encoder inference: fp16 (TensorRT engine, default) or bf16 (torch.compile; "
                             "the numerics of pre-TensorRT checkpoints). Evaluation reads it from the checkpoint.")


def add_learner(parser):
    parser.add_argument("--encoder", default="mean", help=f"arm: {', '.join(ARMS)} or run:<dir>")
    parser.add_argument("--goal", choices=GOALS, default="patch",
                        help="goal representation: patch (goal-crop patches matched against frame patches) "
                             "or global (one CLS + mean-patch vector; earlier checkpoints)")
    add_precision(parser)
    parser.add_argument("--memory-heads", type=Path,
                        help="perception heads (train.py perception) for this encoder: enables the spatial memory")
    parser.add_argument("--explore-fraction", type=float, default=DEFAULTS.explore_fraction,
                        help="fraction of training episodes given the explore command (coverage reward) "
                             "instead of the goal; > 0 makes the policy command-conditioned")
    parser.add_argument("--curriculum", choices=CURRICULUM_MODES, default="adaptive")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--no-gps-fraction", type=float, default=DEFAULTS.no_gps_fraction,
                        help="probability that a training episode runs without GPS")
    parser.add_argument("--aux-coefficient", type=float, default=DEFAULTS.aux_coefficient,
                        help="weight of the visibility + localization auxiliary loss (visual arms)")
    parser.add_argument("--appearance", type=float, default=DEFAULTS.appearance,
                        help="randomized appearance strength during training (0 = plain render)")
    parser.add_argument("--goal-refresh-steps", type=int, default=DEFAULTS.goal_refresh_steps,
                        help="refresh the goal from a visible simulator target box every N control steps "
                             "(0 = fixed reference); no scripted actions")
    parser.add_argument("--start-distance", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="start box-distance range in metres for training, validation and test "
                             "(default: the native 8-30 m)")
    parser.add_argument("--total-steps", type=int, default=DEFAULTS.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULTS.rollout_steps)
    parser.add_argument("--sequence-length", type=int, default=DEFAULTS.sequence_length)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--minibatch-sequences", type=int, default=DEFAULTS.minibatch_sequences)
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--gamma", type=float, default=DEFAULTS.gamma)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULTS.gae_lambda)
    parser.add_argument("--clip", type=float, default=DEFAULTS.clip)
    parser.add_argument("--entropy", type=float, default=DEFAULTS.entropy)
    parser.add_argument("--value-coefficient", type=float, default=DEFAULTS.value_coefficient)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULTS.max_grad_norm)


def parser():
    root = argparse.ArgumentParser(description=f"City drone visual-goal learner ({TASK}, ABI v{ABI_VERSION})")
    commands = root.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    add_native(train)
    add_control(train)
    add_learner(train)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--eval-envs", type=int, default=384)
    train.add_argument("--validation-every", type=int, default=1_000_000)
    train.add_argument("--wall-seconds", type=float, default=3600,
                       help="wall budget in seconds; 0 trains to --total-steps without a time cutoff")
    train.add_argument("--final-reserve-seconds", type=float, default=300)
    train.add_argument("--patience", type=int, default=0,
                       help="stop after this many validations without a new best (0: never)")
    train.set_defaults(func=train_command)

    evaluation = commands.add_parser("eval")
    add_native(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--eval-envs", type=int, default=384)
    evaluation.add_argument("--out", type=Path)
    evaluation.set_defaults(func=eval_command)

    mine = commands.add_parser("mine", help="mine perception failures of a frozen visual policy")
    add_native(mine)
    mine.add_argument("--checkpoint", type=Path, required=True)
    mine.add_argument("--allow-task-transfer", action="store_true",
                      help="mine with an older-task visual policy (perception only; observation contract checked)")
    mine.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train", "validation", "test"])
    mine.add_argument("--num-envs", type=int, default=256)
    mine.add_argument("--rounds", type=int, default=2, help="fresh resets per split")
    mine.add_argument("--horizons", type=int, default=2, help="episode horizons flown per round")
    mine.add_argument("--stride", type=int, default=15, help="control steps between labelled samples")
    mine.add_argument("--normal-rate", type=float, default=0.1, help="fraction of correct samples kept")
    mine.add_argument("--appearance", type=float, default=1.0)
    mine.add_argument("--out", type=Path, required=True)
    mine.set_defaults(func=mine_command)

    post = commands.add_parser("post-train", help="fine-tune a distilled student + policy saliency head on mined failures")
    post.add_argument("--base", type=Path, default=ROOT / "runs/mean", help="`visenc distill` run directory")
    post.add_argument("--policy", type=Path, required=True, help="policy checkpoint whose saliency head initializes")
    post.add_argument("--allow-task-transfer", action="store_true")
    post.add_argument("--mined", type=Path, required=True, help="mine output holding train/")
    post.add_argument("--held-out", type=Path, required=True, help="mine output holding validation/")
    post.add_argument("--teacher", default="lingbot:large")
    post.add_argument("--real", type=Path, default=ROOT / "data/uavid/train_rgb.npy",
                      help="unlabelled real frames for the teacher anchor (UAVid train; val stays held out)")
    post.add_argument("--sim", type=Path, default=ROOT / "data/train-v2/rgb_random1.npy")
    post.add_argument("--train-blocks", type=int, default=4)
    post.add_argument("--steps", type=int, default=3000)
    post.add_argument("--batch-size", type=int, default=32)
    post.add_argument("--real-batch", type=int, default=8)
    post.add_argument("--sim-batch", type=int, default=8)
    post.add_argument("--lr", type=float, default=2e-5)
    post.add_argument("--head-lr", type=float, default=3e-4)
    post.add_argument("--warmup", type=int, default=200)
    post.add_argument("--anchor-weight", type=float, default=1.0)
    post.add_argument("--validate-every", type=int, default=500)
    post.add_argument("--seed", type=int, default=0)
    post.add_argument("--out", type=Path, required=True)
    post.set_defaults(func=post_train_command)

    confusion = commands.add_parser("confusion-probe", help="fresh saliency head per frozen encoder on mined frames")
    confusion.add_argument("--encoders", nargs="+", required=True, help="visenc encoder specs, e.g. run:runs/mean")
    confusion.add_argument("--mined", type=Path, required=True)
    confusion.add_argument("--held-out", type=Path, required=True, help="mine output holding test/")
    confusion.add_argument("--heads", nargs="+", choices=("global", "patch"), default=["global", "patch"],
                           help="global: goal vector (policy head); patch: patch-by-patch goal matching")
    confusion.add_argument("--train-frames", type=int, default=8192)
    confusion.add_argument("--epochs", type=int, default=20)
    confusion.add_argument("--batch-size", type=int, default=128)
    confusion.add_argument("--learning-rate", type=float, default=1e-3)
    confusion.add_argument("--seed", type=int, default=0)
    confusion.add_argument("--out", type=Path, required=True)
    confusion.set_defaults(func=confusion_probe_command)

    perception = commands.add_parser("perception", help="per-patch depth and class heads on frozen encoders")
    add_native(perception)
    add_precision(perception)
    perception.add_argument("--encoders", nargs="+", default=["mean"], help=f"arms: {', '.join(ARMS)} or run:<dir>")
    perception.add_argument("--num-envs", type=int, default=64)
    perception.add_argument("--train-frames", type=int, default=12288)
    perception.add_argument("--test-frames", type=int, default=3072)
    perception.add_argument("--stride", type=int, default=15, help="control steps between kept frames")
    perception.add_argument("--start-distance", type=float, nargs=2, default=[8.0, 60.0], metavar=("MIN", "MAX"))
    perception.add_argument("--appearance", type=float, default=1.0)
    perception.add_argument("--epochs", type=int, default=8)
    perception.add_argument("--batch-size", type=int, default=64)
    perception.add_argument("--learning-rate", type=float, default=2e-3)
    perception.add_argument("--out", type=Path, required=True)
    perception.set_defaults(func=perception_command)

    counting = commands.add_parser("count-eval", help="count cars seen on exploratory flights (perception + memory)")
    add_native(counting)
    add_precision(counting)
    counting.add_argument("--encoder", default="mean")
    counting.add_argument("--heads", type=Path, required=True, help="perception heads for --encoder")
    counting.add_argument("--num-envs", type=int, default=64)
    counting.add_argument("--rounds", type=int, default=4, help="flights per environment (fresh resets)")
    counting.add_argument("--steps", type=int, default=600, help="control steps per flight (30 per second)")
    counting.add_argument("--start-distance", type=float, nargs=2, default=[8.0, 60.0], metavar=("MIN", "MAX"))
    counting.add_argument("--appearance", type=float, default=1.0)
    counting.add_argument("--out", type=Path, required=True)
    counting.set_defaults(func=count_eval_command)

    cohorts = commands.add_parser("eval-cohorts", help="paired fixed-goal hidden and far-target evaluations")
    add_native(cohorts)
    cohorts.add_argument("--checkpoint", type=Path, action="append", required=True)
    cohorts.add_argument("--baseline-checkpoint", type=Path, action="append", default=[],
                         help="explicitly evaluate a v2/ABI5 visual policy under the new hold objective; "
                              "all encoder, observation and flight-control contracts must match")
    cohorts.add_argument("--eval-envs", type=int, default=256)
    cohorts.add_argument("--goal-refresh-steps", type=int, default=0,
                         help="0 measures deployable fixed goals; nonzero uses oracle simulator boxes")
    cohorts.add_argument("--out", type=Path, required=True)
    cohorts.set_defaults(func=cohorts_command)

    probe = commands.add_parser("probe", help="offline goal-localization probe on frozen encoder features")
    add_native(probe)
    add_control(probe)
    probe.add_argument("--encoders", type=lambda text: [item.strip() for item in text.split(",") if item.strip()],
                       default=DEFAULT_ENCODERS.split(","),
                       help="comma-separated visual arms (default: pretrained,baseline,mean)")
    probe.add_argument("--probe-envs", type=int, default=128)
    add_precision(probe)
    probe.add_argument("--train-frames", type=int, default=8192, help="protocol minimum 8192")
    probe.add_argument("--test-frames", type=int, default=4096, help="protocol minimum 4096")
    probe.add_argument("--epochs", type=int, default=20)
    probe.add_argument("--batch-size", type=int, default=128)
    probe.add_argument("--learning-rate", type=float, default=1e-3)
    probe.add_argument("--appearance", type=float, default=DEFAULTS.appearance,
                       help="appearance strength of catalog goal views and probe episodes (0 = plain render)")
    probe.add_argument("--out", type=Path, required=True, help="output directory for probe.json and probe.md")
    probe.set_defaults(func=probe_command)

    benchmark = commands.add_parser("benchmark")
    add_native(benchmark)
    add_control(benchmark)
    add_learner(benchmark)
    benchmark.add_argument("--batch-sizes", type=lambda text: [int(item) for item in text.split(",")],
                           default=[64, 128])
    benchmark.add_argument("--iterations", type=int, default=20)
    benchmark.add_argument("--out", type=Path)
    benchmark.set_defaults(func=benchmark_command)

    profile = commands.add_parser("profile")
    add_native(profile)
    add_control(profile)
    add_learner(profile)
    profile.add_argument("--iterations", type=int, default=1)
    profile.add_argument("--out", type=Path, required=True)
    profile.set_defaults(func=profile_command)

    memory = commands.add_parser("memory-check")
    add_native(memory)
    add_control(memory)
    add_learner(memory)
    memory.add_argument("--repeats", type=int, default=3)
    memory.add_argument("--growth-tolerance-bytes", type=int, default=1 << 20)
    memory.add_argument("--out", type=Path)
    memory.set_defaults(func=memory_check_command)

    verify = commands.add_parser("verify-trace")
    verify.add_argument("--trace", type=Path, required=True)
    verify.set_defaults(func=verify_trace_command)
    return root


def main(process_started=None):
    started = time.monotonic() if process_started is None else process_started
    args = parser().parse_args()
    if getattr(args, "eval_envs", MIN_EPISODES) < MIN_EPISODES:
        raise SystemExit(f"--eval-envs must be at least {MIN_EPISODES} (one first episode per environment)")
    args.func(args, started)
