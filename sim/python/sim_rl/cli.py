"""Timed native visual-RL experiments. All GPU work shares the native stream."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from .features import ARMS, PREPROCESSING, FrozenFeatures
from .native import CudaEnv
from .ppo import Config, Policy, Trainer, evaluate, gpu_range
from .profiling import capture, verify_trace

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_SEED, TEST_SEED = 100_003, 200_003
VALIDATION_STYLE, TEST_STYLE = 1, 2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(args, features, env, config):
    packages = {}
    for name in ("torch", "transformers", "lingbot-vision", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver = ct.CDLL("libcuda.so.1")
    version = ct.c_int()
    if driver.cuDriverGetVersion(ct.byref(version)) != 0:
        raise RuntimeError("cannot query CUDA driver version")
    sources = list((ROOT / "sim" / "python").rglob("*.py"))
    sources += list((ROOT / "sim" / "crates").rglob("*.rs"))
    sources += list((ROOT / "sim" / "crates").rglob("Cargo.toml"))
    sources += [ROOT / "src" / "visenc" / "models.py", ROOT / "sim" / "Cargo.toml", ROOT / "sim" / "Cargo.lock"]
    return {"encoder": features.spec, "encoder_sha256": features.fingerprint,
            "preprocessing": PREPROCESSING, "seed": args.seed, "width": env.width, "height": env.height,
            "num_envs": env.n, "feature_dim": features.dim, "ppo": asdict(config),
            "native_abi": "sim_env C ABI v1, fields 0..7, CUDA-only borrowed allocations",
            "native_library": str(env.library), "native_sha256": sha256(env.library),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(sources)},
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "torch_cuda": torch.version.cuda, "cuda_driver": version.value,
            "gpu": torch.cuda.get_device_name(0), "command": sys.argv,
            "train_style": 0, "validation_style": VALIDATION_STYLE, "final_test_style": TEST_STYLE,
            "validation_seed": VALIDATION_SEED, "final_test_seed": TEST_SEED,
            "task": "random-gap flight; 20Hz control; 120Hz physics; 150-step terminal horizon",
            "actor_inputs": "pooled visual tokens plus proprio[8]; no gap coordinates" if features.encoder is not None
                            else "privileged gap[4] plus proprio[8]" if features.spec == "state" else "proprio[8] only"}


def make_config(args):
    config = Config(total_steps=args.total_steps, rollout_steps=args.rollout_steps,
                    sequence_length=args.sequence_length, epochs=args.epochs,
                    minibatch_sequences=args.minibatch_sequences, hidden_size=args.hidden_size,
                    learning_rate=args.learning_rate, entropy=args.entropy)
    config.validate()
    return config


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def resources(args, n=None):
    env = CudaEnv(args.library, n or args.num_envs, args.seed, args.width, args.height, 0)
    try:
        with env.execution():
            features = FrozenFeatures(args.encoder, env.device)
            seed_all(args.seed)  # Backbone loading must not change policy initialization.
            policy = Policy(features.dim, args.hidden_size).to(env.device)
            trainer = Trainer(env, features, policy, make_config(args))
            trainer.warmup(args.seed)
        return env, features, policy, trainer
    except BaseException:
        env.close()
        raise


def checkpoint(path, trainer, metadata, elapsed, status):
    """Checkpoint serialization is an explicit logging/end-batch boundary."""
    trainer.env.stream.synchronize()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"format": "sim-visual-ppo-v1", "policy": trainer.policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(), "steps": trainer.steps,
                "elapsed_seconds": elapsed, "budget_status": status, "metadata": metadata,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
    temporary.replace(path)


def load_checkpoint(path, args):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation requires a trained checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "sim-visual-ppo-v1" or saved.get("steps", 0) < 1:
        raise ValueError("not an actually trained native visual PPO checkpoint")
    metadata = saved["metadata"]
    if metadata["preprocessing"] != PREPROCESSING:
        raise ValueError("checkpoint preprocessing is incompatible")
    if sha256(args.library) != metadata["native_sha256"]:
        raise ValueError("native environment library differs from training provenance")
    return saved


def run_evaluation(args, features, policy, seed, style, blind=True, env=None):
    # Reuse the validation environment/stream throughout training: Torch's
    # caching allocator and cuBLAS workspaces are associated with that stream.
    # The caller finishes training first; evaluate waits at each batch end.
    manager = (CudaEnv(args.library, args.eval_envs, seed, args.width, args.height, style)
               if env is None else nullcontext(env))
    with manager as env:
        if (env.n, env.width, env.height, env.style) != (args.eval_envs, args.width, args.height, style):
            raise ValueError("evaluation environment configuration differs from requested split")
        with env.execution():
            env.reset(seed)
            result = {"visual" if features.encoder is not None else features.spec: evaluate(env, features, policy)}
            if blind:
                for heuristic in ("center", "goal_centering", "random"):
                    env.reset(seed)
                    result[heuristic] = evaluate(env, features, heuristic=heuristic, seed=seed)
        memory = cuda_memory(env)
    result["seed"], result["style"] = seed, style
    result["cuda_memory"] = memory
    return result


def cuda_memory(env):
    """Read allocator/global device metrics only at completed batch boundaries."""
    free, total = torch.cuda.mem_get_info(env.device)
    return {"stream": env.stream.cuda_stream, "allocated_bytes": torch.cuda.memory_allocated(env.device),
            "reserved_bytes": torch.cuda.memory_reserved(env.device),
            "global_free_bytes": free, "global_total_bytes": total}


def evaluation_summary(evaluation, features):
    key = "visual" if features.encoder is not None else features.spec
    score = evaluation[key]["success"]
    blind = max(evaluation[name]["success"] for name in ("center", "goal_centering", "random"))
    enough = evaluation[key]["episodes"] >= 128
    return {"policy_success": score, "best_blind_success": blind, "absolute_gain": score - blind,
            "episodes": evaluation[key]["episodes"],
            "meaningful_visual_goal_met": bool(features.encoder is not None and enough and score >= 0.8 and score - blind >= 0.2),
            "criterion": "visual success >=0.8 and >=0.2 absolute above best blind heuristic on >=128 held-out episodes"}


def readable_report(out, result):
    rows = ["# Native GPU visual RL", "", "```json", json.dumps(result, indent=2, allow_nan=False), "```", ""]
    (Path(out) / "report.md").write_text("\n".join(rows))


def train(args, process_started):
    if args.wall_seconds <= 0 or args.eval_envs < 128:
        raise ValueError("positive wall budget and >=128 held-out evaluation episodes required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite run: {out}")
    deadline = process_started + args.wall_seconds
    seed_all(args.seed)
    env, features, policy, trainer = resources(args)
    timings = {"initialization_seconds": time.monotonic() - process_started,
               "collection_seconds": 0.0, "ppo_seconds": 0.0, "validation_seconds": 0.0,
               "final_evaluation_seconds": 0.0, "checkpoint_seconds": 0.0}
    metadata = provenance(args, features, env, trainer.config)
    write_json(out / "provenance.json", metadata)
    best, validations, last_validation = -math.inf, [], 0
    evaluation_seconds = 0.0
    status = "step_budget_complete"
    final_evaluation = None
    last_update_seconds = 0.0
    validation_env = None
    try:
        # Calibrate the actual held-out evaluation cost before spending the
        # training budget. Its result is NOT used for selection (zero updates).
        started = time.monotonic()
        validation_env = CudaEnv(args.library, args.eval_envs, VALIDATION_SEED,
                                 args.width, args.height, VALIDATION_STYLE)
        warm_validation = run_evaluation(args, features, policy, VALIDATION_SEED, VALIDATION_STYLE,
                                         blind=False, env=validation_env)
        evaluation_seconds = time.monotonic() - started
        timings["validation_seconds"] += evaluation_seconds
        write_json(out / "initial-validation.json", warm_validation)
        reserve = max(args.final_reserve_seconds, evaluation_seconds * 3.8 + 5)
        train_deadline = deadline - reserve
        with (out / "learning-curves.jsonl").open("a", buffering=1) as log, env.execution():
            while trainer.steps < args.total_steps:
                # CPU clock decisions only at complete rollout boundaries.
                if time.monotonic() + last_update_seconds >= train_deadline:
                    status = "wall_budget_incomplete"
                    break
                remaining = args.total_steps - trainer.steps
                block = env.n * trainer.config.sequence_length
                if remaining < block:
                    status = "step_budget_complete_rounded_down"
                    break
                length = min(trainer.config.rollout_steps, remaining // block * trainer.config.sequence_length)
                begin = time.monotonic()
                trainer.collect(length)
                env.stream.synchronize()  # Explicit end-batch wait outside sim.collect.
                collected = time.monotonic()
                losses = trainer.update(length)
                metrics = trainer.metrics(length, losses)
                env.stream.synchronize()  # Explicit end-update boundary.
                updated = time.monotonic()
                values = metrics.cpu().tolist()  # One post-rollout logging readback.
                last_update_seconds = updated - begin
                timings["collection_seconds"] += collected - begin
                timings["ppo_seconds"] += updated - collected
                episodes = values[1]
                record = {"steps": trainer.steps, "elapsed_seconds": updated - process_started,
                          "transitions_per_second_process": trainer.steps / (updated - process_started),
                          "collection_transitions_per_second": length * env.n / (collected - begin),
                          "update_seconds": updated - collected, "reward_mean": values[0],
                          "completed_episodes": episodes, "success": values[2] / max(1, episodes),
                          "collision": values[3] / max(1, episodes), "episode_return": values[4] / max(1, episodes)}
                record.update(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"), values[5:]))
                log.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
                validate = trainer.steps - last_validation >= args.validation_every
                if validate and time.monotonic() + evaluation_seconds + last_update_seconds < train_deadline:
                    started = time.monotonic()
                    validation = run_evaluation(args, features, policy, VALIDATION_SEED, VALIDATION_STYLE,
                                                blind=False, env=validation_env)
                    evaluation_seconds = time.monotonic() - started
                    timings["validation_seconds"] += evaluation_seconds
                    key = "visual" if features.encoder is not None else features.spec
                    score = validation[key]["success"]
                    validation.update(steps=trainer.steps, elapsed_seconds=time.monotonic() - process_started)
                    validations.append(validation)
                    write_json(out / "validation.json", validations)
                    last_validation = trainer.steps
                    if score > best:
                        best = score
                        started = time.monotonic()
                        checkpoint(out / "best.pt", trainer, metadata, started - process_started, "validation_selected")
                        timings["checkpoint_seconds"] += time.monotonic() - started
                    reserve = max(reserve, evaluation_seconds * 3.8 + 5)
                    train_deadline = deadline - reserve
        if trainer.steps < 1:
            raise RuntimeError("initialization/evaluation exhausted the wall budget before a single PPO update")
        # Always save actual executed steps; never label a timed stop converged.
        started = time.monotonic()
        checkpoint(out / "last.pt", trainer, metadata, started - process_started, status)
        timings["checkpoint_seconds"] += time.monotonic() - started
        if not (out / "best.pt").exists():
            started = time.monotonic()
            validation = run_evaluation(args, features, policy, VALIDATION_SEED, VALIDATION_STYLE,
                                        blind=False, env=validation_env)
            timings["validation_seconds"] += time.monotonic() - started
            validation.update(steps=trainer.steps, elapsed_seconds=time.monotonic() - process_started)
            validations.append(validation)
            write_json(out / "validation.json", validations)
            checkpoint(out / "best.pt", trainer, metadata, time.monotonic() - process_started, "validation_selected_only_candidate")
        selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        with env.execution():
            policy.load_state_dict(selected["policy"], strict=True)
        env.stream.synchronize()
        # The final-test split gets one distinct environment only after the
        # persistent validation environment has released its native resources.
        validation_env.close()
        validation_env = None
        started = time.monotonic()
        final_evaluation = run_evaluation(args, features, policy, TEST_SEED, TEST_STYLE, blind=True)
        timings["final_evaluation_seconds"] = time.monotonic() - started
        write_json(out / "final-test.json", final_evaluation)
    finally:
        if validation_env is not None:
            validation_env.close()
        env.close()
    elapsed = time.monotonic() - process_started
    result = {"status": status, "actual_steps": trainer.steps, "requested_steps": args.total_steps,
              "selected_steps": selected["steps"], "elapsed_seconds_including_initialization_training_evaluation": elapsed,
              "wall_budget_seconds": args.wall_seconds, "within_wall_budget": elapsed <= args.wall_seconds,
              "compile_excluded": True, "convergence_claim": False, "timings": timings,
              "final_test": final_evaluation, "acceptance": evaluation_summary(final_evaluation, features),
              "checkpoint": str(out / "best.pt"), "last_checkpoint": str(out / "last.pt")}
    write_json(out / "summary.json", result)
    readable_report(out, result)
    return result


def evaluation_command(args, process_started):
    if args.eval_envs < 128:
        raise ValueError("final evaluation requires >=128 independent episodes")
    saved = load_checkpoint(args.checkpoint, args)
    metadata = saved["metadata"]
    args.encoder, args.width, args.height = metadata["encoder"], metadata["width"], metadata["height"]
    args.hidden_size = metadata["ppo"]["hidden_size"]
    seed_all(args.seed)
    # A tiny owner env provides a primary context/stream for model upload only.
    with CudaEnv(args.library, 1, args.seed, args.width, args.height, 0) as owner:
        with owner.execution():
            features = FrozenFeatures(args.encoder, owner.device)
            if features.fingerprint != metadata["encoder_sha256"]:
                raise ValueError("actual frozen encoder weights differ from checkpoint")
            policy = Policy(features.dim, args.hidden_size).to(owner.device)
            policy.load_state_dict(saved["policy"], strict=True)
            policy.eval().requires_grad_(False)
        owner.stream.synchronize()
        seed, style = (VALIDATION_SEED, VALIDATION_STYLE) if args.split == "validation" else (TEST_SEED, TEST_STYLE)
        result = run_evaluation(args, features, policy, seed, style)
    result["acceptance"] = evaluation_summary(result, features)
    result["elapsed_seconds_including_initialization_evaluation"] = time.monotonic() - process_started
    result["checkpoint_steps"] = saved["steps"]
    write_json(Path(args.out) / f"{args.split}.json", result)
    readable_report(args.out, result)
    return result


def timed_gpu(stream, function, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream.synchronize()
    begin = time.monotonic()
    start.record(stream)
    for _ in range(iterations):
        function()
    end.record(stream)
    end.synchronize()  # Explicit benchmark end-batch boundary.
    return {"gpu_seconds": start.elapsed_time(end) / 1000, "wall_seconds": time.monotonic() - begin}


def benchmark(args, process_started):
    results = []
    for n in args.batch_sizes:
        env, features, policy, trainer = resources(args, n)
        try:
            with env.execution(), torch.no_grad():
                actions = torch.ones(n, device=env.device, dtype=torch.int64)
                native = timed_gpu(env.stream, lambda: env.step(actions), args.iterations)
                encoder = timed_gpu(env.stream, lambda: features(env), args.iterations)
            with env.execution():
                trainer.collect()
                trainer.update(trainer.config.rollout_steps)  # Warm optimizer/allocator before timing.
                def full_update():
                    length = trainer.collect()
                    trainer.update(length)
                full = timed_gpu(env.stream, full_update, args.ppo_iterations)
                # A short actual CUDA trace classifies physics/task/preprocess
                # kernels, and certifies both collection and PPO zero-staging.
                verification = capture(trainer, Path(args.out) / f"batch-{n}", 1)
            frames = args.iterations * n
            transitions = args.ppo_iterations * trainer.config.rollout_steps * n
            result = {"batch_size": n, "native_physics_render_preprocess": native, "encoder": encoder,
                      "full_ppo": full, "native_frames_per_second": frames / native["wall_seconds"],
                      "encoder_frames_per_second": frames / encoder["wall_seconds"],
                      "full_training_transitions_per_second": transitions / full["wall_seconds"],
                      "allocated_cuda_bytes": torch.cuda.max_memory_allocated(), "trace_verification": verification}
            results.append(result)
            write_json(Path(args.out) / "benchmark.json", results)
            print(json.dumps(result, allow_nan=False), flush=True)
        finally:
            env.close()
        del trainer, policy, features
        torch.cuda.empty_cache()  # Batch boundary, never hidden OOM recovery.
        torch.cuda.reset_peak_memory_stats()
    result = {"batches": results, "elapsed_seconds": time.monotonic() - process_started}
    readable_report(args.out, result)
    return result


def profile_command(args, process_started):
    seed_all(args.seed)
    env, features, policy, trainer = resources(args)
    try:
        with env.execution():
            trainer.collect()
            trainer.update(trainer.config.rollout_steps)
            result = capture(trainer, args.out, args.profile_iterations)
    finally:
        env.close()
    result["elapsed_seconds"] = time.monotonic() - process_started
    return result


def memory_check(args, process_started):
    """Repeat validation on one resource/stream, with no optimizer updates."""
    if args.validation_repeats < 2 or args.max_growth_mib < 0:
        raise ValueError("at least two measured repeats and nonnegative growth tolerance required")
    seed_all(args.seed)
    env, features, policy, trainer = resources(args)
    samples = []
    try:
        with CudaEnv(args.library, args.eval_envs, VALIDATION_SEED,
                     args.width, args.height, VALIDATION_STYLE) as validation_env:
            for repetition in range(args.validation_repeats + 1):
                before = cuda_memory(validation_env)
                evaluation = run_evaluation(args, features, policy, VALIDATION_SEED,
                                            VALIDATION_STYLE, blind=False, env=validation_env)
                sample = {"repetition": repetition, "warmup": repetition == 0,
                          "elapsed_seconds": time.monotonic() - process_started,
                          "before": before,
                          **evaluation["cuda_memory"]}
                samples.append(sample)
                print(json.dumps(sample), flush=True)
                write_json(Path(args.out) / "memory-samples.json", samples)
    finally:
        env.close()
    baseline = samples[0]
    allocated_growth = max(sample["allocated_bytes"] for sample in samples) - baseline["allocated_bytes"]
    reserved_growth = max(sample["reserved_bytes"] for sample in samples) - baseline["reserved_bytes"]
    tolerance = args.max_growth_mib * 1024 * 1024
    bounded = allocated_growth <= tolerance and reserved_growth <= tolerance
    result = {"bounded_after_warmup": bounded, "samples": samples,
              "allocated_growth_after_warmup_bytes": allocated_growth,
              "reserved_growth_after_warmup_bytes": reserved_growth,
              "growth_tolerance_bytes": tolerance, "optimizer_updates": 0,
              "elapsed_seconds": time.monotonic() - process_started,
              "global_memory_note": "Global free memory includes other processes and is reported, not attributed"}
    write_json(Path(args.out) / "memory-check.json", result)
    if not bounded:
        raise RuntimeError("persistent validation memory grew beyond tolerance: " + json.dumps(result))
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "benchmark", "profile", "memory-check"):
        p = commands.add_parser(name)
        p.add_argument("--library", default=str(ROOT / ".sim-target" / "debug" / "libsim_env.so"))
        p.add_argument("--encoder", default="pretrained", help="pretrained|baseline|mean|state|proprio|run:<dir>")
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--num-envs", type=int, default=64)
        p.add_argument("--width", type=int, default=192)
        p.add_argument("--height", type=int, default=128)
        p.add_argument("--total-steps", type=int, default=1_000_000)
        p.add_argument("--rollout-steps", type=int, default=64)
        p.add_argument("--sequence-length", type=int, default=16)
        p.add_argument("--epochs", type=int, default=4)
        p.add_argument("--minibatch-sequences", type=int, default=32)
        p.add_argument("--hidden-size", type=int, default=128)
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--entropy", type=float, default=0.01)
        p.add_argument("--eval-envs", type=int, default=128)
        if name == "train":
            p.add_argument("--wall-seconds", type=float, default=3300)
            p.add_argument("--final-reserve-seconds", type=float, default=90)
            p.add_argument("--validation-every", type=int, default=65_536)
        elif name == "eval":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--split", choices=("validation", "test"), default="test")
        elif name == "benchmark":
            p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 128])
            p.add_argument("--iterations", type=int, default=16)
            p.add_argument("--ppo-iterations", type=int, default=2)
        elif name == "profile":
            p.add_argument("--profile-iterations", type=int, default=1)
        elif name == "memory-check":
            p.add_argument("--validation-repeats", type=int, default=4)
            p.add_argument("--max-growth-mib", type=float, default=16)
    p = commands.add_parser("verify-trace")
    p.add_argument("trace")
    p.add_argument("--scope", default="sim.collect")
    return cli


def main(process_started=None):
    process_started = time.monotonic() if process_started is None else process_started
    args = parser().parse_args()
    if args.command == "verify-trace":
        result = verify_trace(args.trace, args.scope)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no GPU/encoder fallback")
        result = {"train": train, "eval": evaluation_command, "benchmark": benchmark,
                  "profile": profile_command, "memory-check": memory_check}[args.command](args, process_started)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
