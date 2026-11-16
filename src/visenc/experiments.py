"""Independent, budget-matched experiment runners; execute inside `nix develop`.

Every treatment changes one factor. Existing full-budget fixed-city controls are
reused only after checking their saved training arguments and final log step.
No incomplete run is deleted or silently treated as a completed checkpoint.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "experiments"
REPORTS = ROOT / "reports" / "experiments"
BASELINES = {
    ("own", 0): "baseline", ("own", 1): "baseline-seed1",
    ("mean", 0): "mean", ("mean", 1): "mean-seed1",
}


def execute(arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(arguments), "LOG", log, flush=True)
    with log.open("w") as output:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-8000:], file=sys.stderr)
        raise RuntimeError(f"command exited {result.returncode}; see {log}")
    print("DONE", log, flush=True)


def completed_training(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    if not (path / "args.json").exists() or not (path / "log.jsonl").exists():
        raise ValueError(f"incomplete run at {path}; retain it and select a new output directory")
    actual = json.loads((path / "args.json").read_text())
    for key, value in expected.items():
        got = actual.get(key)
        if key == "data":
            got = [got] if isinstance(got, str) else got
            got = [str((ROOT / p).resolve()) for p in got]
            value = [str((ROOT / p).resolve()) for p in value]
        if got != value:
            raise ValueError(f"{path}: {key}={got!r}, expected {value!r}")
    last = json.loads((path / "log.jsonl").read_text().splitlines()[-1])
    if last["step"] != expected["steps"] - 1 or not (path / "student").is_dir():
        raise ValueError(f"{path}: training incomplete; no destructive automatic restart")
    return True


def train_variant(name: str, data: list[str], target: str, seed: int, args, extra: str | None = None) -> str:
    out = RUNS / name
    # 64 teacher/student images per optimizer step in both conditions:
    # simulation-only = 32 poses x 2 looks; mixed = 24 x 2 + 16 real frames.
    batch_size = 24 if extra else 32
    expected = dict(data=data, views=["random1", "random2"], pair_target=target,
                    teacher="lingbot:large", student="lingbot:small", student_init="pretrained",
                    student_aug=True, steps=args.steps, batch_size=batch_size, seed=seed,
                    extra_images=extra, extra_batch_size=16, lr=2e-4, warmup=min(1000, args.steps // 10))
    if not completed_training(out, expected):
        command = ["-m", "visenc.cli", "distill", "--data", *data, "--pair-target", target,
                   "--teacher", "lingbot:large", "--student", "lingbot:small", "--steps", str(args.steps),
                   "--batch-size", str(batch_size), "--seed", str(seed), "--workers", str(args.workers),
                   "--warmup", str(expected["warmup"]), "--out", str(out)]
        if extra:
            command += ["--extra-images", extra, "--extra-batch-size", "16"]
        execute(command, REPORTS / "logs" / f"{name}.train.log")
    return f"run:{out}"


def fixed_controls(args) -> dict[str, list[str]]:
    specs = {"own": [], "mean": []}
    for target in specs:
        for seed in args.seeds:
            old = ROOT / "runs" / BASELINES.get((target, seed), "not-an-existing-control")
            expected = dict(data=["data/train-v2"], views=["random1", "random2"], pair_target=target,
                            teacher="lingbot:large", student="lingbot:small", student_init="pretrained",
                            student_aug=True, steps=args.steps, batch_size=32, seed=seed,
                            extra_images=None, lr=2e-4, warmup=min(1000, args.steps // 10))
            if args.steps == 10000 and old.is_dir() and completed_training(old, expected):
                spec = f"run:{old}"
            else:
                spec = train_variant(f"fixed-{target}-seed{seed}-steps{args.steps}",
                                     ["data/train-v2"], target, seed, args)
            specs[target].append(spec)
    return specs


def save_protocol(name: str, value: dict) -> None:
    out = REPORTS / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"{path}: protocol differs; select a distinct experiment output")
    path.write_text(json.dumps(value, indent=2) + "\n")


def mixed(args) -> None:
    real = ROOT / "data/uavid"
    meta = json.loads((real / "meta.json").read_text())
    train_sequences = {p.split("_")[0] for p in meta["train"]}
    val_sequences = {p.split("_")[0] for p in meta["val"]}
    if train_sequences & val_sequences:
        raise ValueError("real-image train/validation video sequences overlap")
    controls = fixed_controls(args)
    treatments = {target: [train_variant(f"mixed-{target}-seed{seed}-steps{args.steps}",
                                         ["data/train-v2"], target, seed, args,
                                         str(real / "train_rgb.npy")) for seed in args.seeds]
                  for target in ("own", "mean")}
    specs = ["lingbot:small", *sum(controls.values(), []), *sum(treatments.values(), [])]
    save_protocol("mixed", dict(steps=args.steps, seeds=args.seeds, images_per_step=64,
        controls=controls, treatments=treatments, sim_poses_per_step={"control": 32, "mixed": 24},
        real_frames_per_step=16, unlabelled_source=str(real / "train_rgb.npy"),
        training_sequences=sorted(train_sequences), validation_sequences=sorted(val_sequences),
        labels_used_for_distillation=False,
        interpretation="Real-data-assisted adaptation; equal image/optimizer budget, less simulator exposure."))
    execute(["-m", "visenc.cli", "eval", "--data", "data/eval-v2", "--heldout-region", "0", "0", "245", "245",
             "--batch-size", "16", "--encoders", *specs, "--out", str(REPORTS / "mixed/sim")],
            REPORTS / "logs/mixed.sim.log")
    execute(["-m", "visenc.cli", "real-eval", "--real", str(real), "--encoders", *specs,
             "--baseline", "lingbot:small", "--out", str(REPORTS / "mixed/real")],
            REPORTS / "logs/mixed.real.log")


def geometry(args) -> None:
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    controls = fixed_controls(args)
    training = manifest["arms"]["multi"]["train"]
    treatments = {target: [train_variant(f"geometry-multi-{target}-seed{seed}-steps{args.steps}",
                                          training, target, seed, args) for seed in args.seeds]
                  for target in ("own", "mean")}
    save_protocol("geometry", dict(steps=args.steps, seeds=args.seeds, images_per_step=64,
        manifest=str(manifest_path), controls=controls, treatments=treatments,
        probe_protocol="Fit familiar canonical only; apply unchanged to heldout geometry and appearance."))
    for arm, groups in (("fixed", controls), ("multi", treatments)):
        specs = ["lingbot:small", *sum(groups.values(), [])]
        execute(["-m", "visenc.geometry", "eval", "--manifest", str(manifest_path), "--arm", arm,
                 "--encoders", *specs, "--batch-size", "16", "--out", str(REPORTS / f"geometry/{arm}")],
                REPORTS / "logs" / f"geometry.{arm}.log")
    specs = ["lingbot:small", *sum(controls.values(), []), *sum(treatments.values(), [])]
    execute(["-m", "visenc.cli", "real-eval", "--encoders", *specs, "--baseline", "lingbot:small",
             "--out", str(REPORTS / "geometry/real")], REPORTS / "logs/geometry.real.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["mixed", "geometry"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", default="data/geometry/geometry.json")
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("steps must be positive and seeds unique")
    {"mixed": mixed, "geometry": geometry}[args.experiment](args)


if __name__ == "__main__":
    main()
