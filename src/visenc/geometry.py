"""Matched fixed/multi geometry preparation and frozen cross-geometry probes.

Examples (run from repository root, under the renderer's X/EGL environment)::

    python -m visenc.geometry prepare --out data/geometry --phase build
    python -m visenc.geometry prepare --out data/geometry --phase render
    python -m visenc.geometry eval --manifest data/geometry/geometry.json \
        --arm multi --encoders run:runs/multi-mean --out reports/geometry-multi

Training manifests fixed.json/multi.json contain an ordered ``splits`` directory list;
pass those directories to distillation with identical views, steps, batch size, and seed.
The consistency axis is the trainer's existing mean versus baseline objective, not a
change in the rendered data. Reports must compare all four geometry x objective arms.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .city import OBSERVABLE_STATE, STATE_GROUPS, STATE_NAMES
from .data import Split, generate
from .evaluate import Features, Ridge, SKY, TRAIN_PATCHES_PER_IMAGE, dense_targets, r2
from .models import PATCH, load_encoder

TRAIN_VIEWS = ["canonical", "random1", "random2"]
EVAL_VIEWS = TRAIN_VIEWS + ["novel1", "novel2"]
SCHEMA = "visenc.geometry.v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_new(path: Path, value: dict) -> None:
    """Resume identical preparation only; never replace a different experiment."""
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"{path} already describes a different experiment; choose a new output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _counts(total: int, parts: int) -> list[int]:
    return [total // parts + (i < total % parts) for i in range(parts)]


def _layout(scene: Path) -> dict:
    return _json(scene / "layout.json")


def _scene_record(scene: Path) -> dict:
    layout = _layout(scene)
    return {"path": str(scene), "layout_sha256": hashlib.sha256((scene / "layout.json").read_bytes()).hexdigest(),
            "seed": layout["seed"], "family": layout.get("seed_family", "legacy-fixed"),
            "blocks": layout["blocks"], "builder_version": layout.get("builder_version", 1)}


def _build(args, scene: Path, seed: int, family: str) -> None:
    if scene.exists() and any(scene.iterdir()):
        layout = _layout(scene)
        if (layout.get("seed"), layout.get("seed_family"), layout.get("blocks"),
                layout.get("builder_version")) != (seed, family, args.blocks, 2):
            raise ValueError(f"existing scene has incompatible provenance: {scene}")
        missing = [c for c in layout["classes"] if c != "sky" and not (scene / f"{c}.glb").exists()]
        if missing:
            raise ValueError(f"incomplete scene {scene}: missing {missing}; preserve it and use another scene root")
        return
    subprocess.run([args.blender, "-b", "--python", str(args.builder.resolve()), "--",
                    "--assets", str(args.assets.resolve()), "--out", str(scene),
                    "--blocks", str(args.blocks), "--seed", str(seed), "--family", family], check=True)


def _render(args, scene: Path, out: Path, count: int, seed: int, evaluation: bool, exclusion) -> dict:
    views = EVAL_VIEWS if evaluation else TRAIN_VIEWS
    expected = {"count": count, "width": args.width, "height": args.height, "seed": seed,
                "views": views, "exclude_region": exclusion}
    if out.exists():
        meta = _json(out / "meta.json")
        if any(meta.get(k) != v for k, v in expected.items()) or Path(meta["scene"]).resolve() != scene:
            raise ValueError(f"existing split has incompatible metadata: {out}")
        # A completed meta.json is the generator's commit marker. Missing arrays are not a resumable split.
        for name in ["state", "semantic", "depth", *[f"rgb_{v}" for v in views]]:
            if not (out / f"{name}.npy").exists():
                raise ValueError(f"incomplete split {out}: missing {name}.npy")
    else:
        generate(scene, out, seed, count, args.width, args.height, 2, 2 if evaluation else 0,
                 args.workers, tuple(exclusion) if exclusion else None)
    return {"split": str(out), "count": count, "scene": str(scene), "pose_seed": seed}


def prepare(args) -> None:
    out = args.out.resolve()
    scenes = args.scene_root.resolve()
    fixed = args.fixed_scene.resolve()
    if len(args.train_seeds) != 2 or len(args.heldout_seeds) != 2:
        raise ValueError("this matched design requires two new train seeds and two heldout seeds")
    if len(set(args.train_seeds)) != 2 or len(set(args.heldout_seeds)) != 2:
        raise ValueError("layout seeds must be unique within each family")
    if args.train_count < 3 or args.eval_count < 24 or args.workers < 1:
        raise ValueError("train count must be at least 3, eval count at least 24, and workers positive")
    if args.width % PATCH or args.height % PATCH:
        raise ValueError("image dimensions must be multiples of 16")
    train_scenes = [fixed] + [scenes / f"train-{s}" for s in args.train_seeds]
    heldout_scenes = [scenes / f"heldout-{s}" for s in args.heldout_seeds]
    source = _json(args.fixed_train / "meta.json") if args.fixed_train else None
    if source and Path(source["scene"]).resolve() != fixed:
        raise ValueError("fixed training data was not rendered from --fixed-scene")
    exclusion = source.get("exclude_region") if source else args.exclude_region
    train_seed = source["seed"] if source else args.train_pose_seed
    if train_seed == args.eval_pose_seed:
        raise ValueError("training and evaluation pose seeds must differ")
    config = {"train_count": args.train_count, "eval_count": args.eval_count, "width": args.width,
              "height": args.height, "blocks": args.blocks, "train_seeds": args.train_seeds,
              "heldout_seeds": args.heldout_seeds, "train_pose_seed": train_seed,
              "eval_pose_seed": args.eval_pose_seed, "exclude_region": exclusion,
              "fixed_scene": str(fixed), "scene_root": str(scenes),
              "fixed_train": str(args.fixed_train.resolve()) if args.fixed_train else None,
              "builder_sha256": hashlib.sha256(args.builder.read_bytes()).hexdigest()}
    _save_new(out / "preparation.json", config)
    if args.phase in ("build", "all"):
        for scene, seed in zip(train_scenes[1:], args.train_seeds):
            _build(args, scene, seed, "train")
        for scene, seed in zip(heldout_scenes, args.heldout_seeds):
            _build(args, scene, seed, "heldout")
    if args.phase == "build":
        print(f"scene builds complete; render with --phase render and the same flags: {out}")
        return
    records = [_scene_record(s) for s in train_scenes + heldout_scenes]
    for record, seed, family in zip(records[1:], args.train_seeds + args.heldout_seeds,
                                    ["train", "train", "heldout", "heldout"]):
        if (record["seed"], record["family"], record["blocks"], record["builder_version"]) != (seed, family, args.blocks, 2):
            raise ValueError(f"scene does not match the declared build seed/family: {record['path']}")
    classes = _layout(fixed)["classes"]
    if any(_layout(s)["classes"] != classes for s in train_scenes + heldout_scenes):
        raise ValueError("semantic class definitions differ between scenes")
    # A fixed-trained baseline is reusable only for the exact declared rendering budget.
    reuse = source and all(source.get(k) == v for k, v in {
        "count": args.train_count, "width": args.width, "height": args.height, "views": TRAIN_VIEWS}.items())
    fixed_train = ({"split": str(args.fixed_train.resolve()), "count": args.train_count,
                    "scene": str(fixed), "pose_seed": train_seed} if reuse else
                   _render(args, fixed, out / "train-fixed", args.train_count, train_seed, False, exclusion))
    multi_train = [_render(args, scene, out / f"train-multi-{i}", count, train_seed, False, exclusion)
                   for i, (scene, count) in enumerate(zip(train_scenes, _counts(args.train_count, 3)))]
    familiar_count = args.eval_count // 2
    heldout_count = args.eval_count - familiar_count
    # Shared physical split, sliced by the manifest: fixed and multi have equal total familiar poses.
    fixed_eval = _render(args, fixed, out / "eval-fixed", familiar_count, args.eval_pose_seed, True, exclusion)
    familiar_multi = []
    for i, (scene, count) in enumerate(zip(train_scenes, _counts(familiar_count, 3))):
        familiar_multi.append({**fixed_eval, "count": count} if i == 0 else
                              _render(args, scene, out / f"eval-train-{i}", count,
                                      args.eval_pose_seed, True, exclusion))
    heldout = [_render(args, scene, out / f"eval-heldout-{i}", count, args.eval_pose_seed, True, exclusion)
               for i, (scene, count) in enumerate(zip(heldout_scenes, _counts(heldout_count, 2)))]
    arms = {"fixed": {"train": [fixed_train["split"]], "train_records": [fixed_train], "familiar": [fixed_eval]},
            "multi": {"train": [e["split"] for e in multi_train], "train_records": multi_train,
                      "familiar": familiar_multi}}
    for arm, entries in arms.items():
        _save_new(out / f"{arm}.json", {"schema": "visenc.geometry.train.v1", "arm": arm,
                  "splits": entries["train"], "count": args.train_count,
                  "views": TRAIN_VIEWS})
    manifest = {"schema": SCHEMA, "config": config, "scenes": records, "arms": arms,
                "heldout": heldout, "evaluation_views": EVAL_VIEWS,
                "budget": {"train_poses_per_arm": args.train_count, "train_images_per_arm": 3 * args.train_count,
                           "eval_poses_per_arm": args.eval_count, "familiar_poses_per_arm": familiar_count,
                           "heldout_poses_per_arm": heldout_count},
                "design": "fixed versus three layouts crossed with baseline versus mean consistency; identical training budgets"}
    _save_new(out / "geometry.json", manifest)
    print(json.dumps({"manifest": str(out / "geometry.json"), "budget": manifest["budget"],
                      "training": {a: b["train"] for a, b in arms.items()}}, indent=2))


def _indices(entry: dict, fit_frac: float) -> tuple[Split, np.ndarray, np.ndarray]:
    split = Split(Path(entry["split"]))
    count = entry["count"]
    if count > len(split):
        raise ValueError(f"manifest count exceeds split length: {split.root}")
    n_fit = int(count * fit_frac)
    if n_fit < 2 or count - n_fit < 1:
        raise ValueError("each familiar scene needs at least two probe-fit poses and one test pose")
    return split, np.arange(n_fit), np.arange(n_fit, count)


def _fit(encoder, familiar: list[dict], args, device) -> tuple[dict, dict]:
    globals_, states, rows, sems, depths, masks = [], [], [], [], [], []
    n_classes = None
    provenance = []
    for entry in familiar:
        split, idx, _ = _indices(entry, args.fit_frac)
        if "canonical" not in split.views:
            raise ValueError(f"canonical fit view missing: {split.root}")
        n_classes = len(split.meta["classes"])
        rng = np.random.default_rng(0)
        n_patches = (split.size[0] // PATCH) * (split.size[1] // PATCH)
        keep = {int(i): np.sort(rng.choice(n_patches, min(TRAIN_PATCHES_PER_IMAGE, n_patches), replace=False)) for i in idx}
        g, patches = Features(encoder, split, device, args.batch_size).extract(split.rgb["canonical"], idx, keep)
        frac, logd = dense_targets(split, idx, n_classes)
        pos = torch.from_numpy(np.stack([keep[int(i)] for i in idx]))
        sem = torch.gather(frac, 1, pos[..., None].expand(-1, -1, n_classes)).reshape(-1, n_classes)
        globals_.append(g)
        states.append(torch.from_numpy(np.asarray(split.state[idx])))
        rows.append(torch.cat([patches[int(i)] for i in idx]))
        sems.append(sem)
        depths.append(torch.gather(logd, 1, pos).reshape(-1, 1))
        masks.append(sem[:, SKY] < 0.5)
        provenance.append({"split": str(split.root), "view": "canonical", "indices": [0, len(idx)]})
    g, y, p, sem, logd, valid = (torch.cat(v) for v in (globals_, states, rows, sems, depths, masks))
    if valid.sum() < 2:
        raise ValueError("too few geometry patches for depth probe")
    probes = {"state": Ridge(g, y, device), "depth": Ridge(p[valid], logd[valid], device),
              "semantic": Ridge(p, sem, device), "global_mean": g.mean(0),
              "patch_mean": p.float().mean(0), "classes": n_classes}
    return probes, {"source": "shared original-city canonical only, identical probe-fit images for every training arm", "splits": provenance,
                    "fit_poses": len(g), "fit_patches": len(p),
                    "lambda": {k: probes[k].lam for k in ("state", "depth", "semantic")}}


def _score(encoder, probes, split: Split, idx: np.ndarray, args, device) -> dict:
    """Score all appearances together, retaining at most one batch of patch tokens."""
    classes = probes["classes"]
    if classes != len(split.meta["classes"]):
        raise ValueError("class definitions differ between probe fitting and evaluation")
    if not set(EVAL_VIEWS).issubset(split.views):
        raise ValueError(f"evaluation split lacks required appearances: {split.root}")
    feats = Features(encoder, split, device, args.batch_size)
    accum = {v: {"global": [], "state": [], "confusion": torch.zeros(classes, classes, dtype=torch.int64),
                 "depth_n": 0, "depth_absrel": 0.0, "depth_sq": 0.0, "depth_log_sq": 0.0,
                 "depth_abs": 0.0, "depth_delta1": 0.0, "patch_cos_sum": 0.0, "patch_n": 0} for v in EVAL_VIEWS}
    for start in range(0, len(idx), args.batch_size):
        ids = idx[start:start + args.batch_size]
        frac, logd = dense_targets(split, ids, classes)
        valid = frac[..., SKY] < 0.5
        truth = frac.argmax(-1)
        canonical = None
        for view in EVAL_VIEWS:
            a = accum[view]
            g, patches = feats.extract(split.rgb[view], ids, {int(i): None for i in ids})
            tok = torch.stack([patches[int(i)] for i in ids])
            a["global"].append(g)
            a["state"].append(probes["state"](g).cpu())
            flat = tok.flatten(0, 1)
            predicted_log = probes["depth"](flat).cpu().reshape(len(ids), -1)
            pred_log, true_log = predicted_log[valid], logd[valid]
            pred, target = pred_log.exp(), true_log.exp()
            a["depth_n"] += len(pred)
            a["depth_absrel"] += ((pred - target).abs() / target).sum().item()
            a["depth_abs"] += (pred - target).abs().sum().item()
            a["depth_sq"] += ((pred - target) ** 2).sum().item()
            a["depth_log_sq"] += ((pred_log - true_log) ** 2).sum().item()
            a["depth_delta1"] += (torch.maximum(pred / target, target / pred) < 1.25).sum().item()
            pred_cls = probes["semantic"](flat).argmax(-1).cpu().reshape_as(truth)
            a["confusion"] += torch.bincount((truth * classes + pred_cls).flatten(), minlength=classes * classes).reshape(classes, classes)
            if view == "canonical":
                canonical = tok.float() - probes["patch_mean"]
            else:
                cos = F.cosine_similarity(canonical, tok.float() - probes["patch_mean"], dim=-1)
                a["patch_cos_sum"] += cos.sum().item()
                a["patch_n"] += cos.numel()
    state = torch.from_numpy(np.asarray(split.state[idx]))
    output = {}
    canonical_g = torch.cat(accum["canonical"]["global"])
    qa = F.normalize(canonical_g - probes["global_mean"], dim=1)
    for view, a in accum.items():
        prediction = torch.cat(a["state"])
        state_r2 = r2(prediction, state)
        error = prediction - state
        cm = a["confusion"].double()
        union = cm.sum(0) + cm.sum(1) - cm.diag()
        present = cm.sum(1) > 0
        iou = cm.diag() / union.clamp_min(1)
        n = a["depth_n"]
        metric = {"n": len(idx), "state_r2": state_r2.mean().item(),
                  "pose_r2": float(np.mean([state_r2[STATE_NAMES.index(k)].item() for k in OBSERVABLE_STATE])),
                  "state_r2_per_dim": dict(zip(STATE_NAMES, state_r2.tolist())),
                  "state_mae_per_dim": dict(zip(STATE_NAMES, error.abs().mean(0).tolist())),
                  "state_rmse_per_dim": dict(zip(STATE_NAMES, error.square().mean(0).sqrt().tolist())),
                  **{f"state_r2_{k}": state_r2[s].mean().item() for k, s in STATE_GROUPS.items()},
                  "depth_valid_patches": n,
                  "depth_absrel": a["depth_absrel"] / n if n else None,
                  "depth_mae_m": a["depth_abs"] / n if n else None,
                  "depth_rmse_m": math.sqrt(a["depth_sq"] / n) if n else None,
                  "depth_rmse_log": math.sqrt(a["depth_log_sq"] / n) if n else None,
                  "depth_delta1": a["depth_delta1"] / n if n else None,
                  "semantic_miou": iou[present].mean().item(),
                  "semantic_accuracy": (cm.diag().sum() / cm.sum()).item(),
                  "semantic_iou_per_class": {name: iou[c].item() if present[c] else None for c, name in enumerate(split.meta["classes"])}}
        if view != "canonical":
            qb = F.normalize(torch.cat(a["global"]) - probes["global_mean"], dim=1)
            correct = 0
            for start in range(0, len(idx), args.retrieval_chunk):
                sim = qa[start:start + args.retrieval_chunk] @ qb.T
                correct += (sim.argmax(1) == torch.arange(start, start + len(sim))).sum().item()
            metric["pairs"] = {"retrieval_at_1": correct / len(idx), "retrieval_candidates": len(idx),
                               "global_cosine": (qa * qb).sum(1).mean().item(),
                               "patch_cosine": a["patch_cos_sum"] / a["patch_n"]}
        output[view] = metric
    return output


def _mean_metrics(rows: list[tuple[dict, float]]) -> dict:
    """Sample-weighted scene/view macro metrics; raw per-scene scores remain in JSON."""
    keys = set().union(*(r.keys() for r, _ in rows))
    result = {}
    for key in sorted(keys):
        if key in {"n", "depth_valid_patches"}:
            continue
        present = [(r[key], w) for r, w in rows if r.get(key) is not None]
        if not present:
            result[key] = None
        elif isinstance(present[0][0], dict):
            result[key] = _mean_metrics(present)
        elif isinstance(present[0][0], (int, float)):
            result[key] = sum(v * w for v, w in present) / sum(w for _, w in present)
    return result


def _summary(scenes: list[dict]) -> dict:
    result = {}
    for geometry in ("familiar", "heldout"):
        entries = [s for s in scenes if s["geometry"] == geometry]
        for appearance, views in (("canonical", ["canonical"]), ("seen", ["random1", "random2"]),
                                  ("novel", ["novel1", "novel2"])):
            rows = [(s["views"][v], s["n"]) for s in entries for v in views]
            result[f"{geometry}/{appearance}"] = _mean_metrics(rows)
            result[f"{geometry}/{appearance}"]["n_poses"] = sum(s["n"] for s in entries)
            result[f"{geometry}/{appearance}"]["n_view_samples"] = sum(s["n"] for s in entries) * len(views)
    return result


def _table(results: list[dict]) -> str:
    lines = ["# Matched geometry × consistency evaluation", "",
             "Every encoder uses the same original-city canonical probe-fit images, original-city test suffix, and heldout scenes.",
             "Probes are fitted once per encoder and applied unchanged to every evaluated appearance and geometry.",
             "Scene/view metrics below are pose-weighted macro averages; raw scene scores and all state dimensions are in JSON.",
             "Appearance-only = familiar seen/novel; geometry-only = heldout canonical; both = heldout seen/novel.", "",
             "| arm | encoder | shift | pose R² (no yaw) | depth AbsRel | depth log RMSE | seg mIoU | retrieval@1 | patch cos |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        for shift, values in result["summary"].items():
            pair = values.get("pairs", {})
            metrics = [values.get("pose_r2"), values.get("depth_absrel"), values.get("depth_rmse_log"),
                       values.get("semantic_miou"), pair.get("retrieval_at_1"), pair.get("patch_cosine")]
            cells = ["—" if x is None else f"{x:.4f}" for x in metrics]
            lines.append("| " + " | ".join([result["arm"], result["spec"], shift, *cells]) + " |")
    return "\n".join(lines) + "\n"


def evaluate(args) -> None:
    if not 0 < args.fit_frac < 1 or args.batch_size < 1 or args.retrieval_chunk < 1:
        raise ValueError("fit fraction must be between zero and one; batch and chunk sizes positive")
    manifest = _json(args.manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported geometry manifest schema")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    # Training geometry is the intervention; changing readout data would confound the comparison.
    # Keep multi-layout familiar records available in the manifest for separate diagnostics only.
    familiar = manifest["arms"]["fixed"]["familiar"]
    results = []
    for spec in args.encoders:
        encoder = load_encoder(spec).to(device).eval()
        probes, fit_info = _fit(encoder, familiar, args, device)
        scenes = []
        for geometry, entries in (("familiar", familiar), ("heldout", manifest["heldout"])):
            for entry in entries:
                if geometry == "familiar":
                    split, _, idx = _indices(entry, args.fit_frac)
                else:
                    split = Split(Path(entry["split"]))
                    idx = np.arange(entry["count"])
                if geometry == "heldout" and any(Path(entry["scene"]).resolve() == Path(e["scene"]).resolve() for e in familiar):
                    raise ValueError("heldout scene overlaps familiar probe-fitting geometry")
                scores = _score(encoder, probes, split, idx, args, device)
                scenes.append({"geometry": geometry, "scene": entry["scene"], "split": entry["split"],
                               "n": len(idx), "indices": [int(idx[0]), int(idx[-1]) + 1], "views": scores})
        results.append({"spec": spec, "arm": args.arm, "fit": fit_info, "scenes": scenes, "summary": _summary(scenes)})
        del encoder, probes
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload = {"schema": "visenc.geometry.results.v1", "arm": args.arm,
                   "manifest": str(args.manifest.resolve()),
                   "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   "fit_frac": args.fit_frac, "budget": manifest["budget"],
                   "protocol": {"probe_fit_geometry": "shared original fixed city",
                                "probe_fit_view": "canonical",
                                "familiar_test_geometry": "shared original fixed city",
                                "heldout_test_geometry": "shared manifest heldout scenes",
                                "arm_semantics": "encoder training geometry label only",
                                "multi_training_layout_diagnostics": "not included in primary summary"},
                   "aggregation": "pose-weighted scene/view macro; R2 uses each scene test variance; retrieval candidates within scene",
                   "results": results}
        # This newly-created output directory belongs exclusively to this invocation.
        (out / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        (out / "report.md").write_text(_table(results))
    print(_table(results))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="build/render a deterministic matched geometry experiment")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--phase", choices=["build", "render", "all"], default="all")
    prep.add_argument("--scene-root", type=Path, default=Path("assets/geometry"))
    prep.add_argument("--fixed-scene", type=Path, default=Path("assets/city"))
    prep.add_argument("--fixed-train", type=Path, default=Path("data/train-v2"), help="reuse matching fixed split; also inherits its seed and exclusion")
    prep.add_argument("--assets", type=Path, default=Path("assets/kenney"))
    prep.add_argument("--builder", type=Path, default=Path("scene/build_city.py"))
    prep.add_argument("--blender", default="blender")
    prep.add_argument("--blocks", type=int, default=12)
    prep.add_argument("--train-seeds", nargs=2, type=int, default=[101, 102])
    prep.add_argument("--heldout-seeds", nargs=2, type=int, default=[10001, 10002])
    prep.add_argument("--train-count", type=int, default=20000, help="total poses PER ARM, not per layout")
    prep.add_argument("--eval-count", type=int, default=3000, help="total familiar + heldout poses per arm BEFORE probe fitting")
    prep.add_argument("--width", type=int, default=320)
    prep.add_argument("--height", type=int, default=240)
    prep.add_argument("--workers", type=int, default=2)
    prep.add_argument("--train-pose-seed", type=int, default=11, help="used only when no fixed training split is supplied programmatically")
    prep.add_argument("--eval-pose-seed", type=int, default=22)
    prep.add_argument("--exclude-region", type=float, nargs=4, default=[0.0, 0.0, 245.0, 245.0],
                      help="fallback exclusion; fixed split metadata takes precedence")
    prep.set_defaults(func=prepare)
    ev = sub.add_parser("eval", help="fit familiar canonical probes once, apply unchanged to heldout geometry")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--arm", choices=["fixed", "multi"], required=True,
                    help="encoder training geometry label; probe-fit and test images are identical across arms")
    ev.add_argument("--encoders", nargs="+", required=True)
    ev.add_argument("--fit-frac", type=float, default=0.5)
    ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--retrieval-chunk", type=int, default=256)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--out", type=Path, required=True, help="new output directory; existing artifacts are never replaced")
    ev.set_defaults(func=evaluate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
