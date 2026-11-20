"""Encoder failure taxonomy on mined flight frames, from fresh frozen-feature probes.

Inputs: a mined split (``mine.py``: frames with per-patch object labels) and the per-frame probe
samples of one or more encoders on that split (``confusion-probe --samples``). Each frame is
classified per encoder:

  target visible      correct | background | other_class | look_alike_appearance |
                      look_alike_context | look_alike_identical
  target not visible  correct | false_background | false_other_class | false_look_alike_appearance |
                      false_look_alike_context | false_look_alike_identical

A wrong peak (or a false "in view") is attributed to the object under the probe's peak patch:
none (background), another class, the same class but another asset (it looks different:
learnable from appearance), or the same asset. Same-asset instances are identical meshes with the
same palette and appearance draw, so they differ only by surroundings: ``context`` when the
assets within CONTEXT_RADIUS_M of the two objects differ (learnable if that context is in view),
``identical`` when they do not (not learnable from the image; training on them teaches location
memorisation). The neighbourhood test is a layout proxy for visible context.

With two encoders (student, teacher) the joint table splits student failures into those the
teacher gets right (hard-example distillation targets) and those both miss (need a new signal).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from visenc.evaluate import sample_name

CONTEXT_RADIUS_M = 10.0
VISIBLE = ("correct", "background", "other_class", "look_alike_appearance", "look_alike_context",
           "look_alike_identical")
ABSENT = ("correct", "false_background", "false_other_class", "false_look_alike_appearance",
          "false_look_alike_context", "false_look_alike_identical")
TARGET_OBJECT = 27   # diagnostics lane: layout index of the episode's target object


def neighbourhoods(objects, radius=CONTEXT_RADIUS_M):
    """Per layout object: frozenset of the assets whose box centres lie within `radius` (xz)."""
    centres = np.array([[(o["min"][0] + o["max"][0]) / 2, (o["min"][2] + o["max"][2]) / 2] for o in objects])
    assets = [o["asset"] for o in objects]
    out = []
    for i, c in enumerate(centres):
        near = np.flatnonzero(np.hypot(*(centres - c).T) <= radius)
        out.append(frozenset(assets[j] for j in near if j != i))
    return out


def relation(peak_object, target, objects, context):
    """How the object under the peak relates to the target (suffix of the category names)."""
    if peak_object < 0:
        return "background"
    if peak_object == target:
        return None
    a, b = objects[peak_object], objects[target]
    if a["class"] != b["class"]:
        return "other_class"
    if a["asset"] != b["asset"]:
        return "look_alike_appearance"
    return "look_alike_context" if context[peak_object] != context[target] else "look_alike_identical"


def classify(split, samples, objects, context):
    """Category name per frame for one encoder's probe samples on a mined split."""
    patch_object = np.load(split / "patch_object.npy")
    target = np.load(split / "diagnostics.npy")[:, TARGET_OBJECT].astype(np.int64)
    visible, hit, absent_ok = samples["outcomes"].T.astype(bool)
    peak = samples["peak"]
    under_peak = patch_object[np.arange(len(peak)), peak]
    names = []
    for i in range(len(peak)):
        if visible[i]:
            names.append("correct" if hit[i] else
                         relation(under_peak[i], target[i], objects, context) or "background")
        elif absent_ok[i]:
            names.append("correct")
        else:
            names.append("false_" + (relation(under_peak[i], target[i], objects, context) or "background"))
    return np.array(names), visible


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, required=True, help="mined split directory (with patch_object.npy)")
    parser.add_argument("--samples", type=Path, required=True, help="confusion-probe --samples directory")
    parser.add_argument("--encoders", nargs="+", required=True, help="specs as given to confusion-probe")
    parser.add_argument("--head", choices=("patch", "global"), default="patch")
    parser.add_argument("--layout", type=Path, default=Path("assets/city/layout.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    objects = json.loads(args.layout.read_text())["objects"]
    context = neighbourhoods(objects)
    report, labels = {"context_radius_m": CONTEXT_RADIUS_M, "head": args.head, "encoders": {}}, {}
    for spec in args.encoders:
        samples = np.load(args.samples / f"{sample_name(spec)}-{args.head}.npz")
        names, visible = classify(args.split, samples, objects, context)
        labels[spec] = names
        report["encoders"][spec] = {
            "visible_frames": int(visible.sum()), "absent_frames": int((~visible).sum()),
            "visible": {k: int(v) for k, v in Counter(names[visible]).items()},
            "absent": {k: int(v) for k, v in Counter(names[~visible]).items()},
        }
    if len(args.encoders) == 2:
        student, teacher = (labels[s] for s in args.encoders)
        s_fail, t_fail = student != "correct", teacher != "correct"
        report["joint"] = {
            "student_fails_teacher_correct": Counter(student[s_fail & ~t_fail]),
            "both_fail": Counter(student[s_fail & t_fail]),
            "teacher_fails_student_correct": Counter(teacher[t_fail & ~s_fail]),
        }
        report["joint"] = {k: {name: int(n) for name, n in v.items()} for k, v in report["joint"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    np.savez(args.out.with_suffix(".npz"), **{sample_name(s): labels[s] for s in args.encoders})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
