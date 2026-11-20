"""Feature-level encoder post-training on mined drone frames (no task head).

The student (a `visenc distill` run) adapts its last ``train_blocks`` blocks; the frozen teacher
and the student's frozen projection heads define the distillation space. Every step mixes three
image sources and three losses:

  anchor distillation   cosine to the teacher (CLS + every patch) on UAVid real frames (weighted
                        ``real_weight``) and simulator renders: keeps general and real-image quality
  hard distillation     the same cosine on mined drone frames, each patch weighted by how far the
                        student is from the teacher there (normalized to mean 1): it concentrates on
                        the patches the student gets most wrong, mostly small and distant objects
  instance contrast     supervised contrastive loss (SupCon) on patch features of pairs of frames
                        from the same flight: patches of one layout object are positives across
                        frames, every other object is a negative (same-class look-alikes included);
                        small objects are always sampled. A throwaway projection head carries it.

There is no model selection on a training-time score: the run trains a fixed schedule and saves
the final student. Acceptance is decided only by ``encoder_gate.py``.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from visenc.models import Student, load_encoder, load_lingbot, save_backbone

from .cli import emit
from .posttrain import anchor_batch, image_array

SMALL_PATCHES = 8          # an object covering at most this many patches in a frame counts as small
PATCHES_PER_OBJECT = 8     # patches sampled per small object per frame
PATCHES_PER_FRAME = 24     # labelled patches sampled per frame
PARTNER_STEPS = 100        # a frame's partner is another frame of the same flight within this many steps


class Flights:
    """A mined split (memory-mapped frames, per-patch object labels) and same-flight partner frames."""

    def __init__(self, path: Path):
        self.images = np.load(path / "images.npy", mmap_mode="r")                  # [N,3,H,W] uint8
        self.objects = np.load(path / "patch_object.npy", mmap_mode="r")           # [N,P] int32
        catalog, step = np.load(path / "catalog.npy"), np.load(path / "step.npy")
        # A mined frame's flight is (round, catalog entry); partners are its other frames nearby in time.
        flight = (step // 10_000) * 1_000_000 + catalog
        order = np.lexsort((step, flight))
        self.partners = [np.empty(0, dtype=np.int64)] * len(catalog)
        start = 0
        while start < len(order):
            end = start
            while end < len(order) and flight[order[end]] == flight[order[start]]:
                end += 1
            members = order[start:end]
            for i in members:
                near = members[(np.abs(step[members] - step[i]) <= PARTNER_STEPS) & (members != i)]
                self.partners[i] = near
            start = end
        self.paired = np.flatnonzero([len(p) > 0 for p in self.partners])

    def sample_pairs(self, count, rng):
        anchors = rng.choice(self.paired, count, replace=False)
        partners = np.array([rng.choice(self.partners[i]) for i in anchors])
        return np.concatenate((anchors, partners))

    def batch(self, ids, device):
        frames = torch.from_numpy(np.ascontiguousarray(self.images[np.sort(ids)])).to(device).float().div_(255)
        order = np.argsort(np.argsort(ids))          # undo the sort used for sequential mmap reads
        labels = torch.from_numpy(np.ascontiguousarray(self.objects[np.sort(ids)])).to(device).long()
        return frames[order], labels[order]


def sample_patches(labels, rng):
    """(frame, patch) indices and object labels of labelled patches: every small object in a frame
    (up to PATCHES_PER_OBJECT patches each) first, then random patches of larger objects."""
    rows, cols, objects = [], [], []
    host = labels.cpu().numpy()
    for frame, patch_labels in enumerate(host):
        present, counts = np.unique(patch_labels[patch_labels >= 0], return_counts=True)
        chosen = []
        for obj in present[counts <= SMALL_PATCHES]:
            where = np.flatnonzero(patch_labels == obj)
            chosen.extend(rng.choice(where, min(len(where), PATCHES_PER_OBJECT), replace=False))
        large = np.flatnonzero(np.isin(patch_labels, present[counts > SMALL_PATCHES]))
        room = max(PATCHES_PER_FRAME - len(chosen), 0)
        if room and len(large):
            chosen.extend(rng.choice(large, min(room, len(large)), replace=False))
        rows.extend([frame] * len(chosen))
        cols.extend(chosen)
        objects.extend(patch_labels[chosen])
    device = labels.device
    return (torch.tensor(rows, device=device), torch.tensor(cols, device=device),
            torch.tensor(objects, device=device))


def view_batch(views, count, rng, device):
    """`count` poses from uint8 [N,H,W,3] view arrays of the same poses, scene-major
    [p0v0, p0v1, p1v0, ...] float [count*V,3,H,W]; all views of a pose share one flip."""
    ids = np.sort(rng.choice(len(views[0]), count, replace=False))
    flip = torch.from_numpy(rng.random(count) < 0.5).to(device)
    images = torch.stack([torch.from_numpy(np.ascontiguousarray(view[ids])) for view in views], 1)
    images = images.to(device).permute(0, 1, 4, 2, 3).float().div_(255)
    images = torch.where(flip[:, None, None, None, None], images.flip(-1), images)
    return images.flatten(0, 1)


def supcon(features, labels, temperature):
    """Supervised contrastive loss (Khosla et al., 2020) over anchors that have a positive."""
    z = F.normalize(features.float(), dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -1e9)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = logits - logits.logsumexp(1, keepdim=True)
    count = positives.sum(1)
    valid = count > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    return -((log_prob * positives).sum(1)[valid] / count[valid]).mean()


def refine(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    student = Student(load_lingbot(args.base / "student", pretrained=True), 1024)
    heads = torch.load(args.base / "heads.pt", map_location="cpu")
    student.cls_head.load_state_dict(heads["cls_head"])
    student.patch_head.load_state_dict(heads["patch_head"])
    backbone = student.backbone
    blocks = backbone.model.blocks
    trainable = list(blocks[len(blocks) - args.train_blocks:]) + [backbone.model.norm]
    student.requires_grad_(False)
    for module in trainable:
        module.requires_grad_(True)
    student = student.to(device)
    projector = nn.Sequential(nn.Linear(backbone.dim, 512), nn.GELU(), nn.Linear(512, 128)).to(device)
    teacher = load_encoder(args.teacher).to(device).eval().requires_grad_(False)

    flights = Flights(args.mined)
    real = image_array(args.real)
    sim_views = [image_array(path) for path in args.sim_views]
    if len({len(view) for view in sim_views}) != 1:
        raise ValueError("--sim-views must be views of the same poses (equal lengths)")
    backbone_params = [p for module in trainable for p in module.parameters()]
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": args.lr, "weight_decay": 0.05},
                                   {"params": list(projector.parameters()), "lr": args.head_lr, "weight_decay": 1e-4}])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    log = (out / "log.jsonl").open("w")
    started = time.perf_counter()
    for step in range(args.steps):
        warm = min(1.0, (step + 1) / args.warmup)
        decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * warm * (0.1 + 0.9 * decay)
        frames, labels = flights.batch(flights.sample_pairs(args.pairs, rng), device)
        anchors_real = anchor_batch(real, args.real_batch, rng, device)
        anchors_sim = view_batch(sim_views, args.sim_batch, rng, device)
        images = torch.cat((frames, anchors_real, anchors_sim))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_cls, t_patch = teacher(images)
                # Simulator anchors keep the original distillation's target: the teacher features
                # averaged over a pose's appearance views (one representation per physical state).
                t_cls, t_patch = F.normalize(t_cls.float(), dim=-1), F.normalize(t_patch.float(), dim=-1)
                v, start = len(sim_views), len(frames) + len(anchors_real)
                t_cls[start:] = t_cls[start:].unflatten(0, (-1, v)).mean(1, keepdim=True).expand(-1, v, -1).flatten(0, 1)
                t_patch[start:] = (t_patch[start:].unflatten(0, (-1, v)).mean(1, keepdim=True)
                                   .expand(-1, v, -1, -1).flatten(0, 1))
            cls, tokens = backbone(images)
            s_cls, s_patch = student.cls_head(cls), student.patch_head(tokens)
        cls_gap = 1 - F.cosine_similarity(s_cls.float(), t_cls.float(), dim=-1)            # [B]
        patch_gap = 1 - F.cosine_similarity(s_patch.float(), t_patch.float(), dim=-1)       # [B,P]
        m, r = len(frames), len(anchors_real)
        # Hard distillation on drone frames: patches weighted by their (detached) student-teacher gap.
        weight = patch_gap[:m].detach()
        weight = weight / weight.mean().clamp_min(1e-6)
        loss_hard = (weight * patch_gap[:m]).mean() + cls_gap[:m].mean()
        loss_anchor = (args.real_weight * (patch_gap[m:m + r].mean() + cls_gap[m:m + r].mean())
                       + patch_gap[m + r:].mean() + cls_gap[m + r:].mean())
        rows, cols, objects = sample_patches(labels, rng)
        loss_contrast = supcon(projector(tokens[rows, cols].float()), objects, args.temperature)
        loss = loss_anchor + args.hard_weight * loss_hard + args.contrast_weight * loss_contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params + list(projector.parameters()), 1.0)
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.steps:
            record = {"step": step, "loss_anchor": loss_anchor.item(), "loss_hard": loss_hard.item(),
                      "loss_contrast": loss_contrast.item(), "contrast_patches": len(objects),
                      "real_patch_gap": patch_gap[m:m + r].mean().item(),
                      "seconds": round(time.perf_counter() - started, 1),
                      "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    log.close()
    save_backbone(student.backbone, out / "student")
    torch.save({"cls_head": student.cls_head.state_dict(), "patch_head": student.patch_head.state_dict()},
               out / "heads.pt")
    emit({"base": str(args.base), "teacher": args.teacher, "mined": str(args.mined), "steps": args.steps,
          "frames": len(flights.images), "paired_frames": len(flights.paired)}, out / "summary.json")
