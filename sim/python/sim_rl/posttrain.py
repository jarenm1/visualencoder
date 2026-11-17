"""Perception post-training on mined policy failures, and the matching confusion probe.

post-train: fine-tune a distilled student (`visenc distill` run) and the policy's TargetSaliency
head on mined frames (``sim_rl.mine``), sampling failures and correct frames equally. The
saliency loss is the policy's own box-region soft cross-entropy. A cosine distillation term to
the frozen teacher, through the student's original projection heads (frozen), on unlabelled real
frames (UAVid train) and simulator renders anchors the general features against forgetting.
Goals are re-encoded from stored goal-crop images by the encoder being trained.

confusion-probe: the paper comparison. Each frozen encoder gets a fresh TargetSaliency trained on
the same mined training frames (identical seed and schedule), scored on held-out mined frames.

Metrics (both): ``visible_on_target`` = peak patch on the target among frames where it is visible;
``absent_correct`` = P(in view) < 0.5 among frames where it is not; ``score`` = their mean. The
held-out sets keep every sampled observation (unbiased); ``*_failed`` restricts to observations the
mining policy got wrong, which is biased against the mining encoder by construction.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from visenc.distill import cosine_loss
from visenc.models import Student, load_encoder, load_lingbot, save_backbone

from .cli import emit, load_checkpoint, sha256
from .model import GOAL_POOL, PatchMatchSaliency, TargetSaliency, goal_tokens, saliency_loss, saliency_targets

GRID = (15, 20)
TOKEN_CHUNK = 128


class Mined:
    """One mined split held in host memory; batches are moved to the GPU on demand."""

    def __init__(self, path: Path, limit: int | None = None):
        load = lambda name: torch.from_numpy(np.load(path / f"{name}.npy"))
        self.images, self.privileged, self.kind = load("images"), load("privileged").float(), load("kind")
        self.goal_index, self.goal_images = load("goal_index"), load("goal_images")
        if limit is not None and limit < len(self.kind):
            keep = torch.randperm(len(self.kind), generator=torch.Generator().manual_seed(0))[:limit].sort().values
            self.images, self.privileged = self.images[keep], self.privileged[keep]
            self.kind, self.goal_index = self.kind[keep], self.goal_index[keep]

    def __len__(self):
        return len(self.kind)

    def batch(self, ids, device):
        frames = self.images[ids].to(device, non_blocking=True).float().div_(255)
        goals = self.goal_images[self.goal_index[ids]].to(device, non_blocking=True).float().div_(255)
        return frames, goals, self.privileged[ids].to(device)


def encode(backbone, images):
    """Same normalization as FrozenFeatures.encode: layer-normed CLS and patch tokens (float)."""
    cls, patches = backbone(images)
    return F.layer_norm(cls.float(), (cls.shape[-1],)), F.layer_norm(patches.float(), (patches.shape[-1],))


def goal_embedding(backbone, goals):
    cls, patches = encode(backbone, goals)
    return torch.cat((cls, patches.mean(1)), dim=-1)


class Tally:
    """Running counts for the saliency metrics; one host read at the end."""

    def __init__(self, device):
        self.counts = torch.zeros(8, device=device)

    @torch.no_grad()
    def add(self, logits, privileged, failed):
        _, visible, on_target = saliency_targets(privileged, *GRID)
        hit = on_target.gather(1, logits[:, :-1].argmax(-1, keepdim=True)).squeeze(-1)
        absent_ok = logits.softmax(-1)[:, -1] > 0.5
        rows = torch.stack((visible, visible & hit, ~visible, ~visible & absent_ok,
                            visible & failed, visible & failed & hit, ~visible & failed,
                            ~visible & failed & absent_ok)).float().sum(-1)
        self.counts += rows

    def result(self):
        c = self.counts.tolist()
        ratio = lambda a, b: a / b if b else None
        out = {"visible_frames": int(c[0]), "visible_on_target": ratio(c[1], c[0]),
               "absent_frames": int(c[2]), "absent_correct": ratio(c[3], c[2]),
               "visible_failed_frames": int(c[4]), "visible_on_target_failed": ratio(c[5], c[4]),
               "absent_failed_frames": int(c[6]), "absent_correct_failed": ratio(c[7], c[6])}
        parts = [out["visible_on_target"], out["absent_correct"]]
        out["score"] = sum(parts) / 2 if None not in parts else None
        return out


@torch.no_grad()
def evaluate(backbone, saliency, data, device, batch_size=64):
    tally = Tally(device)
    for begin in range(0, len(data), batch_size):
        ids = torch.arange(begin, min(begin + batch_size, len(data)))
        frames, goals, privileged = data.batch(ids, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, tokens = encode(backbone, frames)
            goal = goal_embedding(backbone, goals)
        _, logits = saliency(goal.float(), cls, tokens)
        tally.add(logits.float(), privileged, data.kind[ids].to(device) > 0)
    return tally.result()


def load_policy_saliency(path, token_dim, goal_dim, allow_task_transfer):
    saved = load_checkpoint(path, for_task_transfer=allow_task_transfer)
    saliency = TargetSaliency(token_dim, goal_dim, *GRID)
    state = {key.removeprefix("attention."): value for key, value in saved["model"].items()
             if key.startswith("attention.")}
    saliency.load_state_dict(state)
    return saliency, saved["metadata"]


def image_array(path, limit=None):
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{path}: expected uint8 [N,H,W,3]")
    return array if limit is None else array[:limit]


def anchor_batch(array, count, generator, device):
    ids = np.sort(generator.choice(len(array), count, replace=False))
    images = torch.from_numpy(np.ascontiguousarray(array[ids])).to(device).permute(0, 3, 1, 2).float().div_(255)
    flip = torch.from_numpy(generator.random(count) < 0.5).to(device)
    return torch.where(flip[:, None, None, None], images.flip(-1), images)


def post_train(args):
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device("cuda:0")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    base = args.base
    student = Student(load_lingbot(base / "student", pretrained=True), 1024)
    heads = torch.load(base / "heads.pt", map_location="cpu")
    student.cls_head.load_state_dict(heads["cls_head"])
    student.patch_head.load_state_dict(heads["patch_head"])
    backbone = student.backbone
    blocks = backbone.model.blocks
    # Only the last blocks and the final norm adapt (nothing with --train-blocks 0: head-only ablation);
    # earlier features and the projection heads stay fixed.
    trainable = list(blocks[len(blocks) - args.train_blocks:]) + [backbone.model.norm] if args.train_blocks else []
    student.requires_grad_(False)
    for module in trainable:
        module.requires_grad_(True)
    student = student.to(device)
    saliency, policy_meta = load_policy_saliency(args.policy, backbone.dim, 2 * backbone.dim,
                                                 args.allow_task_transfer)
    if Path(policy_meta["encoder"].removeprefix("run:")).resolve() != base.resolve():
        raise ValueError(f"policy saliency head was trained on {policy_meta['encoder']}, not {base}")
    saliency = saliency.to(device)
    teacher = load_encoder(args.teacher).to(device).eval().requires_grad_(False)

    train = Mined(args.mined / "train")
    validation = Mined(args.held_out / "validation")
    real = image_array(args.real)
    sim = image_array(args.sim)
    failed = (train.kind > 0).nonzero().squeeze(-1)
    correct = (train.kind == 0).nonzero().squeeze(-1)

    backbone_params = [p for module in trainable for p in module.parameters()]
    groups = [{"params": list(saliency.parameters()), "lr": args.head_lr, "weight_decay": 1e-4}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr, "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(groups)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def set_lr(step):
        warm = min(1.0, (step + 1) / args.warmup)
        decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * warm * (0.1 + 0.9 * decay)

    def validate(step):
        student.eval()
        saliency.eval()
        result = evaluate(backbone, saliency, validation, device)
        student.train()
        saliency.train()
        return {"step": step, **result}

    log = (out / "log.jsonl").open("w")
    first = validate(0)
    log.write(json.dumps({"validation": first}) + "\n")
    print(json.dumps({"validation": first}), flush=True)
    best = first
    save(student, saliency, out)
    started = time.perf_counter()
    half = args.batch_size // 2
    for step in range(args.steps):
        set_lr(step)
        ids = torch.cat((failed[torch.randint(len(failed), (half,))],
                         correct[torch.randint(len(correct), (args.batch_size - half,))]))
        frames, goals, privileged = train.batch(ids, device)
        anchors = torch.cat((anchor_batch(real, args.real_batch, generator, device),
                             anchor_batch(sim, args.sim_batch, generator, device)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, tokens = encode(backbone, frames)
            goal = goal_embedding(backbone, goals)
            with torch.no_grad():
                t_cls, t_patch = teacher(anchors)
                t_cls, t_patch = F.normalize(t_cls.float(), dim=-1), F.normalize(t_patch.float(), dim=-1)
            s_cls, s_patch = student(anchors)
        _, logits = saliency(goal.float(), cls, tokens)
        loss_saliency, accuracy = saliency_loss(logits.float(), privileged, *GRID)
        loss_anchor = cosine_loss(s_cls, t_cls) + cosine_loss(s_patch, t_patch)
        loss = loss_saliency + args.anchor_weight * loss_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params + list(saliency.parameters()), 1.0)
        optimizer.step()
        if step % 50 == 0:
            record = {"step": step, "loss_saliency": loss_saliency.item(), "loss_anchor": loss_anchor.item(),
                      "train_patch_accuracy": accuracy.item(), "seconds": time.perf_counter() - started,
                      "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            result = validate(step + 1)
            log.write(json.dumps({"validation": result}) + "\n")
            log.flush()
            print(json.dumps({"validation": result}), flush=True)
            if result["score"] > best["score"]:
                best = result
                save(student, saliency, out)
    log.close()
    emit({"base": str(base), "policy": str(args.policy), "policy_sha256": sha256(args.policy),
          "teacher": args.teacher, "initial_validation": first, "best_validation": best}, out / "summary.json")


def save(student, saliency, out):
    save_backbone(student.backbone, out / "student")
    torch.save({"cls_head": student.cls_head.state_dict(), "patch_head": student.patch_head.state_dict()},
               out / "heads.pt")
    torch.save(saliency.state_dict(), out / "saliency.pt")


@torch.no_grad()
def encode_all(encoder, data, device):
    """Frozen encoder pass on the GPU: cls float [N,D], tokens bf16 [N,P,D], and per unique goal crop
    a goal vector float [U,2D] and goal tokens bf16 [U,1+K,D]; rows map through ``data.goal_index``."""
    cls_all = tokens_all = None
    for begin in range(0, len(data), TOKEN_CHUNK):
        ids = torch.arange(begin, min(begin + TOKEN_CHUNK, len(data)))
        frames = data.images[ids].to(device).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, tokens = encode(encoder, frames)
        if tokens_all is None:   # preallocate: concatenating chunks would double peak memory (ViT-L)
            cls_all = torch.empty((len(data), cls.shape[-1]), device=device)
            tokens_all = torch.empty((len(data), *tokens.shape[1:]), dtype=torch.bfloat16, device=device)
        cls_all[begin:begin + len(ids)] = cls.float()
        tokens_all[begin:begin + len(ids)] = tokens.to(torch.bfloat16)
    vectors, matches = [], []
    for begin in range(0, len(data.goal_images), TOKEN_CHUNK):
        goals = data.goal_images[begin:begin + TOKEN_CHUNK].to(device).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, patches = encode(encoder, goals)
        vectors.append(torch.cat((cls, patches.mean(1)), dim=-1).float())
        matches.append(goal_tokens(cls, patches, *GRID).to(torch.bfloat16))
    return cls_all, tokens_all, torch.cat(vectors), torch.cat(matches)


def confusion_probe(args):
    device = torch.device("cuda:0")
    train = Mined(args.mined / "train", limit=args.train_frames)
    test = Mined(args.held_out / "test")
    train_goal, test_goal = train.goal_index.to(device), test.goal_index.to(device)
    results = {}
    for spec in args.encoders:
        encoder = load_encoder(spec).to(device).eval().requires_grad_(False)
        cls, tokens, vectors, matches = encode_all(encoder, train, device)   # one split resident at a time
        dim, heads = tokens.shape[-1], {}
        privileged = train.privileged.to(device)
        for head in args.heads:
            torch.manual_seed(args.seed)
            heads[head] = saliency = (TargetSaliency(dim, 2 * dim, *GRID) if head == "global"
                                      else PatchMatchSaliency(dim, *GRID)).to(device)
            optimizer = torch.optim.Adam(saliency.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            order_generator = torch.Generator(device=device).manual_seed(args.seed)
            for _ in range(args.epochs):
                order = torch.randperm(len(train), device=device, generator=order_generator)
                for begin in range(0, len(train), args.batch_size):
                    ids = order[begin:begin + args.batch_size]
                    goal = vectors[train_goal[ids]] if head == "global" else matches[train_goal[ids]]
                    _, logits = saliency(goal, cls[ids], tokens[ids])
                    loss, _ = saliency_loss(logits, privileged[ids], *GRID)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            del optimizer
        del cls, tokens, vectors, matches, privileged
        torch.cuda.empty_cache()
        cls, tokens, vectors, matches = encode_all(encoder, test, device)
        del encoder
        for head, saliency in heads.items():
            tally = Tally(device)
            with torch.no_grad():
                for begin in range(0, len(test), 256):
                    ids = torch.arange(begin, min(begin + 256, len(test)), device=device)
                    goal = vectors[test_goal[ids]] if head == "global" else matches[test_goal[ids]]
                    _, logits = saliency(goal, cls[ids], tokens[ids])
                    tally.add(logits, test.privileged[ids.cpu()].to(device), test.kind[ids.cpu()].to(device) > 0)
            results.setdefault(spec, {"dim": int(dim)})[head] = tally.result()
            print(json.dumps({spec: {head: results[spec][head]}}), flush=True)
        del cls, tokens, vectors, matches, heads
        torch.cuda.empty_cache()
    emit({"mined": str(args.mined), "held_out": str(args.held_out), "train_frames": len(train),
          "test_frames": len(test), "epochs": args.epochs, "batch_size": args.batch_size,
          "learning_rate": args.learning_rate, "seed": args.seed, "goal_pool": list(GOAL_POOL),
          "encoders": results}, args.out)
