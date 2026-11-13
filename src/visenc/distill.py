"""Feature distillation from a frozen DINOv3 teacher into a ViT student.

Per pose, every requested appearance view goes through one shared random-resized crop and
flip (so patch grids stay aligned across views and between teacher and student). The
teacher sees the clean crop; the student optionally sees a photometrically corrupted copy.

Targets (cosine loss on CLS and on every patch token):
  own   each view distils the teacher's features of that same view
  mean  every view of a pose distils the view-averaged teacher features, i.e. the student
        is pushed to give one physical state the same representation under any appearance
Unpaired extra images (e.g. real frames) always use `own`.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader

from .data import ImageFolder, PairedViews, Split
from .models import Student, load_encoder, load_student, save_backbone


def shared_crop(views: torch.Tensor, scale: tuple[float, float]) -> torch.Tensor:
    """views: float [B,V,3,H,W]. One crop+flip per scene, applied to all V views."""
    b, _, _, h, w = views.shape
    out = torch.empty_like(views)
    for i in range(b):
        top, left, ch, cw = T.RandomResizedCrop.get_params(views[i, 0], scale, (w / h * 3 / 4, w / h * 4 / 3))
        crop = TF.resized_crop(views[i], top, left, ch, cw, [h, w], antialias=True)
        out[i] = TF.horizontal_flip(crop) if torch.rand(()) < 0.5 else crop
    return out


class PhotometricAug:
    """Appearance-only corruption for the student input (geometry untouched)."""

    def __init__(self):
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.gray = T.RandomGrayscale(p=0.1)
        self.blur = T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.3)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        out = torch.stack([self.blur(self.gray(self.jitter(im))) for im in images])
        sigma = torch.rand(len(out), 1, 1, 1, device=out.device) * 0.03
        return (out + sigma * torch.randn_like(out)).clamp(0, 1)


def cosine_loss(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(student.float(), target.float(), dim=-1)).mean()


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    split = Split(args.data)
    size = split.size
    poses = PairedViews(split, args.views)
    loader = DataLoader(poses, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
    batches = _cycle(loader)
    extra = None
    if args.extra_images:
        extra_ds = ImageFolder(args.extra_images, size)
        extra = _cycle(
            DataLoader(extra_ds, args.extra_batch_size, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)
        )

    teacher = load_encoder(args.teacher).to(device).eval().requires_grad_(False)
    backbone = load_student(args.student, args.student_init)
    student = Student(backbone, teacher.dim).to(device).train()
    params = [p for p in student.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    photometric = PhotometricAug() if args.student_aug else None
    log = (out / "log.jsonl").open("a")
    started = time.perf_counter()
    n_views = len(args.views)
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        views = next(batches).to(device, non_blocking=True).float().div_(255)  # [B,V,3,H,W]
        views = shared_crop(views, (args.crop_min_scale, 1.0))
        b = views.shape[0]
        clean = views.flatten(0, 1)  # scene-major: [b0v0, b0v1, b1v0, ...]
        paired_count = len(clean)
        if extra is not None:
            real = next(extra).to(device, non_blocking=True).float().div_(255)
            clean = torch.cat([clean, shared_crop(real, (args.crop_min_scale, 1.0)).flatten(0, 1)])
        student_in = photometric(clean) if photometric else clean

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_cls, t_patch = teacher(clean)
                t_cls, t_patch = F.normalize(t_cls.float(), dim=-1), F.normalize(t_patch.float(), dim=-1)
                if args.pair_target == "mean" and n_views > 1:
                    pc = t_cls[:paired_count].view(b, n_views, -1).mean(1, keepdim=True).expand(-1, n_views, -1)
                    pp = t_patch[:paired_count].unflatten(0, (b, n_views)).mean(1, keepdim=True).expand(-1, n_views, -1, -1)
                    t_cls = torch.cat([pc.flatten(0, 1), t_cls[paired_count:]])
                    t_patch = torch.cat([pp.flatten(0, 1), t_patch[paired_count:]])
            s_cls, s_patch = student(student_in)
        loss_cls = cosine_loss(s_cls, t_cls)
        loss_patch = cosine_loss(s_patch, t_patch)
        loss = args.cls_weight * loss_cls + args.patch_weight * loss_patch

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - started
            rec = {
                "step": step,
                "loss": loss.item(),
                "loss_cls": loss_cls.item(),
                "loss_patch": loss_patch.item(),
                "grad_norm": grad_norm.item(),
                "lr": lr_at(step),
                "images_per_s": (step + 1) * len(clean) / elapsed,
                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save(student, out)
    log.close()


def save(student: Student, out: Path) -> None:
    save_backbone(student.backbone, out / "student")
    torch.save({"cls_head": student.cls_head.state_dict(), "patch_head": student.patch_head.state_dict()}, out / "heads.pt")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="split directory from `visenc data`")
    p.add_argument("--views", nargs="+", default=["random1", "random2"], help="appearance views per pose, e.g. random1 random2")
    p.add_argument("--pair-target", choices=["own", "mean"], default="own")
    p.add_argument("--extra-images", help="folder of unlabelled images (e.g. real frames), distilled with own targets")
    p.add_argument("--extra-batch-size", type=int, default=16)
    p.add_argument("--teacher", default="lingbot:large", help="encoder spec (see visenc.models.load_encoder)")
    p.add_argument("--student", default="lingbot:small", help="hf:<id> or lingbot:<variant>; architecture (and weights if pretrained)")
    p.add_argument("--student-init", choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--student-aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--crop-min-scale", type=float, default=0.4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32, help="poses per step (images = poses x views); ~1.6 GiB + 0.14 GiB/pose")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--patch-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="new run directory")
