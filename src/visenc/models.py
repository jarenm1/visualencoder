"""Encoders with a common interface: uint8/float images in, (cls [B,D], patches [B,P,D]) out."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16


class Encoder(nn.Module):
    dim: int

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class HFViT(Encoder):
    """DINOv3-style HF ViT. Token layout: [CLS, registers..., patches] after the final norm."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        config = model.config
        if config.patch_size != PATCH:
            raise ValueError(f"expected patch size {PATCH}, got {config.patch_size}")
        self.num_registers = getattr(config, "num_register_tokens", 0)
        self.dim = config.hidden_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images: float [B,3,H,W] in [0,1], H and W multiples of 16."""
        x = (images - self.mean) / self.std
        tokens = self.model(pixel_values=x).last_hidden_state
        return tokens[:, 0], tokens[:, 1 + self.num_registers :]


class Pixels(Encoder):
    """Floor baseline: cls = image average-pooled to the patch grid; patches = raw 16x16 RGB patches."""

    dim = 3 * PATCH * PATCH

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool2d(images, PATCH).flatten(1)
        patches = F.unfold(images, PATCH, stride=PATCH).transpose(1, 2)
        return pooled, patches


class LingBotViT(Encoder):
    """LingBot-Vision ViT (DINOv3-style: registers, RoPE). Same token outputs as HFViT."""

    def __init__(self, model: nn.Module, variant: str):
        super().__init__()
        self.model, self.variant = model, variant
        self.dim = model.embed_dim
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward_features((images - self.mean) / self.std)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "lingbot.json").write_text(json.dumps({"variant": self.variant}))


def load_hf(name_or_path: str | Path, pretrained: bool) -> HFViT:
    if pretrained:
        return HFViT(AutoModel.from_pretrained(name_or_path))
    return HFViT(AutoModel.from_config(AutoConfig.from_pretrained(name_or_path)))


def load_lingbot(variant_or_dir: str | Path, pretrained: bool) -> LingBotViT:
    """variant_or_dir: small|base|large|giant, or a directory written by LingBotViT.save."""
    from lingbot_vision import load_backbone, load_backbone_state, load_config
    from lingbot_vision.build import build_backbone_from_cfg
    from lingbot_vision.loader import PRETRAINED_VARIANTS, resolve_pretrained_files

    local = Path(variant_or_dir) / "lingbot.json"
    if local.exists():
        variant = json.loads(local.read_text())["variant"]
        checkpoint = Path(variant_or_dir) / "model.pt"
    else:
        variant = str(variant_or_dir)
        if variant not in PRETRAINED_VARIANTS:
            raise ValueError(f"unknown LingBot-Vision variant {variant!r}; use {sorted(PRETRAINED_VARIANTS)}")
        checkpoint = resolve_pretrained_files(variant=variant)[1] if pretrained else None
    cfg = load_config(PRETRAINED_VARIANTS[variant]["config_file"])
    if checkpoint is None:
        model, _ = build_backbone_from_cfg(cfg)
    else:
        model, _ = load_backbone(cfg, load_backbone_state(checkpoint), device="cpu", dtype=torch.float32, verbose=False)
    return LingBotViT(model.float().requires_grad_(True), variant)  # load_backbone returns it frozen


def load_encoder(spec: str) -> Encoder:
    """Specs:
    pixels                  raw-pixel floor
    hf:<id|path>            pretrained HF backbone (e.g. hf:facebook/dinov3-vits16-pretrain-lvd1689m)
    random:<id|path>        same HF architecture, random init
    lingbot:<variant>       pretrained LingBot-Vision (small|base|large|giant)
    lingbot-random:<variant> same LingBot architecture, random init
    run:<dir>               distilled student saved by `visenc distill`
    """
    kind, _, arg = spec.partition(":")
    if kind == "pixels":
        return Pixels()
    if kind == "hf":
        return load_hf(arg, pretrained=True)
    if kind == "random":
        torch.manual_seed(0)
        return load_hf(arg, pretrained=False)
    if kind == "lingbot":
        return load_lingbot(arg, pretrained=True)
    if kind == "lingbot-random":
        torch.manual_seed(0)
        return load_lingbot(arg, pretrained=False)
    if kind == "run":
        student = Path(arg) / "student"
        if (student / "lingbot.json").exists():
            return load_lingbot(student, pretrained=True)
        return load_hf(student, pretrained=True)
    raise ValueError(f"unknown encoder spec {spec!r}")


def load_student(spec: str, init: str) -> HFViT | LingBotViT:
    """Trainable backbone from an hf:<id> or lingbot:<variant> spec; init is pretrained|random."""
    kind, _, arg = spec.partition(":")
    if kind == "hf":
        return load_hf(arg, pretrained=init == "pretrained")
    if kind == "lingbot":
        return load_lingbot(arg, pretrained=init == "pretrained")
    raise ValueError(f"student spec must be hf:<id> or lingbot:<variant>, got {spec!r}")


def save_backbone(backbone: HFViT | LingBotViT, path: Path) -> None:
    if isinstance(backbone, LingBotViT):
        backbone.save(path)
    else:
        backbone.model.save_pretrained(path)


class Student(nn.Module):
    """Student backbone plus linear heads projecting into the teacher's feature space."""

    def __init__(self, backbone: HFViT | LingBotViT, teacher_dim: int):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(backbone.dim, teacher_dim)
        self.patch_head = nn.Linear(backbone.dim, teacher_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, patches = self.backbone(images)
        return self.cls_head(cls), self.patch_head(patches)
