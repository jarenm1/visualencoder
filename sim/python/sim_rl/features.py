"""Frozen visual backbones: online GPU inference, CLS + patch-token outputs only."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from visenc.models import load_encoder

ARMS = {"pretrained": "lingbot:small", "baseline": "run:runs/baseline", "mean": "run:runs/mean",
        "state": "state", "hint": "hint"}
PREPROCESSING = "native-NCHW-float01/ImageNet/bf16/CLS+patch-grid16/token-LayerNorm/v3"
GOAL_CHUNK = 8


class FrozenFeatures:
    """Visual arms own a frozen encoder; ``state`` and ``hint`` arms own none."""

    def __init__(self, spec, device):
        self.spec = ARMS.get(spec, spec)
        self.device = device
        self.encoder = None
        self.dim = 0
        if self.spec in ("state", "hint"):
            self.fingerprint = f"city-drone-{self.spec}-only-v1"
            return
        if self.spec != "lingbot:small" and not self.spec.startswith("run:"):
            raise ValueError("visual arms must use lingbot:small or an actual distilled run:<dir>")
        encoder = load_encoder(self.spec).eval().requires_grad_(False)
        digest = hashlib.sha256()
        # Hash the actual loaded parameters before the one-time model upload.
        for name, tensor in sorted(encoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        self.dim = encoder.dim
        self.encoder = encoder.to(device=device, dtype=torch.bfloat16)

    @property
    def visual(self):
        return self.encoder is not None

    @torch.no_grad()
    def encode(self, images):
        """images: float32 [m,3,H,W] in [0,1]. Returns layer-normalized (cls f32 [m,D], patches bf16 [m,P,D])."""
        if self.encoder is None:
            raise ValueError(f"{self.spec} arm has no visual encoder")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, patches = self.encoder(images)
        height, width = images.shape[2] // 16, images.shape[3] // 16
        if patches.shape[1] != height * width:
            raise ValueError("encoder patch layout is not the common 16px grid")
        cls = F.layer_norm(cls.float(), (cls.shape[-1],))
        patches = F.layer_norm(patches.float(), (patches.shape[-1],)).to(torch.bfloat16)
        return cls, patches

    @torch.no_grad()
    def observe(self, env):
        """Encode the current env.obs; the only visual pass per control step."""
        return self.encode(env.obs)

    @torch.no_grad()
    def goal_table(self, goal_images):
        """Per-design goal embedding [K, 2D] = concat(CLS, mean patch token); computed once per env."""
        if self.encoder is None:
            raise ValueError("goal embeddings require a visual arm")
        rows = []
        for start in range(0, goal_images.shape[0], GOAL_CHUNK):
            cls, patches = self.encode(goal_images[start:start + GOAL_CHUNK])
            rows.append(torch.cat((cls, patches.float().mean(1)), dim=-1))
        return torch.cat(rows, dim=0)
