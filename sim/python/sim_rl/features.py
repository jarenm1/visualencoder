"""Actual frozen backbones, online GPU inference, rollout-only feature reuse."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from visenc.models import load_encoder

ARMS = {"pretrained": "lingbot:small", "baseline": "run:runs/baseline", "mean": "run:runs/mean",
        "state": "state", "proprio": "proprio"}
PREPROCESSING = "native-NCHW-float01/ImageNet/bf16/CLS+spatial2x3/token-LayerNorm/v1"


class FrozenFeatures:
    def __init__(self, spec, device):
        self.spec = ARMS.get(spec, spec)
        self.encoder = None
        self.device = device
        if self.spec in ("state", "proprio"):
            self.dim = 4 if self.spec == "state" else 0
            self.fingerprint = f"city-{self.spec}-only-v1"
        else:
            if self.spec != "lingbot:small" and not self.spec.startswith("run:"):
                raise ValueError("visual arms must use lingbot:small or an actual distilled run:<dir>")
            encoder = load_encoder(self.spec).eval().requires_grad_(False)
            digest = hashlib.sha256()
            # Initialization only; hash the actual loaded parameters before the
            # one-time model upload, not a checkpoint name or config substitute.
            for name, tensor in sorted(encoder.state_dict().items()):
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self.fingerprint = digest.hexdigest()
            self.dim = encoder.dim * 7
            self.encoder = encoder.to(device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def __call__(self, env):
        if self.spec == "state":
            return env.privileged
        if self.spec == "proprio":
            return env.proprio[:, :0]
        return self.encode(env)[0]

    @torch.no_grad()
    def encode(self, env):
        """Policy tokens plus the unpooled [N, D, H/16, W/16] patch grid from one encoder pass."""
        if self.encoder is None:
            raise ValueError(f"{self.spec} arm has no visual encoder")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls, patches = self.encoder(env.obs)
        height, width = env.height // 16, env.width // 16
        if patches.shape[1] != height * width:
            raise ValueError("encoder patch layout is not the common 16px grid")
        grid = patches.float().transpose(1, 2).reshape(env.n, -1, height, width)
        pooled = F.adaptive_avg_pool2d(grid, (2, 3)).flatten(2).transpose(1, 2)
        tokens = torch.cat((cls.float().unsqueeze(1), pooled), dim=1)
        return F.layer_norm(tokens, (tokens.shape[-1],)).flatten(1), grid
