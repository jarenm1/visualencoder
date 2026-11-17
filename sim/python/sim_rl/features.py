"""Frozen visual backbones: online GPU inference, CLS + patch-token outputs only."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from visenc.models import load_encoder

ARMS = {"pretrained": "lingbot:small", "baseline": "run:runs/baseline", "mean": "run:runs/mean",
        "state": "state", "hint": "hint"}
# Recorded in checkpoints; each string fixes the encoder numerics the policy was trained on.
PREPROCESSING = {
    "bf16": "native-NCHW-float01/ImageNet/bf16/CLS+patch-grid16/token-LayerNorm/v3",
    "fp16": "native-NCHW-float01/ImageNet/fp16-tensorrt/CLS+patch-grid16/token-LayerNorm/v3",
}
PRECISIONS = tuple(PREPROCESSING)


def precision_of(preprocessing):
    """Encoder precision recorded by a checkpoint's ``preprocessing`` string."""
    for precision, text in PREPROCESSING.items():
        if text == preprocessing:
            return precision
    raise ValueError(f"unknown encoder preprocessing {preprocessing!r}")


def crop_goals(images, crops):
    """Bilinear crop-and-resize: images float32 [m,3,H,W], crops [m,4] normalized [x0,y0,x1,y1]
    (top-left origin) -> [m,3,H,W]. The same operation turns any box in a frame (catalog goal
    reference, operator tap, language-model detection) into the policy's goal image."""
    x0, y0, x1, y1 = crops.unbind(-1)
    zeros = torch.zeros_like(x0)
    theta = torch.stack((torch.stack((x1 - x0, zeros, x0 + x1 - 1), -1),
                         torch.stack((zeros, y1 - y0, y0 + y1 - 1), -1)), 1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


class FrozenFeatures:
    """Visual arms own a frozen encoder; ``state`` and ``hint`` arms own none.

    ``precision`` selects the inference path: ``bf16`` is the torch.compile'd encoder under bf16
    autocast (the numerics of every pre-TensorRT checkpoint); ``fp16`` is a TensorRT FP16 engine.
    """

    def __init__(self, spec, device, precision):
        self.spec = ARMS.get(spec, spec)
        self.device = device
        self.precision = precision
        self.preprocessing = PREPROCESSING[precision]
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
        if precision == "bf16":
            self.encoder = torch.compile(encoder.to(device=device, dtype=torch.bfloat16))
        else:
            from .tensorrt_encoder import TensorRTEncoder
            self.encoder = TensorRTEncoder(encoder, self.fingerprint, device)

    @property
    def visual(self):
        return self.encoder is not None

    @torch.no_grad()
    def encode(self, images):
        """images: float32 [m,3,H,W] in [0,1]. Returns layer-normalized (cls f32 [m,D], patches bf16 [m,P,D])."""
        if self.encoder is None:
            raise ValueError(f"{self.spec} arm has no visual encoder")
        if self.precision == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls, patches = self.encoder(images)
        else:
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
    def goal_embedding(self, images):
        """Goal embedding [m, 2D] = concat(CLS, mean patch token) of goal images [m,3,H,W]."""
        if self.encoder is None:
            raise ValueError("goal embeddings require a visual arm")
        cls, patches = self.encode(images)
        return torch.cat((cls, patches.float().mean(1)), dim=-1)

    @torch.no_grad()
    def catalog_goals(self, env, appearance):
        """bf16 [catalog_size, 2D] goal embedding per catalog entry, built once per env, encoder and
        appearance strength.

        Startup work outside collection: the env renders each entry's goal reference view (with that
        entry's randomized appearance at ``appearance`` strength), which is cropped to its goal box
        and encoded. Episodes look up their row by ``env.catalog_index``.
        """
        cache = env.__dict__.setdefault("goal_tables", {})
        key = (self.fingerprint, float(appearance))
        if key not in cache:
            restore = env.curriculum.clone()
            env.curriculum[4] = appearance
            table = torch.empty((env.catalog_size, 2 * self.dim), device=env.device, dtype=torch.bfloat16)
            for first in range(0, env.catalog_size, env.n):
                count = min(env.n, env.catalog_size - first)
                env.render_goals(first)
                goals = crop_goals(env.goal_views[:count], env.goal_crops[:count])
                table[first:first + count] = self.goal_embedding(goals).to(torch.bfloat16)
            env.curriculum.copy_(restore)
            cache[key] = table
        return cache[key]
