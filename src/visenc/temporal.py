"""Temporal adapter over a frozen LingBot-Vision ViT (offline pilot).

The backbone's first ``depth - 2`` blocks are frozen and run once per frame (``prefix``). The last two
blocks plus the final norm are the frozen ``tail``. A recurrent adapter reads a per-sequence memory:
before the tail for ``reset``/``fifo``/``inside``, after the tail for ``late``.

Contracts
- images: float NCHW in [0,1], H and W multiples of 16. Normalised internally with the backbone's mean/std.
- prefix features: [B,L,D] for one frame, [B,T,L,D] for a sequence, L = 1 + n_storage + P.
- grid: (h, w) patch grid of a frame, needed for RoPE in the tail. Inferred as square when omitted;
  non-square frames must pass ``grid=``.
- state: ``None`` (no history) or ``(memory [B,M,width] float32, n_frames)``. Owned by the caller; the module
  keeps no state between calls.
- Dtype: parameters are float32; features are cast to float32 and run under the caller's autocast if any.

Cold start: with no valid state the adapter is skipped, so cls/patches are exactly the frozen backbone's
output (``cold`` gives the same tokens for distillation targets). Readout gates start at zero, so the adapter
is also the identity at initialisation for every mode.

Modes
- ``reset``: memoryless control. The readout attends to 16 learned constant slots at every frame and the
  state is always ``None``. Trainable: readout + slots + prediction head.
- ``fifo``: four snapshots, spaced three input frames apart, of four pooled tokens each. Each
  snapshot uses learned pooling queries. This spans the pilot's two-second gap at 4 Hz.
- ``inside``: 16 learned gated slots (``width``) updated recurrently by cross-attention over each frame's tokens.
- ``late``: the ``inside`` memory, applied after the frozen tail on final normalised tokens.

Memory visible at frame t comes only from frames < t, so predictions do not depend on future frames.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODES = ("reset", "fifo", "inside", "late")
PATCH = 16
MEM = 16  # memory tokens visible to the readout (16 slots / 4 frames x 4 pooled tokens / 16 constants)
FIFO_FRAMES = 4
FIFO_STRIDE = 3
FIFO_TOKENS = 4  # pooled tokens per frame

TemporalState = tuple[Tensor, int] | None


def _attend(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, heads: int) -> Tensor:
    """q [N,Lq,A], k/v [N,Lk,A], mask [N,Lk] bool (True = attend; at least one True per row) -> [N,Lq,A]."""
    n, lq, a = q.shape
    lk = k.shape[1]
    q = q.reshape(n, lq, heads, a // heads).transpose(1, 2)
    k = k.reshape(n, lk, heads, a // heads).transpose(1, 2)
    v = v.reshape(n, lk, heads, a // heads).transpose(1, 2)
    attn_mask = None if mask is None else mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.transpose(1, 2).reshape(n, lq, a)


class _Writer(nn.Module):
    """Queries [B,Q,width] attend over one frame's tokens [B,L,dim] -> [B,Q,width]."""

    def __init__(self, dim: int, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_ln = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.ln = nn.LayerNorm(dim)
        self.k = nn.Linear(dim, width)
        self.v = nn.Linear(dim, width)

    def forward(self, queries: Tensor, tokens: Tensor) -> Tensor:
        t = self.ln(tokens)
        return _attend(self.q(self.q_ln(queries)), self.k(t), self.v(t), None, self.heads)


class _Reader(nn.Module):
    """Tokens [N,L,dim] attend over memory [N,M,width]; output enters through a zero-initialised gate."""

    def __init__(self, dim: int, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, dim)
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor, mem: Tensor, mask: Tensor) -> Tensor:
        out = _attend(self.q(self.ln(x)), self.k(mem), self.v(mem), mask, self.heads)
        return x + self.gate * self.o(out)


class _Head(nn.Module):
    """Trainable prediction MLP on [cls, mean patch] -> (bearing rad, range/20, confidence logit)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, cls: Tensor, patches: Tensor) -> Tensor:
        return self.net(torch.cat([cls, patches.mean(-2)], -1))


class TemporalEncoder(nn.Module):
    """Temporal wrapper over ``backbone`` (a ``visenc.models.LingBotViT``; uses ``.model``, ``.mean``, ``.std``).

    Construction freezes the backbone (``requires_grad_(False)``) and keeps it in eval mode even under ``train()``.
    """

    def __init__(self, backbone: nn.Module, mode: str, *, width: int = 256, heads: int = 4, head_hidden: int = 256):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if width % heads:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.mode = mode
        self.backbone = backbone.requires_grad_(False).eval()
        vit = backbone.model
        self.dim = vit.embed_dim
        self.n_storage = vit.n_storage_tokens
        self.width = width
        self.heads = heads
        if len(vit.blocks) < 3:
            raise ValueError("backbone needs at least 3 blocks to split prefix and tail")

        # Memory parameters (mode-specific; see module docstring).
        if mode in ("reset", "inside", "late"):
            self.slots = nn.Parameter(torch.randn(MEM, width) * 0.02)
        if mode in ("inside", "late"):
            self.write = _Writer(self.dim, width, heads)
            self.cand = nn.Linear(width, width)
            self.gate = nn.Linear(2 * width, width)
            nn.init.constant_(self.gate.bias, -1.0)
        if mode == "fifo":
            self.write = _Writer(self.dim, width, heads)
            self.fifo_q = nn.Parameter(torch.randn(FIFO_TOKENS, width) * 0.02)
        self.reader = _Reader(self.dim, width, heads)
        self.head = _Head(self.dim, head_hidden)

    def train(self, mode: bool = True) -> "TemporalEncoder":
        super().train(mode)
        self.backbone.eval()  # no dropout / RoPE coordinate augmentation in the frozen backbone
        return self

    # ----- frozen backbone pieces -----

    @property
    def _vit(self) -> nn.Module:
        return self.backbone.model

    def _rope(self, h: int, w: int):
        rope = self._vit.rope_embed
        return None if rope is None else rope(H=h, W=w)

    def _tail(self, tokens: Tensor, rope) -> Tensor:
        for blk in self._vit.blocks[-2:]:
            tokens = blk(tokens, rope)
        return self._vit.norm(tokens)

    def _grid(self, length: int, grid: tuple[int, int] | None) -> tuple[int, int]:
        patches = length - 1 - self.n_storage
        if grid is None:
            side = math.isqrt(patches)
            if side * side != patches:
                raise ValueError(f"{patches} patch tokens is not square; pass grid=(h, w)")
            return side, side
        if grid[0] * grid[1] != patches:
            raise ValueError(f"grid {grid} does not match {patches} patch tokens")
        return grid

    @torch.no_grad()
    def prefix(self, images: Tensor) -> Tensor:
        """images float [B,3,H,W] in [0,1] -> frozen prefix tokens [B,L,D] (blocks[:-2] applied, pre-norm)."""
        h, w = images.shape[-2] // PATCH, images.shape[-1] // PATCH
        tokens, _ = self._vit.prepare_tokens_with_masks((images - self.backbone.mean) / self.backbone.std)
        rope = self._rope(h, w)
        for blk in self._vit.blocks[:-2]:
            tokens = blk(tokens, rope)
        return tokens

    @torch.no_grad()
    def cold(self, features: Tensor, *, grid: tuple[int, int] | None = None) -> tuple[Tensor, Tensor]:
        """Unadapted backbone tokens for prefix features [B,T,L,D] -> (cls [B,T,D], patches [B,T,P,D]).

        Use as the reconstruction target for the adapted tokens returned by ``step_cached``."""
        b, t_len, length, d = features.shape
        h, w = self._grid(length, grid)
        tokens = self._tail(features.flatten(0, 1).to(self.reader.q.weight.dtype), self._rope(h, w))
        tokens = tokens.reshape(b, t_len, length, d)
        return tokens[:, :, 0], tokens[:, :, 1 + self.n_storage :]

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Conventional cold-start encoder: images [B,3,H,W] in [0,1] -> (cls [B,D], patches [B,P,D]).

        Identical to the frozen backbone's ``forward_features`` (no adapter, no state)."""
        h, w = images.shape[-2] // PATCH, images.shape[-1] // PATCH
        tokens = self._tail(self.prefix(images), self._rope(h, w))
        return tokens[:, 0], tokens[:, 1 + self.n_storage :]

    # ----- recurrent memory -----

    def _read(self, state: TemporalState, batch: int) -> tuple[Tensor, Tensor] | None:
        """(memory [B,MEM,width], valid mask [MEM]) visible to the next frame; None when no history exists."""
        valid_all = torch.ones(MEM, dtype=torch.bool, device=self.reader.q.weight.device)
        if self.mode == "reset":
            return self.slots.unsqueeze(0).expand(batch, -1, -1), valid_all
        if state is None:
            return None
        mem, n = state
        if self.mode == "fifo":
            count = min(FIFO_FRAMES, (n + FIFO_STRIDE - 1) // FIFO_STRIDE)
            return mem, torch.arange(MEM, device=mem.device) >= MEM - FIFO_TOKENS * count
        return mem, valid_all

    def _advance(self, state: TemporalState, tokens: Tensor, batch: int) -> TemporalState:
        """Fold one frame's tokens [B,L,D] into the state. Depends on frozen-prefix tokens only."""
        if self.mode == "reset":
            return None
        if self.mode == "fifo":
            if state is None:
                old, n = tokens.new_zeros(batch, MEM, self.width, dtype=torch.float32), 0
            else:
                old, n = state
            if n % FIFO_STRIDE != 0:
                return old, n + 1
            new = self.write(self.fifo_q.unsqueeze(0).expand(batch, -1, -1), tokens)
            return torch.cat([old[:, FIFO_TOKENS:], new.float()], 1), n + 1
        if state is None:
            slots, n = self.slots.unsqueeze(0).expand(batch, -1, -1), 0
        else:
            slots, n = state
        att = self.write(slots, tokens)
        cand = torch.tanh(self.cand(att))
        g = torch.sigmoid(self.gate(torch.cat([slots, cand], -1)))
        return ((1 - g) * slots + g * cand).float(), n + 1

    def _run(self, features: Tensor, state: TemporalState, grid: tuple[int, int] | None,
             incoming: list[Tensor] | None = None):
        """features [B,T,L,D] -> final tokens [B,T,L,D], state after the T frames."""
        b, t_len, length, d = features.shape
        h, w = self._grid(length, grid)
        rope = self._rope(h, w)
        features = features.to(self.reader.q.weight.dtype)
        if self.mode == "late":
            with torch.no_grad():
                src = self._tail(features.flatten(0, 1), rope).reshape(b, t_len, length, d)
        else:
            src = features

        # Recurrence first (cheap): memory seen at frame t is the state after frames < t.
        reads = []
        for t in range(t_len):
            reads.append(self._read(state, b))
            if incoming is not None:
                incoming.append(torch.zeros_like(self.slots).expand(b, -1, -1) if state is None else state[0])
            state = self._advance(state, src[:, t], b)

        # Frames without history (cold start) skip the readout, so they equal the frozen baseline exactly.
        start = next((i for i, r in enumerate(reads) if r is not None), t_len)
        x = src
        if start < t_len:
            n = t_len - start
            mem = torch.stack([r[0] for r in reads[start:]], 1).flatten(0, 1)
            mask = torch.stack([r[1] for r in reads[start:]]).unsqueeze(0).expand(b, -1, -1).reshape(b * n, MEM)
            adapted = self.reader(src[:, start:].flatten(0, 1), mem, mask).reshape(b, n, length, d)
            x = adapted if start == 0 else torch.cat([src[:, :start], adapted], 1)

        if self.mode == "late":
            return x, state
        return self._tail(x.flatten(0, 1), rope).reshape(b, t_len, length, d), state

    def _sequence(self, features: Tensor, state: TemporalState, grid):
        tokens, state = self._run(features, state, grid)
        cls, patches = tokens[:, :, 0], tokens[:, :, 1 + self.n_storage :]
        return cls, patches, self.head(cls, patches), state

    def forward_cached(self, features: Tensor, *, grid: tuple[int, int] | None = None) -> Tensor:
        """Prefix features [B,T,L,D] of one sequence from cold start -> prediction [B,T,3].

        Channels: bearing (radians, raw), range/20 (raw), confidence (logit)."""
        return self._sequence(features, None, grid)[2]

    def supervised_cached(self, features: Tensor, *, grid: tuple[int, int]):
        """Predictions and incoming recurrent slots at every frame, for auxiliary supervision."""
        if self.mode not in ("inside", "late"):
            raise ValueError("slot supervision requires recurrent state")
        incoming = []
        tokens, _ = self._run(features, None, grid, incoming)
        return self.head(tokens[:, :, 0], tokens[:, :, 1 + self.n_storage :]), torch.stack(incoming, dim=1)

    def prefix_memory(self, features: Tensor) -> Tensor:
        """Incoming state [B,T,16,width] without the visual readout, for inside and FIFO modes."""
        if self.mode not in ("inside", "fifo"):
            raise ValueError("prefix_memory requires a prefix-based writer")
        features = features.to(self.reader.q.weight.dtype)
        state = None
        incoming = []
        for frame in features.unbind(dim=1):
            incoming.append(frame.new_zeros(len(frame), MEM, self.width) if state is None else state[0])
            state = self._advance(state, frame, len(frame))
        return torch.stack(incoming, dim=1)

    def step_cached(self, features: Tensor, state: TemporalState, *, grid: tuple[int, int] | None = None):
        """One frame: prefix features [B,L,D], state None (cold) or the previous returned state.

        Returns (cls [B,D], patches [B,P,D], prediction [B,3], state). Pass ``state=None`` to start a sequence.
        ``reset`` always returns ``None``."""
        cls, patches, pred, state = self._sequence(features.unsqueeze(1), state, grid)
        return cls[:, 0], patches[:, 0], pred[:, 0], state
