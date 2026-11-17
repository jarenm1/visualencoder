"""Goal-conditioned continuous actor-critic for the quadrotor visual-goal task."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .native import PRIVILEGED_WIDTH, PROPRIO_WIDTH

LOG_TWO_PI = math.log(2 * math.pi)
MODES = ("visual", "state", "hint")
SALIENCY_RANK = 64               # low-rank goal x token interaction
VALUE = 64                       # saliency-pooled token projection
GLOBAL_POOL = (2, 3)
GLOBAL = 128                     # projected CLS + 2x3 pooled patch tokens
HIDDEN = 256
INITIAL_LOG_STD = -0.5
# expected NDC (2), peak patch logit (1), P(in view) (1), pooled value, global projection
VISUAL_WIDTH = 2 + 1 + 1 + VALUE + GLOBAL   # 196
# Privileged lanes used by the auxiliary localization target.
VISIBLE_LANE, NDC_LANES = 14, slice(18, 20)


def _orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class TargetSaliency(nn.Module):
    """Per-patch target saliency over frozen patch tokens.

    Patch logit = goal-free linear score + low-rank (token x goal) bilinear score; a
    "not in view" logit from CLS and goal completes a softmax over P+1 classes. This
    low-capacity form is what the offline per-patch diagnostic showed to generalize to
    held-out districts and designs (~84% of visible targets within one patch); the
    earlier learned-position query attention fit training frames but was near chance on
    held-out ones.

    forward -> (features [n, 196], logits [n, P+1]). Features: saliency-expected target
    NDC (2), peak patch logit (1), P(in view) (1), saliency-pooled tokens (64) and the
    projected CLS + 2x3 pooled global context (128).
    """

    def __init__(self, token_dim, goal_dim, grid_h, grid_w):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.saliency = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.saliency.weight)
        nn.init.zeros_(self.saliency.bias)
        self.token_rank = nn.Linear(token_dim, SALIENCY_RANK, bias=False)
        self.goal_rank = nn.Linear(goal_dim, SALIENCY_RANK, bias=False)
        nn.init.normal_(self.token_rank.weight, std=0.02)
        nn.init.normal_(self.goal_rank.weight, std=0.02)
        self.absent = nn.Sequential(_orthogonal(nn.Linear(token_dim + goal_dim, HIDDEN), math.sqrt(2)), nn.Tanh(),
                                    _orthogonal(nn.Linear(HIDDEN, 1), 0.01))
        self.value = _orthogonal(nn.Linear(token_dim, VALUE), math.sqrt(2))
        self.global_projection = _orthogonal(nn.Linear(7 * token_dim, GLOBAL), math.sqrt(2))
        rows, columns = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
        # Row-major patch order; NDC x right, y up (matches privileged lanes 18-19).
        ndc = torch.stack(((columns.reshape(-1) + 0.5) / grid_w * 2 - 1,
                           1 - (rows.reshape(-1) + 0.5) / grid_h * 2), dim=-1)
        self.register_buffer("ndc", ndc.float(), persistent=False)

    def forward(self, goal, cls, tokens):
        """goal [n, G], cls [n, D], tokens [n, P, D] -> (features [n, 196], logits [n, P+1])."""
        tokens = tokens.float()
        n, _, dim = tokens.shape
        patch = self.saliency(tokens).squeeze(-1) + (self.token_rank(tokens) * self.goal_rank(goal).unsqueeze(1)).sum(-1)
        absent = self.absent(torch.cat((cls.float(), goal), dim=-1))
        logits = torch.cat((patch, absent), dim=-1)
        in_view = 1 - logits.softmax(-1)[:, -1:]
        where = patch.softmax(-1)
        expected = where @ self.ndc
        pooled_value = torch.einsum("np,npv->nv", where, self.value(tokens))
        grid = tokens.transpose(1, 2).reshape(n, dim, self.grid_h, self.grid_w)
        pooled = F.adaptive_avg_pool2d(grid, GLOBAL_POOL).flatten(2).transpose(1, 2)
        context = self.global_projection(torch.cat((cls.float().unsqueeze(1), pooled), dim=1).flatten(1))
        features = torch.cat((expected, patch.amax(-1, keepdim=True), in_view, pooled_value, context), dim=-1)
        return features, logits


def saliency_targets(privileged, grid_h, grid_w):
    """Class per frame: the patch containing the visible target centre, else P (not in view)."""
    visible = privileged[..., VISIBLE_LANE] > 0.5
    column = ((privileged[..., 18] + 1) / 2 * grid_w).floor().clamp(0, grid_w - 1)
    row = ((1 - privileged[..., 19]) / 2 * grid_h).floor().clamp(0, grid_h - 1)
    patch = (row * grid_w + column).long()
    return torch.where(visible, patch, torch.full_like(patch, grid_h * grid_w)), visible


def saliency_loss(logits, privileged, grid_h, grid_w):
    """Cross-entropy over P+1 classes, and exact-patch accuracy on visible frames (detached)."""
    target, visible = saliency_targets(privileged, grid_h, grid_w)
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        hits = (logits[:, :-1].argmax(-1) == target).float() * visible.float()
        accuracy = hits.sum() / visible.float().sum().clamp_min(1.0)
    return loss, accuracy


class ActorCritic(nn.Module):
    """Gaussian actor (GRU) over one of three information modes, plus a privileged critic.

    visual: target saliency over frozen tokens + proprioception (+ saliency auxiliary loss).
    state:  privileged + proprioception (no vision).
    hint:   proprioception only.
    The critic always sees privileged + proprioception (asymmetric training).
    """

    def __init__(self, mode, token_dim, goal_dim, grid_h, grid_w, action_dim=4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown policy mode {mode}")
        self.mode, self.action_dim, self.hidden_size = mode, action_dim, HIDDEN
        self.attention = TargetSaliency(token_dim, goal_dim, grid_h, grid_w) if mode == "visual" else None
        width = {"visual": VISUAL_WIDTH + PROPRIO_WIDTH, "state": PRIVILEGED_WIDTH + PROPRIO_WIDTH,
                 "hint": PROPRIO_WIDTH}[mode]
        self.trunk = nn.Sequential(_orthogonal(nn.Linear(width, HIDDEN), math.sqrt(2)),
                                   nn.LayerNorm(HIDDEN), nn.Tanh())
        self.recurrent = nn.GRUCell(HIDDEN, HIDDEN)
        self.mean = _orthogonal(nn.Linear(HIDDEN, action_dim), 0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), INITIAL_LOG_STD))
        self.critic = nn.Sequential(
            _orthogonal(nn.Linear(PRIVILEGED_WIDTH + PROPRIO_WIDTH, HIDDEN), math.sqrt(2)), nn.Tanh(),
            _orthogonal(nn.Linear(HIDDEN, HIDDEN), math.sqrt(2)), nn.Tanh(),
            _orthogonal(nn.Linear(HIDDEN, 1), 1.0))

    def actor_inputs(self, cls, tokens, goal, proprio, privileged):
        return self.actor_inputs_and_features(cls, tokens, goal, proprio, privileged)[0]

    def actor_inputs_and_features(self, cls, tokens, goal, proprio, privileged):
        """Returns (actor inputs, saliency logits or None) so training reuses one saliency pass."""
        if self.mode == "visual":
            features, logits = self.attention(goal, cls, tokens)
            return torch.cat((features, proprio), dim=-1), logits
        if self.mode == "state":
            return torch.cat((privileged, proprio), dim=-1), None
        return proprio, None

    def value(self, privileged, proprio):
        return self.critic(torch.cat((privileged, proprio), dim=-1)).squeeze(-1)

    def step(self, inputs, hidden, starts):
        """One control tick: inputs [n, W], hidden [n, H], starts [n] (1 = first tick of an episode)."""
        hidden = self.recurrent(self.trunk(inputs), hidden * (1 - starts).unsqueeze(-1))
        return self.mean(hidden), hidden

    def sequence(self, inputs, hidden, starts):
        """inputs [L, B, W], hidden [B, H] at sequence start, starts [L, B] -> means [L, B, A]."""
        encoded = self.trunk(inputs)
        outputs = []
        for step in range(encoded.shape[0]):
            hidden = self.recurrent(encoded[step], hidden * (1 - starts[step]).unsqueeze(-1))
            outputs.append(hidden)
        return self.mean(torch.stack(outputs))


def gaussian_log_prob(mean, log_std, actions):
    scaled = (actions - mean) * torch.exp(-log_std)
    return (-0.5 * scaled.square() - log_std - 0.5 * LOG_TWO_PI).sum(-1)


def gaussian_entropy(log_std):
    """Entropy of the diagonal Gaussian (state-independent, so a scalar)."""
    return (log_std + 0.5 * (1 + LOG_TWO_PI)).sum()
