"""Goal-conditioned continuous actor-critic for the quadrotor visual-goal task."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_TWO_PI = math.log(2 * math.pi)
MODES = ("visual", "state", "hint")
HEADS, HEAD_DIM = 4, 64
ATTENTION = HEADS * HEAD_DIM      # 256 attended features
GLOBAL_POOL = (2, 3)
GLOBAL = 128                      # projected CLS + 2x3 pooled patch tokens
HIDDEN = 256
INITIAL_LOG_STD = -0.5
PROPRIO_WIDTH, PRIVILEGED_WIDTH = 18, 18
VISUAL_WIDTH = ATTENTION + 2 * HEADS + HEADS + GLOBAL   # 396


def _orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class GoalAttention(nn.Module):
    """Goal-query attention over frozen patch tokens with learned 2D positions.

    Output per sample: attended values (256), expected patch coordinates per
    head (8), max logit per head (4), and projected CLS + 2x3 pooled global
    tokens (128); total 396 features.
    """

    def __init__(self, token_dim, goal_dim, grid_h, grid_w):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.query = nn.Sequential(
            _orthogonal(nn.Linear(goal_dim, HIDDEN), math.sqrt(2)), nn.Tanh(),
            _orthogonal(nn.Linear(HIDDEN, ATTENTION), 0.01))
        self.key = _orthogonal(nn.Linear(token_dim, ATTENTION), math.sqrt(2))
        self.value = _orthogonal(nn.Linear(token_dim, ATTENTION), math.sqrt(2))
        self.global_projection = _orthogonal(nn.Linear(7 * token_dim, GLOBAL), math.sqrt(2))
        self.row_embedding = nn.Parameter(torch.randn(grid_h, token_dim) * 0.02)
        self.column_embedding = nn.Parameter(torch.randn(grid_w, token_dim) * 0.02)
        rows, columns = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
        rows, columns = rows.reshape(-1), columns.reshape(-1)  # row-major, matches patch order
        coords = torch.stack(((columns + 0.5) / grid_w, (rows + 0.5) / grid_h), dim=-1)
        self.register_buffer("row_index", rows, persistent=False)
        self.register_buffer("column_index", columns, persistent=False)
        self.register_buffer("coords", coords.float(), persistent=False)

    def forward(self, goal, cls, tokens):
        """goal [n, G], cls [n, D], tokens [n, P, D] -> [n, 396]."""
        tokens = tokens.float()
        n, patches, dim = tokens.shape
        position = self.row_embedding[self.row_index] + self.column_embedding[self.column_index]
        query = self.query(goal).view(n, HEADS, HEAD_DIM)
        positioned = tokens + position
        keys = self.key(positioned).view(n, patches, HEADS, HEAD_DIM)
        values = self.value(positioned).view(n, patches, HEADS, HEAD_DIM)
        logits = torch.einsum("nhd,nphd->nhp", query, keys) / math.sqrt(HEAD_DIM)
        attention = logits.softmax(-1)
        attended = torch.einsum("nhp,nphd->nhd", attention, values).reshape(n, ATTENTION)
        expected = torch.einsum("nhp,pc->nhc", attention, self.coords).reshape(n, 2 * HEADS)
        peak = logits.amax(-1)
        grid = tokens.transpose(1, 2).reshape(n, dim, self.grid_h, self.grid_w)
        pooled = F.adaptive_avg_pool2d(grid, GLOBAL_POOL).flatten(2).transpose(1, 2)
        global_tokens = torch.cat((cls.float().unsqueeze(1), pooled), dim=1).flatten(1)
        return torch.cat((attended, expected, peak, self.global_projection(global_tokens)), dim=-1)


class ActorCritic(nn.Module):
    """Gaussian actor (GRU) over one of three information modes, plus a privileged critic.

    visual: goal attention over frozen tokens + proprioception.
    state:  privileged + proprioception (no vision).
    hint:   proprioception only.
    The critic always sees privileged + proprioception (asymmetric training).
    """

    def __init__(self, mode, token_dim, goal_dim, grid_h, grid_w, action_dim=4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown policy mode {mode}")
        self.mode, self.action_dim, self.hidden_size = mode, action_dim, HIDDEN
        self.attention = GoalAttention(token_dim, goal_dim, grid_h, grid_w) if mode == "visual" else None
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
        if self.mode == "visual":
            return torch.cat((self.attention(goal, cls, tokens), proprio), dim=-1)
        if self.mode == "state":
            return torch.cat((privileged, proprio), dim=-1)
        return proprio

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
