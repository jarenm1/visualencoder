"""Goal-conditioned continuous actor-critic for the quadrotor visual-goal task."""
from __future__ import annotations

import functools
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
# Privileged lanes used by the auxiliary localization target: anchor visibility, anchor NDC,
# and the target box's screen bounds [x_min, y_min, x_max, y_max] (NDC, y up).
VISIBLE_LANE, NDC_LANES, BOX_LANES = 14, slice(18, 20), slice(25, 29)


@functools.cache
def patch_centres(grid_h, grid_w, device):
    """Row-major patch-centre NDC [P, 2] (x right, y up; matches privileged lanes 18-19 and 25-28)."""
    rows, columns = torch.meshgrid(torch.arange(grid_h, device=device), torch.arange(grid_w, device=device),
                                   indexing="ij")
    return torch.stack(((columns.reshape(-1) + 0.5) / grid_w * 2 - 1,
                        1 - (rows.reshape(-1) + 0.5) / grid_h * 2), dim=-1).float()


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
        self.register_buffer("ndc", patch_centres(grid_h, grid_w, torch.device("cpu")).clone(), persistent=False)

    def scores(self, goal, cls, tokens):
        """Patch logits [n, P] and the "not in view" logit [n, 1] for a goal vector [n, G]."""
        patch = self.saliency(tokens).squeeze(-1) + (self.token_rank(tokens) * self.goal_rank(goal).unsqueeze(1)).sum(-1)
        return patch, self.absent(torch.cat((cls.float(), goal), dim=-1))

    def forward(self, goal, cls, tokens):
        """goal [n, G] (or [n, 1+K, D] for PatchMatchSaliency), cls [n, D], tokens [n, P, D]
        -> (features [n, 196], logits [n, P+1])."""
        tokens = tokens.float()
        n, _, dim = tokens.shape
        patch, absent = self.scores(goal.float(), cls.float(), tokens)
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


GOAL_POOL = (6, 8)               # goal-crop patch grid kept for matching (48 tokens of the 15x20)


def goal_tokens(cls, patches, grid_h, grid_w, pool=GOAL_POOL):
    """Goal-crop tokens for PatchMatchSaliency: [n, 1+K, D] = goal CLS, then its patch grid pooled
    to ``pool`` (the crop is a resized object view, so pooling keeps parts, not background)."""
    n, _, dim = patches.shape
    grid = patches.float().transpose(1, 2).reshape(n, dim, grid_h, grid_w)
    pooled = F.adaptive_avg_pool2d(grid, pool).flatten(2).transpose(1, 2)
    return torch.cat((cls.float().unsqueeze(1), pooled), dim=1)


class PatchMatchSaliency(TargetSaliency):
    """TargetSaliency whose goal term matches every frame patch against every goal-crop patch.

    A single goal vector (CLS + mean patch) cannot keep the parts that separate look-alike objects
    (a van from a sedan of the same colour). Here frame and goal patches share a learned rank-64
    projection; a frame patch scores by its soft-max cosine similarity to the goal's patches
    (temperature learned), added to the goal-free linear term. The "not in view" logit also sees the
    best match, so absence is judged by "nothing here matches", not by global appearance alone.
    Same features and logits as TargetSaliency; the goal is ``goal_tokens`` output [n, 1+K, D].
    """

    def __init__(self, token_dim, grid_h, grid_w):
        super().__init__(token_dim, 2 * token_dim, grid_h, grid_w)
        del self.goal_rank
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.absent = nn.Sequential(_orthogonal(nn.Linear(2 * token_dim + 2, HIDDEN), math.sqrt(2)), nn.Tanh(),
                                    _orthogonal(nn.Linear(HIDDEN, 1), 0.01))

    def scores(self, goal, cls, tokens):
        goal = goal.reshape(goal.shape[0], -1, tokens.shape[-1])   # flat policy goal -> [n, 1+K, D]
        query = F.normalize(self.token_rank(tokens), dim=-1)
        key = F.normalize(self.token_rank(goal[:, 1:]), dim=-1)
        scale = self.log_scale.exp()
        similarity = torch.einsum("npr,nkr->npk", query, key) * scale
        match = (similarity.logsumexp(-1) - math.log(key.shape[1])) / scale   # soft max cosine, [-1, 1]
        patch = self.saliency(tokens).squeeze(-1) + scale * match
        evidence = torch.stack((match.amax(-1), match.mean(-1)), dim=-1)
        return patch, self.absent(torch.cat((cls, goal[:, 0], evidence), dim=-1))


def saliency_targets(privileged, grid_h, grid_w):
    """Soft target over P+1 classes per frame: uniform over the patches on the visible target (centre
    inside the box projection, plus the anchor patch), else all mass on P (not in view).
    Returns (target [N, P+1], visible [N] bool, on_target [N, P] bool)."""
    visible = privileged[..., VISIBLE_LANE] > 0.5
    centres = patch_centres(grid_h, grid_w, privileged.device)
    box = privileged[..., BOX_LANES]
    x, y = centres[:, 0], centres[:, 1]
    inside = ((x >= box[:, 0:1]) & (x <= box[:, 2:3]) & (y >= box[:, 1:2]) & (y <= box[:, 3:4]))
    column = ((privileged[..., 18] + 1) / 2 * grid_w).floor().clamp(0, grid_w - 1)
    row = ((1 - privileged[..., 19]) / 2 * grid_h).floor().clamp(0, grid_h - 1)
    anchor = F.one_hot((row * grid_w + column).long(), grid_h * grid_w).bool()
    on_target = (inside | anchor) & visible.unsqueeze(-1)
    mass = on_target.float() / on_target.sum(-1, keepdim=True).clamp_min(1)
    return torch.cat((mass, (~visible).float().unsqueeze(-1)), dim=-1), visible, on_target


def saliency_loss(logits, privileged, grid_h, grid_w):
    """Soft cross-entropy over P+1 classes, and the fraction of visible frames whose peak patch
    lies on the target (detached)."""
    target, visible, on_target = saliency_targets(privileged, grid_h, grid_w)
    loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
    with torch.no_grad():
        hits = on_target.gather(1, logits[:, :-1].argmax(-1, keepdim=True)).squeeze(-1) & visible
        accuracy = hits.float().sum() / visible.float().sum().clamp_min(1.0)
    return loss, accuracy


class ActorCritic(nn.Module):
    """Tanh-squashed Gaussian actor (GRU) over one of three information modes, plus a privileged critic.

    The Gaussian lives in pre-squash space; commands are tanh(raw) so they stay in [-1, 1] without
    clipping. (Clipping an unbounded mean removed the gradient once the mean left the box, which
    left the v4 policy pinned at full yaw rate.)

    visual: target saliency over frozen tokens + proprioception (+ saliency auxiliary loss).
    state:  privileged + proprioception (no vision).
    hint:   proprioception only.
    The critic always sees privileged + proprioception (asymmetric training).
    """

    def __init__(self, mode, token_dim, goal_dim, grid_h, grid_w, action_dim=4, goal="global"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown policy mode {mode}")
        self.mode, self.action_dim, self.hidden_size = mode, action_dim, HIDDEN
        if mode != "visual":
            self.attention = None
        elif goal == "patch":
            self.attention = PatchMatchSaliency(token_dim, grid_h, grid_w)
        else:
            self.attention = TargetSaliency(token_dim, goal_dim, grid_h, grid_w)
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


def squashed_log_prob(mean, log_std, raw):
    """Log-density of action tanh(raw) when raw ~ N(mean, exp(log_std)); stable tanh Jacobian."""
    jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
    return gaussian_log_prob(mean, log_std, raw) - jacobian.sum(-1)


def squashed_entropy(mean, log_std):
    """Reparameterized one-sample estimate of the tanh-Gaussian entropy, averaged over the batch.

    Unlike the pre-squash Gaussian entropy it is bounded: widening the Gaussian past saturation
    lowers it, so the bonus cannot drive the policy into bang-bang noise.
    """
    raw = mean + log_std.exp() * torch.randn_like(mean)
    return -squashed_log_prob(mean, log_std, raw).mean()


def deterministic_action(mean):
    """Evaluation command: the squashed mean."""
    return torch.tanh(mean)
