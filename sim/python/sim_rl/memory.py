"""Spatial memory: a top-down map of where the goal matched and what the camera has looked at.

Monocular and deployable. The pose is dead-reckoned from the drone's own noisy sensors (gyro yaw
rate about the gravity-derived up axis, body velocity projected onto the level heading frame),
never the simulator's position. Each frame, every patch ray is cast to the depth predicted by
the frozen perception head (``perception.PatchHeads``); the cell it hits takes that patch's goal
evidence (the policy's own saliency: P(target at this patch rather than not in view)), and cells
part-way along the ray are marked looked-at with zero evidence. A cell's evidence is replaced by
its newest observation, so a false match fades once the drone looks again ("looked here, not
there"), and a match stays on the map after it leaves the view.

The actor reads an ego-centric, heading-aligned crop of the map and the strongest candidate's
offset; nothing here issues a command. Maps live on the GPU in the episode's start frame
(x = initial forward, y = initial left) and reset with the episode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .native import CONTROL_HZ
from .perception import ground_prior, patch_rays

GRID = 128                       # allocentric cells per side
CELL = 1.5                       # metres per allocentric cell (+-96 m)
EGO = 32                         # ego crop cells per side
EGO_RADIUS = 48.0                # metres from the drone to the crop edge
CHANNELS = 2                     # goal evidence, looked-at
CANDIDATE_WIDTH = 4              # forward, right, distance (/EGO_RADIUS), evidence
FREE_FRACTIONS = (0.25, 0.5, 0.75)
CANDIDATE_FLOOR = 0.05           # weaker best evidence reports no candidate
RATE_SCALE, VELOCITY_SCALE = 3.5, 5.0   # proprio normalisation (native sensors 3-5, 6-8)


class SpatialMemory:
    def __init__(self, n, grid_h, grid_w, camera, device):
        self.n, self.device = n, device
        self.map = torch.zeros((n, CHANNELS, GRID, GRID), device=device)
        self.position = torch.zeros((n, 2), device=device)
        self.yaw = torch.zeros(n, device=device)
        self.rays = patch_rays(grid_h, grid_w, camera, device)                 # [P, 3]
        self.fractions = torch.tensor((*FREE_FRACTIONS, 1.0), device=device)  # [S]
        self.offsets = torch.arange(n, device=device)[:, None, None] * GRID * GRID

    def reset(self, starts):
        keep = (starts < 0.5).float()
        self.map.mul_(keep[:, None, None, None])
        self.position.mul_(keep[:, None])
        self.yaw.mul_(keep)

    def update(self, proprio, depth, evidence):
        """proprio [n, 19] (native sensors), depth [n, P] metres (0: none), evidence [n, P] in [0, 1]."""
        dt = 1.0 / CONTROL_HZ
        up = F.normalize(-proprio[:, 0:3], dim=-1)
        rates, velocity = proprio[:, 3:6] * RATE_SCALE, proprio[:, 6:9] * VELOCITY_SCALE
        body_forward = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(up)
        rise = (body_forward * up).sum(-1, keepdim=True)                       # vertical part of forward
        forward = F.normalize(body_forward - rise * up, dim=-1)                 # level heading, body frame
        right = torch.cross(forward, up, dim=-1)
        # Heading rate of the body's forward axis projected on the ground (not omega . up, which
        # drifts whenever the body is tilted while it pitches or rolls).
        level = (1 - rise.square()).clamp_min(1e-4).sqrt()
        self.yaw.add_(((rates * up).sum(-1) - (rise * (rates * forward).sum(-1, keepdim=True) / level).squeeze(-1)) * dt)
        heading = torch.stack((self.yaw.cos(), self.yaw.sin()), -1)
        side = torch.stack((self.yaw.sin(), -self.yaw.cos()), -1)
        self.position.add_(dt * ((velocity * forward).sum(-1, keepdim=True) * heading
                                 + (velocity * right).sum(-1, keepdim=True) * side))

        offset = self.rays[None] * depth[..., None]                             # [n, P, 3]
        along = (offset * forward[:, None]).sum(-1, keepdim=True)
        across = (offset * right[:, None]).sum(-1, keepdim=True)
        reach = along * heading[:, None] + across * side[:, None]               # [n, P, 2]
        self.points = self.position[:, None] + reach                            # this frame's patch hits
        points = self.position[:, None, None] + self.fractions[None, None, :, None] * reach[:, :, None]
        values = torch.where(self.fractions == 1.0, evidence[..., None], torch.zeros_like(evidence[..., None]))
        cells = (points / CELL).floor().long() + GRID // 2                      # [n, P, S, 2]
        valid = (depth[..., None] > 0) & (cells >= 0).all(-1) & (cells < GRID).all(-1)
        flat = (self.offsets + cells[..., 1] * GRID + cells[..., 0])[valid]
        current = torch.full((self.n * GRID * GRID,), -1.0, device=self.device)
        current.scatter_reduce_(0, flat, values[valid], reduce="amax")
        current = current.view(self.n, GRID, GRID)
        observed = current >= 0
        self.map[:, 0] = torch.where(observed, current, self.map[:, 0])
        self.map[:, 1] = torch.maximum(self.map[:, 1], observed.float())

    def ego(self):
        """[n, CHANNELS, EGO, EGO]: heading-aligned crop, forward up, drone at the centre."""
        scale, (c, s) = 2.0 / (GRID * CELL), (self.yaw.cos(), self.yaw.sin())
        theta = torch.stack((torch.stack((scale * EGO_RADIUS * s, -scale * EGO_RADIUS * c, scale * self.position[:, 0]), -1),
                             torch.stack((-scale * EGO_RADIUS * c, -scale * EGO_RADIUS * s, scale * self.position[:, 1]), -1)), 1)
        grid = F.affine_grid(theta, [self.n, CHANNELS, EGO, EGO], align_corners=False)
        return F.grid_sample(self.map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def candidate(self):
        """[n, 4]: strongest evidence cell as (forward, right, distance) / EGO_RADIUS and its evidence."""
        value, index = self.map[:, 0].flatten(1).max(-1)
        cell = torch.stack((index % GRID, index // GRID), -1).float() - GRID // 2 + 0.5
        relative = cell * CELL - self.position
        heading = torch.stack((self.yaw.cos(), self.yaw.sin()), -1)
        side = torch.stack((self.yaw.sin(), -self.yaw.cos()), -1)
        rows = torch.stack(((relative * heading).sum(-1), (relative * side).sum(-1), relative.norm(dim=-1)), -1)
        present = (value >= CANDIDATE_FLOOR).float()
        return torch.cat((rows / EGO_RADIUS, value[:, None]), -1) * present[:, None]

    def relative(self, env, point):
        """(forward, right) metres from environment ``env``'s drone to a memory-frame ``point`` [2]."""
        offset = point - self.position[env]
        c, s = float(self.yaw[env].cos()), float(self.yaw[env].sin())
        return float(offset[0] * c + offset[1] * s), float(offset[0] * s - offset[1] * c)


@torch.no_grad()
def observe(policy, memory, cls, tokens, goal, proprio, starts, goal_mask=None):
    """One memory tick before the actor acts: reset finished episodes, project this frame, and
    return the actor's memory inputs (ego crop, candidate). ``goal_mask`` [n] zeroes the goal
    evidence of environments without a goal (explore commands)."""
    memory.reset(starts)
    _, logits = policy.attention(goal, cls, tokens)
    evidence = torch.sigmoid(logits[:, :-1] - logits[:, -1:])
    if goal_mask is not None:
        evidence = evidence * goal_mask[:, None]
    depth, memory.classes = policy.perception.predict(tokens, ground_prior(proprio, memory.rays))
    memory.update(proprio, depth, evidence)
    memory.depth = depth
    return memory.ego(), memory.candidate()
