"""Solvability verifier: can a catalog episode be completed at all?

Hard-case post-training samples the episodes the pilot fails; without a verifier that sample fills
up with impossible ones (a target boxed in by clutter, a standoff band with no free air), and RL on
them teaches nothing. A privileged-state policy is no verifier here: it sees the target but not the
obstacles, so its failures mix "impossible" with "crashed into what it could not see".

This is a geometric check on the simulator's own collision geometry (native ``occupancy``: the
control kernel's crash test at voxel centres). Free space is a VOXEL_M grid of the episode's
district between the landable surface and the ceiling where a sphere of the drone radius plus
CLEARANCE_M touches nothing. An episode is solvable when the start's free component holds a voxel
in the standoff band (box distance <= STANDOFF_MAX) with a clear line of sight (a thin-sphere grid,
ray-marched) to the nearest point of the target box. Voxels quantize space, so the check errs only
by VOXEL_M: a gap narrower than the drone plus clearance counts as closed.

    train.py verify --seed 0 --split train --start-distance 8 60 --out runs/verifier/train-s0.pt
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .native import CudaEnv

VOXEL_M = 1.0
RADIUS_M = 0.25                  # sim_state::drone::RADIUS
CLEARANCE_M = 0.25               # margin for the controller's tracking error
SIGHT_RADIUS_M = 0.05            # line-of-sight grid: geometry thicker than this blocks the view
STANDOFF_MAX = 6.0               # sim_state::drone::STANDOFF_MAX
CEILING_M = 40.0                 # sim_state::drone::CEILING, above the landable surface
SIGHT_STEP_M = 0.5
SIGHT_STOP_M = 0.75              # the march stops this short of the aim point (the target's surface)
LOS_CANDIDATES = 256             # standoff voxels checked for line of sight per episode
REASONS = ("solvable", "start_blocked", "no_reachable_standoff", "no_line_of_sight")


def catalog_entries(env, progress):
    """Start, target box, object, bucket and class of every catalog entry, read back through the
    native reset override (next_catalog)."""
    n, size = env.n, env.catalog_size
    rows = {key: np.zeros((size, width), np.float32) for key, width in
            (("start", 3), ("center", 3), ("half", 3), ("meta", 4))}
    for first in range(0, size, n):
        count = min(n, size - first)
        wanted = torch.full((n,), -1, dtype=torch.int32, device=env.device)
        wanted[:count] = torch.arange(first, first + count, device=env.device, dtype=torch.int32)
        env.next_catalog.copy_(wanted)
        env.reset(first)
        d = env.diagnostics[:count].cpu().numpy()
        rows["start"][first:first + count] = d[:, 0:3]
        rows["center"][first:first + count] = d[:, 3:6]
        rows["half"][first:first + count] = d[:, 24:27]
        rows["meta"][first:first + count] = np.stack((d[:, 27], d[:, 15], d[:, 14], d[:, 16]), -1)
        progress(first + count, size)
    assert (rows["meta"][:, 3] == np.arange(size)).all(), "catalog override did not select every entry"
    return rows


class FreeSpace:
    """Voxel free space of one district from the native collision geometry."""

    def __init__(self, env):
        metadata = env.metadata
        self.half_extent = float(metadata["city"]["half_extent_m"])
        self.ground = float(metadata["landable"]["height_m"])
        side, levels = int(np.ceil(2 * self.half_extent / VOXEL_M)), int(np.ceil(CEILING_M / VOXEL_M))
        self.shape = (side, levels, side)                                   # x, y, z
        origin = (-self.half_extent, self.ground, -self.half_extent)
        blocked = env.occupancy(origin, VOXEL_M, self.shape, RADIUS_M + CLEARANCE_M).cpu().numpy().astype(bool)
        self.sight = env.occupancy(origin, VOXEL_M, self.shape, SIGHT_RADIUS_M).cpu().numpy().astype(bool)
        xs = (np.arange(side) + 0.5) * VOXEL_M - self.half_extent
        heights = self.ground + (np.arange(levels) + 0.5) * VOXEL_M
        district = metadata["district"]
        margin = district["margin_m"]
        inside_x = (xs >= district["min"][0] + margin) & (xs <= district["max"][0] - margin)
        inside_z = (xs >= district["min"][1] + margin) & (xs <= district["max"][1] - margin)
        plan = inside_x[:, None] & inside_z[None, :]
        for rect in district["exclusions"]:                      # {"min": [x, z], "max": [x, z]}
            (lo_x, lo_z), (hi_x, hi_z) = rect["min"], rect["max"]
            plan &= ~(((xs >= lo_x) & (xs <= hi_x))[:, None] & ((xs >= lo_z) & (xs <= hi_z))[None, :])
        free = ~blocked & plan[:, None, :]
        free[:, heights < self.ground + RADIUS_M, :] = False   # landed starts sit at ground + RADIUS
        self.free = free
        self.labels, _ = ndimage.label(free)
        self.centers = xs, heights

    @staticmethod
    def _range(lo, hi, count, offset):
        """Voxel index range [first, last) whose centres lie in [lo, hi] along an axis starting at
        -offset, clipped to [0, count] (empty, never a negative slice end, when outside)."""
        first = min(max(int(np.ceil((lo + offset) / VOXEL_M - 0.5)), 0), count)
        last = min(max(int(np.floor((hi + offset) / VOXEL_M - 0.5)) + 1, first), count)
        return first, last

    def _index(self, points):
        """[..., 3] metres -> integer voxel indices (unclipped)."""
        return np.stack((np.floor((points[..., 0] + self.half_extent) / VOXEL_M),
                         np.floor((points[..., 1] - self.ground) / VOXEL_M),
                         np.floor((points[..., 2] + self.half_extent) / VOXEL_M)), -1).astype(np.int64)

    def start_label(self, start):
        """Free component of the start voxel, or of a free voxel touching it (0: blocked)."""
        i, j, k = self._index(np.asarray(start))
        for di in (0, -1, 1):
            for dj in (0, -1, 1):
                for dk in (0, -1, 1):
                    a, b, c = i + di, j + dj, k + dk
                    if 0 <= a < self.shape[0] and 0 <= b < self.shape[1] and 0 <= c < self.shape[2]:
                        if self.labels[a, b, c]:
                            return int(self.labels[a, b, c])
        return 0

    def verdict(self, start, center, half, rng):
        """REASONS index for one episode."""
        label = self.start_label(start)
        if not label:
            return 1
        reach = half + STANDOFF_MAX
        xs, ys = self.centers
        i0, i1 = self._range(center[0] - reach[0], center[0] + reach[0], self.shape[0], self.half_extent)
        k0, k1 = self._range(center[2] - reach[2], center[2] + reach[2], self.shape[2], self.half_extent)
        j0, j1 = self._range(center[1] - reach[1], center[1] + reach[1], self.shape[1], -self.ground)
        region = self.labels[i0:i1, j0:j1, k0:k1] == label
        if not region.any():
            return 2
        points = np.stack(np.meshgrid(xs[i0:i1], ys[j0:j1], xs[k0:k1], indexing="ij"), -1)[region]
        distance = np.linalg.norm(np.maximum(np.abs(points - center) - half, 0), axis=-1)
        points = points[distance <= STANDOFF_MAX]
        if not len(points):
            return 2
        if len(points) > LOS_CANDIDATES:
            points = points[rng.choice(len(points), LOS_CANDIDATES, replace=False)]
        aim = np.clip(points, center - half, center + half)
        return 0 if self._clear(points, aim).any() else 3

    def _clear(self, eye, aim):
        """[C] sight lines eye -> aim (stopping SIGHT_STOP_M short) through no occupied sight voxel."""
        span = np.linalg.norm(aim - eye, axis=-1)
        steps = int(np.ceil(max(span.max() - SIGHT_STOP_M, 0.0) / SIGHT_STEP_M)) + 1
        along = np.arange(steps) * SIGHT_STEP_M                                      # [S]
        direction = (aim - eye) / np.maximum(span, 1e-6)[:, None]
        samples = eye[:, None] + direction[:, None] * along[None, :, None]          # [C, S, 3]
        valid = along[None, :] <= (span[:, None] - SIGHT_STOP_M)
        index = self._index(samples)
        inside = ((index >= 0) & (index < np.array(self.shape))).all(-1)
        a, b, c = (np.clip(index[..., axis], 0, self.shape[axis] - 1) for axis in range(3))
        hit = self.sight[a, b, c] & inside & valid
        return ~hit.any(-1)


def verify(library, city, seed, style, start_distance, catalog_size, num_envs, progress=None):
    """Per catalog entry: solvable (bool) and REASONS index, with the entry's geometry."""
    with CudaEnv(library, num_envs, seed, 320, 240, style, "velocity", city, start_distance=start_distance,
                 catalog_size=catalog_size) as env, env.execution():
        rows = catalog_entries(env, progress or (lambda done, total: None))
        space = FreeSpace(env)
        fingerprint = env.metadata["city_fingerprint"]
    rng = np.random.default_rng(seed)
    reasons = np.array([space.verdict(rows["start"][e], rows["center"][e], rows["half"][e], rng)
                        for e in range(len(rows["start"]))], np.uint8)
    return {"solvable": reasons == 0, "reason": reasons, "reasons": REASONS, **rows,
            "city_fingerprint": fingerprint, "seed": seed, "style": style,
            "start_distance": start_distance, "catalog_size": catalog_size}
