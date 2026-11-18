"""Object tracks on the spatial memory's ground plane: what lets the drone count.

Each frame, patches the perception head calls a given class (P >= ``CLASS_THRESHOLD``) within
``MAX_RANGE_M`` are placed at their projected ground points (``SpatialMemory.points``: predicted
depth, dead-reckoned pose), then merged: patches of one object in one frame join within
``FRAME_MERGE_M`` (clusters of fewer than ``MIN_PATCHES`` are dropped); each frame cluster then
updates the nearest track within its association gate or starts a new one. Monocular range error
grows with distance, so the gate is ``GATE_M + GATE_PER_M * range`` with the larger of the
observation's range and the track's effective range (harmonic mean of its observations' ranges, so
a car first placed from far away still accepts a precise near sighting), and observations update a
track with weight 1 / range. After each
frame, tracks closer than ``GATE_M`` merge. A track counts once seen in ``CONFIRM_HITS`` frames, so a
single misclassified patch does not count. Counting is a query on these tracks (``count``), not a
flight behaviour.

Fixed capacity (``CAPACITY`` tracks per environment) on the GPU; association loops over the few
clusters per frame, so this is an evaluation and query-time tool, not part of PPO collection.
"""
from __future__ import annotations

import torch

CAPACITY = 128
CLASS_THRESHOLD = 0.5
MAX_RANGE_M = 35.0
FRAME_MERGE_M = 3.0
MIN_PATCHES = 2                  # a frame cluster needs this many patches (one patch is often a false call)
GATE_M, GATE_PER_M = 3.0, 0.25
CONFIRM_HITS = 3


class ObjectTracks:
    def __init__(self, n, device):
        self.n, self.device = n, device
        self.position = torch.zeros((n, CAPACITY, 2), device=device)
        self.weight = torch.zeros((n, CAPACITY), device=device)
        self.hits = torch.zeros((n, CAPACITY), device=device)
        self.used = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, starts):
        keep = starts < 0.5
        self.hits.mul_(keep[:, None])
        self.weight.mul_(keep[:, None])
        self.used.mul_(keep)

    @torch.no_grad()
    def update(self, points, ranges, members):
        """points [n, P, 2] patch ground positions (memory frame), ranges [n, P] metres from the drone,
        members [n, P] bool: patches of the tracked class with valid depth."""
        members = members & (ranges <= MAX_RANGE_M)
        for env in members.any(-1).nonzero().flatten().tolist():
            cloud, reach = points[env, members[env]], ranges[env, members[env]]
            # Greedy frame clustering: seed at the first unassigned point, absorb points within reach.
            unassigned = torch.ones(cloud.shape[0], dtype=torch.bool, device=self.device)
            while bool(unassigned.any()):
                seed = cloud[unassigned.nonzero()[0, 0]]
                near = unassigned & ((cloud - seed).norm(dim=-1) <= FRAME_MERGE_M)
                if int(near.sum()) >= MIN_PATCHES:
                    self._associate(env, cloud[near].mean(0), float(reach[near].mean()))
                unassigned &= ~near
            self._merge(env)

    def _associate(self, env, centre, reach):
        used, weight = int(self.used[env]), 1.0 / max(reach, 1.0)
        if used:
            distance = (self.position[env, :used] - centre).norm(dim=-1)
            spread = torch.clamp(self.hits[env, :used] / self.weight[env, :used].clamp_min(1e-6), min=reach)
            best = int(distance.argmin())
            if float(distance[best]) <= GATE_M + GATE_PER_M * float(spread[best]):
                total = self.weight[env, best] + weight
                self.position[env, best] = (self.position[env, best] * self.weight[env, best] + centre * weight) / total
                self.weight[env, best] = total
                self.hits[env, best] += 1
                return
        if used < CAPACITY:
            self.position[env, used], self.weight[env, used], self.hits[env, used] = centre, weight, 1
            self.used[env] = used + 1

    def _merge(self, env):
        """Fold tracks that converged within GATE_M of each other into the heavier one."""
        used = int(self.used[env])
        if used < 2:
            return
        position, weight, hits = self.position[env, :used], self.weight[env, :used], self.hits[env, :used]
        distance = torch.cdist(position, position)
        distance.fill_diagonal_(float("inf"))
        if float(distance.min()) > GATE_M:
            return
        keep = torch.ones(used, dtype=torch.bool, device=self.device)
        for a, b in (distance <= GATE_M).nonzero().tolist():
            if a < b and keep[a] and keep[b]:
                total = weight[a] + weight[b]
                position[a] = (position[a] * weight[a] + position[b] * weight[b]) / total
                weight[a], hits[a] = total, hits[a] + hits[b]
                keep[b] = False
        count = int(keep.sum())
        self.position[env, :count] = position[keep]
        self.weight[env, :count] = weight[keep]
        self.hits[env, :count] = hits[keep]
        self.used[env] = count

    def count(self):
        """[n] confirmed tracks per environment."""
        slots = torch.arange(CAPACITY, device=self.device)[None] < self.used[:, None]
        return ((self.hits >= CONFIRM_HITS) & slots).sum(-1)

    def confirmed(self, env):
        """Memory-frame positions [2] of environment ``env``'s confirmed tracks."""
        used = int(self.used[env])
        return [self.position[env, k] for k in range(used) if float(self.hits[env, k]) >= CONFIRM_HITS]
