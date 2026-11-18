"""Training on several cities at once: one ``CudaEnv`` per city behind the trainer's single-env interface.

A single city lets the policy memorize its layout (training success far above held-out districts).
``CityPool`` owns K native environments, one per city directory, each simulating n/K drones on its
own CUDA stream. The pool runs on its own stream; every call fans out to the cities (each waits for
the pool's pending work first) and the results are gathered into concatenated [n, ...] buffers with
device copies, so the trainer, encoder and memory see one batch of n environments. Catalog indices
are offset per city, so ``torch.cat`` of the per-city goal tables is indexed exactly as before.

Only what training reads is exposed; free flight, retargeting and goal rendering stay per city.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .native import CURRICULUM_WIDTH, CudaEnv

GATHERED = ("obs", "proprio", "reward", "done", "success", "privileged", "collision", "diagnostics",
            "ended_bucket", "out_of_bounds", "ended_hint", "episode_diagnostics", "patch_depth", "patch_object")
SHARED = ("device", "width", "height", "style", "control_mode", "library", "action_dim", "horizon",
          "control_hz", "target_classes", "layout_classes", "action_names", "action_scales", "metadata",
          "task_contract")


class CityPool:
    def __init__(self, library, n, seed, width, height, style, control_mode, city_paths, start_distance=None,
                 catalog_size=None):
        if len(city_paths) < 2 or n % len(city_paths):
            raise ValueError(f"{n} environments do not split evenly over {len(city_paths)} cities")
        self.n, share = n, n // len(city_paths)
        self.cities = []
        try:
            for k, path in enumerate(city_paths):
                self.cities.append(CudaEnv(library, share, seed + 7919 * k, width, height, style, control_mode,
                                           path, start_distance=start_distance, catalog_size=catalog_size))
            first = self.cities[0]
            for city in self.cities[1:]:
                differing = sorted(key for key in first.task_contract.keys() | city.task_contract.keys()
                                   if first.task_contract.get(key) != city.task_contract.get(key))
                if differing:
                    raise ValueError(f"{city.city_path} differs from {first.city_path} in task contract {differing}")
            for name in SHARED:
                setattr(self, name, getattr(first, name))
            self.city_paths = tuple(city.city_path for city in self.cities)
            self.city_fingerprints = tuple(city.metadata["city_fingerprint"] for city in self.cities)
            self.catalog_size = sum(city.catalog_size for city in self.cities)
            self.slices = [slice(k * share, (k + 1) * share) for k in range(len(self.cities))]
            self.stream = torch.cuda.Stream(self.device)
            with self.execution():
                for name in GATHERED:
                    source = getattr(first, name)
                    setattr(self, name, torch.empty((n, *source.shape[1:]), device=self.device, dtype=source.dtype))
                self.catalog_index = torch.empty(n, device=self.device, dtype=first.catalog_index.dtype)
                offsets = torch.tensor([0] + [city.catalog_size for city in self.cities[:-1]]).cumsum(0)
                self.catalog_offset = offsets.repeat_interleave(share).to(self.device, first.catalog_index.dtype)
                self.curriculum = torch.zeros(CURRICULUM_WIDTH, device=self.device)
                self._gather()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def execution(self):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            yield self

    def _fan(self, call, *tensors):
        """Run ``call(city, k)`` on every city's stream after the pool's pending work; ``tensors``
        the cities read asynchronously stay allocated until they have."""
        for k, city in enumerate(self.cities):
            city.stream.wait_stream(self.stream)
            for tensor in tensors:
                tensor.record_stream(city.stream)
            with city.execution():
                call(city, k)
        for city in self.cities:
            self.stream.wait_stream(city.stream)

    def _gather(self):
        for city, rows in zip(self.cities, self.slices):
            for name in GATHERED:
                getattr(self, name)[rows].copy_(getattr(city, name))
            self.catalog_index[rows].copy_(city.catalog_index)
        self.catalog_index.add_(self.catalog_offset)

    def reset(self, seed):
        self._fan(lambda city, k: city.reset(seed + 7919 * k))
        self._gather()

    def step(self, actions):
        if actions.shape != (self.n, self.action_dim):
            raise ValueError(f"actions must be [{self.n}, {self.action_dim}]")
        self._fan(lambda city, k: city.step(actions[self.slices[k]]), actions)
        self._gather()

    def write_curriculum(self, lanes):
        self.curriculum.copy_(lanes)
        self._fan(lambda city, k: city.write_curriculum(lanes), lanes)

    def joined(self, build):
        """torch.cat of ``build(city)`` over cities in pool order, each built on its city's stream."""
        parts = []
        for city in self.cities:
            city.stream.wait_stream(self.stream)
            with city.execution():
                parts.append(build(city))
        for city in self.cities:
            self.stream.wait_stream(city.stream)
        return torch.cat(parts)

    def close(self):
        for city in self.cities:
            city.close()
        self.cities = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
