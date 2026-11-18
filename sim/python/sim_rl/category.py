"""Category goals: search with a reference of the object's type, finish on the actual instance.

The commander may name an object the drone has never seen ("find the truck"). It then supplies a
reference image of that type from a library rather than of the instance; once the instance is in
view it is boxed and its crop replaces the reference. Success is only ever the actual instance.

Training mirrors that: a ``fraction`` of goal episodes start with the goal-table row of another
catalog entry whose target is a different object of the same type (layout asset, e.g. another
"cars:truck"); the goal refresh then swaps in the true target's crop once it is visible. Until
then, flying toward any object of the type is the right behaviour, so look-alike masking only
applies to episodes whose goal is already the instance (``RefreshedGoals.instance``).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .verifier import catalog_entries

RETRIES = 4                      # redraws when the alternate entry shows the same object


def category_references(env):
    """CategoryReferences for ``env``'s catalogue (reads every entry once through the reset override)."""
    rows = catalog_entries(env, lambda done, total: None)
    layout = json.loads((env.city_path / "layout.json").read_text())["objects"]
    return CategoryReferences(rows["meta"][:, 0].astype(np.int64), layout, env.device)


class CategoryReferences:
    def __init__(self, entry_objects, layout, device):
        """``entry_objects``: layout object index of every catalog entry (verifier.catalog_entries)."""
        assets = {}
        object_type = np.array([assets.setdefault(o["asset"], len(assets)) for o in layout])
        entry_objects = np.asarray(entry_objects, np.int64)
        entry_type = object_type[entry_objects]
        order = np.argsort(entry_type, kind="stable")
        counts = np.bincount(entry_type, minlength=len(assets))
        self.objects = torch.as_tensor(entry_objects, device=device)
        self.types = torch.as_tensor(entry_type, device=device)
        self.members = torch.as_tensor(order, device=device)
        self.first = torch.as_tensor(np.concatenate(([0], np.cumsum(counts)[:-1])), device=device)
        self.count = torch.as_tensor(counts, device=device)

    def alternate(self, index, generator=None):
        """Per catalog entry in ``index``: an entry whose target is another object of the same type
        (the entry itself when the type has a single object)."""
        group = self.types[index]
        pick = index.clone()
        for _ in range(RETRIES):
            draw = torch.rand(index.shape, device=index.device, generator=generator)
            offset = (draw * self.count[group]).long().clamp_max(self.count[group] - 1)
            candidate = self.members[self.first[group] + offset]
            pick = torch.where(self.objects[pick] == self.objects[index], candidate, pick)
        return torch.where(self.objects[pick] != self.objects[index], pick, index)
