"""Old visual-policy comparisons must not bypass encoder or flight-contract checks."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest

import torch

from sim_rl.cli import CHECKPOINT_FORMAT, TRANSFER_SOURCE, load_checkpoint, native_provenance, validate_checkpoint
from sim_rl.native import ABI_VERSION, HORIZON, TASK


def _fixture(directory: str):
    library = Path(directory) / "native.so"
    library.write_bytes(b"provenance fixture")
    env = SimpleNamespace(
        action_dim=4, horizon=HORIZON, control_hz=30, control_mode="velocity", catalog_size=256,
        width=224, height=224, library=library,
        task_contract={"task": TASK, "abi_version": ABI_VERSION, "horizon": HORIZON,
                       "dynamics": {"velocity_scales": [4.0, 2.0, 1.5]},
                       "sensors": {"proprio_width": 19}, "privileged": {"width": 29},
                       "success": {"hold_steps": 90}, "reward": {"timeout": -10.0},
                       "camera_effects": {"vignette": 0.35}, "patch_labels": {"patch_px": 16}})
    features = SimpleNamespace(visual=True, spec="mean", dim=64, fingerprint="frozen-encoder", goal="global",
                               preprocessing={"engine": "tensorrt", "precision": "fp16"})
    recorded: dict[str, Any] = deepcopy(native_provenance(env, features))
    source_task, source_abi = TRANSFER_SOURCE
    recorded.update(task=source_task, abi_version=source_abi, horizon=450)
    contract = recorded["native_task_contract"]
    contract.update(task=source_task, abi_version=source_abi, horizon=450, success={"hold_steps": 90, "timing": "deadline"})
    del contract["camera_effects"], contract["patch_labels"]
    saved: dict[str, Any] = {"format": CHECKPOINT_FORMAT, "metadata": recorded, "model": {}}
    return env, features, saved


class TaskTransferTests(unittest.TestCase):
    def test_old_task_requires_explicit_evaluation_transfer(self):
        with TemporaryDirectory() as directory:
            env, features, old = _fixture(directory)
            path = Path(directory) / "old.pt"
            torch.save(old, path)
            with self.assertRaisesRegex(ValueError, "incompatible native ABI or task"):
                load_checkpoint(path)
            saved = load_checkpoint(path, for_task_transfer=True)
            with self.assertRaisesRegex(ValueError, "explicit visual-policy evaluation"):
                validate_checkpoint(saved, env, features)
            validate_checkpoint(saved, env, features, for_task_transfer=True)

    def test_transfer_still_rejects_changed_flight_sensors_and_encoder(self):
        with TemporaryDirectory() as directory:
            env, features, old = _fixture(directory)
            for field in ("dynamics", "sensors", "privileged"):
                with self.subTest(field=field):
                    saved = deepcopy(old)
                    saved["metadata"]["native_task_contract"][field] = {"incompatible": True}
                    with self.assertRaisesRegex(ValueError, "dynamics, sensors or observation"):
                        validate_checkpoint(saved, env, features, for_task_transfer=True)
            saved = deepcopy(old)
            saved["metadata"]["encoder_fingerprint"] = "different-encoder"
            with self.assertRaisesRegex(ValueError, "encoder_fingerprint differs"):
                validate_checkpoint(saved, env, features, for_task_transfer=True)
            features.visual, features.spec = False, "state"
            with self.assertRaisesRegex(ValueError, "explicit visual-policy evaluation"):
                validate_checkpoint(old, env, features, for_task_transfer=True)


if __name__ == "__main__":
    unittest.main()
