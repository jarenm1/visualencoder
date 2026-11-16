"""CPU-only contracts for matched budgets, disjoint probes, and shift reporting."""

import json

import numpy as np
import pytest

from visenc.geometry import _counts, _indices, _save_new, _summary


def _split(root, n=12):
    root.mkdir()
    meta = {"count": n, "width": 16, "height": 16,
            "views": ["canonical"], "classes": ["sky", "building"]}
    (root / "meta.json").write_text(json.dumps(meta))
    np.save(root / "state.npy", np.zeros((n, 10), dtype=np.float32))
    np.save(root / "depth.npy", np.ones((n, 16, 16), dtype=np.float16))
    np.save(root / "semantic.npy", np.ones((n, 16, 16), dtype=np.uint8))
    np.save(root / "rgb_canonical.npy", np.zeros((n, 16, 16, 3), dtype=np.uint8))
    return {"split": str(root), "count": n}


def test_equal_total_budget_not_per_layout():
    counts = _counts(20000, 3)
    assert counts == [6667, 6667, 6666]
    assert sum(counts) == 20000
    assert sum(_counts(1500, 3)) == sum(_counts(1500, 2)) == 1500


def test_probe_fit_and_familiar_test_are_disjoint_prefixes(tmp_path):
    entry = _split(tmp_path / "split")
    entry["count"] = 8  # A shared fixed evaluation split can be sliced for the multi arm.
    _, fit, test = _indices(entry, 0.5)
    assert fit.tolist() == [0, 1, 2, 3]
    assert test.tolist() == [4, 5, 6, 7]
    assert not set(fit) & set(test)


def test_too_few_probe_fit_samples_are_rejected(tmp_path):
    entry = _split(tmp_path / "split", n=3)
    with pytest.raises(ValueError, match="at least two"):
        _indices(entry, 0.5)


def test_preparation_resume_never_overwrites_different_artifacts(tmp_path):
    path = tmp_path / "manifest.json"
    _save_new(path, {"count": 20000})
    _save_new(path, {"count": 20000})
    with pytest.raises(ValueError, match="different experiment"):
        _save_new(path, {"count": 100})
    assert json.loads(path.read_text()) == {"count": 20000}


def test_shift_summary_weights_scenes_and_retains_geometry_separation():
    def scene(geometry, n, value):
        return {"geometry": geometry, "n": n,
                "views": {v: {"n": n, "pose_r2": value + offset,
                               "state_r2_per_dim": {"pos_x": value + offset}}
                          for v, offset in [("canonical", 0), ("random1", 1),
                                            ("random2", 3), ("novel1", 4), ("novel2", 6)]}}
    result = _summary([scene("familiar", 1, 0), scene("familiar", 3, 4),
                       scene("heldout", 2, -10)])
    assert result["familiar/canonical"]["pose_r2"] == 3
    assert result["familiar/seen"]["pose_r2"] == 5
    assert result["familiar/novel"]["state_r2_per_dim"]["pos_x"] == 8
    assert result["heldout/canonical"]["pose_r2"] == -10
    assert result["familiar/seen"]["n_poses"] == 4
    assert result["familiar/seen"]["n_view_samples"] == 8
    assert "n" not in result["familiar/seen"]
