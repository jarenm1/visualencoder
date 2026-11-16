"""CPU-only regressions for exact ranking metrics and sequence-block uncertainty.

Run: python -m unittest discover -s tests -p 'test_real_metrics.py'
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from visenc.real import (
    METRICS,
    add_args,
    _blocks,
    _block_scores,
    _ci,
    _summarize_cache,
    patch_targets,
    rescore,
    scores,
    sequence_ids,
    sequence_resamples,
    validate_provenance,
)


def fixture():
    # Unequal frame counts deliberately expose frame/patch bootstrap substitutions.
    frac = np.zeros((3, 4, 8), dtype=np.float32)
    frac[..., 1] = 1
    frac[:, 0, 1], frac[:, 0, 5] = 0.8, 0.2
    frac[:, 1, 1], frac[:, 1, 5] = 0.95, 0.05  # ambiguous detection examples
    frac[0, 2, 1], frac[0, 2, 7] = 0.1, 0.9  # ignored segmentation, retained detection
    label, valid = patch_targets(frac)
    pred = label.copy()
    pred[2] = 2
    vehicle_score = np.array([[0.8, 99, 0.8, 0.1], [0.8, -99, 0.5, 0.1], [0.2, 77, 0.6, 0.1]])
    output = np.zeros((3, 4, 7))
    output[..., 5] = vehicle_score
    frames = ["seq16_000000.png", "seq16_000100.png", "seq17_000000.png"]
    meta = {"train": ["seq01_000000.png"], "val": frames}
    return {
        "spec": np.asarray("baseline"),
        "provenance": np.asarray(json.dumps(meta)),
        "probe_fit": np.asarray(json.dumps({"sim": "historical-fixture"})),
        "frames": np.asarray(frames),
        "sequence_ids": sequence_ids(frames),
        "frac": frac, "label": label, "valid": valid,
        "sim_to_real_pred": pred, "sim_to_real_scores": output,
        "real_to_real_pred": label.copy(), "real_to_real_scores": output.copy(),
    }


class RankingTests(unittest.TestCase):
    def evaluate_binary(self, car_fractions, values, valid=None):
        frac = np.zeros((1, len(values), 8), dtype=np.float32)
        frac[..., 1] = 1
        frac[..., 5] = car_fractions
        label = np.ones((1, len(values)), dtype=int)
        if valid is None:
            valid = np.ones_like(label, dtype=bool)
        return scores(label, label, valid, frac, np.asarray([values], dtype=float))

    def test_all_ties_have_half_auc_and_prevalence_ap(self):
        result = self.evaluate_binary([0.2, 0.2, 0, 0, 0], [1, 1, 1, 1, 1])
        self.assertEqual(result["vehicle_auroc"], 0.5)
        self.assertAlmostEqual(result["vehicle_ap"], 0.4)
        self.assertEqual(result["vehicle_positive_count"], 2)
        self.assertEqual(result["vehicle_negative_count"], 3)
        self.assertEqual(result["vehicle_prevalence"], 0.4)

    def test_auc_equals_exact_pairwise_tie_credit(self):
        positive, negative = np.array([0.1, 0.5, 0.5, 0.9]), np.array([0.1, 0.1, 0.5, 1.0])
        expected = ((positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative)).mean()
        for order in (np.arange(8), np.array([7, 3, 5, 0, 4, 2, 1, 6])):
            fractions = np.array([0.2] * 4 + [0.0] * 4)[order]
            values = np.concatenate([positive, negative])[order]
            self.assertAlmostEqual(self.evaluate_binary(fractions, values)["vehicle_auroc"], expected)

    def test_ap_groups_threshold_ties_not_arbitrary_patch_order(self):
        # A tied positive+negative at score .9: precision .5, then final positive: 2/3.
        result = self.evaluate_binary([0.2, 0, 0.2], [0.9, 0.9, 0.1])
        reversed_tie = self.evaluate_binary([0, 0.2, 0.2], [0.9, 0.9, 0.1])
        self.assertAlmostEqual(result["vehicle_ap"], 7 / 12)
        self.assertEqual(result["vehicle_ap"], reversed_tie["vehicle_ap"])

    def test_ambiguous_excluded_ignore_mask_does_not_change_detection(self):
        result = self.evaluate_binary([0.1, 0, 0.05], [0.2, 0.1, 100], np.array([[False, True, True]]))
        self.assertEqual(result["vehicle_positive_count"], 1)
        self.assertEqual(result["vehicle_negative_count"], 1)
        self.assertEqual(result["vehicle_prevalence"], 0.5)
        self.assertEqual(result["vehicle_auroc"], 1)
        self.assertEqual(result["vehicle_ap"], 1)

    def test_undefined_class_metrics_and_ci_draw_counts(self):
        result = self.evaluate_binary([0, 0], [1, 1])
        self.assertTrue(np.isnan(result["vehicle_auroc"]))
        self.assertTrue(np.isnan(result["vehicle_ap"]))
        self.assertEqual(_ci(np.array([np.nan, 0.2, 0.8]))["valid_resamples"], 2)
        self.assertEqual(_ci(np.array([np.nan])), {"low": None, "high": None, "valid_resamples": 0})

    def test_semantic_targets_preserve_ignore_cutoff_and_shared_argmax(self):
        frac = np.zeros((1, 3, 8))
        frac[0, :, 7] = [0.49, 0.5, 0.8]
        frac[0, :, 3] = [0.3, 0.4, 0.1]
        frac[0, :, 5] = [0.2, 0.1, 0.1]
        label, valid = patch_targets(frac)
        np.testing.assert_array_equal(label, [[3, 3, 3]])
        np.testing.assert_array_equal(valid, [[True, False, False]])


class SequenceTests(unittest.TestCase):
    def test_train_val_leakage_uses_sequence_not_frame_identity(self):
        meta = {"train": ["seq16_000000.png"], "val": ["seq16_000100.png"]}
        with self.assertRaisesRegex(ValueError, "sequence leakage"):
            validate_provenance(meta, 1, 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_provenance({"train": ["seq01_000000.png"] * 2, "val": ["seq02_000000.png"]}, 2, 1)
        with self.assertRaises(ValueError):
            sequence_ids(["unidentified.png"])

    def test_block_statistics_equal_resampling_whole_sequences(self):
        cache = fixture()
        ids = cache["sequence_ids"]
        unique, weights = sequence_resamples(ids, 50, seed=19)
        np.testing.assert_array_equal(weights, sequence_resamples(ids, 50, seed=19)[1])
        np.testing.assert_array_equal(weights.sum(1), np.full(50, len(unique)))
        self.assertTrue(np.any(weights[:, 0] == 2))
        blocks = _blocks(cache["sim_to_real_pred"], cache["label"], cache["valid"], cache["frac"], cache["sim_to_real_scores"][..., 5], ids)
        for weight in weights:
            # Repeat ALL frames in a chosen video each time that video is drawn.
            index = np.concatenate([np.tile(np.flatnonzero(ids == sequence), count) for sequence, count in zip(unique, weight)])
            actual = _block_scores(blocks, weight)
            expected = scores(cache["sim_to_real_pred"][index], cache["label"][index], cache["valid"][index], cache["frac"][index], cache["sim_to_real_scores"][index, ..., 5])
            for metric in (*METRICS, "accuracy", "vehicle_prevalence"):
                self.assertAlmostEqual(actual[metric], expected[metric])

    def test_gap_uses_same_sequence_draw_for_both_directions(self):
        cache = fixture()
        weights = sequence_resamples(cache["sequence_ids"], 100, seed=4)[1]
        row, draws = _summarize_cache(cache, weights)
        for metric in METRICS:
            np.testing.assert_allclose(draws["gap"][metric], draws["real_to_real"][metric] - draws["sim_to_real"][metric])
            self.assertEqual(row["gap"]["ci95"][metric], _ci(draws["gap"][metric]))
        # The segmentation gap is not a patch-level binomial interval.
        self.assertGreater(row["gap"]["ci95"]["miou"]["high"], row["gap"]["ci95"]["miou"]["low"])

    def test_zero_resamples_keeps_points_and_marks_intervals_unavailable(self):
        cache = fixture()
        weights = sequence_resamples(cache["sequence_ids"], 0, seed=1)[1]
        row, _ = _summarize_cache(cache, weights)
        self.assertTrue(np.isfinite(row["sim_to_real"]["miou"]))
        self.assertEqual(row["sim_to_real"]["ci95"]["miou"], {"low": None, "high": None, "valid_resamples": 0})

    def test_existing_eval_arguments_keep_historical_fit_defaults(self):
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(["--encoders", "lingbot:small", "--out", "report"])
        self.assertEqual(args.sim, "data/eval-v2")
        self.assertEqual(args.sim_fit, 1200)
        self.assertEqual(args.sim_view, "canonical")
        self.assertEqual(args.bootstrap_resamples, 1000)
        self.assertEqual(args.bootstrap_seed, 0)
        self.assertIsNone(args.baseline)
        self.assertFalse(args.append)

    def test_invalid_bootstrap_settings(self):
        with self.assertRaises(ValueError):
            sequence_resamples(np.array(["seq01"]), -1)
        with self.assertRaises(ValueError):
            sequence_resamples(np.array([]), 10)


class CachedRescoreTests(unittest.TestCase):
    def save_report(self, directory, cache, name):
        archive = directory / f"{name}.npz"
        np.savez_compressed(archive, **cache)
        report = directory / f"{name}.json"
        report.write_text(json.dumps([{"spec": str(cache["spec"].item()), "predictions": archive.name}]))
        return report

    def test_merge_paired_differences_and_repeat_without_model_loading(self):
        baseline = fixture()
        improved = {**fixture(), "spec": np.asarray("improved")}
        improved["sim_to_real_pred"] = improved["label"].copy()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reports = [self.save_report(directory, baseline, "baseline"), self.save_report(directory, improved, "improved")]
            with contextlib.redirect_stdout(io.StringIO()):
                rows = rescore(reports, directory / "combined", 80, 23, "baseline")
                again = rescore([directory / "combined" / "report.json"], directory / "again", 80, 23, "baseline")
            self.assertEqual(rows, again)
            zero = rows[0]["paired_vs_baseline"]["sim_to_real"]["miou"]
            self.assertEqual(zero["difference"], 0)
            self.assertEqual(zero["ci95"]["low"], 0)
            self.assertEqual(zero["ci95"]["high"], 0)
            weights = sequence_resamples(baseline["sequence_ids"], 80, 23)[1]
            _, a = _summarize_cache(baseline, weights)
            _, b = _summarize_cache(improved, weights)
            paired = rows[1]["paired_vs_baseline"]["sim_to_real"]["miou"]
            self.assertEqual(paired["ci95"], _ci(b["sim_to_real"]["miou"] - a["sim_to_real"]["miou"]))
            self.assertGreater(paired["difference"], 0)
            protocol = json.loads((directory / "combined" / "protocol.json").read_text())
            self.assertEqual(protocol["bootstrap"]["unit"], "whole UAVid validation video sequence")
            self.assertTrue(protocol["sequence_disjoint_checked"])

    def test_mismatched_frame_order_cannot_be_paired(self):
        first, second = fixture(), {**fixture(), "spec": np.asarray("second")}
        second["frames"] = second["frames"][::-1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reports = [self.save_report(directory, first, "first"), self.save_report(directory, second, "second")]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "identical validation"):
                rescore(reports, directory / "combined", 20, 0, "baseline")

    def test_cached_sequence_ids_must_match_provenance(self):
        cache = fixture()
        cache["sequence_ids"] = np.array(["seq99"] * 3)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.save_report(directory, cache, "bad")
            with self.assertRaisesRegex(ValueError, "disagree with metadata"):
                rescore([report], directory / "out", 10)

    def test_old_report_requires_one_real_prediction_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = directory / "old.json"
            report.write_text(json.dumps([{"spec": "historical", "sim_to_real": {"miou": 0.5}}]))
            with self.assertRaisesRegex(ValueError, "historical reports need"):
                rescore([report], directory / "out", 10)


if __name__ == "__main__":
    unittest.main()
