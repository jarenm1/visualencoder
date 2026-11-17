"""CPU-only checks: rank metrics, box-region auxiliary targets, GPS heuristic, curriculum lanes."""
import math
import unittest

import torch

from sim_rl.curriculum import Curriculum
from sim_rl.evaluation import gps_pursuit
from sim_rl.model import saliency_loss, saliency_targets
from sim_rl.native import PRIVILEGED_WIDTH, PROPRIO_WIDTH
from sim_rl.ppo import Config
from sim_rl.stats import auroc, average_precision, wilson

VELOCITY_SCALES = (4.0, 4.0, 2.0, 1.5)


class RankMetricsTest(unittest.TestCase):
    def test_auroc_matches_hand_value(self):
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)

    def test_auroc_without_both_classes_is_none(self):
        self.assertIsNone(auroc([0.1, 0.2], [1, 1]))

    def test_average_precision_matches_hand_value(self):
        self.assertAlmostEqual(average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]), 5 / 6)

    def test_wilson_is_none_without_trials(self):
        self.assertIsNone(wilson(0, 0))
        low, high = wilson(50, 100)
        self.assertTrue(low < 0.5 < high)


class SaliencyTargetTest(unittest.TestCase):
    def test_visible_target_spreads_over_its_box_and_hidden_goes_to_absent_class(self):
        privileged = torch.zeros(3, PRIVILEGED_WIDTH)
        # Visible target whose projection covers the top-left 2x2 patches (patches are 0.1 x 0.133 NDC).
        privileged[0, 14], privileged[0, 18:20] = 1.0, torch.tensor([-0.95, 0.95])
        privileged[0, 25:29] = torch.tensor([-1.0, 0.75, -0.85, 1.0])
        # Visible target smaller than a patch: only the anchor patch (bottom-right) carries mass.
        privileged[1, 14], privileged[1, 18:20] = 1.0, torch.tensor([0.99, -0.99])
        privileged[1, 25:29] = torch.tensor([0.985, -0.995, 0.995, -0.985])
        target, visible, on_target = saliency_targets(privileged, 15, 20)
        self.assertEqual(sorted(on_target[0].nonzero().flatten().tolist()), [0, 1, 20, 21])
        torch.testing.assert_close(target[0, [0, 1, 20, 21]], torch.full((4,), 0.25))
        self.assertEqual(on_target[1].nonzero().flatten().tolist(), [14 * 20 + 19])
        self.assertEqual(target[2].argmax().item(), 300)
        self.assertEqual(visible.tolist(), [True, True, False])
        torch.testing.assert_close(target.sum(-1), torch.ones(3))

    def test_accuracy_counts_peaks_on_the_target_in_visible_frames_only(self):
        privileged = torch.zeros(2, PRIVILEGED_WIDTH)
        privileged[0, 14], privileged[0, 18:20] = 1.0, torch.tensor([-0.95, 0.95])
        privileged[0, 25:29] = torch.tensor([-1.0, 0.75, -0.85, 1.0])
        logits = torch.full((2, 301), -10.0)
        logits[0, 21] = 10.0     # a patch inside the visible target's box, not the anchor patch
        logits[1, 5] = 10.0      # hidden frame: patch choice must not count
        loss, accuracy = saliency_loss(logits, privileged, 15, 20)
        self.assertEqual(accuracy.item(), 1.0)
        self.assertTrue(math.isfinite(loss.item()))


class GpsPursuitTest(unittest.TestCase):
    def proprio(self, right=0.0, up=0.0, forward=0.0, gps=1.0):
        values = torch.zeros(1, PROPRIO_WIDTH)
        values[0, 14], values[0, 15], values[0, 16], values[0, 18] = right / 32, up / 32, -forward / 32, gps
        return values

    def test_target_to_the_right_commands_right_and_nose_right(self):
        actions = gps_pursuit(self.proprio(right=1.0), VELOCITY_SCALES)
        torch.testing.assert_close(actions, torch.tensor([[0.0, 0.125, 0.0, -1.0]]))

    def test_target_ahead_commands_forward_without_turn(self):
        actions = gps_pursuit(self.proprio(forward=1.0), VELOCITY_SCALES)
        torch.testing.assert_close(actions, torch.tensor([[0.125, 0.0, 0.0, 0.0]]))

    def test_no_gps_is_hover(self):
        actions = gps_pursuit(self.proprio(right=1.0, up=1.0, forward=1.0, gps=0.0), VELOCITY_SCALES)
        torch.testing.assert_close(actions, torch.zeros(1, 4))


class CurriculumLaneTest(unittest.TestCase):
    def test_lanes_carry_bucket_weights_then_no_gps_then_appearance(self):
        lanes = Curriculum("adaptive", torch.device("cpu"), 0.25, 0.5).lanes()
        torch.testing.assert_close(lanes, torch.tensor([0.8, 0.15, 0.05, 0.25, 0.5]))

    def test_uniform_mode_starts_uniform(self):
        lanes = Curriculum("uniform", torch.device("cpu"), 0.0).lanes()
        torch.testing.assert_close(lanes, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0, 1.0]))

    def test_no_gps_fraction_must_be_a_probability(self):
        with self.assertRaises(ValueError):
            Config(no_gps_fraction=1.5).validate()

    def test_appearance_must_be_a_strength(self):
        with self.assertRaises(ValueError):
            Config(appearance=-0.1).validate()


if __name__ == "__main__":
    unittest.main()
