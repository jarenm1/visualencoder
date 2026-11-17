"""Curriculum weight transitions: promotion, demotion, floors, normalization. CPU only."""
import unittest

import torch

from sim_rl.curriculum import update_weights


def tensor(values):
    return torch.tensor(values, dtype=torch.float32)


class UpdateWeightsTest(unittest.TestCase):
    floors = tensor([0.10, 0.10, 0.05])

    def test_promotion_moves_step_to_next_bucket(self):
        actual = update_weights(tensor([0.8, 0.15, 0.05]), tensor([0.9, 0.5, 0.5]), self.floors)
        torch.testing.assert_close(actual, tensor([0.75, 0.20, 0.05]))

    def test_promotion_respects_floor(self):
        weights = tensor([0.10, 0.50, 0.40])
        actual = update_weights(weights, tensor([0.9, 0.5, 0.5]), self.floors)
        torch.testing.assert_close(actual, weights)

    def test_demotion_returns_bounded_mass_to_lower_bucket(self):
        actual = update_weights(tensor([0.2, 0.3, 0.5]), tensor([0.0, 0.5, 0.1]), self.floors)
        torch.testing.assert_close(actual, tensor([0.2, 0.325, 0.475]))

    def test_result_is_normalized(self):
        actual = update_weights(tensor([0.4, 0.4, 0.2]), tensor([1.0, 1.0, 0.0]), self.floors)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
