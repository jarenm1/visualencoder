"""Diagonal and tanh-squashed Gaussian helpers against torch.distributions. CPU only."""
import unittest

import torch

from sim_rl.model import deterministic_action, gaussian_log_prob, squashed_entropy, squashed_log_prob


class GaussianTest(unittest.TestCase):
    def test_log_prob_matches_torch(self):
        torch.manual_seed(0)
        mean, actions = torch.randn(5, 4), torch.randn(5, 4)
        log_std = torch.randn(4) * 0.3
        expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
        torch.testing.assert_close(gaussian_log_prob(mean, log_std, actions), expected)

    def test_squashed_entropy_peaks_then_falls_as_the_gaussian_widens(self):
        torch.manual_seed(1)
        mean = torch.zeros(20000, 4)
        entropy = {s: float(squashed_entropy(mean, torch.full((4,), s))) for s in (-2.0, 0.0, 2.0)}
        # The pre-squash Gaussian entropy grows linearly in log_std; the squashed one must not.
        self.assertGreater(entropy[0.0], entropy[-2.0])
        self.assertGreater(entropy[0.0], entropy[2.0])

    def test_squashed_log_prob_matches_tanh_transform(self):
        torch.manual_seed(2)
        mean, raw = torch.randn(6, 4), torch.randn(6, 4) * 1.5  # reaches |tanh| > 0.99, still invertible
        log_std = torch.randn(4) * 0.3
        squashed = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, log_std.exp()), [torch.distributions.TanhTransform()])
        expected = squashed.log_prob(torch.tanh(raw)).sum(-1)
        torch.testing.assert_close(squashed_log_prob(mean, log_std, raw), expected, rtol=1e-4, atol=1e-3)

    def test_deterministic_action_is_bounded_and_monotone(self):
        mean = torch.tensor([-50.0, -0.5, 0.0, 0.5, 50.0])
        action = deterministic_action(mean)
        self.assertTrue(bool((action.abs() <= 1).all()))
        self.assertTrue(bool((action[1:] > action[:-1]).all()))


if __name__ == "__main__":
    unittest.main()
