"""Diagonal Gaussian helpers against torch.distributions.Normal. CPU only."""
import unittest

import torch

from sim_rl.model import gaussian_entropy, gaussian_log_prob


class GaussianTest(unittest.TestCase):
    def test_log_prob_matches_torch(self):
        torch.manual_seed(0)
        mean, actions = torch.randn(5, 4), torch.randn(5, 4)
        log_std = torch.randn(4) * 0.3
        expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
        torch.testing.assert_close(gaussian_log_prob(mean, log_std, actions), expected)

    def test_entropy_matches_torch(self):
        torch.manual_seed(1)
        log_std = torch.randn(4) * 0.3
        expected = torch.distributions.Normal(torch.zeros(4), log_std.exp()).entropy().sum()
        torch.testing.assert_close(gaussian_entropy(log_std), expected)


if __name__ == "__main__":
    unittest.main()
