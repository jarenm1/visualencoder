"""Small deterministic edge tests; run with unittest, no simulator required."""
import unittest

import torch

from sim_rl.ppo import Policy, terminal_gae


class TerminalGAETest(unittest.TestCase):
    def test_terminal_cuts_bootstrap_and_next_episode(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        done = torch.tensor([[0.0], [1.0], [0.0]])
        actual = terminal_gae(rewards, values, done, 0.5, 1.0)
        torch.testing.assert_close(actual, torch.tensor([[-8.0], [-18.0], [-7.0]]))

    def test_clock_terminal_never_bootstraps_reset_observation(self):
        rewards = torch.tensor([[5.0]])
        done = torch.ones_like(rewards)
        first = terminal_gae(rewards, torch.tensor([[2.0], [1000.0]]), done, 0.99, 0.95)
        second = terminal_gae(rewards, torch.tensor([[2.0], [-1000.0]]), done, 0.99, 0.95)
        torch.testing.assert_close(first, torch.tensor([[3.0]]))
        torch.testing.assert_close(second, first)

    def test_gru_start_discards_previous_episode_memory(self):
        torch.manual_seed(7)
        policy = Policy(4, action_count=9, hidden_size=8)
        self.assertEqual(policy.actor.out_features, 9)
        features, proprio = torch.randn(3, 4), torch.randn(3, 8)
        old = torch.randn(3, 8)
        a = policy(features, proprio, old, torch.ones(3))
        b = policy(features, proprio, torch.zeros_like(old), torch.zeros(3))
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
