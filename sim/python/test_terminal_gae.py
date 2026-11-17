"""Small deterministic edge tests; run with unittest, no simulator required."""
import unittest

import torch

from sim_rl.model import ActorCritic
from sim_rl.ppo import terminal_gae


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
        policy = ActorCritic("hint", 0, 0, 15, 20, action_dim=4)
        proprio = torch.randn(3, 18)
        inputs = policy.actor_inputs(None, None, None, proprio, torch.randn(3, 18))
        old = torch.randn(3, policy.hidden_size)
        mean_a, hidden_a = policy.step(inputs, old, torch.ones(3))
        mean_b, hidden_b = policy.step(inputs, torch.zeros_like(old), torch.zeros(3))
        self.assertEqual(tuple(mean_a.shape), (3, 4))
        torch.testing.assert_close(mean_a, mean_b)
        torch.testing.assert_close(hidden_a, hidden_b)


if __name__ == "__main__":
    unittest.main()
