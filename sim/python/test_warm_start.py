"""A command-conditioned policy warm-started from a single-command one must act identically at first."""
import unittest

import torch

from sim_rl.cli import warm_start
from sim_rl.model import GOAL_COMMAND, ActorCritic
from sim_rl.native import PRIVILEGED_WIDTH, PROPRIO_WIDTH


class WarmStartTest(unittest.TestCase):
    def test_new_command_lanes_start_inert(self):
        torch.manual_seed(3)
        source = ActorCritic("state", 0, 0, 15, 20)
        target = ActorCritic("state", 0, 0, 15, 20, commands=2)
        warm_start(target, source.state_dict())
        proprio, privileged = torch.randn(5, PROPRIO_WIDTH), torch.randn(5, PRIVILEGED_WIDTH)
        command = torch.full((5,), GOAL_COMMAND)
        hidden, starts = torch.zeros(5, source.hidden_size), torch.ones(5)
        before, _ = source.step(source.actor_inputs(None, None, None, proprio, privileged), hidden, starts)
        after, _ = target.step(target.actor_inputs(None, None, None, proprio, privileged, command=command), hidden, starts)
        self.assertTrue(torch.allclose(before, after))
        self.assertTrue(torch.allclose(source.value(privileged, proprio), target.value(privileged, proprio, command)))


if __name__ == "__main__":
    unittest.main()
