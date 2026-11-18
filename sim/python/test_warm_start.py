"""Warm-started policies with new input lanes (command, depth view) must act identically at first."""
import unittest

import torch

from sim_rl.cli import warm_start
from sim_rl.memory import CANDIDATE_WIDTH, CHANNELS, EGO, VIEW_WIDTH
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

    def test_depth_view_lanes_start_inert(self):
        torch.manual_seed(4)
        source = ActorCritic("visual", 16, 32, 3, 4, goal="global", memory=True, commands=2)
        target = ActorCritic("visual", 16, 32, 3, 4, goal="global", memory=True, commands=2, depth_view=True)
        warm_start(target, source.state_dict())
        n = 5
        cls, tokens, goal = torch.randn(n, 16), torch.randn(n, 12, 16), torch.randn(n, 32)
        proprio, privileged = torch.randn(n, PROPRIO_WIDTH), torch.randn(n, PRIVILEGED_WIDTH)
        ego, candidate = torch.rand(n, CHANNELS, EGO, EGO), torch.rand(n, CANDIDATE_WIDTH)
        view = torch.rand(n, VIEW_WIDTH)
        command = torch.full((n,), GOAL_COMMAND)
        hidden, starts = torch.zeros(n, source.hidden_size), torch.ones(n)
        before, _ = source.step(source.actor_inputs(cls, tokens, goal, proprio, privileged, (ego, candidate), command),
                                hidden, starts)
        inputs = target.actor_inputs(cls, tokens, goal, proprio, privileged,
                                     (ego, torch.cat((candidate, view), -1)), command)
        after, _ = target.step(inputs, hidden, starts)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        self.assertEqual(target.memory_vector_width, CANDIDATE_WIDTH + VIEW_WIDTH)


if __name__ == "__main__":
    unittest.main()
