"""Live diagnostics: distinguish a policy guess from target truth across episode boundaries."""
import threading
import time
from types import SimpleNamespace
import unittest

import torch

from live import Pilot


class LiveDiagnosticsTest(unittest.TestCase):
    def pilot(self):
        pilot = Pilot.__new__(Pilot)
        pilot.waypoint = False
        d = torch.zeros(1, 28)
        d[0, 9], d[0, 27] = 1, 7
        d[0, 24:27] = torch.tensor([2., 10., 2.])
        p = torch.zeros(1, 29)
        p[0, 14], p[0, 20] = 1, 1
        p[0, 25:29] = torch.tensor([-1., .5, -.5, 1.])
        pilot.env = SimpleNamespace(height=240, width=320, control_hz=30, layout_classes=['building'],
                                    obs=torch.zeros(1, 3, 240, 320),
                                    diagnostics=d, privileged=p, proprio=torch.zeros(1, 19),
                                    done=torch.zeros(1), success=torch.zeros(1), collision=torch.zeros(1))
        pilot.lock = threading.Lock()
        pilot.frame_id = 0
        pilot.paused, pilot.error, pilot.last_outcome = False, None, None
        pilot.goal_info = {'source': 'old crop', 'revision': 1}
        pilot.goal_started = time.monotonic()
        pilot.object_info = {7: {'asset': 'tower', 'object_index': 7},
                             8: {'asset': 'house', 'object_index': 8}}
        pilot.checkpoint = {'steps': 3006464, 'live_goal_refresh': False}
        pilot.gemma = None
        pilot.spatial, pilot.memory_image, pilot.memory_info = None, b"", None
        pilot.command = torch.zeros(1, dtype=torch.long)
        pilot.agent, pilot.agent_info, pilot.steps = None, None, 0
        return pilot

    def test_patch_agreement_uses_y_up_and_requires_comparable_ground_truth(self):
        pilot = self.pilot()
        d, p = pilot.env.diagnostics[0].numpy(), pilot.env.privileged[0].numpy()
        for peak, belief, visible, expected in (
                (0, .9, 1, True), (299, .9, 1, False), (0, .2, 1, None), (0, .9, 0, None)):
            with self.subTest(peak=peak, belief=belief, visible=visible):
                p[14] = visible
                pilot.publish(b'frame', p, d, peak, belief, None)
                self.assertIs(pilot.status['saliency']['on_target'], expected)
                self.assertEqual(pilot.status['target']['asset'], 'tower')

    def test_collision_frame_keeps_its_goal_before_catalog_autoreset(self):
        pilot = self.pilot()
        pilot.goal = torch.zeros(1, 768)
        pilot.hidden, pilot.starts = torch.zeros(1, 256), torch.ones(1)
        pilot.features = SimpleNamespace(observe=lambda env: (None, None))
        logits = torch.full((1, 301), -10.)
        logits[0, 0] = 10
        pilot.policy = SimpleNamespace(actor_inputs_and_features=lambda *args: (None, logits),
                                       step=lambda *args: (torch.zeros(1, 4), pilot.hidden))

        def step(action):
            pilot.env.done.fill_(1)
            pilot.env.collision.fill_(1)
            pilot.env.diagnostics = pilot.env.diagnostics.clone()
            pilot.env.diagnostics[0, 27] = 8

        def next_goal():
            pilot.goal_info = {'source': 'new crop', 'revision': 2}

        pilot.env.step = step
        pilot.catalog_goal = next_goal
        # A frame encoder is irrelevant here; preserve real publication and transition ordering.
        from unittest.mock import patch
        with patch('live.jpeg', return_value=b'old camera frame'):
            pilot.tick()
        self.assertEqual(pilot.frame, b'old camera frame')
        self.assertEqual(pilot.status['goal']['source'], 'old crop')
        self.assertEqual(pilot.status['target']['asset'], 'tower')
        self.assertEqual(pilot.status['last_outcome'], 'collision')
        self.assertEqual(pilot.goal_info['source'], 'new crop')


if __name__ == '__main__':
    unittest.main()
