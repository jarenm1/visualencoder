"""Spatial memory dead reckoning and car-track association."""
import math
import unittest

import torch

from sim_rl.memory import SpatialMemory
from sim_rl.native import CONTROL_HZ
from sim_rl.tracks import ObjectTracks

CAMERA = {"uptilt_rad": math.pi / 12, "vertical_fov_rad": 5 * math.pi / 12, "aspect": 4 / 3}


def quat(axis, angle):
    s = math.sin(angle / 2)
    return torch.tensor([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)], dtype=torch.float64)


def multiply(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return torch.stack((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2))


def rotate(q, v):
    u, w = q[:3], q[3]
    t = 2 * torch.linalg.cross(u, v)
    return v + w * t + torch.linalg.cross(u, t)


def inverse(q):
    return torch.cat((-q[:3], q[3:]))


def attitude(t):
    """Yaw, then a varying pitch, then a varying roll: tilted while turning, pitching and rolling."""
    yaw = quat((0, 1, 0), 0.7 * t + 0.4 * math.sin(1.3 * t))
    pitch = quat((1, 0, 0), -0.35 + 0.25 * math.sin(2.1 * t))
    roll = quat((0, 0, 1), 0.3 * math.sin(1.7 * t))
    return multiply(multiply(yaw, pitch), roll)


def heading(q):
    forward = rotate(q, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
    return math.atan2(-forward[0], -forward[2])   # angle from initial -z toward -x (nose left positive)


class DeadReckoningTest(unittest.TestCase):
    def test_heading_and_position_follow_tilted_turning_flight(self):
        memory = SpatialMemory(1, 15, 20, CAMERA, torch.device("cpu"))
        dt, steps = 1.0 / CONTROL_HZ, 10 * CONTROL_HZ
        position, start = torch.zeros(2, dtype=torch.float64), heading(attitude(0.0))
        for k in range(steps):
            q, q_next = attitude(k * dt), attitude((k + 1) * dt)
            # Body rates from the attitude change; the world velocity flies 3 m/s along the heading.
            delta = multiply(inverse(q), q_next)
            rates = 2 * delta[:3] / dt * (1 if delta[3] >= 0 else -1)
            angle = heading(q) - start
            velocity = torch.tensor([3 * -math.sin(heading(q)), 0.0, 3 * -math.cos(heading(q))], dtype=torch.float64)
            proprio = torch.zeros(1, 19)
            proprio[0, 0:3] = rotate(inverse(q), torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)).float()
            proprio[0, 3:6] = (rates / 3.5).float()
            proprio[0, 6:9] = (rotate(inverse(q), velocity) / 5).float()
            memory.update(proprio, torch.zeros(1, 300), torch.zeros(1, 300))
            # Truth in the memory frame: x = initial forward, y = initial left.
            position += dt * 3 * torch.tensor([math.cos(angle + (heading(q_next) - heading(q))),
                                               math.sin(angle + (heading(q_next) - heading(q)))], dtype=torch.float64)
        drift = math.remainder(float(memory.yaw[0]) - (heading(attitude(steps * dt)) - start), 2 * math.pi)
        self.assertLess(abs(drift), 0.02)
        self.assertLess(float((memory.position[0].double() - position).norm()), 0.5)


class TrackAssociationTest(unittest.TestCase):
    def observe(self, tracks, points, ranges):
        points = torch.tensor([points], dtype=torch.float32)
        tracks.update(points, torch.tensor([ranges]), torch.ones(points.shape[:2], dtype=torch.bool))

    def test_one_car_seen_from_far_and_near_is_one_track(self):
        tracks = ObjectTracks(1, torch.device("cpu"))
        for _ in range(3):   # far: monocular range error places the car ~5 m long
            self.observe(tracks, [[35.0, 0.0], [35.5, 0.3]], [30.0, 30.0])
        for _ in range(3):   # near: accurate
            self.observe(tracks, [[30.0, 0.0], [30.4, 0.2]], [8.0, 8.0])
        self.assertEqual(int(tracks.count()[0]), 1)

    def test_two_parked_cars_stay_two_tracks(self):
        tracks = ObjectTracks(1, torch.device("cpu"))
        for _ in range(4):
            self.observe(tracks, [[10.0, 0.0], [10.3, 0.2], [10.0, 7.0], [10.3, 7.2]], [9.0, 9.0, 11.0, 11.0])
        self.assertEqual(int(tracks.count()[0]), 2)


if __name__ == "__main__":
    unittest.main()
