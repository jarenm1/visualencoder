"""Start-camera projection and box matching used to score language-grounded goals. CPU only."""
import math
import unittest

import numpy as np

from gemma_goal import iou, project_boxes, start_camera

CAMERA = {"vertical_fov_rad": 5 * math.pi / 12, "aspect": 4 / 3, "uptilt_rad": math.pi / 12,
          "offset_body_m": [0.0, 0.05, -0.1], "near_m": 0.05}


def diagnostics(position, quaternion):
    row = np.zeros((1, 28))
    row[0, 0:3], row[0, 6:10] = position, quaternion
    return row


class GroundingGeometryTest(unittest.TestCase):
    def test_identity_body_camera_looks_down_minus_z_with_uptilt(self):
        eye, forward, up = start_camera(diagnostics([1.0, 2.0, 3.0], [0, 0, 0, 1]), {"camera": CAMERA})
        np.testing.assert_allclose(eye[0], [1.0, 2.05, 2.9], atol=1e-9)
        np.testing.assert_allclose(forward[0], [0, math.sin(math.pi / 12), -math.cos(math.pi / 12)], atol=1e-9)
        self.assertAlmostEqual(float(forward[0] @ up[0]), 0.0)

    def test_box_on_the_optical_axis_projects_centred_and_left_box_lands_left(self):
        eye, forward, up = np.zeros(3), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
        centers = np.array([[0.0, 0.0, -10.0], [-6.0, 0.0, -10.0], [0.0, 0.0, 10.0]])
        boxes = project_boxes(eye, forward, up, centers, np.full((3, 3), 0.5), CAMERA)
        np.testing.assert_allclose((boxes[0, 0] + boxes[0, 2]) / 2, 0.5, atol=1e-9)
        np.testing.assert_allclose((boxes[0, 1] + boxes[0, 3]) / 2, 0.5, atol=1e-9)
        self.assertLess(boxes[1, 2], 0.5)       # left of centre in the image
        self.assertTrue(np.isnan(boxes[2]).all())  # behind the camera

    def test_iou_prefers_the_overlapping_box(self):
        boxes = np.array([[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.1, 0.1], [np.nan] * 4])
        overlap = iou(np.array([0.42, 0.4, 0.6, 0.62]), boxes)
        self.assertEqual(int(overlap.argmax()), 0)
        self.assertGreater(overlap[0], 0.8)
        self.assertEqual(overlap[2], 0.0)


if __name__ == "__main__":
    unittest.main()
