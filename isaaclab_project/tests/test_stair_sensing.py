"""Sensor-only stair observations and success-gated curriculum, no Isaac needed."""
import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/stair_sensing.py'
SPEC = importlib.util.spec_from_file_location('stair_sensing', PATH)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class StairSensingTests(unittest.TestCase):
    def test_geometry_and_ground_height(self):
        h = torch.tensor([0.02, 0.12])
        lo, hi = S.stair_boxes(h)
        self.assertEqual(lo.shape, (2, 3, 3))
        torch.testing.assert_close(hi[:, -1, 2], 3 * h)
        torch.testing.assert_close(S.terrain_height(torch.tensor([0.1, 0.7]), h), torch.tensor([0., 0.24]))
        with self.assertRaises(ValueError):
            S.stair_boxes(torch.tensor([0.]))

    def test_riser_occludes_top_and_background(self):
        lo, hi = S.stair_boxes(torch.tensor([0.10]))
        origins = torch.tensor([[0., 0., 0.05]])
        rays = torch.tensor([[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]])
        distance, valid = S.raycast_stairs(origins, rays, lo, hi)
        torch.testing.assert_close(distance[0, :2], torch.tensor([0.4, 0.05]))
        self.assertEqual(valid.tolist(), [[True, True, False]])

    def test_blind_zone_does_not_reveal_surface_behind(self):
        lo, hi = S.stair_boxes(torch.tensor([0.10]))
        distance, valid = S.raycast_stairs(torch.tensor([[0.39, 0., 0.05]]),
                                          torch.tensor([[[1., 0., 0.]]]), lo, hi)
        self.assertFalse(bool(valid[0, 0]))
        self.assertAlmostEqual(float(distance[0, 0]), 0.01, places=5)

    def test_top_surface_parallel_axes_and_rotated_ray(self):
        lo, hi = S.stair_boxes(torch.tensor([0.10]))
        d, v = S.raycast_stairs(torch.tensor([[0.5, 0., 0.3]]),
                                torch.tensor([[[0., 0., -1.]]]), lo, hi)
        self.assertTrue(bool(v.all()))
        self.assertAlmostEqual(float(d[0, 0]), 0.2, places=5)

    def test_unknown_cells_have_explicit_mask_and_no_hidden_height(self):
        points = torch.tensor([[[0.05, 0.05, -0.1], [0.05, 0.05, 0.1], [0.15, 0.05, 0.3]]])
        features = S.observed_height_grid(points, torch.tensor([[True, True, False]]))
        self.assertEqual(features.shape, (1, 192))
        self.assertEqual(float(features[:, 96:].sum()), 1)
        self.assertAlmostEqual(float(features[:, :96].max()), 1 / 3, places=5)
        unknown = S.observed_height_grid(points, torch.zeros((1, 3), dtype=torch.bool))
        self.assertEqual(float(unknown.abs().sum()), 0.)

    def test_fov_and_finite_outputs(self):
        rays = S.lidar_directions('cpu')
        self.assertEqual(rays.shape, (612, 3))
        torch.testing.assert_close(rays.norm(dim=-1), torch.ones(612))
        lo, hi = S.stair_boxes(torch.tensor([0.02, 0.10]))
        distances, valid = S.raycast_stairs(torch.tensor([[0., 0., 0.15], [0., 0., 0.15]]),
                                           rays.expand(2, -1, -1), lo, hi)
        self.assertTrue(bool(torch.isfinite(distances).all()))
        self.assertGreater(int(valid.sum()), 0)
        self.assertEqual(S.OBSERVATIONS, 240)

    def test_only_three_consecutive_successes_advance(self):
        levels = torch.tensor([0, 2, 5, 2])
        streaks = torch.tensor([2, 2, 2, 2])
        levels, streaks = S.next_levels(levels, torch.tensor([True, False, True, False]),
                                        torch.tensor([False, False, False, True]), streaks, 5)
        self.assertEqual(levels.tolist(), [1, 2, 5, 1])
        self.assertEqual(streaks.tolist(), [0, 0, 3, 0])


if __name__ == '__main__':
    unittest.main()
