"""Exercise scenario geometry sensing without importing Isaac Lab or Isaac Sim."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/scenario_sensing.py"
    )
    SPEC = importlib.util.spec_from_file_location("dksh_scenario_sensing", MODULE_PATH)
    assert SPEC is not None and SPEC.loader is not None
    SENSING = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SENSING)


@unittest.skipIf(torch is None, "PyTorch is required; use the Isaac Lab Python environment")
class ScenarioSensingTests(unittest.TestCase):
    def tensor(self, values):
        return torch.tensor(values, dtype=torch.float64)

    def test_cardinal_rays_find_nearest_box_in_each_direction(self):
        centers = self.tensor([[[1.0, 0.0, 0.5], [0.0, 1.5, 0.5], [-1.0, 0.0, 0.5], [0.0, -1.0, 0.5]]])
        sizes = self.tensor([[0.25, 0.25, 0.5]] * 4)
        result = SENSING.planar_box_ranges(self.tensor([[0.0, 0.0, 0.5]]), self.tensor([0.0]), centers, sizes)
        torch.testing.assert_close(result[:, [0, 4, 8, 12]], self.tensor([[0.375, 0.625, 0.375, 0.375]]))
        self.assertEqual(result.shape, (1, 16))
        self.assertEqual(result[0, 2].item(), 1.0)

    def test_parallel_ray_on_slab_edge_hits_without_nan(self):
        args = (
            self.tensor([[0.0, 0.5, 0.5]]),
            self.tensor([0.0]),
            self.tensor([[[1.0, 0.0, 0.5]]]),
            self.tensor([[0.25, 0.5, 0.5]]),
        )
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                result = SENSING.planar_box_ranges(*(value.to(dtype) for value in args), ray_count=4)
                expected = torch.tensor([[0.375, 1.0, 1.0, 1.0]], dtype=dtype)
                torch.testing.assert_close(result, expected)

    def test_origin_inside_or_touching_box_reports_zero_in_all_directions(self):
        result = SENSING.planar_box_ranges(
            self.tensor([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]]),
            self.tensor([0.0, 0.2]),
            self.tensor([[[0.0, 0.0, 0.5]], [[0.0, 0.0, 0.5]]]),
            self.tensor([[0.5, 0.5, 0.5]]),
        )
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_translation_and_yaw_preserve_relative_ranges(self):
        # Identical relative geometry at two different environment origins.
        result = SENSING.planar_box_ranges(
            self.tensor([[0.0, 0.0, 0.5], [8.0, -3.0, 4.5]]),
            self.tensor([0.0, math.pi / 2]),
            self.tensor([[[1.0, 0.0, 0.5]], [[9.0, -3.0, 4.5]]]),
            self.tensor([[0.25, 0.25, 0.5]]),
        )
        torch.testing.assert_close(result[1], torch.roll(result[0], shifts=-4))

    def test_vertical_filter_and_max_range(self):
        result = SENSING.planar_box_ranges(
            self.tensor([[0.0, 0.0, 0.5]]),
            self.tensor([0.0]),
            self.tensor([[[0.5, 0.0, 2.0], [0.0, 0.5, -1.0], [-3.0, 0.0, 0.5]]]),
            self.tensor([[0.2, 0.2, 0.2]] * 3),
        )
        torch.testing.assert_close(result, torch.ones_like(result))

    def test_diagonal_slab_intersection_and_closest_of_overlapping_boxes(self):
        result = SENSING.planar_box_ranges(
            self.tensor([[0.0, 0.0, 0.0]]),
            self.tensor([math.pi / 4]),
            self.tensor([[[1.5, 1.5, 0.0], [1.0, 1.0, 0.0]]]),
            self.tensor([[0.5, 0.5, 0.5], [0.25, 0.25, 0.5]]),
        )
        self.assertAlmostEqual(result[0, 0].item(), 0.75 * math.sqrt(2.0) / 2.0)

    def test_batched_half_sizes_are_independent(self):
        result = SENSING.planar_box_ranges(
            self.tensor([[0.0, 0.0, 0.0]] * 2),
            self.tensor([0.0, 0.0]),
            self.tensor([[[1.0, 0.0, 0.0]]] * 2),
            self.tensor([[[0.25, 0.25, 0.5]], [[0.5, 0.5, 0.5]]]),
        )
        torch.testing.assert_close(result[:, 0], self.tensor([0.375, 0.25]))

    def test_empty_obstacles_return_full_range_and_base_height(self):
        centers = torch.empty((2, 0, 3), dtype=torch.float64)
        sizes = torch.empty((0, 3), dtype=torch.float64)
        ranges = SENSING.planar_box_ranges(torch.zeros((2, 3), dtype=torch.float64), self.tensor([0.0, 1.0]), centers, sizes)
        torch.testing.assert_close(ranges, torch.ones((2, 16), dtype=torch.float64))
        points = torch.zeros((2, 9, 2), dtype=torch.float64)
        torch.testing.assert_close(SENSING.sample_box_heights(points, centers, sizes), torch.zeros((2, 9), dtype=torch.float64))
        heights = SENSING.sample_box_heights(points, centers, sizes, self.tensor([1.0, 2.0]))
        torch.testing.assert_close(heights, self.tensor([[1.0] * 9, [2.0] * 9]))

    def test_height_samples_use_highest_overlapping_surface_and_base(self):
        heights = SENSING.sample_box_heights(
            self.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [3.0, 3.0]]]),
            self.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.6], [1.0, 0.0, -0.5]]]),
            self.tensor([[0.5, 0.5, 0.2], [0.25, 0.25, 0.1], [0.25, 0.25, 0.1]]),
            self.tensor([0.1]),
        )
        torch.testing.assert_close(heights, self.tensor([[0.7, 0.4, 0.1, 0.1]]))

    def test_height_sampling_uses_each_environments_geometry(self):
        heights = SENSING.sample_box_heights(
            self.tensor([[[0.4, 0.0], [5.0, 0.0]], [[5.4, 0.0], [0.0, 0.0]]]),
            self.tensor([[[0.0, 0.0, 0.2]], [[5.0, 0.0, 0.8]]]),
            self.tensor([[[0.5, 0.5, 0.2]], [[0.25, 0.25, 0.2]]]),
            self.tensor([0.0, 0.4]),
        )
        torch.testing.assert_close(heights, self.tensor([[0.4, 0.0], [0.4, 0.4]]))

    def test_invalid_configuration_is_rejected(self):
        args = (self.tensor([[0.0, 0.0, 0.0]]), self.tensor([0.0]), self.tensor([[[0.0, 0.0, 0.0]]]), self.tensor([[0.5, 0.5, 0.5]]))
        for distance in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(max_range=distance), self.assertRaises(ValueError):
                SENSING.planar_box_ranges(*args, max_range=distance)
        with self.assertRaises(ValueError):
            SENSING.planar_box_ranges(*args, ray_count=0)

    def test_lidar_measurement_model_preserves_misses_and_models_rm_limits(self):
        ranges = self.tensor([[0.0, 0.01, 0.5, 1.0]])
        noise = self.tensor([[-1.0, 1.0, 0.5, -1.0]])
        result = SENSING.apply_lidar_measurement_model(
            ranges,
            max_range=30.0,
            min_range=0.05,
            accuracy=0.02,
            resolution=0.008,
            noise=noise,
        )
        expected_meters = self.tensor([[0.048, 0.320, 15.008, 30.0]])
        expected_meters[0, 0] = 0.05  # Blind-zone clamp follows quantization.
        torch.testing.assert_close(result[:, :3] * 30.0, expected_meters[:, :3])
        self.assertEqual(result[0, 3].item(), 1.0)

    def test_lidar_measurement_model_rejects_invalid_parameters(self):
        ranges = self.tensor([[0.5]])
        valid = dict(max_range=30.0, min_range=0.05, accuracy=0.02, resolution=0.008)
        for name, value in (
            ("max_range", 0.0), ("min_range", -0.1), ("min_range", 30.0),
            ("accuracy", -0.1), ("resolution", 0.0),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                SENSING.apply_lidar_measurement_model(ranges, **(valid | {name: value}))

    def test_spatial_projection_rotates_with_body_and_detects_ceiling(self):
        positions = self.tensor([[0.0, 0.0, 0.0]])
        identity = self.tensor([[1.0, 0.0, 0.0, 0.0]])
        centers = self.tensor([[[1.0, 0.0, 0.6]]])
        sizes = self.tensor([[0.2, 0.2, 0.2]])
        horizontal_only = SENSING.spatial_obstacle_ranges(
            positions, identity, centers, sizes,
            max_range=2.0, horizontal_count=4, vertical_fov_degrees=90.0, vertical_count=1,
        )
        projected = SENSING.spatial_obstacle_ranges(
            positions, identity, centers, sizes,
            max_range=2.0, horizontal_count=4, vertical_fov_degrees=90.0, vertical_count=3,
        )
        self.assertEqual(horizontal_only[0, 0].item(), 1.0)
        self.assertLess(projected[0, 0].item(), 1.0)

    def test_spatial_projection_includes_dynamic_spheres_and_static_occlusion(self):
        positions = self.tensor([[0.0, 0.0, 0.0]])
        identity = self.tensor([[1.0, 0.0, 0.0, 0.0]])
        spheres = self.tensor([[[0.5, 0.0, 0.0]]])
        clear = SENSING.spatial_obstacle_ranges(
            positions, identity, self.tensor([]).reshape(1, 0, 3), self.tensor([]).reshape(0, 3),
            max_range=2.0, horizontal_count=4, vertical_fov_degrees=0.0, vertical_count=1,
            sphere_centers_w=spheres, sphere_radii=0.1,
        )
        self.assertAlmostEqual(clear[0, 0].item(), 0.2, places=6)
        self.assertEqual(clear[0, 1].item(), 1.0)

        occluded = SENSING.spatial_obstacle_ranges(
            positions, identity, self.tensor([[[0.25, 0.0, 0.0]]]), self.tensor([[0.05, 0.2, 0.2]]),
            max_range=2.0, horizontal_count=4, vertical_fov_degrees=0.0, vertical_count=1,
            sphere_centers_w=spheres, sphere_radii=0.1,
        )
        self.assertAlmostEqual(occluded[0, 0].item(), 0.1, places=6)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu_and_preserves_device(self):
        args = (
            self.tensor([[0.0, 0.0, 0.5]]),
            self.tensor([0.2]),
            self.tensor([[[1.0, 0.0, 0.5]]]),
            self.tensor([[0.25, 0.25, 0.5]]),
        )
        expected = SENSING.planar_box_ranges(*args)
        actual = SENSING.planar_box_ranges(*(value.cuda() for value in args))
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
