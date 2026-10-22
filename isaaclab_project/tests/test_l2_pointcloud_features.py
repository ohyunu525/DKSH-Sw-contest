"""L2 NEGA point-cloud boundaries and parity with the active training config."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from l2_pointcloud_features import (
    L2_MAX_RANGE_M, L2_MIN_RANGE_M, L2_RESOLUTION_M, L2_SECTORS,
    L2_VERTICAL_FOV_DEG, L2_VERTICAL_MIN_DEG, encode_l2_obstacles,
)
from l1_pointcloud_features import encode_l1_obstacles


def point(elevation, distance=1.0, azimuth=0.0):
    elevation, azimuth = math.radians(elevation), math.radians(azimuth)
    return (distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth), distance * math.sin(elevation))


class L2PointcloudFeaturesTests(unittest.TestCase):
    def test_active_isaac_defaults_match_shared_encoder(self):
        cfg = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/spider_navigation_env_cfg.py"
        tree = ast.parse(cfg.read_text(encoding="utf-8"))
        expected = {
            "lidar_min_range_m": L2_MIN_RANGE_M,
            "lidar_max_range_m": L2_MAX_RANGE_M,
            "lidar_measurement_resolution_m": L2_RESOLUTION_M,
            "lidar_vertical_fov_deg": L2_VERTICAL_FOV_DEG,
            "lidar_vertical_min_angle_deg": L2_VERTICAL_MIN_DEG,
            "lidar_observation_bins": L2_SECTORS,
            "lidar_horizontal_scan_frequency_hz": 5.55,
            "lidar_imu_reporting_frequency_hz": 500.0,
        }
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpiderNavigationEnvCfg")
        actual = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in expected}
        self.assertEqual(actual, expected)

    def test_negative_slice_is_accepted_only_by_l2(self):
        points = [point(-5.9)]
        self.assertEqual(encode_l2_obstacles(points)[16], 1.0)
        self.assertEqual(encode_l1_obstacles(points)[16], 0.0)
        for elevation in (-6.1, -45.0):
            self.assertEqual(encode_l2_obstacles([point(elevation)])[16], 0.0)
        for elevation in (-6.0, 0.0, 45.0, 90.0):
            self.assertEqual(encode_l2_obstacles([point(elevation)])[16], 1.0)

    def test_nearest_return_is_quantized_at_4_5_mm(self):
        features = encode_l2_obstacles([(0.503, 0, 0), (1, 0, 0), (2, 0, 0)])
        self.assertAlmostEqual(features[0], 0.504 / 30.0)
        self.assertEqual(features[16:], [1.0] + [0.0] * 15)
        self.assertAlmostEqual(encode_l2_obstacles([(1, 0, 0)])[0], 0.999 / 30.0)

    def test_range_misses_and_cardinal_order(self):
        self.assertEqual(encode_l2_obstacles([(0.01, 0, 0), (30, 0, 0), (math.nan, 0, 0)]),
                         [1.0] * 16 + [0.0] * 16)
        features = encode_l2_obstacles([point(-3, azimuth=a) for a in (0, 90, 180, 270)])
        self.assertEqual([i for i, valid in enumerate(features[16:]) if valid], [0, 4, 8, 12])
        self.assertEqual(encode_l2_obstacles([(29.9999, 0, 0)])[16], 0.0)

    def test_invalid_vertical_limits_are_rejected(self):
        for kwargs in ({"vertical_min_deg": math.nan}, {"vertical_min_deg": -91},
                       {"vertical_min_deg": 0, "vertical_fov_deg": 96}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                encode_l2_obstacles([], **kwargs)


if __name__ == "__main__":
    unittest.main()
