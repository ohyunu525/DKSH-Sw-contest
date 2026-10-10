"""Pure Python checks for the real L1 point-cloud actor contract."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from l1_pointcloud_features import (  # noqa: E402
    L1_MAX_RANGE_M, L1_MIN_RANGE_M, L1_RESOLUTION_M, L1_SECTORS,
    L1_VERTICAL_FOV_DEG, encode_l1_obstacles,
)


class L1PointcloudFeaturesTests(unittest.TestCase):
    def test_empty_frame_and_range_limits(self):
        features = encode_l1_obstacles([(0.01, 0, 0), (30.0, 0, 0), (math.nan, 0, 0)])
        self.assertEqual(features, [1.0] * 16 + [0.0] * 16)

    def test_legacy_l1_defaults_remain_available(self):
        self.assertEqual((L1_MIN_RANGE_M, L1_MAX_RANGE_M, L1_RESOLUTION_M, L1_VERTICAL_FOV_DEG, L1_SECTORS),
                         (0.05, 30.0, 0.008, 90.0, 16))

    def test_nearest_return_quantization_and_validity(self):
        features = encode_l1_obstacles([(1.013, 0, 0), (0.503, 0, 0), (2, 0, 0)])
        self.assertAlmostEqual(features[0], 0.504 / 30.0)
        self.assertEqual(features[16], 1.0)
        self.assertEqual(sum(features[16:]), 1.0)

    def test_quantized_max_range_is_a_miss(self):
        features = encode_l1_obstacles([(29.999, 0, 0)])
        self.assertEqual((features[0], features[16]), (1.0, 0.0))

    def test_cardinal_bins_wrap_and_vertical_fov(self):
        points = [(1, 0, 0), (0, 2, 0), (-3, 0, 0), (0, -4, 0), (0.5, 0, -0.1)]
        features = encode_l1_obstacles(points)
        self.assertEqual([i for i, value in enumerate(features[16:]) if value], [0, 4, 8, 12])
        self.assertAlmostEqual(features[0], 1.0 / 30.0)

    def test_points_near_sector_edges_use_nearest_center(self):
        edge = math.radians(11.25)
        points = [(math.cos(edge - 0.001), math.sin(edge - 0.001), 0),
                  (2 * math.cos(edge + 0.001), 2 * math.sin(edge + 0.001), 0)]
        features = encode_l1_obstacles(points)
        self.assertEqual(features[16:18], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
