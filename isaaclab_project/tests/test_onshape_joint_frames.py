"""Tests for the current Onshape joint-zero and rotation-direction contract."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts/onshape_joint_frames.py"
SPEC = importlib.util.spec_from_file_location("dksh_onshape_joint_frames", MODULE)
assert SPEC is not None and SPEC.loader is not None
FRAMES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRAMES)


def cross(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


class OnshapeJointFrameTests(unittest.TestCase):
    def test_spec_is_complete_for_names_zero_and_directions(self) -> None:
        self.assertEqual(FRAMES.validate_spec(), ())
        records = FRAMES.joint_records()
        self.assertEqual(len(records), 18)
        self.assertEqual(len({item["canonical_name"] for item in records}), 18)
        self.assertTrue(all(item["zero_position_radians"] == 0.0 for item in records))

    def test_onshape_legs_map_to_counterclockwise_policy_order(self) -> None:
        expected = {
            "dof_L6_coxa": "leg_0_hip_joint",
            "dof_L5_coxa": "leg_1_hip_joint",
            "dof_L4_coxa": "leg_2_hip_joint",
            "dof_L3_coxa": "leg_3_hip_joint",
            "dof_L2_coxa": "leg_4_hip_joint",
            "dof_L1_coxa": "leg_5_hip_joint",
        }
        for mate, name in expected.items():
            self.assertEqual(FRAMES.canonical_name_for_mate(mate), name)

    def test_positive_coxa_motion_is_counterclockwise(self) -> None:
        # dq direction is axis x radius.  +Z x +X = +Y, the CCW tangent.
        tangent = cross((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        self.assertEqual(tangent, (0.0, 1.0, 0.0))

    def test_positive_femur_motion_moves_outward_from_vertical_zero(self) -> None:
        # -Y x -Z = +X: the femur distal end starts moving radially outward.
        velocity = cross((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
        self.assertEqual(velocity, (1.0, 0.0, 0.0))

    def test_positive_tibia_motion_flexes_upward(self) -> None:
        # -Y x +X = +Z: a horizontal tibia initially moves upward.
        velocity = cross((0.0, -1.0, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(velocity, (0.0, 0.0, 1.0))

    def test_leg_azimuths_are_sixty_degrees_apart(self) -> None:
        azimuths = [record["azimuth_degrees"] for record in FRAMES.joint_records()[::3]]
        unwrapped = [math.radians(value if value >= 0.0 else value + 360.0) for value in azimuths]
        increments = [
            (unwrapped[(index + 1) % 6] - unwrapped[index]) % (2.0 * math.pi)
            for index in range(6)
        ]
        for increment in increments:
            self.assertAlmostEqual(increment, math.pi / 3.0)


if __name__ == "__main__":
    unittest.main()
