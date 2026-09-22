"""Unit tests for voltage-specific spiderbot servo data."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/servo_specs.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_servo_specs", MODULE)
assert SPEC is not None and SPEC.loader is not None
SERVO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVO)


class ServoSpecificationTests(unittest.TestCase):
    def test_rc920dmg_supplied_physical_data(self) -> None:
        self.assertEqual(SERVO.RC920DMG_OPERATING_VOLTAGE_RANGE_V, (4.8, 8.4))
        self.assertEqual(SERVO.RC920DMG_REFERENCE_VOLTAGE_V, 5.0)
        self.assertEqual(SERVO.RC920DMG_MASS_KG, 0.060)
        self.assertEqual(SERVO.RC920DMG_SIZE_M, (0.040, 0.0205, 0.0405))
        self.assertEqual(SERVO.RC920DMG_BEARING_COUNT, 2)
        self.assertEqual(SERVO.RC920DMG_CABLE_LENGTH_M, 0.32)
        self.assertEqual(SERVO.RC920DMG_SPEC_VERIFICATION_STATUS, "provisional_unverified")

    def test_rc920dmg_5v_actuator_conversions(self) -> None:
        self.assertAlmostEqual(
            SERVO.RC920DMG_STALL_TORQUE_NM_5V,
            19.0 * 0.0980665,
            places=8,
        )
        self.assertAlmostEqual(
            SERVO.RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
            math.radians(60.0) / 0.16,
            places=8,
        )

    def test_rc920dmg_7v4_source_values_are_retained(self) -> None:
        self.assertEqual(SERVO.RC920DMG_STALL_TORQUE_KGF_CM_7V4, 21.0)
        self.assertEqual(SERVO.RC920DMG_SECONDS_PER_60_DEGREES_7V4, 0.14)
        self.assertAlmostEqual(
            SERVO.RC920DMG_STALL_TORQUE_NM_7V4,
            21.0 * 0.0980665,
            places=8,
        )
        self.assertAlmostEqual(
            SERVO.RC920DMG_NO_LOAD_SPEED_RAD_S_7V4,
            math.radians(60.0) / 0.14,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
