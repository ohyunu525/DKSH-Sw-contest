"""Unit tests for the LiDAR payload mass model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/payload_mass.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_payload_mass", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PAYLOAD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAYLOAD
SPEC.loader.exec_module(PAYLOAD)


class PayloadMassTests(unittest.TestCase):
    def test_l1_rm_mass_center_and_inertia_are_combined(self):
        base = PAYLOAD.MassProperties(0.5, (0.0, 0.0, 0.0), (0.002, 0.003, 0.004))
        result = PAYLOAD.add_centered_box_payload(
            base,
            payload_mass=0.230,
            payload_size=(0.075, 0.075, 0.065),
            payload_center=(0.0, 0.0, 0.0625),
        )
        self.assertAlmostEqual(result.mass, 0.730)
        self.assertAlmostEqual(result.center_of_mass[2], 0.230 * 0.0625 / 0.730)
        self.assertGreater(result.diagonal_inertia[0], base.diagonal_inertia[0])
        self.assertGreater(result.diagonal_inertia[1], base.diagonal_inertia[1])
        self.assertGreater(result.diagonal_inertia[2], base.diagonal_inertia[2])
        self.assertAlmostEqual(result.diagonal_inertia[0] - base.diagonal_inertia[0],
                               result.diagonal_inertia[1] - base.diagonal_inertia[1])

    def test_invalid_or_off_center_payload_is_rejected(self):
        base = PAYLOAD.MassProperties(0.5, (0.0, 0.0, 0.0), (0.002, 0.003, 0.004))
        cases = (
            dict(payload_mass=0.0, payload_size=(0.075, 0.075, 0.065), payload_center=(0.0, 0.0, 0.06)),
            dict(payload_mass=0.23, payload_size=(0.0, 0.075, 0.065), payload_center=(0.0, 0.0, 0.06)),
            dict(payload_mass=0.23, payload_size=(0.075, 0.075, 0.065), payload_center=(0.01, 0.0, 0.06)),
            dict(payload_mass=math.nan, payload_size=(0.075, 0.075, 0.065), payload_center=(0.0, 0.0, 0.06)),
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PAYLOAD.add_centered_box_payload(base, **values)


if __name__ == "__main__":
    unittest.main()
