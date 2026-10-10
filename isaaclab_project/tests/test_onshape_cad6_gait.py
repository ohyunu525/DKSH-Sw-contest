"""Fast checks for conservative current-Onshape joint-space references."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import torch


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dksh_isaaclab"
    / "dksh_isaaclab"
    / "tasks"
    / "direct"
    / "spider_navigation"
    / "onshape_cad6_gait.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_onshape_cad6_gait", MODULE)
assert SPEC is not None and SPEC.loader is not None
GAIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GAIT)


class OnshapeCad6GaitTests(unittest.TestCase):
    def test_stationary_reference_is_the_saved_zero_pose(self) -> None:
        phase = torch.linspace(0, 1, 41)
        q = GAIT.wave_joint_positions(phase, torch.zeros_like(phase), 2.4, 0.0, 0.012)
        torch.testing.assert_close(q, torch.zeros_like(q))

    def test_references_remain_well_within_joint_limits(self) -> None:
        phase = torch.linspace(0, 1, 2401)
        commands = torch.tensor([[0.012, 0.006, 0.08]]).expand(len(phase), -1)
        q = GAIT.joint_positions_for_twist(phase, commands, 2.4, 0.005, 0.012)
        self.assertEqual(q.shape, (len(phase), 6, 3))
        self.assertTrue(torch.isfinite(q).all())
        self.assertLess(float(q.abs().max()), math.pi / 2 * 0.30)

    def test_swing_lifts_only_femur_and_tibia_from_zero(self) -> None:
        phase = torch.tensor([0.90])
        q = GAIT.wave_joint_positions(phase, torch.tensor([0.0]), 2.4, 0.004, 0.012)
        self.assertTrue((q[:, :, 1:] >= 0).all())
        self.assertGreater(float(q[:, :, 1:].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
