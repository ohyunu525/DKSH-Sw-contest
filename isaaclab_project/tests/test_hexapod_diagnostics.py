"""Physical-time slip detection and preservation of CAD foot geometry."""
import importlib.util
from pathlib import Path
import unittest

import torch

PATH = Path(__file__).resolve().parents[1] / 'source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/hexapod_diagnostics.py'
SPEC = importlib.util.spec_from_file_location('hexapod_diagnostics', PATH)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class HexapodDiagnosticsTests(unittest.TestCase):
    def test_slip_requires_60ms_at_different_sample_intervals(self):
        for dt, steps in ((0.02, 3), (0.01, 6), (0.005, 12)):
            timer = DIAGNOSTICS.SlipTimer((1, 6), 'cpu')
            for index in range(steps):
                slip = timer.update(torch.ones(1, 6, dtype=torch.bool), torch.full((1, 6), 0.02), dt)
                self.assertEqual(bool(slip.all()), index == steps - 1)

    def test_air_or_low_velocity_interrupts_continuous_slip(self):
        timer = DIAGNOSTICS.SlipTimer((1, 2), 'cpu')
        contact = torch.ones(1, 2, dtype=torch.bool)
        speed = torch.full((1, 2), 0.02)
        timer.update(contact, speed, 0.04)
        timer.update(torch.tensor([[False, True]]), torch.tensor([[0.02, 0.0]]), 0.02)
        self.assertFalse(timer.update(contact, speed, 0.04).any())

    def test_foot_frames_are_six_distinct_nonzero_tibia_offsets(self):
        offsets = DIAGNOSTICS.foot_offsets()
        self.assertEqual(len(offsets), 6)
        self.assertTrue(all(torch.tensor(offset).norm() > 0.01 for offset in offsets))

    def test_binary_cad_mesh_bounds_are_in_metres(self):
        low, high = DIAGNOSTICS._stl_bounds(DIAGNOSTICS.URDF_PATH.parent / 'assets/part_1.stl')
        self.assertTrue((high > low).all())
        self.assertLess(float((high - low).max()), 0.2)

    def test_point_velocity_uses_com_lever_arm(self):
        actual = DIAGNOSTICS.velocity_at_point(
            torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 2.0]]),
            torch.tensor([[0.5, 0.0, 0.0]]), torch.tensor([[0.2, 0.0, 0.0]]),
        )
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.6, 0.0]]))


if __name__ == '__main__':
    unittest.main()
