"""Check IK reachability and continuity over full wave cycles."""
import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/wave_gait.py"
spec = importlib.util.spec_from_file_location("wave_gait", path)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)


class WaveGaitTests(unittest.TestCase):
    def test_full_cycle_ik_is_reachable_and_inside_joint_limits(self):
        phase = torch.linspace(0, 1, 1001, dtype=torch.float64)
        for speed in (0.0, 0.004, 0.01):
            feet = gait.foot_targets(phase, torch.full_like(phase, speed))
            angles = gait.inverse_kinematics(feet)
            self.assertTrue(torch.isfinite(angles).all())
            self.assertLess(float(angles.abs().max()), 1.57079632679 * 0.95)
            torch.testing.assert_close(gait.forward_kinematics(angles), feet, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(angles[0], angles[-1])
            self.assertLess(float((angles[1:] - angles[:-1]).abs().max()), 0.015)

    def test_at_most_one_foot_in_swing(self):
        phase = torch.linspace(0, 1, 1000)
        feet = gait.foot_targets(phase, torch.full_like(phase, 0.008))
        self.assertTrue(((feet[:, :, 2] > -gait.STANCE_DEPTH + 1e-6).sum(-1) <= 1).all())

    def test_standing_matches_initial_joint_angles(self):
        phase = torch.zeros(4, dtype=torch.float64)
        feet = gait.foot_targets(phase, phase, lift=0)
        angles = gait.inverse_kinematics(feet)
        torch.testing.assert_close(angles[:, :, 1], torch.full_like(angles[:, :, 1], gait.STANCE_FEMUR))
        torch.testing.assert_close(angles[:, :, 2], torch.full_like(angles[:, :, 2], gait.STANCE_TIBIA))
