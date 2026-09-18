"""Verify CAD6 kinematics against the URDF, independently of the IK equations."""
import importlib.util
from pathlib import Path
import math
import unittest
import xml.etree.ElementTree as ET
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cad6_gait', ROOT / 'source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/cad6_wave_gait.py')
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)


class Cad6GaitTests(unittest.TestCase):
    def test_cycle_reachable_and_limited_with_residual_margin(self):
        phase = torch.linspace(0, 1, 2401, dtype=torch.float64)
        for speed, period, stride_limit in ((0., 2.4, 0.012), (0.006, 2.4, 0.012), (0.050, 0.6, 0.025)):
            feet = gait.foot_targets(phase, torch.full_like(phase, speed), period, 0.005, stride_limit)
            q = gait.inverse_kinematics(feet)
            torch.testing.assert_close(gait.forward_kinematics(q), feet, atol=1e-8, rtol=1e-7)
            self.assertLess(float(q.abs().max()) + 0.035, math.pi / 2 * 0.95)
            self.assertLess(float((q[1:] - q[:-1]).abs().max()), 0.02)
            self.assertTrue(((feet[:, :, 2] > gait.STANCE_Z + 1e-7).sum(-1) <= 1).all())

    def test_stance_and_zero_pose_follow_urdf_transforms(self):
        robot = ET.parse(ROOT / 'assets/spiderbot_variants/spiderbot_6leg/spiderbot_6leg.urdf').getroot()
        self.assertEqual(len(robot.findall('joint')), 18)
        def translation(name):
            j = robot.find(f"joint[@name='{name}']")
            return torch.tensor([float(v) for v in j.find('origin').get('xyz').split()], dtype=torch.float64)
        def rotate_y(point, q):
            c, s = math.cos(q), math.sin(q)
            return torch.tensor([c * point[0] + s * point[2], point[1], -s * point[0] + c * point[2]])
        sole = torch.tensor([gait.SOLE_X, 0, gait.SOLE_Z], dtype=torch.float64)
        for femur, tibia in ((0., 0.), (gait.STANCE_FEMUR, gait.STANCE_TIBIA), (.4, .15)):
            expected = translation('leg_0_femur_joint') + rotate_y(translation('leg_0_tibia_joint') + rotate_y(sole, tibia), femur)
            q = torch.tensor([[[0., femur, tibia]] * 6], dtype=torch.float64)
            torch.testing.assert_close(gait.forward_kinematics(q)[0, 0], expected)
        phase = torch.zeros(1, dtype=torch.float64)
        stance = gait.inverse_kinematics(gait.foot_targets(phase, phase, lift=0))
        torch.testing.assert_close(stance[:, :, 1], torch.full_like(stance[:, :, 1], gait.STANCE_FEMUR))
        torch.testing.assert_close(stance[:, :, 2], torch.full_like(stance[:, :, 2], gait.STANCE_TIBIA))

    def test_tripod_and_wave_blend_remain_reachable_and_continuous(self):
        phase = torch.linspace(0, 1, 2401, dtype=torch.float64)
        speed = torch.full_like(phase, 0.050)
        weights = torch.linspace(0, 1, len(phase), dtype=torch.float64)
        feet = gait.blended_foot_targets(phase, speed, 0.6, 0.005, 0.025, weights)
        q = gait.inverse_kinematics(feet)
        torch.testing.assert_close(gait.forward_kinematics(q), feet, atol=1e-8, rtol=1e-7)
        self.assertLess(float(q.abs().max()) + 0.035, math.pi / 2 * 0.95)
