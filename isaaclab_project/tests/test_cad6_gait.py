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
    def test_velocity_v1_commands_fit_without_hidden_stride_clipping(self):
        commands = torch.cartesian_prod(
            torch.tensor([-0.025, 0., 0.025, 0.050], dtype=torch.float64),
            torch.tensor([-0.025, 0., 0.025], dtype=torch.float64),
            torch.tensor([-0.20, 0., 0.20], dtype=torch.float64),
        )
        projected = gait.feasible_velocity_command(commands, 0.6, 0.025)
        torch.testing.assert_close(gait.feasible_velocity_command(projected, 0.6, 0.025), projected)
        straight = torch.tensor([[0.050, 0., 0.], [0., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(gait.feasible_velocity_command(straight, 0.6, 0.025), straight)
        phase = torch.linspace(0, 1, 301, dtype=torch.float64)
        for command in projected:
            values = command.expand(len(phase), -1)
            # A larger cap must produce identical targets: no per-foot clipping.
            torch.testing.assert_close(
                gait.directional_foot_targets(phase, values, 0.6, 0.005, 0.025),
                gait.directional_foot_targets(phase, values, 0.6, 0.005, 1.0),
            )
            speed = command[:2].norm() + command[2].abs() * gait.STANCE_RADIUS
            blend = ((speed - 0.025) / 0.01).clamp(0, 1)
            weight = (blend.square() * (3 - 2 * blend)).expand_as(phase)
            feet = gait.blended_directional_foot_targets(phase, values, 0.6, 0.005, 0.025, weight)
            q = gait.inverse_kinematics(feet)
            torch.testing.assert_close(gait.forward_kinematics(q), feet, atol=1e-8, rtol=1e-7)
            self.assertLess(float(q.abs().max()) + 0.035, math.pi / 2 * 0.95)

    def test_velocity_v3_projection_matches_corrected_foot_targets(self):
        commands = torch.cartesian_prod(
            torch.tensor([-0.025, 0.0, 0.035, 0.050], dtype=torch.float64),
            torch.tensor([-0.025, 0.0, 0.025], dtype=torch.float64),
            torch.tensor([-0.08, 0.0, 0.08], dtype=torch.float64),
        )
        projected = gait.feasible_velocity_command(commands, 0.6, 0.025, correct_yaw=True)
        phase = torch.linspace(0, 1, 301, dtype=torch.float64)
        for command in projected:
            values = command.expand(len(phase), -1)
            torch.testing.assert_close(
                gait.directional_foot_targets(phase, values, 0.6, 0.005, 0.025, correct_yaw=True),
                gait.directional_foot_targets(phase, values, 0.6, 0.005, 1.0, correct_yaw=True),
            )

    def test_body_yaw_uses_urdf_hip_mount_and_positive_axis(self):
        robot = ET.parse(ROOT / 'assets/spiderbot_variants/spiderbot_6leg/spiderbot_6leg.urdf').getroot()
        hip_joint = robot.find("joint[@name='leg_0_hip_joint']")
        self.assertEqual(hip_joint.find('axis').get('xyz'), '0 0 1')
        self.assertAlmostEqual(float(hip_joint.find('origin').get('xyz').split()[0]), gait.HIP_MOUNT_RADIUS)
        self.assertAlmostEqual(gait.BODY_STANCE_RADIUS, gait.HIP_MOUNT_RADIUS + gait.STANCE_RADIUS)
        phase = torch.tensor([0.1, 0.10001], dtype=torch.float64)
        command = torch.tensor([[0.0, 0.0, 0.04]] * 2, dtype=torch.float64)
        feet = gait.directional_foot_targets(
            phase, command, 0.6, 0.005, 0.025, offsets=(0,) * 6, correct_yaw=True,
        )
        dt = float((phase[1] - phase[0]) * 0.6)
        # A stationary world foot moves backward in hip-local Y when the
        # base has positive yaw. The full base-to-foot radius determines speed.
        self.assertAlmostEqual(
            float((feet[1, 0, 1] - feet[0, 0, 1]) / dt),
            -0.04 * gait.BODY_STANCE_RADIUS, places=5,
        )

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

    def test_directional_commands_remain_reachable_and_stride_limited(self):
        phase = torch.linspace(0, 1, 2401, dtype=torch.float64)
        commands = (
            (0.012, 0.0, 0.0), (-0.006, 0.006, 0.0),
            (0.0, 0.0, 0.08), (0.012, -0.006, -0.08),
        )
        angles = torch.arange(6, dtype=torch.float64) * (math.pi / 3)
        rest = torch.stack((gait.STANCE_RADIUS * torch.cos(angles), gait.STANCE_RADIUS * torch.sin(angles)), -1)
        for command in commands:
            with self.subTest(command=command):
                values = torch.tensor(command, dtype=torch.float64).expand(len(phase), -1)
                feet = gait.directional_foot_targets(phase, values)
                q = gait.inverse_kinematics(feet)
                torch.testing.assert_close(gait.forward_kinematics(q), feet, atol=1e-8, rtol=1e-7)
                displacement = (feet[:, :, :2] - rest).norm(dim=-1)
                self.assertLessEqual(float(displacement.max()), 0.0060001)
                self.assertLess(float(q.abs().max()) + 0.035, math.pi / 2 * 0.95)

    def test_zero_twist_holds_the_stance_without_lifting(self):
        phase = torch.linspace(0, 1, 61, dtype=torch.float64)
        feet = gait.directional_foot_targets(phase, torch.zeros((len(phase), 3), dtype=torch.float64))
        expected = feet[0].expand_as(feet)
        torch.testing.assert_close(feet, expected)

    def test_yaw_command_moves_opposite_sides_in_opposite_directions(self):
        phase = torch.zeros(1, dtype=torch.float64)
        command = torch.tensor([[0.0, 0.0, 0.08]], dtype=torch.float64)
        feet = gait.directional_foot_targets(phase, command, offsets=(0, 0, 0, 0, 0, 0))
        # At phase zero, positive yaw gives the front and rear feet opposing
        # lateral components around the robot centre.
        self.assertLess(float(feet[0, 0, 1]), 0.0)
        self.assertGreater(float(feet[0, 3, 1]), 0.0)
