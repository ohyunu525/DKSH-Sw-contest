"""Behavioral tests for direct angles, safety limits and effort rewards (CPU only)."""

import importlib.util
from pathlib import Path
import unittest

import torch


FOLDER = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"
SPEC = importlib.util.spec_from_file_location("joint_angle_control", FOLDER / "joint_angle_control.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


class JointAngleControlTests(unittest.TestCase):
    def setUp(self):
        self.limits = torch.tensor([[[-1.4, 1.4], [-1.1, 1.4], [-0.4, 1.4]]])

    def test_actions_cover_full_absolute_working_range(self):
        low = CONTROL.action_to_angles(torch.full((1, 3), -1.0), self.limits)
        high = CONTROL.action_to_angles(torch.ones((1, 3)), self.limits)
        torch.testing.assert_close(low, self.limits[..., 0])
        torch.testing.assert_close(high, self.limits[..., 1])
        torch.testing.assert_close(CONTROL.action_to_angles(torch.zeros((1, 3)), self.limits), self.limits.mean(-1))
        self.assertTrue(((high - low) > 1.0).all())  # Far more than a +/-1 degree residual.
        torch.testing.assert_close(CONTROL.normalize_angles(high, self.limits), torch.ones((1, 3)))

    def test_constant_action_converges_to_angle_without_accumulating_increments(self):
        action = torch.tensor([[0.4, -0.3, 0.1]])
        target = torch.zeros((1, 3))
        expected = CONTROL.action_to_angles(action, self.limits)
        for _ in range(600):
            previous = target.clone()
            target = CONTROL.bounded_angle_targets(action, target, self.limits, dt=0.02, rate_limit=2, smoothing=0.35)
            self.assertLessEqual(float((target - previous).abs().max()), 0.040001)
        torch.testing.assert_close(target, expected)
        again = CONTROL.bounded_angle_targets(action, target, self.limits, dt=0.02, rate_limit=2, smoothing=0.35)
        torch.testing.assert_close(again, target)

    def test_abrupt_actions_remain_within_limits_and_invalid_components_hold(self):
        previous = self.limits.mean(-1)
        for index in range(100):
            action = torch.full((1, 3), 50.0 if index % 2 else -50.0)
            target = CONTROL.bounded_angle_targets(action, previous, self.limits, dt=0.01, rate_limit=1.5, smoothing=1)
            self.assertTrue((target >= self.limits[..., 0]).all())
            self.assertTrue((target <= self.limits[..., 1]).all())
            self.assertLessEqual(float((target - previous).abs().max()), 0.015001)
            previous = target
        invalid = torch.tensor([[float("nan"), float("inf"), -float("inf")]])
        held = CONTROL.bounded_angle_targets(invalid, previous, self.limits, dt=0.02, rate_limit=2, smoothing=1)
        torch.testing.assert_close(held, previous)

    def test_nonpositive_or_nonfinite_control_parameters_fail(self):
        for key, value in (("dt", 0), ("dt", float("nan")), ("rate_limit", -1), ("smoothing", 0), ("smoothing", 1.1)):
            options = dict(dt=0.02, rate_limit=2, smoothing=0.35)
            options[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CONTROL.bounded_angle_targets(torch.zeros((1, 3)), torch.zeros((1, 3)), self.limits, **options)

    def test_rocking_cannot_repeatedly_collect_progress_reward(self):
        best = torch.tensor([1.0])
        earned = []
        for distance in (0.8, 1.0, 0.8, 1.0, 0.8, 0.7):
            progress, best = CONTROL.frontier_progress(best, torch.tensor([distance]), 1)
            earned.append(float(progress))
        torch.testing.assert_close(torch.tensor(earned), torch.tensor([0.2, 0, 0, 0, 0, 0.1]))

    def test_power_does_not_cancel_between_oppositely_driven_joints(self):
        position = self.limits.mean(-1)
        velocity = torch.tensor([[2.0, -2.0, 0.0]])
        torque = torch.tensor([[0.1, 0.1, 0.0]])
        costs = CONTROL.motion_costs(position, velocity, torque, self.limits, torch.ones((1, 3)), 10)
        torch.testing.assert_close(costs["power"], torch.tensor([0.4]))
        self.assertGreater(float(costs["joint_velocity"]), 0)
        limits_cost = CONTROL.motion_costs(self.limits[..., 1], velocity, torque, self.limits, torch.ones((1, 3)), 10)
        self.assertGreater(float(limits_cost["joint_limit"]), float(costs["joint_limit"]))

    def test_planted_sole_is_stationary_despite_link_rotation(self):
        offset = torch.tensor([[[0.1, 0, 0]]])
        omega = torch.tensor([[[0, 0, 2.0]]])
        link_velocity = torch.tensor([[[0, -0.2, 0]]])
        torch.testing.assert_close(CONTROL.point_velocity(link_velocity, omega, offset), torch.zeros((1, 1, 3)))

    def test_robot_dimensions_and_leg_major_action_order(self):
        self.assertEqual(CONTROL.observation_size(18, "flat"), 74)
        self.assertEqual(CONTROL.observation_size(18, "stairs"), 106)
        self.assertEqual(CONTROL.observation_size(24, "flat"), 94)
        self.assertEqual(CONTROL.observation_size(24, "stairs"), 126)
        self.assertEqual(CONTROL.joint_names(6)[:4], ["leg_0_hip_joint", "leg_0_femur_joint", "leg_0_tibia_joint", "leg_1_hip_joint"])


if __name__ == "__main__":
    unittest.main()
