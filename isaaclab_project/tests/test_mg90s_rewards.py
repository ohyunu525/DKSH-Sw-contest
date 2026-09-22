"""Unit tests for MG90S reward terms without starting Isaac Sim."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/mg90s_rewards.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_mg90s_rewards", MODULE)
assert SPEC is not None and SPEC.loader is not None
REWARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARDS)


class SpeedTrackingRewardTests(unittest.TestCase):
    def test_stationary_velocity_has_zero_reward(self):
        command = torch.tensor([0.012, 0.016, 0.020])
        reward = REWARDS.speed_tracking_reward(torch.zeros_like(command), command, 0.006)
        self.assertTrue(torch.allclose(reward, torch.zeros_like(reward)))

    def test_command_velocity_is_preferred_to_under_or_overshoot(self):
        command = torch.tensor([0.016])
        at_command = REWARDS.speed_tracking_reward(command, command, 0.006)
        slower = REWARDS.speed_tracking_reward(torch.tensor([0.011]), command, 0.006)
        faster = REWARDS.speed_tracking_reward(torch.tensor([0.021]), command, 0.006)
        self.assertGreater(float(at_command), float(slower))
        self.assertTrue(torch.allclose(slower, faster))

    def test_narrower_sigma_penalizes_the_same_error_more(self):
        command = torch.tensor([0.016])
        velocity = torch.tensor([0.011])
        narrow = REWARDS.speed_tracking_reward(velocity, command, 0.006)
        broad = REWARDS.speed_tracking_reward(velocity, command, 0.010)
        self.assertLess(float(narrow), float(broad))

    def test_non_positive_sigma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            REWARDS.speed_tracking_reward(torch.tensor([0.0]), torch.tensor([0.0]), 0.0)


class VelocityTrackingRewardTests(unittest.TestCase):
    def test_exact_planar_and_yaw_commands_receive_full_reward(self):
        command = torch.tensor([[0.01, -0.004], [0.0, 0.0]])
        reward = REWARDS.velocity_tracking_reward(command, command, 0.01)
        torch.testing.assert_close(reward, torch.ones(2))

    def test_vector_error_is_symmetric_and_monotonic(self):
        command = torch.zeros((3, 2))
        measured = torch.tensor([[0.002, 0.0], [-0.002, 0.0], [0.004, 0.0]])
        reward = REWARDS.velocity_tracking_reward(measured, command, 0.01)
        self.assertAlmostEqual(float(reward[0]), float(reward[1]))
        self.assertGreater(float(reward[0]), float(reward[2]))

    def test_shape_and_sigma_are_validated(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            REWARDS.velocity_tracking_reward(torch.zeros((1, 2)), torch.zeros((1, 1)), 0.01)
        with self.assertRaisesRegex(ValueError, "positive"):
            REWARDS.velocity_tracking_reward(torch.zeros((1, 2)), torch.zeros((1, 2)), 0.0)


class ArrivalTimingRewardTests(unittest.TestCase):
    def test_exact_scheduled_progress_has_no_arrival_penalty(self):
        target = torch.tensor([1.0, 2.0])
        scheduled = torch.tensor([0.25, 1.50])
        penalty, undershoot, overshoot = REWARDS.proportional_arrival_penalty(
            scheduled, scheduled, target, minimum_total_progress=0.01
        )
        torch.testing.assert_close(penalty, torch.zeros_like(target))
        torch.testing.assert_close(undershoot, torch.zeros_like(target))
        torch.testing.assert_close(overshoot, torch.zeros_like(target))

    def test_arriving_early_and_late_are_both_penalized(self):
        target = torch.tensor([1.0, 1.0])
        scheduled = torch.tensor([0.50, 0.50])
        achieved = torch.tensor([0.25, 0.75])
        penalty, undershoot, overshoot = REWARDS.proportional_arrival_penalty(
            achieved, scheduled, target, minimum_total_progress=0.01
        )
        torch.testing.assert_close(undershoot, torch.tensor([0.0625, 0.0]))
        torch.testing.assert_close(overshoot, torch.tensor([0.0, 0.0625]))
        torch.testing.assert_close(penalty, torch.tensor([0.0625, 0.0625]))

    def test_near_stationary_goal_does_not_create_an_arrival_penalty(self):
        target = torch.tensor([0.005])
        penalty, undershoot, overshoot = REWARDS.proportional_arrival_penalty(
            torch.tensor([0.050]), torch.tensor([0.002]), target, minimum_total_progress=0.010
        )
        torch.testing.assert_close(penalty, torch.zeros(1))
        torch.testing.assert_close(undershoot, torch.zeros(1))
        torch.testing.assert_close(overshoot, torch.zeros(1))

    def test_straight_body_twist_integrates_to_linear_schedule(self):
        position, tangent = REWARDS.body_twist_reference_trajectory(
            torch.tensor([[0.10, -0.04]]), torch.tensor([0.0]), torch.tensor([3.0])
        )
        torch.testing.assert_close(position, torch.tensor([[0.30, -0.12]]))
        torch.testing.assert_close(tangent, torch.tensor([[0.10, -0.04]]))

    def test_yawed_body_twist_curves_its_scheduled_path(self):
        position, tangent = REWARDS.body_twist_reference_trajectory(
            torch.tensor([[1.0, 0.0]]), torch.tensor([1.0]), torch.tensor([torch.pi / 2])
        )
        torch.testing.assert_close(position, torch.tensor([[1.0, 1.0]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(tangent, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
