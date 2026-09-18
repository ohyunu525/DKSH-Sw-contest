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


if __name__ == "__main__":
    unittest.main()
