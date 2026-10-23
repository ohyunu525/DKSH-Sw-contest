"""Behavioral tests for direct angles, safety limits and effort rewards (CPU only)."""

import importlib.util
import math
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
        self.assertEqual(CONTROL.ACTION_SEMANTICS, "absolute_joint_angles_v2")
        for environment in ("flat", "stairs", "rough", "mixed"):
            self.assertEqual(CONTROL.observation_size(18, environment), 140)
            self.assertEqual(CONTROL.observation_size(24, environment), 166)
        self.assertEqual(CONTROL.joint_names(6)[:4], ["leg_0_hip_joint", "leg_0_femur_joint", "leg_0_tibia_joint", "leg_1_hip_joint"])

    def test_signed_progress_never_caps_large_backward_motion(self):
        progress = CONTROL.signed_progress(torch.tensor([1.0, 1.0]), torch.tensor([0.2, 1.8]), 0.01)
        torch.testing.assert_close(progress, torch.tensor([0.01, -0.8]))
        for invalid in (0, -1, math.nan, math.inf):
            with self.subTest(cap=invalid), self.assertRaises(ValueError):
                CONTROL.signed_progress(torch.tensor([1.0]), torch.tensor([0.9]), invalid)

    def test_closed_distance_loops_have_no_positive_progress_for_asymmetric_speeds(self):
        # Slow forward / fast backward exposes the old symmetric-clipping exploit.
        for forward_count, backward_count in ((40, 40), (80, 10), (10, 80)):
            forward = torch.full((forward_count,), 0.016 / forward_count, dtype=torch.float64)
            backward = torch.full((backward_count,), -0.016 / backward_count, dtype=torch.float64)
            displacements = torch.cat((forward, backward))
            distances = 1.0 - torch.cat((torch.zeros(1, dtype=torch.float64), displacements.cumsum(0)))
            progress = CONTROL.signed_progress(distances[:-1], distances[1:], 0.0008)
            with self.subTest(forward_count=forward_count, backward_count=backward_count):
                self.assertAlmostEqual(float(displacements.sum()), 0, places=12)
                self.assertLessEqual(float(progress.sum()), 1e-12)

    def test_speed_cost_is_nonnegative_unclipped_and_convex(self):
        velocity = torch.tensor([0.02, -0.02, 0.0, 1.0], dtype=torch.float64)
        cost = CONTROL.speed_tracking_cost(velocity, 0.02, 0.015)
        self.assertTrue((cost >= 0).all())
        self.assertEqual(float(cost[0]), 0)
        self.assertGreater(float(cost[-1]), 1000)
        self.assertGreaterEqual(float(cost[:2].mean()), float(cost[2]))
        for invalid in (0, -1, math.nan, math.inf):
            with self.subTest(sigma=invalid), self.assertRaises(ValueError):
                CONTROL.speed_tracking_cost(velocity, 0.02, invalid)

    def test_uncapped_progress_preserves_bursts_and_telescopes(self):
        distances = torch.tensor([1.0, 0.2, 1.8, 1.0, 0.9], dtype=torch.float64)
        progress = CONTROL.uncapped_signed_progress(distances[:-1], distances[1:])
        torch.testing.assert_close(progress, torch.tensor([0.8, -1.6, 0.8, 0.1], dtype=torch.float64))
        self.assertAlmostEqual(float(progress[:3].sum()), 0, places=12)
        self.assertAlmostEqual(float(progress.sum()), float(distances[0] - distances[-1]), places=12)

    def test_huber_speed_cost_is_smooth_convex_and_has_unbounded_linear_tails(self):
        error = torch.tensor([-100.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0], dtype=torch.float64)
        velocity = (0.02 + 0.015 * error).requires_grad_()
        cost = CONTROL.huber_speed_tracking_cost(velocity, 0.02, 0.015)
        expected = torch.tensor([99.5, 1.5, 0.5, 0.125, 0, 0.125, 0.5, 1.5, 99.5], dtype=torch.float64)
        torch.testing.assert_close(cost, expected)
        self.assertTrue((cost >= 0).all())
        cost.sum().backward()
        torch.testing.assert_close(velocity.grad, error.clamp(-1, 1) / 0.015)
        self.assertGreaterEqual(float(cost.mean()), float(CONTROL.huber_speed_tracking_cost(velocity.mean(), 0.02, 0.015)))
        for invalid in (0, -1, math.nan, math.inf):
            with self.subTest(sigma=invalid), self.assertRaises(ValueError):
                CONTROL.huber_speed_tracking_cost(velocity, 0.02, invalid)

    def test_revised_objective_prefers_net_advancing_bursty_motion_to_standing(self):
        # Equal durations yield the commanded mean +0.02m/s, despite reverse motion.
        dt, command, sigma = 0.02, 0.02, 0.015
        velocity = torch.tensor([0.08, -0.04], dtype=torch.float64)
        distance = 1.0 - torch.cat((torch.zeros(1, dtype=torch.float64), (velocity * dt).cumsum(0)))
        stillness = -0.02 * torch.relu(0.1 * command - velocity) / sigma
        old_reward = 25 * CONTROL.signed_progress(distance[:-1], distance[1:], command * dt)
        old_reward += (-0.1 * CONTROL.speed_tracking_cost(velocity, command, sigma) + stillness) * dt
        new_reward = 25 * CONTROL.uncapped_signed_progress(distance[:-1], distance[1:])
        new_reward += (-0.1 * CONTROL.huber_speed_tracking_cost(velocity, command, sigma) + stillness) * dt
        standing = torch.zeros(1, dtype=torch.float64)
        standing_hinge = -0.02 * 0.1 * command / sigma
        old_standing = float(-0.1 * CONTROL.speed_tracking_cost(standing, command, sigma) + standing_hinge)
        new_standing = float(-0.1 * CONTROL.huber_speed_tracking_cost(standing, command, sigma) + standing_hinge)
        self.assertAlmostEqual(float(velocity.mean()), command, places=12)
        self.assertAlmostEqual(float(old_reward.mean() / dt), -1.878, places=12)
        self.assertAlmostEqual(float(new_reward.mean() / dt), 0.122, places=12)
        self.assertLess(float(old_reward.mean() / dt), old_standing)
        self.assertGreater(float(new_reward.mean() / dt), new_standing)

    def test_revised_objective_closed_loops_cannot_outperform_standing(self):
        dt, command, sigma = 0.02, 0.02, 0.015
        standing_speed = CONTROL.huber_speed_tracking_cost(torch.tensor(0.0, dtype=torch.float64), command, sigma)
        standing_rate = -0.1 * standing_speed - 0.02 * 0.1 * command / sigma
        for forward_speed, forward_count, backward_speed, backward_count in (
            (0.02, 40, -0.02, 40), (0.08, 10, -0.02, 40), (0.01, 80, -0.08, 10),
        ):
            velocity = torch.cat((
                torch.full((forward_count,), forward_speed, dtype=torch.float64),
                torch.full((backward_count,), backward_speed, dtype=torch.float64),
            ))
            distance = 1.0 - torch.cat((torch.zeros(1, dtype=torch.float64), (velocity * dt).cumsum(0)))
            progress = CONTROL.uncapped_signed_progress(distance[:-1], distance[1:])
            rates = -0.1 * CONTROL.huber_speed_tracking_cost(velocity, command, sigma)
            rates -= 0.02 * torch.relu(0.1 * command - velocity) / sigma
            for orientation in ("upright", "tilted_during_retreat", "tilted_during_advance"):
                upright = torch.ones_like(velocity)
                if orientation == "tilted_during_retreat":
                    upright[forward_count:] = 0.1
                elif orientation == "tilted_during_advance":
                    upright[:forward_count] = 0.1
                reward = 25 * progress + (rates - 2 * (1 - upright).square()) * dt
                with self.subTest(forward=forward_speed, backward=backward_speed, orientation=orientation):
                    self.assertAlmostEqual(float(progress.sum()), 0, places=12)
                    self.assertLessEqual(float(reward.sum()), 1e-12)
                    self.assertLessEqual(float(reward.mean() / dt), float(standing_rate) + 1e-12)

    def test_zero_net_oscillation_cannot_collect_locomotion_reward(self):
        dt = 0.02
        for forward_speed, forward_count, backward_speed, backward_count in (
            (0.02, 40, -0.02, 40), (0.08, 10, -0.02, 40), (0.01, 80, -0.08, 10),
        ):
            velocity = torch.cat((
                torch.full((forward_count,), forward_speed, dtype=torch.float64),
                torch.full((backward_count,), backward_speed, dtype=torch.float64),
            ))
            distance = 1.0 - torch.cat((torch.zeros(1, dtype=torch.float64), (velocity * dt).cumsum(0)))
            progress = CONTROL.signed_progress(distance[:-1], distance[1:], 0.0008)
            cost = CONTROL.speed_tracking_cost(velocity, 0.02, 0.015)
            for orientation in ("upright", "tilted_during_retreat", "tilted_during_advance"):
                upright = torch.ones_like(velocity)
                if orientation == "tilted_during_retreat":
                    upright[forward_count:] = 0.1
                elif orientation == "tilted_during_advance":
                    upright[:forward_count] = 0.1
                # Signed potential progress is intentionally not posture-gated.
                reward = 25 * progress - 2 * cost * dt - 2 * (1 - upright).square() * dt
                with self.subTest(forward=forward_speed, backward=backward_speed, orientation=orientation):
                    self.assertAlmostEqual(float((velocity * dt).sum()), 0, places=12)
                    self.assertLessEqual(float(reward.sum()), 1e-12)
                    self.assertGreaterEqual(float(cost.mean()), float(CONTROL.speed_tracking_cost(velocity.mean(), 0.02, 0.015)) - 1e-12)

    def test_exploration_noise_satisfies_per_joint_gaussian_tail_budget(self):
        limits = torch.tensor([[-math.pi / 2, math.pi / 2], [-1.1, 1.4], [-0.4, 1.4]], dtype=torch.float64)
        std = CONTROL.exploration_std(limits, dt=0.02, rate_limit=1.5, smoothing=0.35)
        self.assertGreater(std, 0)
        self.assertLess(std, 0.03)
        for span in limits[:, 1] - limits[:, 0]:
            angle_std = float(span) * 0.5 * std
            tail = math.erfc((1.5 * 0.02) / (0.35 * angle_std * math.sqrt(2)))
            self.assertLessEqual(tail, 0.01 + 1e-12)
        tighter = CONTROL.exploration_std(limits, dt=0.02, rate_limit=1.5, smoothing=0.35, saturation_probability=0.001)
        self.assertLess(tighter, std)
        doubled_dt = CONTROL.exploration_std(limits, dt=0.04, rate_limit=1.5, smoothing=0.35)
        self.assertAlmostEqual(doubled_dt, 2 * std)

    def test_exploration_noise_tail_matches_requested_rate_saturation(self):
        limits = torch.tensor([[-math.pi / 2, math.pi / 2]], dtype=torch.float64)
        std = CONTROL.exploration_std(limits, dt=0.02, rate_limit=1.5, smoothing=0.35)
        generator = torch.Generator().manual_seed(80421)
        actions = std * torch.randn((500_000, 1), generator=generator, dtype=torch.float64)
        requested = CONTROL.action_to_angles(actions, limits)
        saturation = (0.35 * requested.abs() > 1.5 * 0.02).double().mean()
        self.assertLess(float(saturation), 0.0108)
        self.assertGreater(float(saturation), 0.0092)
        # Tuning exploration must never shrink the policy's full angle range.
        torch.testing.assert_close(CONTROL.action_to_angles(torch.tensor([[-1.0], [1.0]]), limits).flatten(), limits.flatten())

    def test_exploration_noise_rejects_invalid_ranges_or_tail_budgets(self):
        for limits in (torch.empty((0, 2)), torch.tensor([0.0]), torch.tensor([[1.0, -1.0]]), torch.tensor([[0.0, math.inf]])):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                CONTROL.exploration_std(limits, dt=0.02, rate_limit=1.5, smoothing=0.35)
        for key, invalid in (("dt", 0), ("rate_limit", -1), ("smoothing", 0), ("smoothing", 1.1), ("saturation_probability", 0), ("saturation_probability", 1), ("saturation_probability", math.nan)):
            options = dict(dt=0.02, rate_limit=1.5, smoothing=0.35, saturation_probability=0.01)
            options[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                CONTROL.exploration_std(self.limits, **options)


if __name__ == "__main__":
    unittest.main()
