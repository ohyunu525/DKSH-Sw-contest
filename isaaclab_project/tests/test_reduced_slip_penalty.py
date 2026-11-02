"""Check the optional reduced-slip experiment with actual reward routing.

The default coefficient stays six. A coefficient of 0.6 deliberately weakens
the skating deterrent; these tensor fixtures do not simulate contact physics.
"""

import importlib.util
from pathlib import Path
import unittest

import torch


FIXTURE_PATH = Path(__file__).with_name("test_joint_angle_env.py")
SPEC = importlib.util.spec_from_file_location("reduced_slip_environment_fixture", FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


class ReducedSlipPenaltyTests(unittest.TestCase):
    def policy_step(self, coefficient, *, body_speed, stance_speeds, torque=0.0, joint_speed=0.0):
        env = FIXTURE.V11LearningIntegrationTests().make_env(profile="revised_v12")
        self.assertEqual(env.cfg.stance_slip_cost_coefficient, 6.0)
        self.assertEqual(env.cfg.decimation, 20)
        self.assertEqual(len(stance_speeds), env.cfg.decimation)
        self.assertAlmostEqual(env.step_dt, 0.1)
        env.cfg.stance_slip_cost_coefficient = coefficient
        env._pre_physics_step(torch.zeros((2, 18)))
        env._robot.data.root_pos_w[:, 0] += body_speed * env.step_dt
        env._robot.data.root_lin_vel_w[:, 0] = body_speed
        env._robot.data.applied_torque[:] = torque
        env._robot.data.joint_vel[:] = joint_speed
        for substep, stance_speed in enumerate(stance_speeds):
            env._robot.data.body_link_lin_vel_w[:, env._feet] = 0.0
            env._robot.data.body_link_lin_vel_w[:, env._feet, 0] = stance_speed
            env._contact_diagnostic_substep = substep + 1
            env._capture_external_contacts()
        reward = env._get_rewards().clone()
        self.assertFalse(env._strict_contact_failure.any())
        self.assertFalse(env._reached_goal.any())
        return env, reward

    def test_reducing_coefficient_changes_only_slip_cost_by_one_tenth(self):
        options = dict(body_speed=0.02, stance_speeds=[0.01] * 20, torque=0.2, joint_speed=0.3)
        baseline, baseline_reward = self.policy_step(6.0, **options)
        reduced, reduced_reward = self.policy_step(0.6, **options)

        torch.testing.assert_close(
            reduced._episode_sums["foot_slip"], baseline._episode_sums["foot_slip"] / 10,
            atol=1e-8, rtol=0,
        )
        for name in baseline._episode_sums:
            if name != "foot_slip":
                with self.subTest(reward_term=name):
                    torch.testing.assert_close(reduced._episode_sums[name], baseline._episode_sums[name])
        for name in ("progress_m", "reward_progress_m", "speed_reward_per_s", "mechanical_power_w",
                     "mean_stance_slip_speed_mps", "foot_slip_m2_s2"):
            with self.subTest(metric=name):
                torch.testing.assert_close(reduced.step_metrics[name], baseline.step_metrics[name])
        self.assertTrue((baseline._episode_sums["torque"] < 0).all())
        self.assertTrue((baseline._episode_sums["power"] < 0).all())
        torch.testing.assert_close(
            reduced_reward - baseline_reward,
            reduced._episode_sums["foot_slip"] - baseline._episode_sums["foot_slip"],
            atol=1e-8, rtol=0,
        )

    def test_reduced_weight_keeps_touchdown_time_integral_and_duration_scaling(self):
        penalties = {6.0: [], 0.6: []}
        for active_frames in (1, 2, 20):
            samples = [0.02] * active_frames + [0.0] * (20 - active_frames)
            states = {}
            for coefficient in (6.0, 0.6):
                with self.subTest(coefficient=coefficient, active_frames=active_frames):
                    env, _ = self.policy_step(coefficient, body_speed=0.02, stance_speeds=samples)
                    expected_cost = -coefficient * 0.02 * (active_frames * 0.005)
                    torch.testing.assert_close(
                        env._episode_sums["foot_slip"], torch.full((2,), expected_cost),
                        atol=1e-8, rtol=0,
                    )
                    torch.testing.assert_close(
                        env.step_metrics["mean_stance_slip_speed_mps"],
                        torch.full((2,), 0.02 * active_frames / 20),
                    )
                    torch.testing.assert_close(
                        env.step_metrics["foot_slip_m2_s2"], torch.full((2,), 0.02 ** 2 * active_frames / 20),
                    )
                    penalties[coefficient].append(env._episode_sums["foot_slip"].clone())
                    states[coefficient] = env
            torch.testing.assert_close(
                states[0.6].step_metrics["mean_stance_slip_speed_mps"],
                states[6.0].step_metrics["mean_stance_slip_speed_mps"],
            )
            torch.testing.assert_close(
                states[0.6]._episode_sums["foot_slip"], states[6.0]._episode_sums["foot_slip"] / 10,
                atol=1e-8, rtol=0,
            )
        for coefficient, values in penalties.items():
            with self.subTest(coefficient=coefficient):
                torch.testing.assert_close(values[1], 2 * values[0], atol=1e-8, rtol=0)
                torch.testing.assert_close(values[2], 20 * values[0], atol=1e-8, rtol=0)

    def test_reduced_coefficient_deliberately_allows_profitable_all_foot_skating(self):
        # This is the experiment's tradeoff, not a no-skating guarantee.
        # All six supported feet translate with the body on every frame.
        for speed in (0.02, 0.05, 0.1):
            with self.subTest(speed=speed):
                options = dict(body_speed=speed, stance_speeds=[speed] * 20)
                baseline, baseline_reward = self.policy_step(6.0, **options)
                reduced, reduced_reward = self.policy_step(0.6, **options)
                self.assertTrue((baseline_reward < 0).all())
                self.assertTrue((reduced_reward > 0).all())
                self.assertTrue((reduced._strict_contact_state()[0].sum(-1) == 6).all())
                torch.testing.assert_close(
                    reduced._episode_sums["foot_slip"], baseline._episode_sums["foot_slip"] / 10,
                    atol=1e-8, rtol=0,
                )
                torch.testing.assert_close(reduced._episode_sums["progress"], baseline._episode_sums["progress"])
                torch.testing.assert_close(reduced._episode_sums["speed"], baseline._episode_sums["speed"])


if __name__ == "__main__":
    unittest.main()
