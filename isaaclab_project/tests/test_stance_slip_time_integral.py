"""Regress policy-rate slip integration using actual environment methods.

Tensor robot and sensor fixtures exercise reward/contact routing without PhysX.
The advancing trajectory is a reward counterexample, not a recorded robot gait.
"""

import importlib.util
from pathlib import Path
import unittest

import torch


FIXTURE_PATH = Path(__file__).with_name("test_joint_angle_env.py")
SPEC = importlib.util.spec_from_file_location("stance_slip_environment_fixture", FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


class StanceSlipTimeIntegralTests(unittest.TestCase):
    def make_env(self, profile):
        env = FIXTURE.V11LearningIntegrationTests().make_env(profile=profile)
        self.assertEqual(env.cfg.decimation, 20)
        self.assertAlmostEqual(env.step_dt, 0.1)
        return env

    def policy_step(self, env, body_velocity, stance_speeds):
        self.assertEqual(len(stance_speeds), env.cfg.decimation)
        env._pre_physics_step(torch.zeros((2, 18)))
        env._robot.data.root_pos_w[:, 0] += body_velocity * env.step_dt
        env._robot.data.root_lin_vel_w[:, 0] = body_velocity
        previous_slip = env._episode_sums["foot_slip"].clone()
        for substep, stance_speed in enumerate(stance_speeds):
            env._robot.data.body_link_lin_vel_w[:, env._feet] = 0.0
            env._robot.data.body_link_lin_vel_w[:, env._feet, 0] = stance_speed
            env._contact_diagnostic_substep = substep + 1
            env._capture_external_contacts()
        reward = env._get_rewards().clone()
        self.assertFalse(env._strict_contact_failure.any())
        return {
            "reward": reward,
            "slip_reward": env._episode_sums["foot_slip"] - previous_slip,
            "metrics": {name: value.clone() for name, value in env.step_metrics.items()},
        }

    def two_step_burst(self, profile, *, advancing):
        env = self.make_env(profile)
        total = torch.zeros(2)
        slip_total = torch.zeros(2)
        progress_total = torch.zeros(2)
        steps = []
        for body_velocity in ((0.08, -0.035) if advancing else (0.0, 0.0)):
            # One 5 ms touchdown sample per 100 ms policy interval has
            # 20 mm/s stance-foot speed; the other nineteen are stationary.
            stance_speeds = ([0.02] + [0.0] * 19) if advancing else [0.0] * 20
            step = self.policy_step(env, body_velocity, stance_speeds)
            total += step["reward"]
            slip_total += step["slip_reward"]
            progress_total += step["metrics"]["progress_m"]
            steps.append(step)
        return {"reward": total, "slip_reward": slip_total, "progress": progress_total, "steps": steps}

    def test_temporal_rms_no_longer_reverses_advancing_burst_preference(self):
        old_stand = self.two_step_burst("revised_v11", advancing=False)
        old_burst = self.two_step_burst("revised_v11", advancing=True)
        new_stand = self.two_step_burst("revised_v12", advancing=False)
        new_burst = self.two_step_burst("revised_v12", advancing=True)

        # Forward 8 mm, then backward 3.5 mm, advances 4.5 mm in 200 ms.
        torch.testing.assert_close(new_burst["progress"], torch.full((2,), 0.0045), atol=1e-7, rtol=0)
        self.assertTrue((old_burst["reward"] < old_stand["reward"]).all())
        self.assertTrue((new_burst["reward"] > new_stand["reward"]).all())
        torch.testing.assert_close(new_stand["reward"], old_stand["reward"])
        # Only the placement of the square root changes this comparison.
        torch.testing.assert_close(
            new_burst["reward"] - old_burst["reward"],
            new_burst["slip_reward"] - old_burst["slip_reward"],
            atol=1e-7, rtol=0,
        )
        for old, new in zip(old_burst["steps"], new_burst["steps"]):
            torch.testing.assert_close(
                new["metrics"]["foot_slip_m2_s2"], old["metrics"]["foot_slip_m2_s2"],
            )
            torch.testing.assert_close(new["metrics"]["foot_slip_m2_s2"], torch.full((2,), 0.02 ** 2 / 20))
            torch.testing.assert_close(new["metrics"]["mean_stance_slip_speed_mps"], torch.full((2,), 0.02 / 20))
            torch.testing.assert_close(new["slip_reward"], torch.full((2,), -6 * 0.02 * 0.005))

    def test_constant_all_foot_translation_retains_penalty_and_is_worse_than_standing(self):
        for speed in (0.01, 0.02, 0.05, 0.1, 0.2):
            results = {}
            for profile in ("revised_v11", "revised_v12"):
                with self.subTest(speed=speed, profile=profile):
                    stationary = self.policy_step(self.make_env(profile), 0.0, [0.0] * 20)
                    translating = self.policy_step(self.make_env(profile), speed, [speed] * 20)
                    expected = torch.full((2,), -6 * speed * 0.1)
                    torch.testing.assert_close(translating["slip_reward"], expected, atol=1e-7, rtol=0)
                    self.assertTrue((translating["reward"] < stationary["reward"]).all())
                    results[profile] = translating
            torch.testing.assert_close(
                results["revised_v12"]["slip_reward"], results["revised_v11"]["slip_reward"],
                atol=1e-7, rtol=0,
            )
            torch.testing.assert_close(
                results["revised_v12"]["metrics"]["mean_stance_slip_speed_mps"], torch.full((2,), speed),
            )

    def test_touchdown_slip_cost_scales_with_physical_contact_duration(self):
        penalties = []
        for contact_frames in (1, 2, 20):
            with self.subTest(contact_frames=contact_frames):
                env = self.make_env("revised_v12")
                step = self.policy_step(env, 0.0, [0.02] * contact_frames + [0.0] * (20 - contact_frames))
                expected = -6 * 0.02 * (contact_frames * 0.005)
                torch.testing.assert_close(step["slip_reward"], torch.full((2,), expected), atol=1e-7, rtol=0)
                torch.testing.assert_close(
                    step["metrics"]["mean_stance_slip_speed_mps"], torch.full((2,), 0.02 * contact_frames / 20),
                )
                penalties.append(step["slip_reward"])
        torch.testing.assert_close(penalties[1], 2 * penalties[0], atol=1e-7, rtol=0)
        torch.testing.assert_close(penalties[2], 20 * penalties[0], atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
