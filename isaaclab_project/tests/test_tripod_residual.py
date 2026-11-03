"""Test batched tripod profiles and residual action routing without PhysX."""

import ast
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import torch

import test_joint_angle_env
import test_onshape_cad6_gait
from test_joint_angle_env import CONTROL, FOLDER, Environment, load_helper


GAIT = load_helper("onshape_cad6_gait")


def make_tripod_env():
    path = FOLDER / "tripod_residual_env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    namespace = {
        "math": math, "torch": torch, "JointAngleNavigationEnv": Environment,
        **{name: value for name, value in vars(GAIT).items() if not name.startswith("_")},
        **{name: value for name, value in vars(CONTROL).items() if not name.startswith("_")},
    }
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import)
                 and not (isinstance(node, ast.ImportFrom) and node.module != "__future__")
                 and not (isinstance(node, ast.ClassDef) and node.name.endswith("Cfg"))]
    exec(compile(tree, str(path), "exec"), namespace)
    base = test_joint_angle_env.StrictContactEnvironmentTests().strict_env(behavior_profile="revised_v12")
    settings = vars(base.cfg).copy()
    settings.update(gait_profile=GAIT.DEFAULT_GAIT_PROFILE, residual_limit_rad=0.15,
                    robot=SimpleNamespace(copy=lambda: deepcopy(base.cfg.robot)))
    env = namespace["Cad6TripodResidualEnv"](SimpleNamespace(**settings))
    env._reset_idx(torch.tensor([0, 1]))
    env._out_of_bounds = torch.zeros(2, dtype=torch.bool)
    return env


class TripodProfileTests(unittest.TestCase):
    def test_each_lane_uses_its_own_gait_profile(self):
        phase = torch.tensor([0.32, 0.69, 0.88], dtype=torch.float64)
        command = phase.new_tensor([[0.05, 0.0, 0.0], [-0.03, 0.04, 0.05], [0.0, -0.05, 0.0]])
        profiles = phase.new_tensor([[1.2, 0.025, 0.007, 0.55],
                                     [2.1, 0.080, 0.010, 0.65],
                                     [2.8, 0.030, 0.019, 0.74]])

        def targets(phase, command, period, stride, lift, duty):
            return GAIT.joint_positions_for_twist(phase, command, period, lift, stride,
                                                  offsets=GAIT.TRIPOD_OFFSETS, duty=duty)

        actual = targets(phase, command, *profiles.unbind(-1))
        expected = torch.cat([
            targets(phase[i:i + 1], command[i:i + 1], *profile.tolist())
            for i, profile in enumerate(profiles)
        ])
        self.assertEqual(actual.shape, (3, 6, 3))
        torch.testing.assert_close(actual, expected)


class TripodResidualTests(unittest.TestCase):
    def test_peak_swing_lift_is_calibrated_at_the_distal_stance(self):
        env = make_tripod_env()
        neutral = torch.tensor([0.0, -0.3, -0.3])
        env._robot.data.default_joint_pos[:, env._joint_ids] = neutral.repeat(6)
        env.cfg.target_speed = 0.0  # Isolate vertical swing from radial travel.
        env.set_gait_profiles(torch.tensor([[2.0, 0.05, 0.01, 0.60]]).expand(2, -1))
        env.episode_length_buf[:] = 36  # Phase 0.8: leg 0 at swing midpoint.
        _, reference = env._gait_reference()
        fk = test_onshape_cad6_gait.OnshapeCad6GaitTests()
        fk.setUpClass()
        before = fk._cad_foot_position(0, neutral.numpy())
        after = fk._cad_foot_position(0, reference[0, :3].numpy())
        self.assertAlmostEqual(after[2] - before[2], 0.01, delta=0.001)

    def test_reference_observations_follow_per_lane_phase_and_reset(self):
        env = make_tripod_env()
        profiles = torch.tensor([[1.5, 0.035, 0.009, 0.59], [2.4, 0.060, 0.016, 0.68]])
        env.set_gait_profiles(profiles)
        profiles.zero_()
        env.episode_length_buf[:] = torch.tensor([7, 13])
        phase, reference = env._gait_reference()
        torch.testing.assert_close(phase, torch.tensor([0.7 / 1.5, 1.3 / 2.4]))

        observation = env._get_observations()["policy"]
        self.assertEqual(observation.shape, (2, 160))
        self.assertEqual(env.spaces_observations, observation.shape[-1])
        phase_features = torch.stack(((math.tau * phase).sin(), (math.tau * phase).cos()), dim=-1)
        torch.testing.assert_close(observation[:, -20:-18], phase_features)
        torch.testing.assert_close(observation[:, -18:], CONTROL.normalize_angles(reference, env._angle_limits))

        env._reset_idx(torch.tensor([0]))
        reset_phase, reset_reference = env._gait_reference()
        torch.testing.assert_close(reset_phase, torch.tensor([0.0, phase[1]]))
        torch.testing.assert_close(reset_reference[0], env._robot.data.default_joint_pos[0, env._joint_ids])
        torch.testing.assert_close(reset_reference[1], reference[1])

    def test_zero_residual_preserves_reference_and_corrections_use_target_limiter(self):
        env = make_tripod_env()
        env.episode_length_buf[:] = torch.tensor([18, 21])
        _, reference = env._gait_reference()
        env._pre_physics_step(torch.zeros_like(reference))
        torch.testing.assert_close(CONTROL.action_to_angles(env._actions, env._angle_limits), reference)

        actions = torch.full_like(reference, 9.0)
        actions[:, 1::2] = -9.0
        actions[0, 0] = float("nan")
        previous = env.joint_angle_targets_rad.clone()
        env._pre_physics_step(actions)
        lower, upper = env._angle_limits.unbind(-1)
        expected = (reference + actions.sign() * env.cfg.residual_limit_rad).clamp(lower, upper)
        expected[0, 0] = previous[0, 0]
        torch.testing.assert_close(CONTROL.action_to_angles(env._actions, env._angle_limits), expected)

        interval = env.cfg.target_filter_interval_s
        env._advance_angle_targets(interval)
        self.assertLessEqual(float((env.joint_angle_targets_rad - previous).abs().max()),
                             env.cfg.target_rate_limit * interval + 1e-6)

    def test_invalid_profile_update_keeps_previous_profiles(self):
        env = make_tripod_env()
        original = env._gait_profiles.clone()
        for invalid in (original[0], original.clone().fill_(float("nan")), original.clone().fill_(0.0)):
            with self.assertRaises(ValueError):
                env.set_gait_profiles(invalid)
            torch.testing.assert_close(env._gait_profiles, original)


if __name__ == "__main__":
    unittest.main()
