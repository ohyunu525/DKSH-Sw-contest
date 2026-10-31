"""Exercise actual environment methods with tensor assets; no PhysX simulation.

These tests cover target routing, observable filter state, goal gates and partial
resets. They do not claim to validate contact physics or trained locomotion.
"""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


FOLDER = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, FOLDER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = load_helper("joint_angle_control")
CONTACT = load_helper("joint_angle_contact")


class Robot:
    def __init__(self, count, legs):
        # Intentionally different PhysX order from the public leg-major contract.
        self.joint_names = list(reversed(CONTROL.joint_names(legs)))
        self.body_names = ["base"] + [f"leg_{leg}_{part}" for leg in range(legs) for part in ("hip", "femur", "tibia")]
        actions = 3 * legs
        self._ALL_INDICES = torch.arange(count)
        limits = torch.tensor([-1.5, 1.5]).expand(count, actions, 2).clone()
        pos = torch.zeros((count, actions))
        pos[:, [i for i, name in enumerate(self.joint_names) if "femur" in name]] = 0.6
        quat = torch.zeros((count, len(self.body_names), 4))
        quat[..., 0] = 1
        body_pos = torch.zeros((count, len(self.body_names), 3))
        body_pos[..., 0] = -0.120  # Default sole offset lands on the flat surface.
        root_state = torch.zeros((count, 13))
        root_state[:, 2], root_state[:, 3] = 0.132, 1
        self.data = NS(
            soft_joint_pos_limits=limits.clone(), joint_pos_limits=limits.clone(),
            default_joint_pos=pos.clone(), joint_pos=pos.clone(), joint_vel=torch.zeros_like(pos),
            applied_torque=torch.zeros_like(pos), joint_effort_limits=torch.ones_like(pos),
            root_pos_w=torch.tensor([[0, 0, 0.132]]).expand(count, 3).clone(),
            root_quat_w=root_state[:, 3:7].clone(),
            root_lin_vel_w=torch.zeros((count, 3)), root_lin_vel_b=torch.zeros((count, 3)),
            root_ang_vel_b=torch.zeros((count, 3)), projected_gravity_b=torch.tensor([[0, 0, -1.0]]).expand(count, 3).clone(),
            body_link_quat_w=quat, body_link_lin_vel_w=torch.zeros((count, len(self.body_names), 3)),
            body_link_ang_vel_w=torch.zeros((count, len(self.body_names), 3)),
            body_link_pos_w=body_pos, default_root_state=root_state,
        )
        self.targets = pos.clone()

    def write_joint_position_limit_to_sim(self, limits):
        self.data.joint_pos_limits.copy_(limits)

    def write_root_state_to_sim(self, state, env_ids):
        self.data.root_pos_w[env_ids] = state[:, :3]
        self.data.root_quat_w[env_ids] = state[:, 3:7]

    def write_joint_state_to_sim(self, pos, vel, env_ids):
        self.data.joint_pos[env_ids] = pos
        self.data.joint_vel[env_ids] = vel

    def set_joint_position_target(self, target, env_ids=None):
        if env_ids is None:
            self.targets.copy_(target)
        else:
            self.targets[env_ids] = target


class Parent:
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg, self.num_envs, self.device, self.step_dt = cfg, 2, "cpu", 0.02
        self._scenario, self.extras = None, {"log": {}}
        self._robot = Robot(2, cfg.action_space // 3)
        self.scene = NS(env_origins=torch.zeros((2, 3)))
        self._contacts = NS(body_names=self._robot.body_names.copy(), data=NS(net_forces_w_history=torch.zeros((2, 4, len(self._robot.body_names), 3))))
        self._actions = torch.zeros((2, cfg.action_space))
        self._previous_actions = torch.zeros_like(self._actions)
        self._target_positions_w = torch.tensor([[0.5, 0, 0.015]]).expand(2, 3).clone()
        self._previous_goal_distance = torch.full((2,), 0.5)
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.reset_time_outs = torch.zeros(2, dtype=torch.bool)
        self.reset_terminated = torch.zeros(2, dtype=torch.bool)
        self.max_episode_length = 1000
        self._configure_gym_env_spaces()

    def _configure_gym_env_spaces(self):
        self.spaces_observations = self.cfg.observation_space

    def _goal_data(self):
        offset = self._target_positions_w - self._robot.data.root_pos_w
        offset[:, 2] = 0
        distance = torch.linalg.vector_norm(offset[:, :2], dim=-1)
        direction = offset / distance.clamp_min(1e-6)[:, None]
        return distance, direction, direction

    def _fallen(self):
        return (self._robot.data.root_pos_w[:, 2] < 0.06) | (self._robot.data.projected_gravity_b[:, 2] > -0.35)

    def _debris_hit(self):
        return torch.zeros(2, dtype=torch.bool)

    def _reset_idx(self, env_ids):
        self._robot.data.joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids]
        self._previous_goal_distance[env_ids] = self._goal_data()[0][env_ids]
        self.extras["log"] = {}


def quat_rotate(q, v):
    uv = 2 * torch.cross(q[:, 1:], v, dim=-1)
    return v + q[:, :1] * uv + torch.cross(q[:, 1:], uv, dim=-1)


tree = ast.parse((FOLDER / "joint_angle_env.py").read_text(encoding="utf-8"))
definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
namespace = {"torch": torch, "math": math, "SpiderNavigationEnv": Parent, "quat_rotate": quat_rotate,
             "sole_contact_state": CONTACT.sole_contact_state,
             "foot_contact_state": CONTACT.foot_contact_state,
             "closest_foot_support": CONTACT.closest_foot_support,
             "toe_button_contact_state": CONTACT.toe_button_contact_state,
             "distal_foot_contact_state": CONTACT.distal_foot_contact_state,
             "strict_leg_contact_state": CONTACT.strict_leg_contact_state,
             **{name: getattr(CONTROL, name) for name in (
                 "action_to_angles", "bounded_angle_targets", "joint_names", "motion_costs",
                 "normalize_angles", "observation_size", "point_velocity", "signed_progress", "speed_tracking_cost",
                 "uncapped_signed_progress", "huber_speed_tracking_cost",
             )}}
exec(compile(ast.Module(body=[definition], type_ignores=[]), "joint_angle_env.py", "exec"), namespace)
Environment = namespace["JointAngleNavigationEnv"]


def make_env(actions=18, support_offsets=None, *, behavior_profile="revised_v3", toe_button_offsets=None,
             distal_offsets=None, proximal_offsets=None, environment_preset="flat", collision_offsets=None):
    # Read numeric tuning values from the real cfg, keeping external asset setup
    # out of these tensor tests; no duplicate copy of the reward implementation.
    cfg_tree = ast.parse((FOLDER / "joint_angle_env_cfg.py").read_text(encoding="utf-8"))
    classes = [next(node for node in cfg_tree.body if isinstance(node, ast.ClassDef))]
    if behavior_profile in ("revised_v5", "revised_v6", "revised_v7", "revised_v7_com", "revised_v8"):
        classes.append(next(node for node in cfg_tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "MG996RHipCad6JointAngleEnvCfg"))
    settings = {}
    for cls in classes:
        for node in cls.body:
            if isinstance(node, ast.Assign):
                try:
                    settings[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, AttributeError):
                    pass
    settings.update(
        action_space=actions, environment_preset=environment_preset, robot=NS(copy=lambda: NS(spawn=NS(usd_path="test", articulation_props=NS(enabled_self_collisions=False)))),
        minimum_base_height=0.06, robot_height=0.24, max_distance_from_origin=5.5,
        joint_angle_ranges={"hip": (-1.4, 1.4), "femur": (-1.1, 1.4), "tibia": (-0.4, 1.4)},
        servo_velocity_limit=10.0, goal_reward=30, failure_penalty=-10, debris_impact_penalty=-2,
        foot_support_offsets=support_offsets,
        behavior_profile=behavior_profile, toe_button_offsets=toe_button_offsets,
        distal_foot_support_offsets=distal_offsets, proximal_foot_offsets=proximal_offsets,
        non_distal_collision_offsets=collision_offsets,
    )
    env = Environment(NS(**settings))
    env._reset_idx(torch.tensor([0, 1]))
    env._out_of_bounds = torch.zeros(2, dtype=torch.bool)
    return env


class JointAngleEnvironmentTests(unittest.TestCase):
    def test_leg_major_actions_reach_correct_physx_joints_without_pose_offset(self):
        env = make_env()
        actions = torch.linspace(-0.6, 0.6, 18).expand(2, -1)
        for _ in range(500):
            env._pre_physics_step(actions)
        expected = CONTROL.action_to_angles(actions, env._angle_limits)
        torch.testing.assert_close(env._processed_actions[:, env._joint_ids], expected)
        self.assertNotEqual(env._joint_ids, list(range(18)))

    def test_observations_contain_actual_applied_target_state_and_contacts(self):
        for actions in (18, 24):
            env = make_env(actions)
            env._pre_physics_step(torch.ones((2, actions)))
            observation = env._get_observations()["policy"]
            self.assertEqual(observation.shape, (2, CONTROL.observation_size(actions, "flat")))
            self.assertEqual(env.spaces_observations, observation.shape[-1])
            target_section = observation[:, 12 + 2 * actions:12 + 3 * actions]
            torch.testing.assert_close(target_section, CONTROL.normalize_angles(env._processed_actions[:, env._joint_ids], env._angle_limits))
            torch.testing.assert_close(observation[:, 12 + 3 * actions:12 + 4 * actions], env._target_velocity / env.cfg.target_rate_limit)
            torch.testing.assert_close(observation[:, -48:-32], torch.ones((2, 16)))

    def test_stair_heights_and_hazards_reach_actor_with_same_schema_as_flat(self):
        env = make_env()
        env.cfg.environment_preset = "stairs"
        features = torch.arange(48, dtype=torch.float32).expand(2, -1)
        env._scenario = NS(observations=lambda: features, ground_height=lambda _: torch.zeros(2), surface_heights=lambda p: torch.zeros(p.shape[:2]))
        env._configure_gym_env_spaces()
        observation = env._get_observations()["policy"]
        self.assertEqual(observation.shape, (2, 140))
        self.assertEqual(env.cfg.state_space, 0)
        torch.testing.assert_close(observation[:, -48:], features)

    def support_at_goal(self, env):
        env._target_positions_w[:, :2] = env._robot.data.root_pos_w[:, :2] + torch.tensor([0.01, 0])
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2

    def test_goal_requires_supported_upright_hold_and_correct_platform_height(self):
        env = make_env()
        self.support_at_goal(env)
        self.assertFalse(env._get_dones()[0].any())
        for _ in range(12):
            done, _ = env._get_dones()
        self.assertTrue(done.all())
        env._goal_hold_time.zero_()
        env._target_positions_w[:, 2] = 0.30  # Reaching XY beneath a platform is not a goal.
        env._scenario = NS(collect_impacts=lambda: None, layout=NS(corridor_width=1.6), surface_heights=lambda p: torch.zeros(p.shape[:2]))
        for _ in range(12):
            done, _ = env._get_dones()
        self.assertFalse(done.any())

    def test_tumbling_or_body_dragging_cannot_collect_goal_bonus(self):
        env = make_env()
        self.support_at_goal(env)
        env._contacts.data.net_forces_w_history[:, :, 0, 2] = 5
        for _ in range(12):
            env._get_dones()
        self.assertFalse(env._reached_goal.any())
        env._robot.data.projected_gravity_b[:, 2] = 0
        self.assertTrue(env._get_dones()[0].all())
        self.assertFalse(env._reached_goal.any())

    def test_energy_and_contact_penalties_prefer_less_waste_for_equal_progress(self):
        env = make_env()
        env._get_dones()
        baseline = env._get_rewards().clone()
        env._robot.data.applied_torque.fill_(0.5)
        env._robot.data.joint_vel.fill_(2)
        env._contacts.data.net_forces_w_history[:, :, 0, 2] = 5
        wasteful = env._get_rewards()
        self.assertTrue((wasteful < baseline).all())
        self.assertTrue((env.step_metrics["mechanical_power_w"] > 0).all())

    def test_partial_reset_clears_filter_and_goal_state_only_for_reset_env(self):
        env = make_env()
        env._pre_physics_step(torch.ones((2, 18)))
        env._goal_hold_time.fill_(0.10)
        before = {name: getattr(env, name).clone() for name in (
            "_processed_actions", "_actions", "_target_velocity", "_goal_hold_time", "_previous_root_xy",
        )}
        env._reset_idx(torch.tensor([0]))
        for name, old in before.items():
            torch.testing.assert_close(getattr(env, name)[1], old[1])
        self.assertEqual(float(env._goal_hold_time[0]), 0)
        self.assertTrue((env._target_velocity[0] == 0).all())
        torch.testing.assert_close(env._robot.targets[0], env._robot.data.joint_pos[0])

    def test_stationary_rocking_earns_no_positive_motion_reward(self):
        env = make_env()
        # No contact/effort differences: vary only the trajectory and posture.
        total = torch.zeros(2)
        for x, speed, gravity_z in ((0.0004, 0.02, -1.0), (0.0, -0.02, -0.5)):
            env._robot.data.root_pos_w[:, 0] = x
            env._robot.data.root_lin_vel_w[:, 0] = speed
            env._robot.data.projected_gravity_b[:, 2] = gravity_z
            env._get_dones()
            total += env._get_rewards()
        self.assertTrue((total <= 0).all())
        self.assertFalse(hasattr(env, "_best_goal_distance"))

    def test_fixed_evaluation_reset_requires_a_sustained_one_metre_route(self):
        env = make_env()
        env.cfg.evaluation_mode = True
        env._reset_idx(torch.tensor([0]))
        self.assertAlmostEqual(env._goal_data()[0][0].item(), 1.0)
        torch.testing.assert_close(env._target_positions_w[0, :2], torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(env._previous_goal_distance[0].item(), 1.0)

    def test_small_and_command_speed_cycles_are_worse_than_standing(self):
        for speed in (0.002, 0.02):
            env = make_env()
            env.cfg.target_speed, env.cfg.speed_tracking_sigma = 0.02, 0.015
            env._get_dones()
            standing = env._get_rewards().clone()
            cycle = torch.zeros(2)
            for x, velocity in ((speed * env.step_dt, speed), (0.0, -speed)):
                env._robot.data.root_pos_w[:, 0] = x
                env._robot.data.root_lin_vel_w[:, 0] = velocity
                cycle += env._get_rewards()
            self.assertTrue((cycle / 2 <= standing + 1e-7).all())

    def test_shaft_contact_near_goal_is_not_stable_support(self):
        env = make_env()
        self.support_at_goal(env)
        env._robot.data.body_link_pos_w[:, env._feet, 2] = 0.05
        for _ in range(12):
            env._get_dones()
        self.assertFalse(env._reached_goal.any())
        feet, body = env._contact_state()
        self.assertFalse(feet.any())
        self.assertTrue(body.all())

    def test_terminal_rate_saturation_metric_survives_auto_reset(self):
        env = make_env()
        env._pre_physics_step(torch.ones((2, 18)))
        env._get_rewards()
        terminal = env.step_metrics["target_rate_saturation"].clone()
        self.assertTrue((terminal > 0).all())
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.step_metrics["target_rate_saturation"], terminal)
        self.assertEqual(env._rate_saturation[0].item(), 0)

    def test_revised_reward_prefers_advancing_bursts_over_standing(self):
        scores = {}
        for profile in ("legacy_v2", "revised_v3"):
            env = make_env()
            env.cfg.behavior_profile = profile
            env.cfg.target_speed, env.cfg.speed_tracking_sigma = 0.02, 0.015
            env._get_dones()
            standing = env._get_rewards().clone()
            cycle = torch.zeros(2)
            for x, speed in ((0.0016, 0.08), (0.0008, -0.04)):
                env._robot.data.root_pos_w[:, 0] = x
                env._robot.data.root_lin_vel_w[:, 0] = speed
                cycle += env._get_rewards()
            scores[profile] = cycle / 2 - standing
        self.assertTrue((scores["legacy_v2"] < 0).all())
        self.assertTrue((scores["revised_v3"] > 0).all())

    def test_instantaneous_velocity_without_displacement_does_not_improve_revised_reward(self):
        env = make_env()
        env._get_dones()
        standing = env._get_rewards().clone()
        env._robot.data.root_lin_vel_w[:, 0] = env.cfg.target_speed
        unchanged_position = env._get_rewards()
        torch.testing.assert_close(unchanged_position, standing)
        self.assertTrue((env.step_metrics["forward_speed_mps"] > 0).all())
        self.assertTrue((env.step_metrics["step_progress_speed_mps"] == 0).all())

    def test_revised_contact_uses_actual_heel_witness_and_keeps_legacy_reproducible(self):
        offsets = tuple(((0.12, 0.0, 0.0), (0.12, 0.0, 0.01275)) for _ in range(6))
        env = make_env(support_offsets=offsets)
        env._sole_offset = env._sole_offset.clone()
        env._sole_offset[..., 2] = 0.01275
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2
        feet, body = env._contact_state()
        self.assertTrue(feet.all())
        self.assertFalse(body.any())
        env._get_rewards()  # Slip and clearance use the selected heel witness too.
        torch.testing.assert_close(env.step_metrics["mean_foot_clearance_m"], torch.zeros(2))
        env.cfg.behavior_profile = "legacy_v2"
        feet, body = env._contact_state()
        self.assertFalse(feet.any())
        self.assertTrue(body.all())

    def toe_button_env(self, profile):
        offsets = tuple(((0.12, 0.0, 0.0), (0.12, 0.0, 0.01275)) for _ in range(6))
        buttons = tuple((0.12, 0.0, 0.01275) for _ in range(6))
        env = make_env(support_offsets=offsets, behavior_profile=profile, toe_button_offsets=buttons)
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2
        env._robot.data.body_link_lin_vel_w[:, env._feet, 0] = 0.2
        return env

    def test_full_foot_slip_penalizes_heel_sliding_without_changing_button_observations(self):
        for profile in ("revised_v4", "revised_v5", "revised_v6"):
            with self.subTest(profile=profile):
                env = self.toe_button_env(profile)
                feet, body = env._contact_state()
                self.assertFalse(feet.any())
                self.assertFalse(body.any())
                observation = env._get_observations()["policy"]
                self.assertEqual(observation.shape, (2, 140))
                torch.testing.assert_close(observation[:, 84:90], torch.zeros((2, 6)))
                sliding = env._get_rewards().clone()
                expected_slip = 0.04 if profile == "revised_v6" else 0.0
                torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.full((2,), expected_slip))
                torch.testing.assert_close(env.step_metrics["feet_contact"], torch.zeros(2))
                env._robot.data.body_link_lin_vel_w.zero_()
                stationary = env._get_rewards()
                expected_cost = env.cfg.foot_slip_cost_scale * expected_slip * env.step_dt
                torch.testing.assert_close(sliding - stationary, torch.full((2,), expected_cost))
                self.support_at_goal(env)
                for _ in range(12):
                    env._get_dones()
                self.assertFalse(env._reached_goal.any())  # A heel is not a pressed toe button.

    def test_v7_uses_center_of_mass_for_support_velocity(self):
        env = self.toe_button_env("revised_v7_com")
        data = env._robot.data
        data.body_com_pos_w = data.body_link_pos_w.clone()
        data.body_com_pos_w[:, env._feet, 0] += 0.05
        data.body_link_ang_vel_w[:, env._feet, 2] = 1.0
        data.body_link_lin_vel_w[:, env._feet, 0] = 0.0
        data.body_link_lin_vel_w[:, env._feet, 1] = -0.07
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.zeros(2), atol=1e-7, rtol=0)

    def test_full_foot_slip_excludes_airborne_and_unsupported_impacts(self):
        env = self.toe_button_env("revised_v6")
        # One environment is airborne despite a near-ground heel. The other
        # has a tibia impact above the ground, without any supporting witness.
        env._contacts.data.net_forces_w_history[0].zero_()
        env._robot.data.body_link_pos_w[1, env._feet, 2] = 0.05
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.zeros(2))
        torch.testing.assert_close(env.step_metrics["body_contact"], torch.tensor([0.0, 1.0]))

    def test_full_foot_slip_uses_support_point_rotation_and_floor_velocity(self):
        env = self.toe_button_env("revised_v6")
        env._robot.data.body_link_ang_vel_w[:, env._feet, 2] = 1.0
        env._scenario = NS(
            surface_heights=lambda p: torch.zeros(p.shape[:2]),
            floor_velocity=torch.tensor([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        )
        env._get_rewards()
        # The grounded heel has a 0.12 m arm; floor motion cancels only the
        # matching translational component, leaving rotational slip intact.
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.tensor([0.12 ** 2, 0.1 ** 2 + 0.12 ** 2]))

    def test_full_foot_slip_preserves_v5_knee_clearance_penalty(self):
        for profile in ("revised_v5", "revised_v6"):
            env = self.toe_button_env(profile)
            env._robot.data.body_link_pos_w[:, env._knees, 2] = -0.01
            env._get_rewards()
            expected = env.cfg.knee_below_body_penalty_scale * env.step_dt
            torch.testing.assert_close(env._episode_sums["knee_clearance"], torch.full((2,), expected))

    def distal_env(self):
        distal = tuple(((0.12, 0.0, -0.06),) for _ in range(6))
        proximal = tuple(((0.0, 0.0, -0.01),) for _ in range(6))
        hull = tuple(distal[leg] + proximal[leg] for leg in range(6))
        buttons = tuple(points[0] for points in distal)
        env = make_env(support_offsets=hull, behavior_profile="revised_v7", toe_button_offsets=buttons,
                       distal_offsets=distal, proximal_offsets=proximal)
        env._robot.data.body_link_pos_w[:, env._feet, 2] = 0.06
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2
        return env

    def test_distal_support_preserves_button_observation_and_stable_goal(self):
        env = self.distal_env()
        feet, body = env._contact_state()
        self.assertTrue(feet.all())
        self.assertFalse(body.any())
        observation = env._get_observations()["policy"]
        self.assertEqual(observation.shape, (2, 140))
        torch.testing.assert_close(observation[:, 84:90], torch.ones((2, 6)))
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["distal_support_fraction"], torch.ones(2))
        torch.testing.assert_close(env.step_metrics["non_distal_support_fraction"], torch.zeros(2))
        torch.testing.assert_close(env._episode_sums["knee_clearance"], torch.zeros(2))
        self.support_at_goal(env)
        for _ in range(12):
            env._get_dones()
        self.assertTrue(env._reached_goal.all())

    def test_proximal_only_support_is_penalized_and_cannot_complete_goal(self):
        env = self.distal_env()
        # Rotate the tip upwards, leaving the proximal witness at floor height.
        env._robot.data.body_link_quat_w[:, env._feet, 0] = 2 ** -0.5
        env._robot.data.body_link_quat_w[:, env._feet, 2] = -(2 ** -0.5)
        env._robot.data.body_link_pos_w[:, env._feet, 2] = 0.0
        env._robot.data.body_link_lin_vel_w[:, env._feet, 0] = 0.2
        feet, body = env._contact_state()
        self.assertFalse(feet.any())
        self.assertTrue(body.all())
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["distal_support_fraction"], torch.zeros(2))
        torch.testing.assert_close(env.step_metrics["non_distal_support_fraction"], torch.ones(2))
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.full((2,), 0.04))
        expected = env.cfg.non_distal_support_cost_scale * env.step_dt
        torch.testing.assert_close(env._episode_sums["non_distal_support"], torch.full((2,), expected))
        self.support_at_goal(env)
        for _ in range(12):
            env._get_dones()
        self.assertFalse(env._reached_goal.any())

    def test_simultaneous_proximal_and_tip_grounding_keeps_raw_button_but_blocks_goal(self):
        env = self.distal_env()
        env._proximal_foot_offsets = env._proximal_foot_offsets.clone()
        env._proximal_foot_offsets[..., 2] = -0.06
        feet, body = env._contact_state()
        self.assertTrue(feet.all())  # Digital toe switches remain actual switch estimates.
        self.assertTrue(body.all())
        supported, _, invalid = env._distal_contact_state()
        self.assertFalse(supported.any())
        self.assertTrue(invalid.all())
        self.support_at_goal(env)
        for _ in range(12):
            env._get_dones()
        self.assertFalse(env._reached_goal.any())

    def test_knee_clearance_uses_each_knees_terrain_immediately(self):
        env = self.distal_env()
        env._contacts.data.net_forces_w_history.zero_()  # Clearance also guides unloaded knees.
        env._robot.data.body_link_pos_w[:, env._knees, 2] = 0.05
        env._scenario = NS(surface_heights=lambda p: torch.tensor([0.0, 0.035])[:, None].expand(p.shape[:2]),
                           floor_velocity=torch.zeros((2, 3)))
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["mean_knee_ground_clearance_m"], torch.tensor([0.05, 0.015]))
        torch.testing.assert_close(env.step_metrics["knee_ground_clearance_violation_fraction"], torch.tensor([0.0, 1.0]))
        expected = env.cfg.knee_ground_clearance_cost_scale * 0.5 * env.step_dt
        torch.testing.assert_close(env._episode_sums["knee_clearance"], torch.tensor([0.0, expected]))
        torch.testing.assert_close(env.step_metrics["non_distal_support_fraction"], torch.zeros(2))

    def test_distal_profile_rejects_missing_geometry(self):
        with self.assertRaisesRegex(ValueError, "requires distal_foot_support_offsets"):
            make_env(behavior_profile="revised_v7")

    def test_curriculum_requires_existing_distal_support_behavior(self):
        with self.assertRaisesRegex(ValueError, "requires revised_v7"):
            make_env(environment_preset="curriculum")

    def test_curriculum_bounds_use_each_environments_actual_corridor(self):
        env = self.distal_env()
        env._robot.data.root_pos_w[:, 1] = 0.4
        env._scenario = NS(collect_impacts=lambda: None, layout=NS(corridor_width=1.6),
                           corridor_widths=torch.tensor([0.6, 1.6]),
                           surface_heights=lambda p: torch.zeros(p.shape[:2]))
        env._get_dones()
        torch.testing.assert_close(env._out_of_bounds, torch.tensor([True, False]))

    def test_completed_courses_are_snapshotted_before_reset_with_failure_precedence(self):
        env = self.distal_env()
        env.cfg.environment_preset = "curriculum"
        env.cfg.evaluation_mode = True
        env.episode_length_buf[:] = 10
        env.reset_terminated[:] = True
        env._reached_goal[:] = True
        env._robot.data.projected_gravity_b[1, 2] = 0.0
        outcomes = []
        env._scenario = NS(
            episode_metadata=lambda ids: [{"environment_index": int(i), "course_id": f"old_{i}"}
                                          for i in ids],
            record_outcomes=lambda ids, success: outcomes.append((ids.clone(), success.clone())),
            statistics=lambda: {"difficulty/stairs": 0.2},
        )
        env._reset_idx(torch.tensor([1, 0]))
        self.assertEqual([item["course_id"] for item in env.completed_course_records], ["old_0", "old_1"])
        self.assertEqual([item["outcome"] for item in env.completed_course_records], ["goal", "fallen"])
        torch.testing.assert_close(outcomes[0][1], torch.tensor([True, False]))
        self.assertEqual(env.extras["log"]["Curriculum/difficulty/stairs"], 0.2)
        self.assertFalse(env._reached_goal.any())

    def test_manual_or_initial_course_reset_does_not_update_success_rate(self):
        env = self.distal_env()
        env.cfg.environment_preset = "curriculum"
        env.cfg.evaluation_mode = True
        env.episode_length_buf[:] = 10
        env._scenario = NS(statistics=lambda: {},
                           record_outcomes=lambda *args: self.fail("An unfinished episode changed the curriculum"))
        env._reset_idx(torch.tensor([0, 1]))
        self.assertEqual(env.completed_course_records, [])


class StrictContactEnvironmentTests(unittest.TestCase):
    def strict_env(self, *, servo_contact=False):
        distal = tuple(((0.12, 0.0, -0.06),) for _ in range(6))
        proximal = tuple(((0.0, 0.0, -0.01),) for _ in range(6))
        hull = tuple(distal[leg] + proximal[leg] for leg in range(6))
        collision = {"base": ((0.0, 0.0, 0.0),)}
        for leg in range(6):
            for part in ("hip", "femur", "tibia"):
                offset_z = (-0.06 if servo_contact else -0.01) if part == "tibia" else 0.0
                collision[f"leg_{leg}_{part}"] = ((0.0, 0.0, offset_z),)
        env = make_env(
            support_offsets=hull, behavior_profile="revised_v8",
            toe_button_offsets=tuple(points[0] for points in distal),
            distal_offsets=distal, proximal_offsets=proximal, collision_offsets=collision,
        )
        env._robot.data.body_link_pos_w[..., 2] = 0.10
        env._robot.data.body_link_pos_w[:, 0, 2] = 0.132
        env._robot.data.body_link_pos_w[:, env._feet, 2] = 0.06
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2
        return env

    def set_link_contact(self, env, name, *, force=5.0, height=0.0, axis=2, environment=None):
        body = env._robot.body_names.index(name)
        sensor = env._contacts.body_names.index(name)
        selection = slice(None) if environment is None else environment
        env._robot.data.body_link_pos_w[selection, body, 2] = height
        env._contacts.data.net_forces_w_history[selection, 0, sensor, axis] = force

    def airborne_tibias(self, env):
        env._contacts.data.net_forces_w_history.zero_()
        env._robot.data.body_link_pos_w[:, env._feet, 2] = 0.12

    def test_femur_only_support_fails_and_cannot_earn_forward_progress(self):
        env = self.strict_env()
        self.airborne_tibias(env)
        for leg in range(6):
            self.set_link_contact(env, f"leg_{leg}_femur")
        support, body, invalid, loaded, gaps = env._strict_contact_state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(invalid.all()))
        self.assertTrue(bool(loaded.all()))
        torch.testing.assert_close(gaps.amin(-1), torch.zeros(2))
        env._robot.data.root_pos_w[:, 0] += 0.004
        env._robot.data.root_lin_vel_w[:, 0] = env.cfg.target_speed
        terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))
        reward = env._get_rewards()
        self.assertTrue(bool((reward < 0).all()))
        torch.testing.assert_close(env._episode_sums["progress"], torch.zeros(2))
        torch.testing.assert_close(env._episode_sums["goal"], torch.zeros(2))
        torch.testing.assert_close(env.step_metrics["non_distal_collision_fraction"], torch.ones(2))
        torch.testing.assert_close(env.step_metrics["strict_contact_failure"], torch.ones(2))
        torch.testing.assert_close(env.step_metrics["min_non_distal_clearance_m"], torch.zeros(2))

    def test_hip_only_and_base_only_contacts_fail_with_airborne_tibias(self):
        for name in ("leg_3_hip", "base"):
            with self.subTest(name=name):
                env = self.strict_env()
                self.airborne_tibias(env)
                self.set_link_contact(env, name)
                _, body, invalid, loaded, _ = env._strict_contact_state()
                self.assertTrue(bool(body.all()))
                expected_leg_count = 0 if name == "base" else 1
                torch.testing.assert_close(invalid.sum(-1), torch.full((2,), expected_leg_count))
                torch.testing.assert_close(loaded.sum(-1), torch.full((2,), expected_leg_count))
                terminated, _ = env._get_dones()
                self.assertTrue(bool(terminated.all()))

    def test_one_bad_leg_is_not_hidden_by_five_good_distal_contacts(self):
        env = self.strict_env()
        self.set_link_contact(env, "leg_2_femur")
        support, body, invalid, loaded, _ = env._strict_contact_state()
        torch.testing.assert_close(support.sum(-1), torch.full((2,), 5))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(invalid[:, 2].all()))
        torch.testing.assert_close(invalid.sum(-1), torch.ones(2, dtype=torch.long))
        self.assertTrue(bool(loaded.all()))
        buttons, button_body = env._contact_state()
        self.assertFalse(bool(buttons[:, 2].any()))
        torch.testing.assert_close(buttons.sum(-1), torch.full((2,), 5))
        self.assertTrue(bool(button_body.all()))
        env._target_positions_w[:, 0] = 0.02
        for _ in range(12):
            terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))
        self.assertFalse(bool(env._reached_goal.any()))

    def test_tibia_servo_hull_contact_is_rejected_with_a_high_hinge_and_valid_tip(self):
        env = self.strict_env(servo_contact=True)
        hinge_height = env._robot.data.body_link_pos_w[:, env._knees, 2]
        self.assertTrue(bool((hinge_height > env.cfg.knee_ground_clearance_m).all()))
        distal, _, _ = env._distal_contact_state()
        self.assertTrue(bool(distal.all()))
        support, body, invalid, loaded, _ = env._strict_contact_state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(invalid.all()))
        self.assertTrue(bool(loaded.all()))
        terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))

    def test_deep_nonfoot_penetration_fails_even_without_reported_force(self):
        env = self.strict_env()
        self.airborne_tibias(env)
        self.set_link_contact(env, "base", force=0.0, height=-0.003)
        _, body, _, loaded, _ = env._strict_contact_state()
        self.assertTrue(bool(body.all()))
        self.assertFalse(bool(loaded.any()))
        terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))

    def test_deep_proximal_foot_penetration_fails_even_without_reported_force(self):
        env = self.strict_env()
        env._contacts.data.net_forces_w_history.zero_()
        env._proximal_foot_offsets = env._proximal_foot_offsets.clone()
        env._proximal_foot_offsets[..., 2] = -0.07
        _, body, invalid, loaded, gaps = env._strict_contact_state()
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(invalid.all()))
        self.assertTrue(bool(loaded.all()))
        torch.testing.assert_close(gaps.amin(-1), torch.full((2,), -0.01))
        terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["non_distal_collision_fraction"], torch.ones(2))

    def test_nonfinite_non_distal_geometry_cannot_be_classified_as_clear(self):
        for invalid_height in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_height=invalid_height):
                env = self.strict_env()
                femur = env._robot.body_names.index("leg_0_femur")
                env._robot.data.body_link_pos_w[:, femur, 2] = invalid_height
                supported, body, invalid, _, _ = env._strict_contact_state()
                self.assertTrue(bool(body.all()))
                self.assertTrue(bool(invalid[:, 0].all()))
                self.assertFalse(bool(supported[:, 0].any()))
                terminated, _ = env._get_dones()
                self.assertTrue(bool(terminated.all()))
                reward = env._get_rewards()
                self.assertTrue(bool(torch.isfinite(reward).all()))
                self.assertTrue(bool((reward < 0).all()))
                self.assertTrue(bool(torch.isfinite(env.step_metrics["min_non_distal_clearance_m"]).all()))

    def test_valid_distal_support_keeps_slip_cost_when_virtual_button_is_raised(self):
        env = self.strict_env()
        env._toe_button_offsets = env._toe_button_offsets.clone()
        env._toe_button_offsets[..., 2] += 0.02
        env._robot.data.body_link_lin_vel_w[:, env._feet, 0] = 0.2
        supported, body, _, _, _ = env._strict_contact_state()
        buttons, _ = env._contact_state()
        self.assertTrue(bool(supported.all()))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(buttons.any()))
        env._get_rewards()
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.full((2,), 0.04))
        expected = env.cfg.foot_slip_cost_scale * 0.04 * env.step_dt
        torch.testing.assert_close(env._episode_sums["foot_slip"], torch.full((2,), expected))

    def test_unloaded_near_floor_swing_gets_clearance_cost_without_contact_failure(self):
        env = self.strict_env()
        self.airborne_tibias(env)
        self.set_link_contact(env, "leg_0_femur", force=0.0, height=0.004)
        _, body, invalid, loaded, _ = env._strict_contact_state()
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(invalid.any()))
        self.assertFalse(bool(loaded.any()))
        terminated, _ = env._get_dones()
        self.assertFalse(bool(terminated.any()))
        env._get_rewards()
        self.assertTrue(bool((env._episode_sums["non_distal_clearance"] < 0).all()))
        torch.testing.assert_close(env.step_metrics["non_distal_collision_fraction"], torch.zeros(2))
        torch.testing.assert_close(env.step_metrics["strict_contact_failure"], torch.zeros(2))

    def test_clean_distal_support_can_make_progress_and_complete_goal(self):
        env = self.strict_env()
        support, body, invalid, loaded, _ = env._strict_contact_state()
        self.assertTrue(bool(support.all()))
        self.assertTrue(bool(loaded.all()))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(invalid.any()))
        env._robot.data.root_pos_w[:, 0] += 0.004
        env._get_rewards()
        self.assertTrue(bool((env._episode_sums["progress"] > 0).all()))
        env._target_positions_w[:, 0] = 0.02
        for _ in range(12):
            env._get_dones()
        self.assertTrue(bool(env._reached_goal.all()))
        self.assertFalse(bool(env._strict_contact_failure.any()))

    def test_no_contact_flight_is_allowed(self):
        env = self.strict_env()
        self.airborne_tibias(env)
        support, body, invalid, loaded, gaps = env._strict_contact_state()
        self.assertFalse(bool(support.any()))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(invalid.any()))
        self.assertFalse(bool(loaded.any()))
        self.assertTrue(bool((gaps > env.cfg.non_distal_ground_clearance_m).all()))
        terminated, _ = env._get_dones()
        self.assertFalse(bool(terminated.any()))

    def test_contact_failure_remains_until_reset_and_partial_reset_is_local(self):
        env = self.strict_env()
        self.set_link_contact(env, "leg_0_femur")
        env._get_dones()
        env._robot.data.body_link_pos_w[:, env._robot.body_names.index("leg_0_femur"), 2] = 0.10
        env._contacts.data.net_forces_w_history.zero_()
        env._contacts.data.net_forces_w_history[:, :, env._sensor_feet, 2] = 2
        _, body, _, _, _ = env._strict_contact_state()
        self.assertFalse(bool(body.any()))
        terminated, _ = env._get_dones()
        self.assertTrue(bool(terminated.all()))
        self.assertTrue(bool(env._strict_contact_failure.all()))
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env._strict_contact_failure, torch.tensor([False, True]))
        terminated, _ = env._get_dones()
        torch.testing.assert_close(terminated, torch.tensor([False, True]))

    def test_reward_path_latches_contact_failure_without_dones_called_first(self):
        env = self.strict_env()
        self.set_link_contact(env, "leg_4_femur")
        env._robot.data.root_pos_w[:, 0] += 0.004
        env._reached_goal[:] = True
        reward = env._get_rewards()
        self.assertTrue(bool((reward < 0).all()))
        self.assertTrue(bool(env._strict_contact_failure.all()))
        torch.testing.assert_close(env._episode_sums["progress"], torch.zeros(2))
        torch.testing.assert_close(env._episode_sums["goal"], torch.zeros(2))

    def test_strict_contact_mapping_uses_sensor_names_with_reversed_sensor_order(self):
        original_init = Parent.__init__

        def reordered_sensor_init(parent_self, *args, **kwargs):
            original_init(parent_self, *args, **kwargs)
            parent_self._contacts.body_names.reverse()
            parent_self._contacts.data.net_forces_w_history = (
                parent_self._contacts.data.net_forces_w_history.flip(2)
            )

        with patch.object(Parent, "__init__", reordered_sensor_init):
            env = self.strict_env()
        self.assertNotEqual(env._contacts.body_names, env._robot.body_names)
        self.set_link_contact(env, "leg_4_femur", environment=0)
        support, body, invalid, loaded, _ = env._strict_contact_state()
        torch.testing.assert_close(body, torch.tensor([True, False]))
        torch.testing.assert_close(invalid.sum(-1), torch.tensor([1, 0]))
        self.assertTrue(bool(invalid[0, 4]))
        self.assertFalse(bool(support[0, 4]))
        self.assertTrue(bool(support[1].all()))
        self.assertTrue(bool(loaded.all()))
        terminated, _ = env._get_dones()
        torch.testing.assert_close(terminated, torch.tensor([True, False]))


if __name__ == "__main__":
    unittest.main()
