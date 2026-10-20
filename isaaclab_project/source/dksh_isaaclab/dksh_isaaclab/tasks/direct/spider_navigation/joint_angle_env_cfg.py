"""Motor-level policies that choose absolute joint angles, without a reference gait."""

import math

from isaaclab.utils import configclass
from dksh_isaaclab.assets import SPIDERBOT_CFG
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .spider_navigation_env_cfg import SpiderNavigationEnvCfg
from .agents.rsl_rl_ppo_cfg import SpiderNavigationPPORunnerCfg
from .mg90s_env_cfg import mg90s_robot_cfg, MG90S_NO_LOAD_SPEED

CAD_ANGLE_RANGES = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (-0.40, 1.40)}
CAD_SOLE_OFFSET = (0.016120809, 0.0, -0.0803304305)


@configclass
class JointAngleNavigationEnvCfg(SpiderNavigationEnvCfg):
    action_semantics = "absolute_joint_angles_v1"
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    observation_space = 94
    episode_length_s = 60.0
    # Working limits intersect the asset limits. The knee stays on its assembled
    # bend branch; these are simulation settings, not measured PWM calibration.
    joint_angle_ranges = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (0.10, 1.45)}
    target_rate_limit = 3.0  # rad/s, below the modeled MG996R no-load speed
    target_smoothing = 0.35
    servo_velocity_limit = math.radians(400)
    sole_offset = (0.120, 0.0, 0.0)
    contact_threshold = 1.0
    minimum_support_feet = 3
    goal_hold_seconds = 0.20
    goal_min_distance = 0.35
    goal_max_distance = 1.0
    goal_radius = 0.15
    target_speed = 0.10
    speed_tracking_sigma = 0.06
    progress_reward_scale = 25.0
    speed_reward_scale = 2.0
    upright_cost_scale = -2.0
    joint_limit_cost_scale = -1.0
    joint_velocity_cost_scale = -0.05
    torque_cost_scale = -0.05
    power_cost_scale = -0.02
    target_acceleration_cost_scale = -0.02
    body_contact_cost_scale = -2.0
    foot_slip_cost_scale = -2.0
    angular_velocity_cost_scale = -0.10
    stillness_penalty_scale = -0.15
    # Rewards never require matching a static pose or a periodic gait.


@configclass
class Cad8JointAngleEnvCfg(JointAngleNavigationEnvCfg):
    robot = cad_spiderbot_cfg(8).replace(prim_path="/World/envs/env_.*/Robot")
    joint_angle_ranges = CAD_ANGLE_RANGES
    sole_offset = CAD_SOLE_OFFSET
    minimum_base_height = 0.045
    robot_width = 0.46
    robot_height = 0.24
    target_speed = 0.06
    episode_length_s = 90.0


@configclass
class Cad6JointAngleEnvCfg(Cad8JointAngleEnvCfg):
    robot = cad_spiderbot_cfg(6).replace(prim_path="/World/envs/env_.*/Robot")
    action_space = 18
    observation_space = 74


def _mg90s_cad_robot(leg_count):
    robot = cad_spiderbot_cfg(leg_count)
    robot.actuators = mg90s_robot_cfg().actuators
    # Static reset pose only. It is never added to the policy's actions.
    robot.init_state.pos = (0, 0, 0.132)
    robot.init_state.joint_pos = {".*_hip_joint": 0.0, ".*_femur_joint": 0.60, ".*_tibia_joint": -0.20}
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class MG90SJointAngleEnvCfg(JointAngleNavigationEnvCfg):
    robot = mg90s_robot_cfg()
    target_rate_limit = 2.0
    servo_velocity_limit = MG90S_NO_LOAD_SPEED
    minimum_base_height = 0.09
    target_speed = 0.02
    speed_tracking_sigma = 0.015
    goal_min_distance = 0.08
    goal_max_distance = 0.25
    goal_radius = 0.04
    episode_length_s = 180.0


@configclass
class MG90SCad8JointAngleEnvCfg(MG90SJointAngleEnvCfg):
    robot = _mg90s_cad_robot(8)
    joint_angle_ranges = CAD_ANGLE_RANGES
    sole_offset = CAD_SOLE_OFFSET
    minimum_base_height = 0.06
    robot_width = 0.46
    robot_height = 0.24


@configclass
class MG90SCad6JointAngleEnvCfg(MG90SCad8JointAngleEnvCfg):
    robot = _mg90s_cad_robot(6)
    action_space = 18
    observation_space = 74


@configclass
class JointAnglePPORunnerCfg(SpiderNavigationPPORunnerCfg):
    experiment_name = "dksh_joint_angles"
    policy = SpiderNavigationPPORunnerCfg().policy.copy()
    policy.init_noise_std = 0.12
