"""Motor-level policies that choose absolute joint angles, without a reference gait."""

import math
from pathlib import Path

from isaaclab.utils import configclass
from dksh_isaaclab.assets import SPIDERBOT_CFG
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .spider_navigation_env_cfg import SpiderNavigationEnvCfg
from .agents.rsl_rl_ppo_cfg import SpiderNavigationPPORunnerCfg
from .mg90s_env_cfg import historical_mg90s_robot_cfg as mg90s_robot_cfg, mixed_servo_robot_cfg
from .servo_specs import (
    ACTUATOR_CONTRACT,
    MG996R_MASS_KG,
    MG996R_NO_LOAD_SPEED_RAD_S,
    MG996R_REFERENCE_VOLTAGE_V,
    MG996R_STALL_TORQUE_NM,
    MG90S_MASS_KG,
    MG90S_NO_LOAD_SPEED_RAD_S,
    MG90S_REFERENCE_VOLTAGE_V,
    MG90S_STALL_TORQUE_NM,
    RC920DMG_MASS_KG,
    RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
    RC920DMG_REFERENCE_VOLTAGE_V,
    RC920DMG_STALL_TORQUE_NM_5V,
)

CAD_ANGLE_RANGES = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (-0.40, 1.40)}
CAD_SOLE_OFFSET = (0.016120809, 0.0, -0.0803304305)
# Fixed-foot origins in main's normalized Onshape URDF, in policy leg order.
ONSHAPE_SOLE_OFFSETS = (
    (0.137, 0.010, -0.01922),
    (0.137, 0.010, -0.01922),
    (0.137, 0.010, -0.01922),
    (0.137, 0.010, -0.01922),
    (0.010, -0.137, -0.01922),
    (0.0150579, -0.136537, -0.01922),
)


def _cad6_joint_angle_robot():
    robot = cad_spiderbot_cfg(6)
    robot.init_state.pos = (0.0, 0.0, 0.1375)
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class JointAngleNavigationEnvCfg(SpiderNavigationEnvCfg):
    action_semantics = "absolute_joint_angles_v2"
    terrain_observation_contract = "terrain_full48_all_presets_v2"
    observation_contract = "joint_angles_targets_target_velocity_terrain48_v2"
    reward_contract = "signed_progress_sustained_speed_v2"
    contact_contract = "support_surface_vertical_sole_v2"
    control_contract = "absolute_targets_rate_limited_observed_velocity_v2"
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuator_contract = "mg996r_4v8_v1"
    servo_model = "MG996R"
    supply_voltage = MG996R_REFERENCE_VOLTAGE_V
    servo_reference_voltage_v = MG996R_REFERENCE_VOLTAGE_V
    servo_stall_torque_nm = MG996R_STALL_TORQUE_NM
    servo_effort_limit_nm = 0.75 * MG996R_STALL_TORQUE_NM
    servo_mass_kg = MG996R_MASS_KG
    observation_space = 166
    state_space = 0
    use_asset_joint_limits = False
    enable_self_collisions = True
    episode_length_s = 120.0
    joint_reset_noise = 0.0
    evaluation_mode = False
    evaluation_goal_distance = 1.0
    evaluation_min_speed_fraction = 0.5
    evaluation_episode_length_s = 90.0
    max_initial_action_saturation = 0.01
    # Working limits intersect the asset limits. The knee stays on its assembled
    # bend branch; these are simulation settings, not measured PWM calibration.
    joint_angle_ranges = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (0.10, 1.45)}
    target_rate_limit = 3.0  # rad/s, below the modeled MG996R no-load speed
    target_smoothing = 0.35
    servo_velocity_limit = MG996R_NO_LOAD_SPEED_RAD_S
    sole_offset = (0.120, 0.0, 0.0)
    contact_threshold = 1.0
    sole_contact_height_tolerance = 0.012
    minimum_support_feet = 3
    goal_hold_seconds = 0.20
    goal_min_distance = 0.5
    goal_max_distance = 1.5
    goal_radius = 0.08
    goal_reward = 5.0
    failure_penalty = -100.0
    target_speed = 0.10
    speed_tracking_sigma = 0.06
    progress_reward_scale = 25.0
    speed_cost_scale = -0.10
    upright_cost_scale = -2.0
    joint_limit_cost_scale = -1.0
    joint_velocity_cost_scale = -0.05
    torque_cost_scale = -0.05
    power_cost_scale = -0.02
    target_acceleration_cost_scale = -0.02
    body_contact_cost_scale = -2.0
    foot_slip_cost_scale = -2.0
    angular_velocity_cost_scale = -0.10
    stillness_penalty_scale = -0.02
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
    episode_length_s = 180.0


@configclass
class Cad6JointAngleEnvCfg(Cad8JointAngleEnvCfg):
    robot = _cad6_joint_angle_robot()
    action_space = 18
    observation_space = 140
    joint_angle_ranges = {part: (-math.pi / 2, math.pi / 2) for part in ("hip", "femur", "tibia")}
    sole_offset = ONSHAPE_SOLE_OFFSETS
    use_asset_joint_limits = True
    enable_self_collisions = True
    robot_width = 0.56


def _mg90s_cad_robot(leg_count):
    robot = cad_spiderbot_cfg(leg_count)
    if leg_count == 6:
        # Archived all-MG90S v1 policies keep their original geometry and zero.
        robot.spawn.usd_path = str(Path(robot.spawn.usd_path).parent.parent / "spiderbot_6leg/spiderbot_6leg.usd")
    robot.actuators = mg90s_robot_cfg().actuators
    # Static reset pose only. It is never added to the policy's actions.
    robot.init_state.pos = (0, 0, 0.132)
    robot.init_state.joint_pos = {".*_hip_joint": 0.0, ".*_femur_joint": 0.60, ".*_tibia_joint": -0.20}
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class MG90SJointAngleEnvCfg(JointAngleNavigationEnvCfg):
    robot = mg90s_robot_cfg()
    actuator_contract = "mg90s_4v8_historical_v1"
    servo_model = "MG90S"
    supply_voltage = MG90S_REFERENCE_VOLTAGE_V
    servo_reference_voltage_v = MG90S_REFERENCE_VOLTAGE_V
    servo_stall_torque_nm = MG90S_STALL_TORQUE_NM
    servo_effort_limit_nm = 0.75 * MG90S_STALL_TORQUE_NM
    servo_mass_kg = MG90S_MASS_KG
    target_rate_limit = 2.0
    servo_velocity_limit = MG90S_NO_LOAD_SPEED_RAD_S
    minimum_base_height = 0.09
    target_speed = 0.02
    speed_tracking_sigma = 0.015
    goal_min_distance = 0.5
    goal_max_distance = 1.5
    goal_radius = 0.08
    episode_length_s = 240.0


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
    observation_space = 140


def _mg996r_hip_cad6_robot():
    robot = cad_spiderbot_cfg(6)
    robot.actuators = mixed_servo_robot_cfg().actuators
    # q=0 hull bottom is z=-0.126578m: leave >10mm clearance at reset.
    robot.init_state.pos = (0.0, 0.0, 0.1375)
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class MG996RHipCad6JointAngleEnvCfg(MG90SCad6JointAngleEnvCfg):
    """Six MG996R hip servos and twelve provisional RC920DMG leg servos."""
    robot = _mg996r_hip_cad6_robot()
    joint_angle_ranges = Cad6JointAngleEnvCfg().joint_angle_ranges
    sole_offset = ONSHAPE_SOLE_OFFSETS
    use_asset_joint_limits = True
    enable_self_collisions = True
    robot_width = 0.56
    servo_model = None
    servo_reference_voltage_v = None
    servo_stall_torque_nm = None
    servo_effort_limit_nm = None
    servo_mass_kg = None
    target_rate_limit = 1.5
    servo_velocity_limit = tuple(
        speed
        for _ in range(6)
        for speed in (MG996R_NO_LOAD_SPEED_RAD_S,
                      RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
                      RC920DMG_NO_LOAD_SPEED_RAD_S_5V)
    )
    actuator_contract = ACTUATOR_CONTRACT
    supply_voltage = None  # Distinct 4.8 V / 5.0 V reference assumptions; rail topology unverified.
    hip_servo_model = "MG996R"
    leg_servo_model = "RC920DMG"
    hip_servo_reference_voltage_v = MG996R_REFERENCE_VOLTAGE_V
    leg_servo_reference_voltage_v = RC920DMG_REFERENCE_VOLTAGE_V
    hip_servo_stall_torque_nm = MG996R_STALL_TORQUE_NM
    leg_servo_stall_torque_nm = RC920DMG_STALL_TORQUE_NM_5V
    hip_servo_effort_limit_nm = 0.75 * MG996R_STALL_TORQUE_NM
    leg_servo_effort_limit_nm = 0.75 * RC920DMG_STALL_TORQUE_NM_5V
    hip_servo_mass_kg = MG996R_MASS_KG
    leg_servo_mass_kg = RC920DMG_MASS_KG
    known_servo_mass_per_leg_kg = MG996R_MASS_KG + 2 * RC920DMG_MASS_KG
    mass_assumption = (
        "Onshape CAD6 links 1.792339 kg + L2 0.230 kg; known mixed-servo masses "
        "are recorded but not redistributed across provisional link inertias"
    )


@configclass
class JointAnglePPORunnerCfg(SpiderNavigationPPORunnerCfg):
    experiment_name = "dksh_joint_angles"
    policy = SpiderNavigationPPORunnerCfg().policy.copy()
    # Runner derives the exact per-asset cap from limits/rate/smoothing.
    policy.init_noise_std = 0.02
    num_steps_per_env = 128
    algorithm = SpiderNavigationPPORunnerCfg().algorithm.copy()
    algorithm.gamma = 0.9995
    algorithm.lam = 0.98
    algorithm.entropy_coef = 0.0001
