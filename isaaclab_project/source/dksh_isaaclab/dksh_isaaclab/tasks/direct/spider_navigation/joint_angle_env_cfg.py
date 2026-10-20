"""Motor-level policies that choose absolute joint angles, without a reference gait."""

from isaaclab.utils import configclass
from dksh_isaaclab.assets import SPIDERBOT_CFG
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .spider_navigation_env_cfg import SpiderNavigationEnvCfg
from .agents.rsl_rl_ppo_cfg import SpiderNavigationPPORunnerCfg
from .mg90s_env_cfg import mg90s_robot_cfg, mixed_servo_robot_cfg, MG90S_NO_LOAD_SPEED
from dksh_isaaclab.assets.servo_specs import (
    ACTUATOR_CONTRACT,
    MG996R_MASS_KG,
    MG996R_NO_LOAD_SPEED_RAD_S,
    MG996R_REFERENCE_VOLTAGE_V,
    MG996R_STALL_TORQUE_NM,
    MG90S_MASS_KG,
    MG90S_REFERENCE_VOLTAGE_V,
    MG90S_STALL_TORQUE_NM,
    RC920DMG_MASS_KG,
    RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
    RC920DMG_REFERENCE_VOLTAGE_V,
    RC920DMG_STALL_TORQUE_NM_5V,
)

CAD_ANGLE_RANGES = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (-0.40, 1.40)}
CAD_SOLE_OFFSET = (0.016120809, 0.0, -0.0803304305)


@configclass
class JointAngleNavigationEnvCfg(SpiderNavigationEnvCfg):
    action_semantics = "absolute_joint_angles_v1"
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuator_contract = "mg996r_4v8_v1"
    servo_model = "MG996R"
    supply_voltage = MG996R_REFERENCE_VOLTAGE_V
    servo_reference_voltage_v = MG996R_REFERENCE_VOLTAGE_V
    servo_stall_torque_nm = MG996R_STALL_TORQUE_NM
    servo_effort_limit_nm = 0.75 * MG996R_STALL_TORQUE_NM
    servo_mass_kg = MG996R_MASS_KG
    observation_space = 94
    episode_length_s = 60.0
    # Working limits intersect the asset limits. The knee stays on its assembled
    # bend branch; these are simulation settings, not measured PWM calibration.
    joint_angle_ranges = {"hip": (-1.40, 1.40), "femur": (-1.10, 1.40), "tibia": (0.10, 1.45)}
    target_rate_limit = 3.0  # rad/s, below the modeled MG996R no-load speed
    target_smoothing = 0.35
    servo_velocity_limit = MG996R_NO_LOAD_SPEED_RAD_S
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
    actuator_contract = "mg90s_4v8_historical_v1"
    servo_model = "MG90S"
    supply_voltage = MG90S_REFERENCE_VOLTAGE_V
    servo_reference_voltage_v = MG90S_REFERENCE_VOLTAGE_V
    servo_stall_torque_nm = MG90S_STALL_TORQUE_NM
    servo_effort_limit_nm = 0.75 * MG90S_STALL_TORQUE_NM
    servo_mass_kg = MG90S_MASS_KG
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


def _mg996r_hip_cad6_robot():
    robot = cad_spiderbot_cfg(6)
    robot.actuators = mixed_servo_robot_cfg().actuators
    robot.init_state.pos = (0, 0, 0.132)
    robot.init_state.joint_pos = {".*_hip_joint": 0.0, ".*_femur_joint": 0.60, ".*_tibia_joint": -0.20}
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class MG996RHipCad6JointAngleEnvCfg(MG90SCad6JointAngleEnvCfg):
    """Six MG996R hip servos and twelve provisional RC920DMG leg servos."""
    robot = _mg996r_hip_cad6_robot()
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
        "Existing CAD6 link mass/inertia estimates retained; mixed-servo mass is recorded, "
        "not redistributed into link inertias"
    )


@configclass
class JointAnglePPORunnerCfg(SpiderNavigationPPORunnerCfg):
    experiment_name = "dksh_joint_angles"
    policy = SpiderNavigationPPORunnerCfg().policy.copy()
    policy.init_noise_std = 0.12
