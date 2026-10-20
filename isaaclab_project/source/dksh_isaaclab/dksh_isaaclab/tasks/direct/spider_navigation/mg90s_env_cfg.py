"""Provisional mixed-servo feasibility and residual wave-gait training.

Uses the existing primitive geometry and link inertias (2.91816 kg total).
MG996R drives the body-coxa joints. RC920DMG drives the coxa-femur and
femur-tibia joints using its conservative 5 V specification.
Servo masses are known; the allocation of old servo/structure masses is not.
Do not add/subtract servo masses blindly or treat this as a measured robot twin.
"""

from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .wave_gait import STANCE_DEPTH, STANCE_FEMUR, STANCE_TIBIA
from dksh_isaaclab.assets import SPIDERBOT_CFG
from .agents.rsl_rl_ppo_cfg import SpiderNavigationPPORunnerCfg
from .payload_mass import L2_MOUNT_POSITION_B, L2_PAYLOAD_MASS_KG, L2_PAYLOAD_SIZE_M
from .servo_specs import (
    ACTUATOR_CONTRACT,
    MG996R_MASS_KG,
    MG996R_NO_LOAD_SPEED_RAD_S,
    MG996R_REFERENCE_VOLTAGE_V,
    MG996R_STALL_TORQUE_NM,
    PROVISIONAL_EFFORT_DERATING,
    RC920DMG_MASS_KG,
    RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
    RC920DMG_REFERENCE_VOLTAGE_V,
    RC920DMG_SIZE_M,
    RC920DMG_STALL_TORQUE_NM_5V,
)

# Backward-compatible names used by older evaluation helpers.
HIP_SERVO_NO_LOAD_SPEED = MG996R_NO_LOAD_SPEED_RAD_S
RC920DMG_STALL_TORQUE = RC920DMG_STALL_TORQUE_NM_5V
RC920DMG_NO_LOAD_SPEED = RC920DMG_NO_LOAD_SPEED_RAD_S_5V


def mixed_servo_robot_cfg():
    robot = SPIDERBOT_CFG.copy()
    robot.init_state.pos = (0.0, 0.0, STANCE_DEPTH + 0.004)
    robot.init_state.joint_pos = {
        ".*_hip_joint": 0.0, ".*_femur_joint": STANCE_FEMUR, ".*_tibia_joint": STANCE_TIBIA,
    }
    robot.actuators = {
        "mg996r_4v8_hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            saturation_effort=MG996R_STALL_TORQUE_NM,
            # A provisional 25% derating, NOT a published continuous torque rating.
            effort_limit=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            effort_limit_sim=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            velocity_limit=MG996R_NO_LOAD_SPEED_RAD_S,
            velocity_limit_sim=MG996R_NO_LOAD_SPEED_RAD_S,
            stiffness=3.0, damping=0.06, armature=0.0002, friction=0.0,
        ),
        "rc920dmg_5v_leg": DCMotorCfg(
            joint_names_expr=[".*_femur_joint", ".*_tibia_joint"],
            saturation_effort=RC920DMG_STALL_TORQUE,
            # Apply the same provisional derating until continuous torque is measured.
            effort_limit=PROVISIONAL_EFFORT_DERATING * RC920DMG_STALL_TORQUE,
            effort_limit_sim=PROVISIONAL_EFFORT_DERATING * RC920DMG_STALL_TORQUE,
            velocity_limit=RC920DMG_NO_LOAD_SPEED,
            velocity_limit_sim=RC920DMG_NO_LOAD_SPEED,
            # Controller gains/armature remain provisional and require hardware tuning.
            stiffness=3.0, damping=0.06, armature=0.0002, friction=0.0,
        ),
    }
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


def mg90s_robot_cfg():
    """Backward-compatible name for the current mixed-servo robot config."""
    return mixed_servo_robot_cfg()


@configclass
class MG90SWalkEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=True)
    # Both colliding surfaces use the same explicit material. Average of two
    # identical coefficients gives the intended 0.4/0.6/0.8 comparison value.
    ground_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.6, dynamic_friction=0.6, restitution=0.0,
        friction_combine_mode="average", restitution_combine_mode="average",
    )
    observation_contract = "simulation_only_privileged_state_v1"
    foot_contact_sensor = None
    simple_collisions = False
    assumed_foot_radius_m = 0.005
    # Keep the playback viewport user-controlled.  ``asset_root`` rewrites the
    # camera pose on every render frame, which prevents orbit/pan navigation.
    viewer = ViewerCfg(eye=(1.0, 1.0, 0.65), lookat=(0, 0, 0.15), origin_type="world")
    robot = mixed_servo_robot_cfg()
    episode_length_s = 20.0
    action_space = 24
    observation_space = 86
    state_space = 0
    supply_voltage = RC920DMG_REFERENCE_VOLTAGE_V
    actuator_contract = ACTUATOR_CONTRACT
    hip_servo_model = 'MG996R'
    leg_servo_model = 'RC920DMG'
    hip_servo_reference_voltage_v = MG996R_REFERENCE_VOLTAGE_V
    leg_servo_reference_voltage_v = RC920DMG_REFERENCE_VOLTAGE_V
    hip_servo_mass_kg = MG996R_MASS_KG
    leg_servo_mass_kg = RC920DMG_MASS_KG
    known_servo_mass_per_leg_kg = MG996R_MASS_KG + 2 * RC920DMG_MASS_KG
    leg_servo_size_m = RC920DMG_SIZE_M
    mass_assumption = (
        "Primitive links 2.91816 kg + L2 0.230 kg; known mixed-servo masses "
        "are recorded but not redistributed across provisional link inertias"
    )
    lidar_payload_mass_kg = L2_PAYLOAD_MASS_KG
    lidar_payload_size_m = L2_PAYLOAD_SIZE_M
    lidar_mount_position_b = L2_MOUNT_POSITION_B
    gait_period = 2.0
    gait_lift = 0.006
    settling_seconds = 1.0
    command_speed_min = 0.004
    command_speed_max = 0.010
    action_scale = 0.035
    action_smoothing = 0.2
    target_rate_limit = 1.5
    minimum_height = 0.15
    speed_tracking_reward_scale = 2.0
    speed_tracking_sigma = 0.010
    forward_progress_reward_scale = 60.0


@configclass
class MG90SWalkRunnerCfg(SpiderNavigationPPORunnerCfg):
    experiment_name = "dksh_mg90s_walk"
    run_name = "4v8_mass_unmeasured"
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 50

    def __post_init__(self):
        self.policy = self.policy.copy()
        self.algorithm = self.algorithm.copy()
        self.policy.init_noise_std = 0.15
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.schedule = "fixed"
