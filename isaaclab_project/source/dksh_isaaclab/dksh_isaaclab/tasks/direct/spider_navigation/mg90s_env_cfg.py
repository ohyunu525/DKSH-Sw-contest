"""Provisional MG90S 4.8 V feasibility and residual wave-gait training.

Uses the existing primitive geometry and link inertias (2.91816 kg total).
Servo mass is known; the allocation of old servo/structure masses is not.
Do not subtract an assumed MG996R mass or treat this as a measured robot twin.
"""

import math
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from .wave_gait import STANCE_DEPTH, STANCE_FEMUR, STANCE_TIBIA
from dksh_isaaclab.assets import SPIDERBOT_CFG
from .agents.rsl_rl_ppo_cfg import SpiderNavigationPPORunnerCfg

MG90S_STALL_TORQUE = 1.8 * 0.0980665
MG90S_NO_LOAD_SPEED = math.radians(60) / 0.10


def mg90s_robot_cfg():
    robot = SPIDERBOT_CFG.copy()
    robot.init_state.pos = (0.0, 0.0, STANCE_DEPTH + 0.004)
    robot.init_state.joint_pos = {
        ".*_hip_joint": 0.0, ".*_femur_joint": STANCE_FEMUR, ".*_tibia_joint": STANCE_TIBIA,
    }
    robot.actuators = {
        "mg90s_4v8": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            saturation_effort=MG90S_STALL_TORQUE,
            # A provisional 25% derating, NOT a published continuous torque rating.
            effort_limit=0.75 * MG90S_STALL_TORQUE,
            effort_limit_sim=0.75 * MG90S_STALL_TORQUE,
            velocity_limit=MG90S_NO_LOAD_SPEED,
            velocity_limit_sim=MG90S_NO_LOAD_SPEED,
            stiffness=3.0, damping=0.06, armature=0.0002, friction=0.0,
        )
    }
    return robot.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class MG90SWalkEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(1.0, 1.0, 0.65), lookat=(0, 0, 0.15), origin_type="asset_root", asset_name="robot")
    robot = mg90s_robot_cfg()
    episode_length_s = 20.0
    action_space = 24
    observation_space = 86
    state_space = 0
    supply_voltage = 4.8
    servo_mass_kg = 0.0134
    mass_assumption = "Existing primitive link masses retained: 2.91816 kg; hardware mass unmeasured"
    gait_period = 2.0
    gait_lift = 0.006
    settling_seconds = 1.0
    command_speed_min = 0.004
    command_speed_max = 0.010
    action_scale = 0.035
    action_smoothing = 0.2
    target_rate_limit = 1.5
    minimum_height = 0.15


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
