"""Configuration for the DKSH spiderbot navigation environment."""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dksh_isaaclab.assets import SPIDERBOT_CFG


@configclass
class SpiderNavigationEnvCfg(DirectRLEnvCfg):
    """Flat-ground goal navigation with a fully articulated eight-legged robot."""

    decimation = 4
    seed = 42
    episode_length_s = 20.0
    action_space = 24
    observation_space = 84
    state_space = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
    )
    viewer = ViewerCfg(
        eye=(1.2, 1.2, 0.8),
        lookat=(0.0, 0.0, 0.10),
        origin_type="env",
        env_index=0,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=7.0, replicate_physics=True)
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    action_scale = 0.55
    joint_velocity_scale = 0.10
    goal_min_distance = 1.5
    goal_max_distance = 3.5
    goal_radius = 0.30
    max_distance_from_origin = 5.5
    minimum_base_height = 0.07

    progress_reward_scale = 8.0
    velocity_to_goal_reward_scale = 0.35
    heading_reward_scale = 0.04
    upright_reward_scale = 0.04
    action_rate_penalty_scale = -0.003
    torque_penalty_scale = -0.0002
    vertical_velocity_penalty_scale = -0.02
    goal_reward = 12.0
    failure_penalty = -5.0
