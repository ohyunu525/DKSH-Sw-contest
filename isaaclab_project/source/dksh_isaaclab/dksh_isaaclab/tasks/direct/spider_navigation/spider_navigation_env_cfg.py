"""Configuration for the DKSH spiderbot navigation environment."""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dksh_isaaclab.assets import SPIDERBOT_CFG
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg


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

    # Keep early exploration inside a range the MG996R-powered stance can recover from.
    action_scale = 0.30
    joint_velocity_scale = 0.10
    goal_min_distance = 0.75
    goal_max_distance = 2.50
    goal_radius = 0.30
    max_distance_from_origin = 5.5
    minimum_base_height = 0.09

    progress_reward_scale = 10.0
    velocity_to_goal_reward_scale = 0.50
    heading_reward_scale = 0.10
    upright_reward_scale = 0.25
    action_rate_penalty_scale = -0.015
    torque_penalty_scale = -0.0003
    vertical_velocity_penalty_scale = -0.05
    goal_reward = 20.0
    failure_penalty = -3.0


@configclass
class SpiderCad8NavigationEnvCfg(SpiderNavigationEnvCfg):
    """Eight-legged assembly built from the user's CAD leg meshes."""

    robot = cad_spiderbot_cfg(8).replace(prim_path='/World/envs/env_.*/Robot')
    minimum_base_height = 0.045
    action_scale = 0.20


@configclass
class SpiderCad6NavigationEnvCfg(SpiderCad8NavigationEnvCfg):
    """Six-legged version, with 18 actions and 66 observations."""

    action_space = 18
    observation_space = 66
    robot = cad_spiderbot_cfg(6).replace(prim_path='/World/envs/env_.*/Robot')
