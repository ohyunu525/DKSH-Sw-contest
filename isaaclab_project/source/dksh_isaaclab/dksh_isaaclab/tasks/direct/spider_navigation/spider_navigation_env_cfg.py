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
    """Selectable obstacle courses with a fully articulated spiderbot."""

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
        # Track the robot root during playback.  A static environment camera can
        # leave the spawn point behind a corridor wall, making the policy appear
        # to run without a visible robot.
        eye=(1.5, -1.5, 1.0),
        lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=7.0, replicate_physics=True)
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Flat retains the original observation/checkpoint contract. Other presets add
    # the same 32 terrain/hazard observations so policies can transfer between them.
    environment_preset = "flat"
    environment_difficulty = 0.5
    robot_width = 0.69
    robot_height = 0.28
    debris_impact_penalty = -2.0
    debris_impact_threshold = 2.0  # Newtons, filtered contacts with the robot only.

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
    robot_width = 0.46
    robot_height = 0.24


@configclass
class SpiderCad6NavigationEnvCfg(SpiderCad8NavigationEnvCfg):
    """Six-legged version, with 18 actions and 66 observations."""

    action_space = 18
    observation_space = 66
    robot = cad_spiderbot_cfg(6).replace(prim_path='/World/envs/env_.*/Robot')
