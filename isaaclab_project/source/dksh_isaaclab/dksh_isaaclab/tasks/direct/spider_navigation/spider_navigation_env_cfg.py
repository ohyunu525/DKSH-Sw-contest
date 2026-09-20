"""Configuration for the DKSH spiderbot navigation environment."""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dksh_isaaclab.assets import SPIDERBOT_CFG
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg

from .payload_mass import L1_RM_MOUNT_POSITION_B, L1_RM_PAYLOAD_MASS_KG, L1_RM_PAYLOAD_SIZE_M


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

    # Unitree 4D LiDAR L1 RM.  The policy consumes a compact planar projection
    # instead of the complete point cloud so many environments can train in
    # parallel.  Hardware values are from the Unitree L1 user manual v1.1.
    lidar_model = "Unitree 4D LiDAR L1 RM"
    lidar_min_range_m = 0.05
    lidar_max_range_m = 30.0  # 90% reflectivity; 10% reflectivity is 15 m.
    lidar_low_reflectivity_range_m = 15.0
    lidar_horizontal_fov_deg = 360.0
    lidar_vertical_fov_deg = 90.0
    lidar_vertical_projection_bins = 3
    lidar_sampling_frequency_hz = 43_200
    lidar_effective_frequency_hz = 21_600
    lidar_horizontal_scan_frequency_hz = 11.0
    lidar_vertical_scan_frequency_hz = 180.0
    lidar_imu_sampling_frequency_hz = 1_000.0
    lidar_imu_reporting_frequency_hz = 250.0
    lidar_measurement_accuracy_m = 0.02
    lidar_measurement_resolution_m = 0.008
    lidar_observation_bins = 16
    lidar_noise_enabled = True
    lidar_payload_mass_kg = L1_RM_PAYLOAD_MASS_KG
    lidar_payload_size_m = L1_RM_PAYLOAD_SIZE_M
    # Point-cloud origin: bottom center of L1. The 30 mm bracket height is
    # provisional until the physical mounting bracket is measured.
    lidar_mount_position_b = L1_RM_MOUNT_POSITION_B

    # Keep early exploration inside a range the MG996R-powered stance can recover from.
    action_scale = 0.30
    joint_velocity_scale = 0.10
    # Stage 2 curriculum: first learn repeatable short-range locomotion.
    goal_min_distance = 0.50
    goal_max_distance = 1.50
    goal_radius = 0.35
    max_distance_from_origin = 5.5
    minimum_base_height = 0.09

    # Translational progress must dominate the passive reward for standing still.
    progress_reward_scale = 25.0
    velocity_to_goal_reward_scale = 1.50
    heading_reward_scale = 0.05
    upright_reward_scale = 0.05
    action_rate_penalty_scale = -0.015
    torque_penalty_scale = -0.0003
    vertical_velocity_penalty_scale = -0.05
    stillness_penalty_scale = -0.10
    goal_reward = 30.0
    failure_penalty = -10.0


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
