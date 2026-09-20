"""MG90S locomotion on the actual six-leg CAD asset: 18 joints / 68 observations."""
import torch

from isaaclab.utils import configclass
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .mg90s_env import MG90SWalkEnv
from .mg90s_env_cfg import MG90SWalkEnvCfg, MG90SWalkRunnerCfg, mg90s_robot_cfg
from .mg90s_rewards import velocity_tracking_reward
from . import cad6_wave_gait as gait


def robot_cfg():
    robot = cad_spiderbot_cfg(6)
    robot.actuators = mg90s_robot_cfg().actuators
    robot.init_state.pos = (0, 0, -gait.STANCE_Z + 0.016)
    robot.init_state.joint_pos = {'.*_hip_joint': 0., '.*_femur_joint': gait.STANCE_FEMUR,
                                  '.*_tibia_joint': gait.STANCE_TIBIA}
    return robot.replace(prim_path='/World/envs/env_.*/Robot')


@configclass
class MG90SCad6EnvCfg(MG90SWalkEnvCfg):
    robot = robot_cfg()
    action_space = 18
    observation_space = 68
    minimum_height = 0.075
    gait_period = 2.4
    gait_lift = 0.004
    gait_stride_limit = 0.012
    command_speed_min = 0.003
    command_speed_max = 0.006
    tripod_transition_speed = 0.030
    tripod_transition_width = 0.005
    tripod_enabled = False
    mass_assumption = 'CAD6 links 2.31362 kg + L1 RM 0.230 kg; remaining hardware mass unmeasured'


@configclass
class MG90SCad6RunnerCfg(MG90SWalkRunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_walk'
    run_name = 'cad6_4v8'


@configclass
class MG90SCad6SprintEnvCfg(MG90SCad6EnvCfg):
    """A higher-cadence profile kept within the MG90S no-load speed envelope."""
    # Explore a substantially faster command range without increasing the
    # already reach-limited 25 mm stride. At 0.050 m/s the 0.6 s wave cycle
    # produces exactly 25 mm of stance travel (speed * period * 5/6).
    gait_period = 0.6
    gait_lift = 0.005
    gait_stride_limit = 0.025
    command_speed_min = 0.012
    command_speed_max = 0.050
    # Leave a small margin below the modeled 10.47 rad/s no-load speed.
    target_rate_limit = 10.0
    # Low speeds retain wave gait stability. Blend to alternating tripods over
    # 0.025--0.035 m/s to make the high command range physically meaningful.
    tripod_enabled = True
    # Sprint must follow the sampled command instead of maximizing forward
    # velocity regardless of overshoot. Keep the safety penalties unchanged.
    speed_tracking_reward_scale = 6.0
    speed_tracking_sigma = 0.006
    forward_progress_reward_scale = 30.0


@configclass
class MG90SCad6SprintRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_sprint'
    run_name = 'cad6_4v8_sprint'


@configclass
class MG90SCad6VelocityEnvCfg(MG90SCad6EnvCfg):
    """Low-level body-twist tracking for the ROS2/Nav2 ``cmd_vel`` contract."""
    command_forward_min = -0.006
    command_forward_max = 0.012
    command_lateral_min = -0.006
    command_lateral_max = 0.006
    command_yaw_min = -0.08
    command_yaw_max = 0.08
    command_stand_probability = 0.15
    planar_tracking_reward_scale = 3.0
    planar_tracking_sigma = 0.010
    yaw_tracking_reward_scale = 2.0
    yaw_tracking_sigma = 0.08


@configclass
class MG90SCad6VelocityRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_velocity_l1v1'
    run_name = 'cad6_4v8_velocity'


class MG90SCad6Env(MG90SWalkEnv):
    cfg: MG90SCad6EnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self._robot.num_joints == 18 and self._robot.num_bodies == 19
        assert len(self._joint_ids) == 18
        self._viewer_visuals = None

    def _gait_joint_positions(self):
        feet = gait.blended_foot_targets(
            self._phase(), self._speed, self.cfg.gait_period, self.cfg.gait_lift,
            self.cfg.gait_stride_limit, self._tripod_weight(),
        )
        return gait.inverse_kinematics(feet).flatten(1)

    def _tripod_weight(self):
        if not self.cfg.tripod_enabled:
            return torch.zeros_like(self._speed)
        width = self.cfg.tripod_transition_width
        if width <= 0:
            return (self._speed >= self.cfg.tripod_transition_speed).float()
        blend = ((self._speed - (self.cfg.tripod_transition_speed - width)) / (2 * width)).clamp(0, 1)
        return blend.square() * (3 - 2 * blend)

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras["log"]["Metrics/tripod_blend"] = self._tripod_weight().mean()
        return reward

    def _get_observations(self):
        if self.sim.has_gui():
            from .spider_navigation_env import SpiderNavigationEnv
            SpiderNavigationEnv._sync_cad_visual_for_gui(self)
        return super()._get_observations()


class MG90SCad6VelocityEnv(MG90SCad6Env):
    """Track planar velocity and yaw-rate commands without consuming LiDAR."""
    cfg: MG90SCad6VelocityEnvCfg

    def _sample_commands(self, env_ids):
        count = len(env_ids)
        random = torch.rand((count, 4), device=self.device)
        self._commands[env_ids, 0] = self.cfg.command_forward_min + (
            self.cfg.command_forward_max - self.cfg.command_forward_min
        ) * random[:, 0]
        self._commands[env_ids, 1] = self.cfg.command_lateral_min + (
            self.cfg.command_lateral_max - self.cfg.command_lateral_min
        ) * random[:, 1]
        self._commands[env_ids, 2] = self.cfg.command_yaw_min + (
            self.cfg.command_yaw_max - self.cfg.command_yaw_min
        ) * random[:, 2]
        self._commands[env_ids[random[:, 3] < self.cfg.command_stand_probability]] = 0

    def _gait_joint_positions(self):
        feet = gait.blended_directional_foot_targets(
            self._phase(), self._commands, self.cfg.gait_period, self.cfg.gait_lift,
            self.cfg.gait_stride_limit, self._tripod_weight(),
        )
        return gait.inverse_kinematics(feet).flatten(1)

    def _tripod_weight(self):
        if not self.cfg.tripod_enabled:
            return torch.zeros_like(self._speed)
        equivalent_speed = torch.linalg.vector_norm(self._commands[:, :2], dim=-1)
        equivalent_speed += self._commands[:, 2].abs() * gait.STANCE_RADIUS
        width = self.cfg.tripod_transition_width
        if width <= 0:
            return (equivalent_speed >= self.cfg.tripod_transition_speed).float()
        blend = ((equivalent_speed - (self.cfg.tripod_transition_speed - width)) / (2 * width)).clamp(0, 1)
        return blend.square() * (3 - 2 * blend)

    def _get_rewards(self):
        linear_velocity = self._robot.data.root_lin_vel_b
        angular_velocity = self._robot.data.root_ang_vel_b
        planar_track = velocity_tracking_reward(
            linear_velocity[:, :2], self._commands[:, :2], self.cfg.planar_tracking_sigma
        )
        yaw_track = velocity_tracking_reward(
            angular_velocity[:, 2, None], self._commands[:, 2, None], self.cfg.yaw_tracking_sigma
        )
        upright = (-self._robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
        active = (self._age > self.cfg.settling_seconds + 1).float()
        reward = (
            self.cfg.planar_tracking_reward_scale * planar_track
            + self.cfg.yaw_tracking_reward_scale * yaw_track
        ) * upright * active
        reward -= 2.0 * linear_velocity[:, 2].square()
        reward -= 0.5 * angular_velocity[:, :2].square().sum(-1)
        reward -= 2.0 * (1 - upright).square()
        reward -= 0.003 * (self._actions - self._previous_actions).square().sum(-1)
        reward -= 0.01 * self._filtered_actions.square().sum(-1)
        reward = reward * self.step_dt - 5 * (self._fallen | self._escaped).float()
        self.extras["log"].update({
            "Metrics/command_forward_mps": self._commands[:, 0].mean(),
            "Metrics/command_lateral_mps": self._commands[:, 1].mean(),
            "Metrics/command_yaw_rps": self._commands[:, 2].mean(),
            "Metrics/planar_error_mps": torch.linalg.vector_norm(
                linear_velocity[:, :2] - self._commands[:, :2], dim=-1
            ).mean(),
            "Metrics/yaw_error_rps": (angular_velocity[:, 2] - self._commands[:, 2]).abs().mean(),
            "Metrics/planar_tracking_reward": planar_track.mean(),
            "Metrics/yaw_tracking_reward": yaw_track.mean(),
            "Metrics/upright": upright.mean(),
            "Metrics/torque_limit_fraction": (
                self._robot.data.applied_torque.abs() > 0.70 * self.cfg.robot.actuators["mg90s_4v8"].saturation_effort
            ).float().mean(),
        })
        return reward
