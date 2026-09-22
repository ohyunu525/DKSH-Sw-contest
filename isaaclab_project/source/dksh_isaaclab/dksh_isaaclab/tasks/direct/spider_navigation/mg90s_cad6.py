"""Mixed-servo locomotion on the six-leg CAD asset: 18 joints / 68 observations."""
import torch

from isaaclab.utils import configclass
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .mg90s_env import MG90SWalkEnv
from .mg90s_env_cfg import MG90SWalkEnvCfg, MG90SWalkRunnerCfg, mixed_servo_robot_cfg
from .mg90s_rewards import (
    body_twist_reference_trajectory,
    proportional_arrival_penalty,
    velocity_tracking_reward,
)
from . import cad6_wave_gait as gait


def robot_cfg():
    robot = cad_spiderbot_cfg(6)
    robot.actuators = mixed_servo_robot_cfg().actuators
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
    mass_assumption = (
        'CAD6 links 2.31362 kg + L1 RM 0.230 kg; known mixed-servo masses '
        'are recorded but not redistributed across provisional link inertias'
    )


@configclass
class MG90SCad6RunnerCfg(MG90SWalkRunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_walk'
    run_name = 'cad6_4v8'


@configclass
class MG90SCad6SprintEnvCfg(MG90SCad6EnvCfg):
    """A higher-cadence profile kept within the slower leg-servo envelope."""
    # Explore a substantially faster command range without increasing the
    # already reach-limited 25 mm stride. At 0.050 m/s the 0.6 s wave cycle
    # produces exactly 25 mm of stance travel (speed * period * 5/6).
    gait_period = 0.6
    gait_lift = 0.005
    gait_stride_limit = 0.025
    command_speed_min = 0.012
    command_speed_max = 0.050
    # Leave a margin below the RC920DMG 5 V no-load speed (6.545 rad/s).
    target_rate_limit = 6.25
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
    # Legacy velocity tasks retain their 68-element observation and do not
    # enable this shaping term. The v2 task below opts in with an explicit
    # time-to-arrival observation, so old policies are never reinterpreted.
    arrival_timing_enabled = False
    arrival_time_s = 0.0  # use episode_length_s - settling_seconds when zero
    arrival_timing_penalty_scale = 8.0
    arrival_overshoot_weight = 1.0
    arrival_cross_track_weight = 0.5
    arrival_min_planar_goal_m = 0.010
    arrival_min_yaw_goal_rad = 0.020


@configclass
class MG90SCad6VelocityRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_velocity_l1v1'
    run_name = 'cad6_4v8_velocity'


@configclass
class MG90SCad6VelocityV1EnvCfg(MG90SCad6VelocityEnvCfg):
    """Faster twist tracking with commands inside the reference gait envelope."""
    gait_period = 0.6
    gait_lift = 0.005
    gait_stride_limit = 0.025
    target_rate_limit = 6.25
    tripod_enabled = True
    command_forward_min = -0.025
    command_forward_max = 0.050
    command_lateral_min = -0.025
    command_lateral_max = 0.025
    command_yaw_min = -0.08
    command_yaw_max = 0.08
    project_velocity_commands = True
    planar_tracking_reward_scale = 6.0
    planar_tracking_sigma = 0.015
    yaw_tracking_reward_scale = 3.0


@configclass
class MG90SCad6VelocityV1RunnerCfg(MG90SCad6VelocityRunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_velocity_l1v2'
    run_name = 'cad6_mixed_velocity_v1'


@configclass
class MG90SCad6VelocityV2EnvCfg(MG90SCad6VelocityV1EnvCfg):
    """Velocity tracking with a scheduled arrival-time penalty.

    The extra observation is the non-repeating fraction of the requested
    arrival window that has elapsed. It is required for a stationary PPO
    policy to distinguish an early trajectory from the same pose late in an
    episode.
    """
    observation_space = 69
    arrival_timing_enabled = True


@configclass
class MG90SCad6VelocityV2RunnerCfg(MG90SCad6VelocityV1RunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_velocity_l1v3'
    run_name = 'cad6_mixed_velocity_arrival_v2'


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

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        """Extract world yaw from Isaac's wxyz root quaternion."""
        return torch.atan2(
            2.0 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1.0 - 2.0 * (quaternion[:, 1].square() + quaternion[:, 2].square()),
        )

    @staticmethod
    def _wrap_to_pi(angle):
        return torch.atan2(angle.sin(), angle.cos())

    def _on_reset(self, env_ids, root_state):
        """Capture the world pose defining this episode's arrival schedule."""
        if not getattr(self.cfg, 'arrival_timing_enabled', False):
            return
        if not hasattr(self, '_arrival_start_pos_w'):
            self._arrival_start_pos_w = torch.zeros((self.num_envs, 2), device=self.device)
            self._arrival_start_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._arrival_start_pos_w[env_ids] = root_state[:, :2]
        self._arrival_start_yaw_w[env_ids] = self._yaw_from_quaternion(root_state[:, 3:7])

    def _arrival_duration(self):
        configured = self.cfg.arrival_time_s
        duration = configured if configured > 0 else self.cfg.episode_length_s - self.cfg.settling_seconds
        if duration <= 0:
            raise ValueError('arrival_time_s must be positive or leave time after settling')
        return duration

    def _arrival_elapsed(self):
        return (self._age - self.cfg.settling_seconds).clamp(0, self._arrival_duration())

    def _arrival_phase(self):
        return self._arrival_elapsed() / self._arrival_duration()

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
        if getattr(self.cfg, 'project_velocity_commands', False):
            self._commands[env_ids] = gait.feasible_velocity_command(
                self._commands[env_ids], self.cfg.gait_period, self.cfg.gait_stride_limit
            )

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

        arrival_metrics = {}
        if self.cfg.arrival_timing_enabled:
            arrival_penalty, arrival_metrics = self._arrival_timing_penalty()
            reward -= self.cfg.arrival_timing_penalty_scale * arrival_penalty
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
                self._robot.data.applied_torque.abs()
                > 0.70 * self._robot.data.joint_effort_limits
            ).float().mean(),
            **arrival_metrics,
        })
        return reward

    def _arrival_timing_penalty(self):
        """Penalize scheduled-path undershoot and overshoot independently."""
        duration = self._arrival_duration()
        elapsed = self._arrival_elapsed()
        planar_velocity = self._commands[:, :2]
        yaw_rate = self._commands[:, 2]
        expected_position, tangent = body_twist_reference_trajectory(planar_velocity, yaw_rate, elapsed)

        displacement_w = self._robot.data.root_pos_w[:, :2] - self._arrival_start_pos_w
        start_cos, start_sin = self._arrival_start_yaw_w.cos(), self._arrival_start_yaw_w.sin()
        actual_position = torch.stack((
            start_cos * displacement_w[:, 0] + start_sin * displacement_w[:, 1],
            -start_sin * displacement_w[:, 0] + start_cos * displacement_w[:, 1],
        ), dim=-1)
        position_error = actual_position - expected_position
        planar_speed = torch.linalg.vector_norm(planar_velocity, dim=-1)
        tangent_unit = tangent / planar_speed.clamp_min(1.0e-6)[:, None]
        scheduled_distance = planar_speed * elapsed
        achieved_distance = scheduled_distance + (position_error * tangent_unit).sum(dim=-1)
        total_distance = planar_speed * duration
        planar_penalty, planar_under, planar_over = proportional_arrival_penalty(
            achieved_distance,
            scheduled_distance,
            total_distance,
            minimum_total_progress=self.cfg.arrival_min_planar_goal_m,
            overshoot_weight=self.cfg.arrival_overshoot_weight,
        )

        normal = torch.stack((-tangent_unit[:, 1], tangent_unit[:, 0]), dim=-1)
        cross_track = (position_error * normal).sum(dim=-1)
        planar_active = (total_distance >= self.cfg.arrival_min_planar_goal_m).to(planar_speed.dtype)
        cross_track_penalty = planar_active * (
            cross_track / total_distance.clamp_min(self.cfg.arrival_min_planar_goal_m)
        ).square()
        planar_penalty += self.cfg.arrival_cross_track_weight * cross_track_penalty

        current_yaw = self._yaw_from_quaternion(self._robot.data.root_quat_w)
        actual_yaw = self._wrap_to_pi(current_yaw - self._arrival_start_yaw_w)
        scheduled_yaw = yaw_rate.abs() * elapsed
        yaw_error = self._wrap_to_pi(actual_yaw - yaw_rate * elapsed)
        achieved_yaw = scheduled_yaw + yaw_rate.sign() * yaw_error
        total_yaw = yaw_rate.abs() * duration
        yaw_penalty, yaw_under, yaw_over = proportional_arrival_penalty(
            achieved_yaw,
            scheduled_yaw,
            total_yaw,
            minimum_total_progress=self.cfg.arrival_min_yaw_goal_rad,
            overshoot_weight=self.cfg.arrival_overshoot_weight,
        )
        return planar_penalty + yaw_penalty, {
            'Metrics/arrival_phase': self._arrival_phase().mean(),
            'Metrics/arrival_undershoot': (planar_under + yaw_under).mean(),
            'Metrics/arrival_overshoot': (planar_over + yaw_over).mean(),
            'Metrics/arrival_cross_track': cross_track_penalty.mean(),
            'Metrics/arrival_timing_penalty': (planar_penalty + yaw_penalty).mean(),
        }

    def _get_observations(self):
        observations = super()._get_observations()
        if self.cfg.arrival_timing_enabled:
            observations['policy'] = torch.cat((
                observations['policy'], self._arrival_phase()[:, None],
            ), dim=-1)
        return observations
