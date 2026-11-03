"""Single-stage asymmetric PPO over CAD foothold trajectories."""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from .joint_angle_control import normalize_angles
from .joint_angle_env import JointAngleNavigationEnv
from .joint_angle_env_cfg import MG996RHipCad6JointAngleEnvCfg, JointAnglePPORunnerCfg
from .Upgraded_cad6_control import UpgradedCad6Kinematics, UpgradedFootTrajectory


@configclass
class UpgradedCad6EnvCfg(MG996RHipCad6JointAngleEnvCfg):
    action_semantics = "Upgraded_latched_foothold_xyz_v1"
    observation_contract = "Upgraded_imu_command_targets_trajectory_lidar_history_v1"
    control_contract = "Upgraded_cad_ik_50hz_foothold_10hz_v1"
    reward_contract = "Upgraded_distal_slip_v12_ik_rejection_v1"
    history_frames = 4
    gait_period = 2.0
    gait_duty = 0.65
    gait_lift = 0.02
    foothold_offset_limit = 0.015
    ik_clearance = 0.008
    target_speed = 0.03
    imu_angular_noise = 0.01
    imu_gravity_noise = 0.005
    observation_space = 744
    state_space = 884


@configclass
class UpgradedCad6PPORunnerCfg(JointAnglePPORunnerCfg):
    experiment_name = "Upgraded_cad6"
    num_steps_per_env = 64
    policy = JointAnglePPORunnerCfg().policy.copy()
    policy.actor_hidden_dims = [128, 128, 64]
    policy.critic_hidden_dims = [256, 128, 64]
    policy.init_noise_std = 0.12
    algorithm = JointAnglePPORunnerCfg().algorithm.copy()
    algorithm.schedule = "fixed"
    algorithm.learning_rate = 1.5e-4
    algorithm.gamma = 0.995
    algorithm.lam = 0.98


class UpgradedCad6Env(JointAngleNavigationEnv):
    frame_size = 6 + 3 + 18 + 18 + UpgradedFootTrajectory.feature_count + 32

    def _configure_gym_env_spaces(self):
        self.cfg.observation_space = self.frame_size * self.cfg.history_frames
        self.cfg.state_space = self.cfg.observation_space + 140
        DirectRLEnv._configure_gym_env_spaces(self)

    def __init__(self, cfg: UpgradedCad6EnvCfg, render_mode=None, **kwargs):
        if cfg.history_frames < 1 or cfg.action_space != 18 or cfg.behavior_profile != "revised_v12":
            raise ValueError("Upgraded CAD6 requires 18 actions, revised_v12 and positive history length")
        super().__init__(cfg, render_mode, **kwargs)
        self._kinematics = UpgradedCad6Kinematics(
            cfg.toe_button_offsets, cfg.non_distal_collision_offsets, cfg.proximal_foot_offsets,
            device=self.device, dtype=self._actions.dtype,
        )
        neutral = self._robot.data.default_joint_pos[:, self._joint_ids].reshape(self.num_envs, 6, 3)
        home = self._kinematics.forward(neutral)[0]
        home[..., 2] = home[..., 2].amin(-1, keepdim=True)
        self._ik_seed = neutral.clone()
        self._trajectory = UpgradedFootTrajectory(
            home, period=cfg.gait_period, duty=cfg.gait_duty,
            lift=cfg.gait_lift, offset_limit=cfg.foothold_offset_limit,
        )
        self._policy_actions = torch.zeros_like(self._actions)
        self._command = self._actions.new_zeros((self.num_envs, 3))
        self._history = self._actions.new_zeros((self.num_envs, cfg.history_frames, self.frame_size))
        self._history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._history_stamp = -1
        self._ik_rejections = self._actions.new_zeros(self.num_envs)
        self._ik_error = self._actions.new_zeros(self.num_envs)

    def _velocity_command(self):
        distance, _, direction = self._goal_data()
        scale = (distance / self.cfg.goal_radius - 0.5).clamp(0, 1)
        planar = torch.nn.functional.normalize(direction[:, :2], dim=-1, eps=1e-6)
        return torch.cat((planar * (scale * self.cfg.target_speed)[:, None], torch.zeros_like(scale[:, None])), -1)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Upgraded PPO produced non-finite foothold actions")
        # Retain the parent contact/reward housekeeping and absolute servo state.
        super()._pre_physics_step(self._actions.clone())
        self._policy_actions.copy_(actions.clamp(-1, 1))
        self._command.copy_(self._velocity_command())
        self._ik_rejections.zero_()
        self._ik_error.zero_()

    def _advance_angle_targets(self, dt):
        targets = self._trajectory.advance(self._command, self._policy_actions, dt)
        angles, valid, error = self._kinematics.inverse(
            targets, self._ik_seed, self._angle_limits.reshape(self.num_envs, 6, 3, 2),
            clearance=self.cfg.ik_clearance,
        )
        self._ik_seed.copy_(angles)
        self._ik_rejections += (~valid).float().mean(-1) / self._target_filter_updates_per_step
        self._ik_error.copy_(torch.maximum(self._ik_error, error.amax(-1)))
        self._actions.copy_(normalize_angles(angles.flatten(1), self._angle_limits))
        super()._advance_angle_targets(dt)

    def _get_observations(self):
        privileged = super()._get_observations()["policy"]
        new_step = self._history_stamp != self.common_step_counter
        if new_step or self._history_reset.any():
            angular = self._robot.data.root_ang_vel_b.clone()
            gravity = self._robot.data.projected_gravity_b.clone()
            if not self.cfg.evaluation_mode:
                angular += torch.randn_like(angular) * self.cfg.imu_angular_noise
                gravity += torch.randn_like(gravity) * self.cfg.imu_gravity_noise
                gravity = torch.nn.functional.normalize(gravity, dim=-1)
            frame = torch.cat((
                angular, gravity, self._velocity_command(),
                normalize_angles(self.joint_angle_targets_rad, self._angle_limits),
                self._target_velocity / self.cfg.target_rate_limit,
                self._trajectory.features(), privileged[:, -48:-16],
            ), -1)
            if new_step:
                self._history.copy_(torch.cat((self._history[:, 1:], frame[:, None]), dim=1))
            self._history[self._history_reset] = frame[self._history_reset, None]
            self._history_reset.zero_()
            self._history_stamp = self.common_step_counter
        # Rollout storage may retain this tensor until after the next env.step.
        actor = self._history.flatten(1).clone()
        return {"policy": actor, "critic": torch.cat((actor, privileged), dim=-1)}

    def _get_rewards(self):
        reward = super()._get_rewards()
        # Rejecting unreachable foot commands must not be free to the optimizer.
        penalty = self._ik_rejections * self.step_dt
        self.step_metrics["Upgraded_ik_rejection_fraction"] = self._ik_rejections.clone()
        self.step_metrics["Upgraded_ik_error_m"] = self._ik_error.clone()
        self.extras["log"].update({
            "Upgraded/ik_rejection_fraction": self._ik_rejections.mean(),
            "Upgraded/ik_error_m": self._ik_error.mean(),
        })
        return reward - penalty

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_trajectory"):
            if env_ids is None:
                env_ids = self._robot._ALL_INDICES
            self._trajectory.reset(env_ids)
            self._ik_seed[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._joint_ids].reshape(-1, 6, 3)
            self._policy_actions[env_ids] = 0
            self._command[env_ids] = 0
            self._history_reset[env_ids] = True
