"""Phase-conditioned absolute-angle imitation environment for the mixed CAD6."""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from .joint_angle_control import normalize_angles
from .joint_angle_env import JointAngleNavigationEnv
from .joint_angle_env_cfg import MG996RHipCad6JointAngleEnvCfg
from .Upgraded_cad6_control import UpgradedCad6Kinematics
from .Upgraded_cad6_env import UpgradedCad6PPORunnerCfg
from .Upgraded_hexapod_reference import UpgradedCrawlReference


@configclass
class UpgradedImitationEnvCfg(MG996RHipCad6JointAngleEnvCfg):
    action_semantics = 'Upgraded_absolute_joint_angles_imitation_v1'
    observation_contract = 'Upgraded_imu_commands_switches_phase_history_v1'
    control_contract = 'Upgraded_single_foot_cad_reference_50hz_v1'
    reward_contract = 'Upgraded_distal_v12_reference_abort_v1'
    decimation = 4
    sim = MG996RHipCad6JointAngleEnvCfg().sim.replace(render_interval=decimation)
    history_frames = 4
    observation_space = 456
    state_space = 596
    reference_settings = dict(stride=0.03, lift=0.025, probe_speed=0.03, probe_depth=0.02,
                              lift_time=0.5, transfer_time=0.6, shift_time=0.5, settle_time=0.2,
                              debounce_time=0.06, timeout=6.0, support_margin=0.015)
    ik_clearance = 0.008
    target_speed = 0.003
    goal_min_distance = 0.15
    goal_max_distance = 0.20
    goal_radius = 0.03
    episode_length_s = 60.0
    imu_angular_noise = 0.01
    imu_gravity_noise = 0.005


@configclass
class UpgradedImitationPPORunnerCfg(UpgradedCad6PPORunnerCfg):
    experiment_name = 'Upgraded_imitation'
    empirical_normalization = False  # BC and PPO consume the same scaled observations.
    num_steps_per_env = 256
    policy = UpgradedCad6PPORunnerCfg().policy.copy()
    policy.activation = 'elu'
    # 50 Hz instead of PMTG's 10 Hz: approximately preserve its discount horizon.
    algorithm = UpgradedCad6PPORunnerCfg().algorithm.copy()
    algorithm.gamma = 0.999
    algorithm.lam = 0.996


class UpgradedImitationEnv(JointAngleNavigationEnv):
    frame_size = 6 + 3 + 18 + 18 + 18 + 6 + UpgradedCrawlReference.feature_count + 32

    def _configure_gym_env_spaces(self):
        self.cfg.observation_space = self.frame_size * self.cfg.history_frames
        self.cfg.state_space = self.cfg.observation_space + 140
        DirectRLEnv._configure_gym_env_spaces(self)

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.history_frames < 1 or cfg.action_space != 18 or cfg.behavior_profile != 'revised_v12':
            raise ValueError('Imitation requires the mixed CAD6 v12 contract')
        super().__init__(cfg, render_mode, **kwargs)
        if abs(self.step_dt - 0.02) > 1e-8:
            raise ValueError('Imitation labels and actions require the same 50 Hz clock')
        self._kinematics = UpgradedCad6Kinematics(
            cfg.toe_button_offsets, cfg.non_distal_collision_offsets, cfg.proximal_foot_offsets,
            device=self.device, dtype=self._actions.dtype)
        neutral = self._robot.data.default_joint_pos[:, self._joint_ids].reshape(self.num_envs, 6, 3)
        home = self._kinematics.forward(neutral)[0]
        home[..., 2] = home[..., 2].amin(-1, keepdim=True)
        self._reference = UpgradedCrawlReference(home, **cfg.reference_settings)
        self._ik_seed = neutral.clone()
        self.reference_actions = normalize_angles(neutral.flatten(1), self._angle_limits).clone()
        self._reference_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._valid_for_step = self._reference_valid.clone()
        self._fault_for_step = self._reference.fault.clone()
        self._history = self._actions.new_zeros((self.num_envs, cfg.history_frames, self.frame_size))
        self._history_reset = torch.ones_like(self._reference_valid)
        self._history_stamp = -1

    def _command(self):
        distance, _, direction = self._goal_data()
        speed = ((distance / self.cfg.goal_radius - 0.5).clamp(0, 1) * self.cfg.target_speed)
        planar = torch.nn.functional.normalize(direction[:, :2], dim=-1, eps=1e-6)
        return torch.cat((planar * speed[:, None], torch.zeros_like(speed[:, None])), -1)

    def _pre_physics_step(self, actions):
        if actions.shape != self._actions.shape or not torch.isfinite(actions).all():
            raise ValueError('Imitation actions must be finite normalized absolute joint targets')
        self._valid_for_step.copy_(self._reference_valid)
        self._fault_for_step.copy_(self._reference.fault)
        super()._pre_physics_step(actions)

    def _get_observations(self):
        privileged = super()._get_observations()['policy']
        new_step = self._history_stamp != self.common_step_counter
        if new_step or self._history_reset.any():
            contacts, _ = self._contact_state()
            command = self._command()
            update = torch.ones_like(self._history_reset) if new_step else self._history_reset.clone()
            targets = self._reference.advance(contacts, command, self._robot.data.projected_gravity_b,
                                              self.step_dt, mask=update)
            angles, valid, _ = self._kinematics.inverse(
                targets, self._ik_seed, self._angle_limits.reshape(self.num_envs, 6, 3, 2),
                clearance=self.cfg.ik_clearance)
            self._reference.fail(update & ~valid.all(-1), 8)
            self._reference_valid[update] = valid.all(-1)[update] & ~self._reference.failed[update]
            self._ik_seed[update] = angles[update]
            self.reference_actions[update] = normalize_angles(angles.flatten(1), self._angle_limits)[update]
            angular = self._robot.data.root_ang_vel_b.clone()
            gravity = self._robot.data.projected_gravity_b.clone()
            if not self.cfg.evaluation_mode:
                angular += torch.randn_like(angular) * self.cfg.imu_angular_noise
                gravity += torch.randn_like(gravity) * self.cfg.imu_gravity_noise
                gravity = torch.nn.functional.normalize(gravity, dim=-1)
            frame = torch.cat((angular, gravity, command / self.cfg.target_speed,
                               normalize_angles(self.joint_angle_targets_rad, self._angle_limits),
                               self._target_velocity / self.cfg.target_rate_limit, self._actions,
                               contacts.float(), self._reference.features(), privileged[:, -48:-16]), -1)
            if new_step:
                self._history.copy_(torch.cat((self._history[:, 1:], frame[:, None]), 1))
            self._history[self._history_reset] = frame[self._history_reset, None]
            self._history_reset.zero_()
            self._history_stamp = self.common_step_counter
        actor = self._history.flatten(1).clone()
        return {'policy': actor, 'critic': torch.cat((actor, privileged), -1)}

    def _fallen(self):
        fallen = super()._fallen()
        return fallen | self._reference.failed if hasattr(self, '_reference') else fallen

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.step_metrics.update(
            Upgraded_reference_valid=self._valid_for_step.float().clone(),
            Upgraded_reference_fault=self._fault_for_step.clone(),
            Upgraded_tracking_error_max_rad=(self._robot.data.joint_pos[:, self._joint_ids]
                                             - self.joint_angle_targets_rad).abs().amax(-1),
            Upgraded_torque_abs_max_nm=self._robot.data.applied_torque[:, self._joint_ids].abs().amax(-1))
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, '_reference'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._reference.reset(ids)
            self._ik_seed[ids] = self._robot.data.default_joint_pos[ids][:, self._joint_ids].reshape(-1, 6, 3)
            self._reference_valid[ids] = True
            self._history_reset[ids] = True
