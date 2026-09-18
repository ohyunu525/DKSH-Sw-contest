"""Flat, slow forward locomotion with an IK wave gait plus learned corrections."""

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from .mg90s_env_cfg import MG90SWalkEnvCfg, MG90S_STALL_TORQUE
from .mg90s_rewards import speed_tracking_reward
from .wave_gait import foot_targets, inverse_kinematics


class MG90SWalkEnv(DirectRLEnv):
    cfg: MG90SWalkEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids = torch.tensor([
            self._robot.joint_names.index(f"leg_{i}_{part}_joint")
            for i in range(cfg.action_space // 3) for part in ("hip", "femur", "tibia")
        ], device=self.device)
        self._actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._filtered_actions = torch.zeros_like(self._actions)
        self._targets = self._robot.data.default_joint_pos.clone()
        self._reference = self._targets.clone()
        self._age = torch.zeros(self.num_envs, device=self.device)
        self._phase_offset = torch.zeros_like(self._age)
        self._speed = torch.zeros_like(self._age)
        self._fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._escaped = self._fallen.clone()
        self.completed = {name: 0 for name in ("episodes", "fallen", "time_out", "out_of_bounds")}

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self._robot
        light = sim_utils.DomeLightCfg(intensity=2200.0)
        light.func("/World/Light", light)

    def _phase(self):
        return (self._phase_offset + (self._age - self.cfg.settling_seconds).clamp_min(0) / self.cfg.gait_period).remainder(1)

    def _gait_joint_positions(self):
        feet = foot_targets(self._phase(), self._speed, self.cfg.gait_period, self.cfg.gait_lift)
        return inverse_kinematics(feet).flatten(1)

    def _pre_physics_step(self, actions):
        self.extras["log"] = {}
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1, 1))
        self._filtered_actions.lerp_(self._actions, self.cfg.action_smoothing)
        self._age += self.step_dt
        reference = self._robot.data.default_joint_pos.clone()
        reference[:, self._joint_ids] = self._gait_joint_positions()
        # Smoothly leave the reset pose after one second of settling.
        ramp = ((self._age - self.cfg.settling_seconds) / 1.0).clamp(0, 1)[:, None]
        reference = torch.lerp(self._robot.data.default_joint_pos, reference, ramp)
        self._reference.copy_(reference)
        desired = reference + self.cfg.action_scale * self._filtered_actions
        limits = self._robot.data.soft_joint_pos_limits
        desired = desired.clamp(limits[:, :, 0], limits[:, :, 1])
        max_delta = self.cfg.target_rate_limit * self.step_dt
        self._targets += (desired - self._targets).clamp(-max_delta, max_delta)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._targets)

    def _get_observations(self):
        command = torch.zeros((self.num_envs, 3), device=self.device)
        command[:, 0] = self._speed
        phase = 2 * math.pi * self._phase()
        obs = torch.cat((
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            command,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel * 0.1,
            self._filtered_actions,
            torch.sin(phase)[:, None], torch.cos(phase)[:, None],
        ), dim=-1)
        return {"policy": obs}

    def _get_dones(self):
        pos = self._robot.data.root_pos_w - self.scene.env_origins
        self._fallen = (pos[:, 2] < self.cfg.minimum_height) | (self._robot.data.projected_gravity_b[:, 2] > -0.7)
        self._escaped = torch.linalg.vector_norm(pos[:, :2], dim=1) > 0.8
        return self._fallen | self._escaped, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        vel = self._robot.data.root_lin_vel_b
        # Zero forward velocity gets zero tracking reward; moving to the command earns more.
        track = speed_tracking_reward(vel[:, 0], self._speed, self.cfg.speed_tracking_sigma)
        upright = (-self._robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
        active = (self._age > self.cfg.settling_seconds + 1).float()
        progress = vel[:, 0].clamp(-0.03, 0.03)
        reward = (
            self.cfg.speed_tracking_reward_scale * track
            + self.cfg.forward_progress_reward_scale * progress
        ) * upright * active
        reward -= 60 * vel[:, 1].square() + 0.5 * self._robot.data.root_ang_vel_b.square().sum(-1)
        reward -= 2 * (1 - upright).square()
        reward -= 0.003 * (self._actions - self._previous_actions).square().sum(-1)
        reward -= 0.01 * self._filtered_actions.square().sum(-1)
        reward = reward * self.step_dt - 5 * (self._fallen | self._escaped).float()
        self.extras["log"].update({
            "Metrics/forward_speed_mps": vel[:, 0].mean(),
            "Metrics/command_speed_mps": self._speed.mean(),
            "Metrics/speed_error_mps": (vel[:, 0] - self._speed).abs().mean(),
            "Metrics/speed_tracking_reward": track.mean(),
            "Metrics/upright": upright.mean(),
            "Metrics/torque_limit_fraction": (self._robot.data.applied_torque.abs() > 0.70 * MG90S_STALL_TORQUE).float().mean(),
        })
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        valid = env_ids[self._age[env_ids] > 0]
        if len(valid):
            falls = self._fallen[valid]
            escapes = self._escaped[valid] & ~falls
            timeouts = self.reset_time_outs[valid] & ~falls & ~escapes
            counts = {"episodes": len(valid), "fallen": int(falls.sum()), "out_of_bounds": int(escapes.sum()), "time_out": int(timeouts.sum())}
            for name, count in counts.items():
                self.completed[name] += count
            # Actual per-reset counts, not stale metrics carried into later timesteps.
            self.extras["log"].update({f"Episode_Count/{name}": float(count) for name, count in counts.items()})
            self.extras["log"]["Episode/displacement_x_m"] = (self._robot.data.root_pos_w[valid, 0] - self.scene.env_origins[valid, 0]).mean()
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._age[env_ids] = 0
        self._actions[env_ids] = 0
        self._previous_actions[env_ids] = 0
        self._filtered_actions[env_ids] = 0
        self._phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)
        self._speed[env_ids] = self.cfg.command_speed_min + (self.cfg.command_speed_max - self.cfg.command_speed_min) * torch.rand(len(env_ids), device=self.device)
        state = self._robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        pos = self._robot.data.default_joint_pos[env_ids].clone()
        self._targets[env_ids] = pos
        self._reference[env_ids] = pos
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), None, env_ids)
