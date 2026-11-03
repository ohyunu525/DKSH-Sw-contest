"""GA-selected tripod targets with bounded PPO corrections on the mixed-servo CAD6."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .joint_angle_control import normalize_angles
from .joint_angle_env import JointAngleNavigationEnv
from .joint_angle_env_cfg import MG996RHipCad6JointAngleEnvCfg
from .onshape_cad6_gait import (
    DEFAULT_GAIT_PROFILE,
    FOOT_VERTICAL_GAIN_M_PER_RAD,
    GAIT_PROFILE_BOUNDS,
    SWING_JOINT_GAIN,
    TRIPOD_OFFSETS,
    joint_positions_for_twist,
)

# Nominal CAD foot-point derivative for equal femur/tibia increments at the
# mixed-servo reset stance (0, -0.30, -0.30). The legacy gain is for CAD zero.
DISTAL_STANCE_VERTICAL_GAIN_M_PER_RAD = 0.185


@configclass
class Cad6TripodResidualEnvCfg(MG996RHipCad6JointAngleEnvCfg):
    gait_profile: tuple[float, float, float, float] = DEFAULT_GAIT_PROFILE
    residual_limit_rad: float = 0.15
    action_semantics = "tripod_residual_joint_angles_v1"
    observation_contract = "joint_angles_targets_target_velocity_terrain48_tripod_reference_v1"
    observation_space = 160


class Cad6TripodResidualEnv(JointAngleNavigationEnv):
    _extra_policy_observations = 20  # sin/cos of gait phase and 18 reference targets

    def __init__(self, cfg: Cad6TripodResidualEnvCfg, render_mode=None, **kwargs):
        if len(cfg.gait_profile) != len(GAIT_PROFILE_BOUNDS) or not all(
            low <= value <= high for value, (low, high) in zip(cfg.gait_profile, GAIT_PROFILE_BOUNDS)
        ):
            raise ValueError(f"gait_profile must contain period, stride limit, lift and duty in {GAIT_PROFILE_BOUNDS}")
        if not math.isfinite(cfg.residual_limit_rad) or not 0 < cfg.residual_limit_rad <= 0.3:
            raise ValueError("residual_limit_rad must be in (0, 0.3]")
        super().__init__(cfg, render_mode, **kwargs)
        self._gait_profiles = self._actions.new_tensor(cfg.gait_profile).repeat(self.num_envs, 1)

    def set_gait_profiles(self, profiles):
        """Assign one candidate per simulator lane before a GA rollout."""
        profiles = torch.as_tensor(profiles, device=self.device, dtype=self._gait_profiles.dtype)
        if profiles.shape != self._gait_profiles.shape:
            raise ValueError(f"Expected gait profiles with shape {tuple(self._gait_profiles.shape)}")
        low, high = profiles.new_tensor(GAIT_PROFILE_BOUNDS).unbind(-1)
        if not torch.all((profiles >= low) & (profiles <= high)):
            raise ValueError(f"Gait profiles must be in {GAIT_PROFILE_BOUNDS}")
        self._gait_profiles.copy_(profiles)

    def _gait_reference(self):
        period, stride_limit, lift, duty = self._gait_profiles.unbind(-1)
        elapsed_cycles = self.episode_length_buf * self.step_dt / period
        phase = elapsed_cycles.remainder(1.0)
        distance, _, direction_b = self._goal_data()
        planar_direction = torch.nn.functional.normalize(direction_b[:, :2], dim=-1, eps=1e-6)
        motion_scale = (distance / self.cfg.goal_radius - 0.5).clamp(0.0, 1.0)
        linear = self.cfg.target_speed * planar_direction * motion_scale[:, None]
        command = torch.cat((linear, torch.zeros_like(linear[:, :1])), dim=-1)
        calibrated_lift = lift * motion_scale * FOOT_VERTICAL_GAIN_M_PER_RAD / (
            SWING_JOINT_GAIN * DISTAL_STANCE_VERTICAL_GAIN_M_PER_RAD
        )
        offsets = joint_positions_for_twist(
            phase, command, period, calibrated_lift, stride_limit, offsets=TRIPOD_OFFSETS, duty=duty,
        ).flatten(1)
        neutral = self._robot.data.default_joint_pos[:, self._joint_ids]
        # Begin in the distal-support stance and ease into the first full stride.
        targets = neutral + elapsed_cycles.clamp(0.0, 1.0)[:, None] * offsets
        lower, upper = self._angle_limits.unbind(-1)
        return phase, targets.clamp(lower, upper)

    def _pre_physics_step(self, actions):
        _, reference = self._gait_reference()
        desired = reference + self.cfg.residual_limit_rad * actions.clamp(-1.0, 1.0)
        absolute = normalize_angles(desired, self._angle_limits)
        # The absolute-angle path holds the previous target for invalid actions.
        super()._pre_physics_step(torch.where(torch.isfinite(actions), absolute, actions))

    def _get_observations(self):
        observation = super()._get_observations()
        phase, reference = self._gait_reference()
        phase_rad = 2 * math.pi * phase
        observation["policy"] = torch.cat(
            (
                observation["policy"],
                phase_rad.sin()[:, None],
                phase_rad.cos()[:, None],
                normalize_angles(reference, self._angle_limits),
            ),
            dim=-1,
        )
        return observation
