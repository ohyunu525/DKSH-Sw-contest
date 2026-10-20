"""Absolute joint-angle control and costs, independent of Isaac Sim.

Actions use leg-major order: leg_0 hip/femur/tibia, leg_1 hip/femur/tibia, ...
There is no gait generator, default-pose offset or accumulated angle increment.
"""

from __future__ import annotations

import math
import torch


ACTION_SEMANTICS = "absolute_joint_angles_v1"


def joint_names(leg_count: int) -> list[str]:
    return [f"leg_{leg}_{part}_joint" for leg in range(leg_count) for part in ("hip", "femur", "tibia")]


def observation_size(action_count: int, environment: str) -> int:
    return 14 + 3 * action_count + action_count // 3 + (32 if environment != "flat" else 0)


def normalize_angles(angles: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    lower, upper = limits.unbind(-1)
    return 2 * (angles - lower) / (upper - lower).clamp_min(1e-6) - 1


def action_to_angles(actions: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] onto each joint's complete configured working range, in rad."""
    lower, upper = limits.unbind(-1)
    return lower + 0.5 * (actions.clamp(-1, 1) + 1) * (upper - lower)


def bounded_angle_targets(actions, previous_targets, limits, *, dt, rate_limit, smoothing):
    """Filter absolute targets; a constant action converges to a constant angle.

    The only filter state is the last applied target, which is also observed by
    the policy. Invalid components hold their previous target.
    """
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(rate_limit) and rate_limit > 0):
        raise ValueError("dt and rate_limit must be finite and positive")
    if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
        raise ValueError("smoothing must be in (0, 1]")
    safe_actions = torch.where(torch.isfinite(actions), actions, normalize_angles(previous_targets, limits))
    desired = action_to_angles(safe_actions, limits)
    delta = ((desired - previous_targets) * smoothing).clamp(-rate_limit * dt, rate_limit * dt)
    lower, upper = limits.unbind(-1)
    return (previous_targets + delta).clamp(lower, upper)


def motion_costs(joint_pos, joint_vel, torque, limits, effort_limits, velocity_limit):
    """Penalize joint extremes and actual mechanical effort, without pose imitation."""
    normalized = normalize_angles(joint_pos, limits)
    return {
        "joint_limit": ((normalized.abs() - 0.85).clamp_min(0) / 0.15).square().mean(-1),
        "joint_velocity": (joint_vel / velocity_limit).square().mean(-1),
        "torque": (torque / effort_limits.clamp_min(1e-6)).square().mean(-1),
        # Absolute per-joint power prevents cancellation between driving/braking joints.
        "power": (torque * joint_vel).abs().sum(-1),
    }


def frontier_progress(best_distance, distance, max_step):
    """Reward newly covered distance only; rocking over the same area earns zero."""
    return (best_distance - distance).clamp(0, max_step), torch.minimum(best_distance, distance)


def point_velocity(link_velocity, angular_velocity, offset_w):
    """Velocity at the tibia sole, rather than its rotating centre of mass."""
    return link_velocity + torch.linalg.cross(angular_velocity, offset_w, dim=-1)
