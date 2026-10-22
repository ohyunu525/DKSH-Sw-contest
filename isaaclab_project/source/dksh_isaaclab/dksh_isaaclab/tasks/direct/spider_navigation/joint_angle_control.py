"""Absolute joint-angle control and costs, independent of Isaac Sim.

Actions use leg-major order: leg_0 hip/femur/tibia, leg_1 hip/femur/tibia, ...
There is no gait generator, default-pose offset or accumulated angle increment.
"""

from __future__ import annotations

import math
from statistics import NormalDist
import torch


ACTION_SEMANTICS = "absolute_joint_angles_v2"


def joint_names(leg_count: int) -> list[str]:
    return [f"leg_{leg}_{part}_joint" for leg in range(leg_count) for part in ("hip", "femur", "tibia")]


def observation_size(action_count: int, environment: str) -> int:
    """Use one actor contract across flat and terrain courses for staged learning.

    Joint state includes angles, velocities, applied targets and applied target
    velocities. All courses include the same 48 terrain/hazard features.
    """
    return 14 + 4 * action_count + action_count // 3 + 48


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
    """Legacy v1 helper; v2 uses signed_progress with no hidden best-distance state."""
    return (best_distance - distance).clamp(0, max_step), torch.minimum(best_distance, distance)


def signed_progress(
    previous_distance: torch.Tensor, distance: torch.Tensor, max_positive_step: float
) -> torch.Tensor:
    """Return signed distance reduction, limiting only positive progress.

    The uncapped signed differences telescope over a closed trajectory. Capping
    positive terms can only reduce that sum, so any closed loop earns at most
    zero progress, even if backward and forward motion use different speeds.
    Keep this potential term ungated by posture or contact state; posture and
    contact should contribute separate nonpositive costs.
    """
    if not math.isfinite(max_positive_step) or max_positive_step <= 0:
        raise ValueError("max_positive_step must be finite and positive")
    return (previous_distance - distance).clamp(max=max_positive_step)


def speed_tracking_cost(
    forward_velocity: torch.Tensor, command_velocity: torch.Tensor | float, sigma: float
) -> torch.Tensor:
    """Convex, nonnegative speed error; callers apply a negative reward scale.

    Do not clip or multiply this cost by upright: either can make a zero-net
    forward/backward cycle preferable to standing. For equal mean velocity,
    Jensen's inequality makes oscillation at least as costly as steady motion.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("speed tracking sigma must be finite and positive")
    return ((forward_velocity - command_velocity) / sigma).square()


def exploration_std(
    limits: torch.Tensor, *, dt: float, rate_limit: float, smoothing: float,
    saturation_probability: float = 0.01,
) -> float:
    """Bound initial normalized Gaussian noise using the widest joint range.

    Assumes the initial policy mean maps to the current applied target. If the
    half range is h, requested filter displacement has standard deviation
    smoothing*h*std. Choose std so its two-sided Gaussian tail beyond
    rate_limit*dt is at most saturation_probability for every joint. This only
    initializes exploration; it never changes the available absolute angles.
    """
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(rate_limit) and rate_limit > 0):
        raise ValueError("dt and rate_limit must be finite and positive")
    if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
        raise ValueError("smoothing must be in (0, 1]")
    if not math.isfinite(saturation_probability) or not 0 < saturation_probability < 1:
        raise ValueError("saturation_probability must be in (0, 1)")
    limits = torch.as_tensor(limits)
    if limits.ndim < 1 or limits.shape[-1] != 2 or limits.numel() == 0:
        raise ValueError("limits must have nonempty (..., 2) shape")
    spans = limits[..., 1] - limits[..., 0]
    if not torch.isfinite(limits).all() or not (spans > 0).all():
        raise ValueError("joint limits must be finite with positive ranges")
    max_span = float(spans.max().item())
    # inv_cdf(p/2) avoids rounding 1-p/2 to exactly 1 for small tail budgets.
    quantile = -NormalDist().inv_cdf(saturation_probability / 2)
    return 2 * rate_limit * dt / (smoothing * max_span * quantile)


def point_velocity(link_velocity, angular_velocity, offset_w):
    """Velocity at the tibia sole, rather than its rotating centre of mass."""
    return link_velocity + torch.linalg.cross(angular_velocity, offset_w, dim=-1)
