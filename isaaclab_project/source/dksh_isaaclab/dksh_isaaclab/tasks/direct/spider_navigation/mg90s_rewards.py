"""Pure reward helpers shared by MG90S locomotion environments and tests."""

import torch


def speed_tracking_reward(
    forward_velocity: torch.Tensor, command_velocity: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Return a zero-at-rest Gaussian reward centered on the speed command."""
    if sigma <= 0:
        raise ValueError("speed tracking sigma must be positive")
    tracking = torch.exp(-((forward_velocity - command_velocity) / sigma).square())
    stationary_baseline = torch.exp(-(command_velocity / sigma).square())
    return tracking - stationary_baseline


def velocity_tracking_reward(
    measured_velocity: torch.Tensor, command_velocity: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Return a Gaussian tracking reward for equal-shaped velocity vectors."""
    if sigma <= 0:
        raise ValueError("velocity tracking sigma must be positive")
    if measured_velocity.shape != command_velocity.shape:
        raise ValueError("measured and commanded velocities must have the same shape")
    if measured_velocity.ndim < 1:
        raise ValueError("velocity tensors must have at least one dimension")
    return torch.exp(-((measured_velocity - command_velocity) / sigma).square().sum(dim=-1))


def proportional_arrival_penalty(
    achieved_progress: torch.Tensor,
    scheduled_progress: torch.Tensor,
    total_progress: torch.Tensor,
    *,
    minimum_total_progress: float,
    overshoot_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized late and early arrival penalties.

    ``scheduled_progress`` is the distance or angle that should have been
    completed at the current point in an episode. Progress below that value
    is an undershoot (late arrival); progress above it is an overshoot (early
    arrival). Both terms are normalized by the episode target so that a large
    command cannot dominate a small, but non-zero, command. Commands whose
    total target is below ``minimum_total_progress`` are deliberately
    inactive: holding still must not acquire a synthetic arrival objective.
    """
    if achieved_progress.shape != scheduled_progress.shape or achieved_progress.shape != total_progress.shape:
        raise ValueError("arrival progress tensors must have the same shape")
    if minimum_total_progress <= 0:
        raise ValueError("minimum_total_progress must be positive")
    if overshoot_weight < 0:
        raise ValueError("overshoot_weight must be non-negative")

    normalizer = total_progress.abs().clamp_min(minimum_total_progress)
    active = (total_progress.abs() >= minimum_total_progress).to(achieved_progress.dtype)
    undershoot = active * ((scheduled_progress - achieved_progress).clamp_min(0) / normalizer).square()
    overshoot = active * ((achieved_progress - scheduled_progress).clamp_min(0) / normalizer).square()
    return undershoot + overshoot_weight * overshoot, undershoot, overshoot


def body_twist_reference_trajectory(
    planar_velocity: torch.Tensor, yaw_rate: torch.Tensor, elapsed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a constant body-frame twist from an episode's start frame.

    The returned tuple is ``(position, tangent_velocity)`` in the initial
    body frame. It lets an arrival reward distinguish being ahead of the
    scheduled curve from being behind it, including when a command combines
    translation and yaw.
    """
    if planar_velocity.ndim != 2 or planar_velocity.shape[1] != 2:
        raise ValueError("planar_velocity must have shape [num_envs, 2]")
    if yaw_rate.shape != planar_velocity.shape[:1] or elapsed.shape != yaw_rate.shape:
        raise ValueError("yaw_rate and elapsed must have shape [num_envs]")

    vx, vy = planar_velocity.unbind(-1)
    theta = yaw_rate * elapsed
    sin_theta, cos_theta = theta.sin(), theta.cos()
    nonzero_yaw = yaw_rate.abs() > 1.0e-6
    safe_yaw = torch.where(nonzero_yaw, yaw_rate, torch.ones_like(yaw_rate))
    curved_x = (vx * sin_theta + vy * (cos_theta - 1.0)) / safe_yaw
    curved_y = (vx * (1.0 - cos_theta) + vy * sin_theta) / safe_yaw
    position = torch.stack((
        torch.where(nonzero_yaw, curved_x, vx * elapsed),
        torch.where(nonzero_yaw, curved_y, vy * elapsed),
    ), dim=-1)
    tangent = torch.stack((
        cos_theta * vx - sin_theta * vy,
        sin_theta * vx + cos_theta * vy,
    ), dim=-1)
    return position, tangent
