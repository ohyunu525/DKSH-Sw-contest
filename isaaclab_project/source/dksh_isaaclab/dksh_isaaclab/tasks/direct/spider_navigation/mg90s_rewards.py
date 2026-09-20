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
