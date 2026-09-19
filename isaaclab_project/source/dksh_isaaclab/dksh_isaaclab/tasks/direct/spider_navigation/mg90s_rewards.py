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
