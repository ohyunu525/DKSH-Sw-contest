"""Estimate sole support from link forces and terrain height without Isaac Sim.

Isaac Lab 2.1 exposes net link forces, not contact point locations. A tibia's
force alone therefore cannot distinguish its sole from a shaft/wall collision.
This helper requires upward force and a sole close to the supporting surface.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch


def sole_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int], sensor_body: list[int],
    sole_positions_w: torch.Tensor, surface_heights: torch.Tensor,
    force_threshold: float, height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return supported feet ``(N,L)`` and undesired contacts ``(N,)``.

    ``force_history`` is world-frame force history ``(N,H,B,3)``. Foot indices
    and sole positions share leg order; surface heights are sampled separately
    under each sole. A foot needs upward vertical force above the threshold in
    any history sample and current sole height within the inclusive tolerance.
    Tibia force without valid sole support counts as an undesired contact, as
    does sufficiently large force on any configured non-foot body.

    This is a geometric support estimate, not contact-point reconstruction.
    Shape, index and scalar validation avoids device synchronization per step.
    """
    if not math.isfinite(force_threshold) or force_threshold <= 0:
        raise ValueError("force_threshold must be finite and positive")
    if not math.isfinite(height_tolerance) or height_tolerance < 0:
        raise ValueError("height_tolerance must be finite and nonnegative")
    if force_history.ndim != 4 or force_history.shape[-1] != 3 or force_history.shape[1] < 1:
        raise ValueError("force_history must have shape (N,H,B,3) with nonempty history")
    count, _, bodies, _ = force_history.shape
    feet_indices = tuple(sensor_feet)
    body_indices = tuple(sensor_body)
    if not feet_indices:
        raise ValueError("sensor_feet must contain at least one foot index")
    for name, indices in (("sensor_feet", feet_indices), ("sensor_body", body_indices)):
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
            raise ValueError(f"{name} must contain integer body indices")
        if any(index < 0 or index >= bodies for index in indices):
            raise ValueError(f"{name} contains an index outside the sensor body range")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must contain unique indices")
    if set(feet_indices).intersection(body_indices):
        raise ValueError("sensor_feet and sensor_body must be disjoint")
    legs = len(feet_indices)
    if sole_positions_w.shape != (count, legs, 3):
        raise ValueError("sole_positions_w must have shape (N,L,3) matching sensor_feet")
    if surface_heights.shape != (count, legs):
        raise ValueError("surface_heights must have shape (N,L) matching sensor_feet")
    if sole_positions_w.device != force_history.device or surface_heights.device != force_history.device:
        raise ValueError("forces, sole positions and surface heights must share a device")
    if not all(value.is_floating_point() for value in (force_history, sole_positions_w, surface_heights)):
        raise ValueError("forces, sole positions and surface heights must be floating-point tensors")

    # Convert tuples to lists so PyTorch indexes the body dimension rather than
    # interpreting tuple entries as indices of separate tensor dimensions.
    feet_indices = list(feet_indices)
    body_indices = list(body_indices)
    force_magnitudes = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1)
    upward = force_history[:, :, feet_indices, 2].amax(dim=1) > force_threshold
    near_surface = (sole_positions_w[..., 2] - surface_heights).abs() <= height_tolerance
    feet = upward & near_surface
    tibia_without_support = (force_magnitudes[:, feet_indices] > force_threshold) & ~feet
    body = tibia_without_support.any(dim=-1)
    if body_indices:
        body |= (force_magnitudes[:, body_indices] > force_threshold).any(dim=-1)
    return feet, body
