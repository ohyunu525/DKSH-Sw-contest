"""Tensor-only sensing for axis-aligned scenario geometry.

These helpers do not depend on Isaac Sim or Isaac Lab and keep all geometry
calculations on the input device, including when tensors reside on CUDA.
"""

from __future__ import annotations

import math

import torch


def _batched_half_sizes(centers_w: torch.Tensor, half_sizes: torch.Tensor) -> torch.Tensor:
    """Validate geometry shapes without reading tensor values on the host."""
    if centers_w.ndim != 3 or centers_w.shape[-1] != 3:
        raise ValueError("centers_w must have shape (num_envs, num_boxes, 3)")
    if half_sizes.ndim == 2 and half_sizes.shape == centers_w.shape[1:]:
        return half_sizes.unsqueeze(0)
    if half_sizes.ndim == 3 and half_sizes.shape == centers_w.shape:
        return half_sizes
    raise ValueError("half_sizes must have shape (num_boxes, 3) or (num_envs, num_boxes, 3)")


def planar_box_ranges(
    positions_w: torch.Tensor,
    yaw: torch.Tensor,
    centers_w: torch.Tensor,
    half_sizes: torch.Tensor,
    max_range: float = 2.0,
    ray_count: int = 16,
) -> torch.Tensor:
    """Cast horizontal rays from robot bases against world-axis-aligned boxes.

    Args:
        positions_w: Robot base positions, shape ``(N, 3)``.
        yaw: Robot body headings in radians, shape ``(N,)``.
        centers_w: Box centers, shape ``(N, B, 3)``.
        half_sizes: Nonnegative box half extents, shape ``(B, 3)`` or ``(N, B, 3)``.
        max_range: Positive distance at which measurements saturate, in meters.
        ray_count: Number of rays evenly spaced counterclockwise from body +X.

    Returns:
        Normalized nearest-hit distances, shape ``(N, ray_count)``. A value of
        one means no hit within ``max_range``; zero means the origin touches or
        lies inside a box. Boxes outside the base's height are ignored. Inputs
        must use compatible floating dtypes and the same device.
    """
    if not math.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range must be positive and finite")
    if not isinstance(ray_count, int) or ray_count <= 0:
        raise ValueError("ray_count must be a positive integer")
    sizes = _batched_half_sizes(centers_w, half_sizes)
    if positions_w.shape != (centers_w.shape[0], 3):
        raise ValueError("positions_w must have shape (num_envs, 3)")
    if yaw.shape != positions_w.shape[:1]:
        raise ValueError("yaw must have shape (num_envs,)")
    if centers_w.shape[1] == 0:
        return positions_w.new_ones((positions_w.shape[0], ray_count))

    angles = yaw[:, None] + torch.arange(
        ray_count, device=positions_w.device, dtype=positions_w.dtype
    )[None, :] * (2.0 * math.pi / ray_count)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).unsqueeze(2)
    lower = (centers_w[..., :2] - sizes[..., :2] - positions_w[:, None, :2]).unsqueeze(1)
    upper = (centers_w[..., :2] + sizes[..., :2] - positions_w[:, None, :2]).unsqueeze(1)

    # Cardinal rays are only approximately zero after sin/cos. Treat their
    # near-zero component as parallel and avoid both division by zero and 0*inf.
    parallel = directions.abs() <= 4.0 * torch.finfo(directions.dtype).eps
    inverse = torch.where(parallel, torch.ones_like(directions), directions).reciprocal()
    first = lower * inverse
    second = upper * inverse
    near = torch.where(parallel, -torch.inf, torch.minimum(first, second)).amax(dim=-1)
    far = torch.where(parallel, torch.inf, torch.maximum(first, second)).amin(dim=-1)
    parallel_outside = (parallel & ((lower > 0.0) | (upper < 0.0))).any(dim=-1)
    within_height = (
        (positions_w[:, None, 2] >= centers_w[..., 2] - sizes[..., 2])
        & (positions_w[:, None, 2] <= centers_w[..., 2] + sizes[..., 2])
    )
    distance = near.clamp_min(0.0)
    valid = (far >= distance) & ~parallel_outside & within_height[:, None, :]
    hits = torch.where(valid, distance, torch.inf)
    return hits.amin(dim=-1).clamp(max=max_range) / max_range


def sample_box_heights(
    points_w: torch.Tensor,
    centers_w: torch.Tensor,
    half_sizes: torch.Tensor,
    default_height: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample the highest box top at each world XY point, above a base plane.

    ``points_w`` has shape ``(N, P, 2)``, ``centers_w`` has shape ``(N, B, 3)``,
    and ``half_sizes`` has shape ``(B, 3)`` or ``(N, B, 3)``. The optional base
    plane height has shape ``(N,)`` and defaults to zero. Returns ``(N, P)``.
    Box edges count as supported. Callers should supply only terrain/support
    boxes: this top-surface query does not distinguish ceilings or falling debris.
    """
    sizes = _batched_half_sizes(centers_w, half_sizes)
    if points_w.ndim != 3 or points_w.shape[0] != centers_w.shape[0] or points_w.shape[-1] != 2:
        raise ValueError("points_w must have shape (num_envs, num_points, 2)")
    if default_height is None:
        default_height = points_w.new_zeros(points_w.shape[0])
    elif default_height.shape != points_w.shape[:1]:
        raise ValueError("default_height must have shape (num_envs,)")
    base = default_height[:, None].expand(points_w.shape[:2])
    if centers_w.shape[1] == 0:
        return base.clone()

    offset = points_w[:, :, None, :] - centers_w[:, None, :, :2]
    inside = (offset.abs() <= sizes[:, None, :, :2]).all(dim=-1)
    top = centers_w[..., 2] + sizes[..., 2]
    heights = torch.where(inside, top[:, None, :], -torch.inf).amax(dim=-1)
    return torch.maximum(base, heights)
