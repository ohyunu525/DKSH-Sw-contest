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


def spatial_obstacle_ranges(
    positions_w: torch.Tensor,
    quaternion_w: torch.Tensor,
    centers_w: torch.Tensor,
    half_sizes: torch.Tensor,
    *,
    max_range: float,
    horizontal_count: int,
    azimuth_samples_per_bin: int = 1,
    vertical_fov_degrees: float,
    vertical_count: int,
    sphere_centers_w: torch.Tensor | None = None,
    sphere_radii: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Project a body-aligned 3-D scan into nearest ranges per azimuth.

    Static obstacles use axis-aligned boxes and dynamic obstacles may be supplied
    as spheres. Rays rotate with the complete body quaternion, so roll and pitch
    affect ceiling, step, and debris returns. The nearest vertical return is kept
    for each horizontal bin to preserve the existing policy observation size.
    """
    if not math.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range must be positive and finite")
    if not isinstance(horizontal_count, int) or horizontal_count <= 0:
        raise ValueError("horizontal_count must be a positive integer")
    if not isinstance(azimuth_samples_per_bin, int) or azimuth_samples_per_bin <= 0:
        raise ValueError("azimuth_samples_per_bin must be a positive integer")
    if not isinstance(vertical_count, int) or vertical_count <= 0:
        raise ValueError("vertical_count must be a positive integer")
    if not math.isfinite(vertical_fov_degrees) or not 0.0 <= vertical_fov_degrees <= 180.0:
        raise ValueError("vertical_fov_degrees must be finite and between zero and 180")
    sizes = _batched_half_sizes(centers_w, half_sizes)
    if positions_w.shape != (centers_w.shape[0], 3):
        raise ValueError("positions_w must have shape (num_envs, 3)")
    if quaternion_w.shape != (centers_w.shape[0], 4):
        raise ValueError("quaternion_w must have shape (num_envs, 4)")

    dtype, device = positions_w.dtype, positions_w.device
    sector_width = 2.0 * math.pi / horizontal_count
    horizontal = torch.arange(horizontal_count, dtype=dtype, device=device)[:, None] * sector_width
    offsets = (torch.arange(azimuth_samples_per_bin, dtype=dtype, device=device) + 0.5) / azimuth_samples_per_bin - 0.5
    horizontal = (horizontal + offsets[None, :] * sector_width).reshape(-1)
    horizontal_rays = horizontal_count * azimuth_samples_per_bin
    if vertical_count == 1:
        vertical = positions_w.new_zeros(1)
    else:
        # The upright L1 scans the hemisphere above its mounting plane.
        vertical = torch.linspace(0.0, math.radians(vertical_fov_degrees), vertical_count, dtype=dtype, device=device)
    azimuth = horizontal[:, None].expand(horizontal_rays, vertical_count)
    elevation = vertical[None, :].expand(horizontal_rays, vertical_count)
    cos_elevation = elevation.cos()
    local_directions = torch.stack(
        (
            cos_elevation * azimuth.cos(),
            cos_elevation * azimuth.sin(),
            elevation.sin(),
        ),
        dim=-1,
    ).reshape(-1, 3)
    local_directions = local_directions.unsqueeze(0).expand(positions_w.shape[0], -1, -1)

    # Isaac Lab stores quaternions as wxyz. Expand the vector formula instead of
    # relying on a simulator utility so this geometry remains unit-testable.
    quaternion_xyz = quaternion_w[:, None, 1:].expand_as(local_directions)
    uv = 2.0 * torch.cross(quaternion_xyz, local_directions, dim=-1)
    directions_w = (
        local_directions
        + quaternion_w[:, None, :1] * uv
        + torch.cross(quaternion_xyz, uv, dim=-1)
    )

    ray_count = horizontal_rays * vertical_count
    nearest = positions_w.new_full((positions_w.shape[0], ray_count), max_range)
    if centers_w.shape[1] > 0:
        ray_directions = directions_w.unsqueeze(2)
        lower = (centers_w - sizes - positions_w[:, None, :]).unsqueeze(1)
        upper = (centers_w + sizes - positions_w[:, None, :]).unsqueeze(1)
        parallel = ray_directions.abs() <= 4.0 * torch.finfo(dtype).eps
        inverse = torch.where(parallel, torch.ones_like(ray_directions), ray_directions).reciprocal()
        first = lower * inverse
        second = upper * inverse
        near = torch.where(parallel, -torch.inf, torch.minimum(first, second)).amax(dim=-1)
        far = torch.where(parallel, torch.inf, torch.maximum(first, second)).amin(dim=-1)
        parallel_outside = (parallel & ((lower > 0.0) | (upper < 0.0))).any(dim=-1)
        distance = near.clamp_min(0.0)
        valid = (far >= distance) & ~parallel_outside
        box_hits = torch.where(valid, distance, torch.inf).amin(dim=-1)
        nearest = torch.minimum(nearest, box_hits)

    if sphere_centers_w is not None:
        if sphere_centers_w.ndim != 3 or sphere_centers_w.shape[0] != positions_w.shape[0] or sphere_centers_w.shape[-1] != 3:
            raise ValueError("sphere_centers_w must have shape (num_envs, num_spheres, 3)")
        if sphere_radii is None:
            raise ValueError("sphere_radii is required with sphere_centers_w")
        radii = torch.as_tensor(sphere_radii, dtype=dtype, device=device)
        if radii.ndim == 0:
            radii = radii.expand(sphere_centers_w.shape[:2])
        elif radii.ndim == 1 and radii.shape[0] == sphere_centers_w.shape[1]:
            radii = radii.unsqueeze(0).expand(sphere_centers_w.shape[:2])
        elif radii.shape != sphere_centers_w.shape[:2]:
            raise ValueError("sphere_radii must be scalar, (num_spheres,), or (num_envs, num_spheres)")
        offset = positions_w[:, None, :] - sphere_centers_w
        projection = (directions_w[:, :, None, :] * offset[:, None, :, :]).sum(dim=-1)
        center_term = offset.square().sum(dim=-1) - radii.square()
        discriminant = projection.square() - center_term[:, None, :]
        root = discriminant.clamp_min(0.0).sqrt()
        near = -projection - root
        far = -projection + root
        distance = torch.where(center_term[:, None, :] <= 0.0, 0.0, near)
        valid = (discriminant >= 0.0) & (far >= 0.0)
        sphere_hits = torch.where(valid, distance.clamp_min(0.0), torch.inf).amin(dim=-1)
        nearest = torch.minimum(nearest, sphere_hits)

    nearest = nearest.clamp(max=max_range).reshape(
        positions_w.shape[0], horizontal_count, azimuth_samples_per_bin, vertical_count
    )
    return nearest.amin(dim=(-1, -2)) / max_range


def apply_lidar_measurement_model(
    normalized_ranges: torch.Tensor,
    *,
    max_range: float,
    min_range: float,
    accuracy: float,
    resolution: float,
    noise: torch.Tensor | None = None,
    validate_tensors: bool = True,
) -> torch.Tensor:
    """Apply blind-zone, bounded accuracy, and resolution to normalized ranges.

    Values equal to one represent no return and remain exactly one. ``noise``
    is a unitless tensor in ``[-1, 1]``; callers may provide zeros for a
    deterministic measurement or random values for domain randomization.
    """
    values = (max_range, min_range, accuracy, resolution)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LiDAR measurement parameters must be finite")
    if max_range <= 0.0:
        raise ValueError("max_range must be positive")
    if min_range < 0.0 or min_range >= max_range:
        raise ValueError("min_range must be nonnegative and less than max_range")
    if accuracy < 0.0:
        raise ValueError("accuracy must be nonnegative")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if noise is None:
        noise = torch.zeros_like(normalized_ranges)
    if noise.shape != normalized_ranges.shape:
        raise ValueError("noise must have the same shape as normalized_ranges")
    if validate_tensors:
        if not torch.isfinite(normalized_ranges).all() or not torch.isfinite(noise).all():
            raise ValueError("ranges and noise must contain only finite values")
        if (normalized_ranges < 0.0).any() or (normalized_ranges > 1.0).any():
            raise ValueError("normalized_ranges must be between zero and one")
        if (noise.abs() > 1.0).any():
            raise ValueError("noise must be between -1 and one")

    # A geometric intersection inside the L1 near field cannot be measured.
    has_return = (normalized_ranges < 1.0) & (normalized_ranges * max_range >= min_range)
    distance = normalized_ranges * max_range + noise * accuracy
    distance = torch.round(distance / resolution) * resolution
    distance = distance.clamp(min=min_range, max=max_range)
    return torch.where(has_return, distance / max_range, torch.ones_like(distance))


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
