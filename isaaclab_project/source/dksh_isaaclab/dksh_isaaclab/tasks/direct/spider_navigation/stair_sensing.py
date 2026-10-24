"""Torch-only stair geometry and visible-return LiDAR surrogate.

No terrain height query is exposed to the actor. Rays intersect risers, treads
and ground; occluded surfaces cannot enter the local observed-height grid.
This is NOT an exact Unitree scan pattern or a reflectance/self-occlusion model.
"""
import math
import torch

HEIGHTS_M = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
FIRST_RISER_M = 0.40
TREAD_M = 0.25
STEP_COUNT = 3
WIDTH_M = 1.2
PLATFORM_M = 0.60
PATCH_SPACING_M = 2.5
GRID_SHAPE = (12, 8)
OBSERVATIONS = 48 + 2 * GRID_SHAPE[0] * GRID_SHAPE[1]
TASK = 'Isaac-DKSH-MG90S-CAD6-Stairs-Sensors-Direct-v0'


def stair_boxes(heights):
    """Local AABBs [batch, step, xyz], same boxes used for physical collision."""
    if heights.ndim != 1 or not torch.isfinite(heights).all() or (heights <= 0).any():
        raise ValueError('Step heights must be finite and positive')
    index = torch.arange(STEP_COUNT, device=heights.device, dtype=heights.dtype)
    lower = torch.zeros((len(heights), STEP_COUNT, 3), device=heights.device, dtype=heights.dtype)
    lower[:, :, 0] = FIRST_RISER_M + index * TREAD_M
    lower[:, :, 1] = -WIDTH_M / 2
    upper = lower.clone()
    upper[:, :, 0] += TREAD_M
    upper[:, -1, 0] += PLATFORM_M
    upper[:, :, 1] = WIDTH_M / 2
    upper[:, :, 2] = heights[:, None] * (index + 1)
    return lower, upper


def terrain_height(x, heights):
    """Privileged ground truth ONLY for termination/evaluation, never policy."""
    count = (torch.floor((x - FIRST_RISER_M) / TREAD_M) + 1).clamp(0, STEP_COUNT)
    end = FIRST_RISER_M + STEP_COUNT * TREAD_M + PLATFORM_M
    return torch.where(x < end, count * heights, torch.zeros_like(x))


def lidar_directions(device, dtype=torch.float32):
    # Assumed inverted L2: upward hemisphere becomes [-90, +6] degrees.
    # Decimated 10-degree azimuth / 6-degree elevation, not 64k real points/s.
    az = torch.arange(0, 360, 10, device=device, dtype=dtype) * math.pi / 180
    el = torch.arange(-90, 7, 6, device=device, dtype=dtype) * math.pi / 180
    elev, azi = torch.meshgrid(el, az, indexing='ij')
    return torch.stack((elev.cos() * azi.cos(), elev.cos() * azi.sin(), elev.sin()), -1).reshape(-1, 3)


def raycast_stairs(origins, directions, lower, upper, max_range=1.5, min_range=0.05):
    """Nearest visible surface; returns [batch, ray] distance and validity."""
    origin = origins[:, None, None, :]
    direction = directions[:, :, None, :]
    parallel = direction.abs() < 1e-7
    inverse = torch.where(parallel, torch.ones_like(direction), direction).reciprocal()
    a = (lower[:, None] - origin) * inverse
    b = (upper[:, None] - origin) * inverse
    near = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(-1)
    far = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(-1)
    outside_parallel = (parallel & ((origin < lower[:, None]) | (origin > upper[:, None]))).any(-1)
    hit = torch.where(near >= 0, near, far)
    valid_box = ~outside_parallel & (far >= near) & (hit >= 0)
    distance = torch.where(valid_box, hit, torch.inf).amin(-1)
    dz = directions[:, :, 2]
    plane = -origins[:, None, 2] / torch.where(dz.abs() > 1e-7, dz, torch.ones_like(dz))
    plane = torch.where((dz < -1e-7) & (plane >= 0), plane, torch.inf)
    distance = torch.minimum(distance, plane)
    # Do NOT see through a blind-zone hit to a surface behind it.
    valid = torch.isfinite(distance) & (distance >= min_range) & (distance <= max_range)
    return distance.clamp(0, max_range), valid


def observed_height_grid(points_b, valid):
    """Highest observed return per cell + explicit unknown mask."""
    nx, ny = GRID_SHAPE
    ix = torch.floor((points_b[..., 0] + 0.3) / 0.1).long()
    iy = torch.floor((points_b[..., 1] + 0.4) / 0.1).long()
    valid = valid & torch.isfinite(points_b).all(-1) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    indices = (ix.clamp(0, nx - 1) * ny + iy.clamp(0, ny - 1))
    heights = torch.full((len(points_b), nx * ny), -torch.inf, device=points_b.device)
    heights.scatter_reduce_(1, indices, torch.where(valid, points_b[..., 2], -torch.inf), reduce='amax')
    mask = torch.isfinite(heights)
    heights = torch.where(mask, heights / 0.3, 0).clamp(-1, 1)
    return torch.cat((heights, mask.float()), -1)


def next_levels(levels, success, failed, streaks, maximum):
    """Require 3 consecutive full-flight successes; timeout is NOT success."""
    streaks = torch.where(success, streaks + 1, torch.zeros_like(streaks))
    advance = success & (streaks >= 3) & (levels < maximum)
    levels = torch.where(advance, levels + 1, levels)
    levels = torch.where(failed, (levels - 1).clamp_min(0), levels)
    return levels, torch.where(advance, torch.zeros_like(streaks), streaks)
