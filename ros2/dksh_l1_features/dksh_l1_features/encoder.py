"""Pure-Python Unitree L1 RM and L2 point-cloud feature contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


L1_MIN_RANGE_M = 0.05
L1_MAX_RANGE_M = 30.0
L1_RESOLUTION_M = 0.008
L1_VERTICAL_FOV_DEG = 90.0
L1_SECTORS = 16

L2_MIN_RANGE_M = 0.05
L2_MAX_RANGE_M = 30.0
L2_RESOLUTION_M = 0.0045
L2_VERTICAL_FOV_DEG = 96.0
L2_VERTICAL_MIN_DEG = -6.0
L2_SECTORS = 16


def encode_l1_obstacles(
    points_lidar: Iterable[Sequence[float]],
    *,
    sectors: int = L1_SECTORS,
    min_range_m: float = L1_MIN_RANGE_M,
    max_range_m: float = L1_MAX_RANGE_M,
    resolution_m: float = L1_RESOLUTION_M,
    vertical_fov_deg: float = L1_VERTICAL_FOV_DEG,
    vertical_min_deg: float = 0.0,
) -> list[float]:
    """Return normalized nearest ranges followed by return-valid flags.

    Bin zero is centered on sensor +X; bin indices increase toward +Y.
    Input must be one complete obstacle-only frame in the L1 optical frame.
    """
    if sectors <= 0 or not isinstance(sectors, int):
        raise ValueError("sectors must be a positive integer")
    if not (0.0 <= min_range_m < max_range_m < math.inf):
        raise ValueError("range limits must satisfy 0 <= min < max < infinity")
    if not (0.0 < resolution_m < math.inf):
        raise ValueError("resolution_m must be positive and finite")
    if not (0.0 <= vertical_fov_deg <= 180.0):
        raise ValueError("vertical_fov_deg must be between 0 and 180")
    if not (-90.0 <= vertical_min_deg <= 90.0) or vertical_min_deg + vertical_fov_deg > 90.0:
        raise ValueError("vertical elevation limits must be within -90 and 90 degrees")

    nearest = [max_range_m] * sectors
    min_elevation = math.radians(vertical_min_deg)
    max_elevation = math.radians(vertical_min_deg + vertical_fov_deg)
    sector_width = 2.0 * math.pi / sectors
    for point in points_lidar:
        if len(point) < 3:
            raise ValueError("each point must contain x, y, z")
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        planar = math.hypot(x, y)
        distance = math.hypot(planar, z)
        if not min_range_m <= distance < max_range_m:
            continue
        elevation = math.atan2(z, planar)
        if not min_elevation <= elevation <= max_elevation:
            continue
        azimuth = math.atan2(y, x)
        bin_index = math.floor((azimuth + 0.5 * sector_width) / sector_width) % sectors
        nearest[bin_index] = min(nearest[bin_index], distance)

    ranges = [
        min(max(round(distance / resolution_m) * resolution_m, min_range_m), max_range_m) / max_range_m
        if distance < max_range_m else 1.0
        for distance in nearest
    ]
    valid = [float(distance < 1.0) for distance in ranges]
    return ranges + valid


def encode_l2_obstacles(
    points_lidar: Iterable[Sequence[float]],
    *,
    sectors: int = L2_SECTORS,
    min_range_m: float = L2_MIN_RANGE_M,
    max_range_m: float = L2_MAX_RANGE_M,
    resolution_m: float = L2_RESOLUTION_M,
    vertical_fov_deg: float = L2_VERTICAL_FOV_DEG,
    vertical_min_deg: float = L2_VERTICAL_MIN_DEG,
) -> list[float]:
    """Encode one obstacle-only L2 NEGA frame in its optical (+X,+Y,+Z) frame.

    The upright L2 covers elevations -6..90 degrees. Self and ground returns
    must be removed before encoding, including returns in the negative slice.
    """
    return encode_l1_obstacles(
        points_lidar, sectors=sectors, min_range_m=min_range_m,
        max_range_m=max_range_m, resolution_m=resolution_m,
        vertical_fov_deg=vertical_fov_deg, vertical_min_deg=vertical_min_deg,
    )
