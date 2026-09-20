"""Pure-Python Unitree L1 RM point-cloud feature contract."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


L1_MIN_RANGE_M = 0.05
L1_MAX_RANGE_M = 30.0
L1_RESOLUTION_M = 0.008
L1_VERTICAL_FOV_DEG = 90.0
L1_SECTORS = 16


def encode_l1_obstacles(
    points_lidar: Iterable[Sequence[float]],
    *,
    sectors: int = L1_SECTORS,
    min_range_m: float = L1_MIN_RANGE_M,
    max_range_m: float = L1_MAX_RANGE_M,
    resolution_m: float = L1_RESOLUTION_M,
    vertical_fov_deg: float = L1_VERTICAL_FOV_DEG,
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

    nearest = [max_range_m] * sectors
    max_elevation = math.radians(vertical_fov_deg)
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
        if not 0.0 <= elevation <= max_elevation:
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
