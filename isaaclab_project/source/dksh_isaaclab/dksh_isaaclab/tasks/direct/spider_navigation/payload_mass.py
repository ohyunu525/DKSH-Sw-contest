"""Pure helpers for adding a centered box payload to a rigid body."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MassProperties:
    mass: float
    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]


def add_centered_box_payload(
    base: MassProperties,
    *,
    payload_mass: float,
    payload_size: tuple[float, float, float],
    payload_center: tuple[float, float, float],
) -> MassProperties:
    """Combine aligned principal inertias using the parallel-axis theorem.

    The payload and base centers must share X/Y coordinates.  This matches a
    centered LiDAR mount and keeps the combined principal axes aligned with the
    body frame, so a diagonal USD inertia remains exact.
    """
    scalars = (
        base.mass,
        *base.center_of_mass,
        *base.diagonal_inertia,
        payload_mass,
        *payload_size,
        *payload_center,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("mass properties must be finite")
    if base.mass <= 0.0 or payload_mass <= 0.0:
        raise ValueError("base and payload masses must be positive")
    if any(size <= 0.0 for size in payload_size):
        raise ValueError("payload dimensions must be positive")
    if any(inertia <= 0.0 for inertia in base.diagonal_inertia):
        raise ValueError("base diagonal inertia must be positive")
    if not math.isclose(base.center_of_mass[0], payload_center[0], abs_tol=1.0e-7):
        raise ValueError("payload must be centered on the base X coordinate")
    if not math.isclose(base.center_of_mass[1], payload_center[1], abs_tol=1.0e-7):
        raise ValueError("payload must be centered on the base Y coordinate")

    total_mass = base.mass + payload_mass
    combined_z = (
        base.mass * base.center_of_mass[2] + payload_mass * payload_center[2]
    ) / total_mass
    center = (base.center_of_mass[0], base.center_of_mass[1], combined_z)
    size_x, size_y, size_z = payload_size
    payload_inertia = (
        payload_mass * (size_y**2 + size_z**2) / 12.0,
        payload_mass * (size_x**2 + size_z**2) / 12.0,
        payload_mass * (size_x**2 + size_y**2) / 12.0,
    )
    base_dz = base.center_of_mass[2] - combined_z
    payload_dz = payload_center[2] - combined_z
    translated = base.mass * base_dz**2 + payload_mass * payload_dz**2
    inertia = (
        base.diagonal_inertia[0] + payload_inertia[0] + translated,
        base.diagonal_inertia[1] + payload_inertia[1] + translated,
        base.diagonal_inertia[2] + payload_inertia[2],
    )
    return MassProperties(total_mass, center, inertia)
