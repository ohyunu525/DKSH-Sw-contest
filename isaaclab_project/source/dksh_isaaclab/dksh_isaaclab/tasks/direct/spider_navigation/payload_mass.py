"""Mass and USD helpers for the Unitree L2 payload."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


L2_PAYLOAD_MASS_KG = 0.230
L2_PAYLOAD_SIZE_M = (0.075, 0.075, 0.065)
L2_MOUNT_POSITION_B = (0.0, 0.0, 0.030)
L2_PRIM_NAME = "LidarPayload"

# L1 RM and L2 share these published mass/enclosure values. Keep the old
# imports available for tools that inspect archived L1 training artifacts.
L1_RM_PAYLOAD_MASS_KG = L2_PAYLOAD_MASS_KG
L1_RM_PAYLOAD_SIZE_M = L2_PAYLOAD_SIZE_M
L1_RM_MOUNT_POSITION_B = L2_MOUNT_POSITION_B
L1_RM_PRIM_NAME = L2_PRIM_NAME


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


def _quaternion_to_matrix(real: float, imaginary: tuple[float, float, float]) -> np.ndarray:
    """Return the body-frame rotation matrix of a USD principal-axis quaternion."""
    w = float(real)
    x, y, z = (float(value) for value in imaginary)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("principal-axis quaternion must be non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ), dtype=float)


def _matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to USD's (real, i, j, k) layout."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)


def add_oriented_centered_box_payload(
    base: MassProperties,
    *,
    principal_axes: tuple[float, float, float, float],
    payload_mass: float,
    payload_size: tuple[float, float, float],
    payload_center: tuple[float, float, float],
) -> tuple[MassProperties, tuple[float, float, float, float]]:
    """Combine a centered payload with CAD inertias whose principal axes are rotated.

    Onshape exports a valid principal-axis quaternion for the base.  USD stores
    only diagonal inertias, so re-diagonalize the combined body-frame tensor and
    return its new quaternion rather than silently discarding that orientation.
    """
    scalars = (*base.center_of_mass, *base.diagonal_inertia, payload_mass, *payload_size, *payload_center)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("mass properties must be finite")
    if base.mass <= 0.0 or payload_mass <= 0.0 or any(size <= 0.0 for size in payload_size):
        raise ValueError("base/payload mass and payload dimensions must be positive")
    total_mass = base.mass + payload_mass
    center = tuple(
        (base.mass * base.center_of_mass[index] + payload_mass * payload_center[index]) / total_mass
        for index in range(3)
    )
    rotation = _quaternion_to_matrix(principal_axes[0], principal_axes[1:])
    base_tensor = rotation @ np.diag(base.diagonal_inertia) @ rotation.T
    sx, sy, sz = payload_size
    payload_tensor = np.diag((
        payload_mass * (sy**2 + sz**2) / 12.0,
        payload_mass * (sx**2 + sz**2) / 12.0,
        payload_mass * (sx**2 + sy**2) / 12.0,
    ))
    base_offset = np.array(base.center_of_mass) - np.array(center)
    payload_offset = np.array(payload_center) - np.array(center)
    def parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
        return mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    tensor = base_tensor + payload_tensor + parallel_axis(base.mass, base_offset) + parallel_axis(payload_mass, payload_offset)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, 0] *= -1.0
    real, x, y, z = _matrix_to_quaternion(eigenvectors)
    return MassProperties(total_mass, center, tuple(float(value) for value in eigenvalues)), (real, x, y, z)


def apply_l2_payload_to_stage(
    stage,
    base_prim_path: str,
    *,
    payload_mass: float = L2_PAYLOAD_MASS_KG,
    payload_size: tuple[float, float, float] = L2_PAYLOAD_SIZE_M,
    mount_position: tuple[float, float, float] = L2_MOUNT_POSITION_B,
) -> MassProperties:
    """Apply L2 mass properties and a collidable enclosure to one USD base.

    Imports USD lazily so the pure mass helper remains usable in normal Python
    unit tests without an Isaac Sim runtime.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    base_prim = stage.GetPrimAtPath(base_prim_path)
    mass_api = UsdPhysics.MassAPI(base_prim)
    if not base_prim or not mass_api:
        raise RuntimeError("Robot base is missing USD mass properties for the LiDAR payload")

    mass = mass_api.GetMassAttr().Get()
    center = mass_api.GetCenterOfMassAttr().Get()
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    axes = mass_api.GetPrincipalAxesAttr().Get()
    if mass is None or center is None or inertia is None:
        raise RuntimeError("Robot base mass, center of mass, and inertia must be authored")
    payload_center = (
        mount_position[0],
        mount_position[1],
        mount_position[2] + payload_size[2] / 2.0,
    )
    combined, principal_axes = add_oriented_centered_box_payload(
        MassProperties(
            float(mass),
            tuple(float(value) for value in center),
            tuple(float(value) for value in inertia),
        ),
        principal_axes=(float(axes.GetReal()), *(float(value) for value in axes.GetImaginary())) if axes else (1.0, 0.0, 0.0, 0.0),
        payload_mass=payload_mass,
        payload_size=payload_size,
        payload_center=payload_center,
    )
    mass_api.GetMassAttr().Set(combined.mass)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*combined.center_of_mass))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*combined.diagonal_inertia))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(principal_axes[0], Gf.Vec3f(*principal_axes[1:])))

    # A child collision shape belongs to the base rigid body.  This makes low
    # ceilings and falling debris interact with the real enclosure dimensions
    # without introducing an extra joint or rigid body into the articulation.
    payload_path = f"{base_prim_path}/{L2_PRIM_NAME}"
    payload = UsdGeom.Cube.Define(stage, payload_path)
    payload.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(payload.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*payload_center))
    xform.AddScaleOp().Set(Gf.Vec3d(*payload_size))
    payload.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.12)])
    UsdPhysics.CollisionAPI.Apply(payload.GetPrim())
    return combined


apply_l1_rm_payload_to_stage = apply_l2_payload_to_stage
