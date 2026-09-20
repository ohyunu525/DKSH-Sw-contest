"""Mass and USD helpers for the Unitree L1 RM payload."""

from __future__ import annotations

from dataclasses import dataclass
import math


L1_RM_PAYLOAD_MASS_KG = 0.230
L1_RM_PAYLOAD_SIZE_M = (0.075, 0.075, 0.065)
L1_RM_MOUNT_POSITION_B = (0.0, 0.0, 0.030)
L1_RM_PRIM_NAME = "LidarPayload"


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


def apply_l1_rm_payload_to_stage(
    stage,
    base_prim_path: str,
    *,
    payload_mass: float = L1_RM_PAYLOAD_MASS_KG,
    payload_size: tuple[float, float, float] = L1_RM_PAYLOAD_SIZE_M,
    mount_position: tuple[float, float, float] = L1_RM_MOUNT_POSITION_B,
) -> MassProperties:
    """Apply L1 RM mass properties and a collidable enclosure to one USD base.

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
    if axes is not None:
        imaginary = axes.GetImaginary()
        if (
            abs(abs(float(axes.GetReal())) - 1.0) > 1.0e-6
            or max(abs(float(value)) for value in imaginary) > 1.0e-6
        ):
            raise RuntimeError("LiDAR payload requires base principal axes aligned with the body frame")

    payload_center = (
        mount_position[0],
        mount_position[1],
        mount_position[2] + payload_size[2] / 2.0,
    )
    combined = add_centered_box_payload(
        MassProperties(
            float(mass),
            tuple(float(value) for value in center),
            tuple(float(value) for value in inertia),
        ),
        payload_mass=payload_mass,
        payload_size=payload_size,
        payload_center=payload_center,
    )
    mass_api.GetMassAttr().Set(combined.mass)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*combined.center_of_mass))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*combined.diagonal_inertia))

    # A child collision shape belongs to the base rigid body.  This makes low
    # ceilings and falling debris interact with the real enclosure dimensions
    # without introducing an extra joint or rigid body into the articulation.
    payload_path = f"{base_prim_path}/{L1_RM_PRIM_NAME}"
    payload = UsdGeom.Cube.Define(stage, payload_path)
    payload.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(payload.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*payload_center))
    xform.AddScaleOp().Set(Gf.Vec3d(*payload_size))
    payload.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.12)])
    UsdPhysics.CollisionAPI.Apply(payload.GetPrim())
    return combined
