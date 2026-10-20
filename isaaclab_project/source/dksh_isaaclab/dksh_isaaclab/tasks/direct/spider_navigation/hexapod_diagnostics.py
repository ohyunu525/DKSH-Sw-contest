"""CAD foot frames and flat-ground diagnostics; no policy observations added."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import torch


URDF_PATH = Path(__file__).resolve().parents[6] / "assets/spiderbot_variants/onshape_20260928/robot.training.urdf"


def foot_offsets(urdf_path=URDF_PATH):
    """Recover the six massless foot-frame positions relative to their tibias."""
    root = ET.parse(urdf_path).getroot()
    return [tuple(float(x) for x in root.find(f"joint[@name='leg_{i}_foot_fixed']/origin").get("xyz").split())
            for i in range(6)]


def restore_foot_frames(stage, robot_path):
    """Keep CAD frames as Xforms without adding zero-inertia rigid bodies."""
    from pxr import Gf, UsdGeom

    root = ET.parse(URDF_PATH).getroot()
    for i, offset in enumerate(foot_offsets()):
        origin = root.find(f"joint[@name='leg_{i}_foot_fixed']/origin")
        roll, pitch, yaw = (float(x) for x in origin.get("rpy", "0 0 0").split())
        frame = UsdGeom.Xform.Define(stage, f"{robot_path}/leg_{i}_tibia/foot_frame")
        frame.AddTranslateOp().Set(Gf.Vec3d(*offset))
        rotation = (Gf.Rotation(Gf.Vec3d(0, 0, 1), np.degrees(yaw))
                    * Gf.Rotation(Gf.Vec3d(0, 1, 0), np.degrees(pitch))
                    * Gf.Rotation(Gf.Vec3d(1, 0, 0), np.degrees(roll)))
        frame.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))


def _stl_bounds(path):
    """Bounds of binary Onshape STL, in metres; STL export is already SI."""
    with path.open("rb") as stream:
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError(f"Truncated STL: {path}")
        count = struct.unpack_from("<I", header, 80)[0]
        facets = np.frombuffer(stream.read(), dtype=np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
        ]))
        if len(facets) != count or count == 0:
            raise ValueError(f"Invalid binary STL: {path}")
        points = facets["vertices"].reshape(-1, 3)
        return points.min(axis=0), points.max(axis=0)


def apply_simple_collisions(stage, robot_path, foot_radius):
    """Optional conservative mesh-local bounding boxes plus spherical foot soles.

    Bounds enclose each CAD collision part, not the full link (which could fill
    large empty gaps). Mass and inertia remain the original CAD values. Sphere
    radius is an explicit, unmeasured pad assumption.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    robot = stage.GetPrimAtPath(robot_path)
    # Uninstance collisions before disabling the referenced original meshes.
    for prim in list(Usd.PrimRange(robot)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetName() != "LidarPayload":
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    root = ET.parse(URDF_PATH).getroot()
    bounds_cache = {}
    for link in root.findall("link"):
        for index, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename")
            if filename not in bounds_cache:
                bounds_cache[filename] = _stl_bounds(URDF_PATH.parent / filename)
            low, high = bounds_cache[filename]
            origin = collision.find("origin")
            xyz = tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
            rpy = tuple(float(x) for x in origin.get("rpy", "0 0 0").split())
            frame = UsdGeom.Xform.Define(stage, f"{robot_path}/{link.get('name')}/SimpleCollision_{index}")
            frame.AddTranslateOp().Set(Gf.Vec3d(*xyz))
            # USD rotateXYZ applies Rx then Ry then Rz, matching URDF fixed RPY.
            frame.AddRotateXYZOp().Set(Gf.Vec3f(*np.degrees(rpy)))
            box = UsdGeom.Cube.Define(stage, f"{frame.GetPath()}/box")
            box.CreateSizeAttr(1.0)
            box.AddTranslateOp().Set(Gf.Vec3d(*((low + high) * 0.5).astype(float)))
            box.AddScaleOp().Set(Gf.Vec3d(*(high - low).astype(float)))
            box.CreateVisibilityAttr().Set("invisible")
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    if not np.isfinite(foot_radius) or foot_radius <= 0:
        raise ValueError("Assumed foot radius must be finite and positive")
    for i in range(6):
        sphere = UsdGeom.Sphere.Define(stage, f"{robot_path}/leg_{i}_tibia/foot_frame/SimpleSole")
        sphere.CreateRadiusAttr(foot_radius)
        # Pad-center offset is a provisional local-frame assumption, not a
        # measured sole thickness or a claim about the frame's world +Z axis.
        sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, foot_radius))
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())


class SlipTimer:
    """Qualify continuous foot slip by elapsed seconds instead of step count."""

    def __init__(self, shape, device, duration_s=0.060):
        if duration_s <= 0:
            raise ValueError("Slip duration must be positive")
        self.elapsed = torch.zeros(shape, device=device)
        self.duration_s = duration_s

    def update(self, in_contact, planar_speed, dt, speed_threshold=0.01):
        if dt <= 0:
            raise ValueError("Sample interval must be positive")
        slipping = in_contact & (planar_speed > speed_threshold)
        self.elapsed = torch.where(slipping, self.elapsed + dt, torch.zeros_like(self.elapsed))
        return self.elapsed >= self.duration_s - 1e-7


def foot_kinematics(robot, body_ids, offsets):
    """World foot point positions/velocities including the tibia angular term."""
    from isaaclab.utils.math import quat_apply

    rotated = quat_apply(robot.data.body_quat_w[:, body_ids], offsets.expand(robot.num_instances, -1, -1))
    position = robot.data.body_pos_w[:, body_ids] + rotated
    # PhysX body_lin_vel_w is the COM velocity, whereas body_pos_w is the
    # actor/link origin. Use the foot-to-COM lever arm, not the URDF offset.
    velocity = velocity_at_point(
        robot.data.body_lin_vel_w[:, body_ids], robot.data.body_ang_vel_w[:, body_ids],
        position, robot.data.body_com_pos_w[:, body_ids],
    )
    return position, velocity


def velocity_at_point(com_velocity, angular_velocity, point_position, com_position):
    """Rigid-body point velocity with both positions in the same world frame."""
    return com_velocity + torch.cross(angular_velocity, point_position - com_position, dim=-1)


def heading_increment(current, previous):
    """Shortest signed world-heading change, including a +/- pi crossing."""
    difference = current - previous
    return torch.atan2(torch.sin(difference), torch.cos(difference))
