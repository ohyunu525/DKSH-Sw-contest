"""Verify fixed-foot support witnesses against the actual Onshape assets."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import re
import struct
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
MODULE = PROJECT / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_foot_geometry.py"
ASSET_DIR = PROJECT / "assets/spiderbot_variants/onshape_20260928"


def load_module(name="foot_geometry_under_test"):
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEOMETRY = load_module()

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None

try:
    import numpy as np
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None


def identity():
    return tuple(tuple(float(i == j) for j in range(4)) for i in range(4))


def multiply(left, right):
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(4))
                       for j in range(4)) for i in range(4))


def transform(matrix, point):
    return tuple(sum(matrix[i][j] * point[j] for j in range(3)) + matrix[i][3]
                 for i in range(3))


def vector(value):
    return tuple(map(float, value.split()))


def rotation(axis, angle):
    length = math.sqrt(sum(value * value for value in axis))
    x, y, z = (value / length for value in axis)
    c, s, v = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return ((x*x*v+c, x*y*v-z*s, x*z*v+y*s, 0.0),
            (x*y*v+z*s, y*y*v+c, y*z*v-x*s, 0.0),
            (x*z*v-y*s, y*z*v+x*s, z*z*v+c, 0.0),
            (0.0, 0.0, 0.0, 1.0))


def origin(element):
    if element is None:
        return identity()
    roll, pitch, yaw = vector(element.get("rpy", "0 0 0"))
    matrix = multiply(rotation((0, 0, 1), yaw),
                      multiply(rotation((0, 1, 0), pitch), rotation((1, 0, 0), roll)))
    position = vector(element.get("xyz", "0 0 0"))
    return tuple(tuple(matrix[i][j] if j != 3 or i == 3 else position[i]
                       for j in range(4)) for i in range(4))


def stl_points(path):
    data = path.read_bytes()
    triangles = struct.unpack_from("<I", data, 80)[0]
    points = set()
    for index in range(triangles):
        for offset in (12, 24, 36):
            points.add(struct.unpack_from("<fff", data, 84 + 50 * index + offset))
    return tuple(sorted(points))


class FootGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = ET.parse(ASSET_DIR / "robot.training.urdf").getroot()
        cls.joints = {joint.find("child").get("link"): joint
                      for joint in cls.urdf.findall("joint")}
        cls.foot_points = []
        for leg in range(6):
            link = cls.urdf.find(f"link[@name='leg_{leg}_tibia']")
            foot_collisions = [collision for collision in link.findall("collision")
                               if collision.find("geometry/mesh").get("filename").endswith("/part_1__8.stl")]
            if len(foot_collisions) != 1:
                raise AssertionError("Each tibia must contain exactly one actual fixed-foot mesh")
            collision = foot_collisions[0]
            mesh = collision.find("geometry/mesh")
            points = stl_points(ASSET_DIR / mesh.get("filename"))
            scale = vector(mesh.get("scale", "1 1 1"))
            matrix = origin(collision.find("origin"))
            cls.foot_points.append(tuple(transform(matrix, tuple(x*s for x, s in zip(point, scale)))
                                         for point in points))

    def world_frame(self, link_name, angles):
        if link_name == "base":
            return identity()
        joint = self.joints[link_name]
        matrix = origin(joint.find("origin"))
        if joint.get("type") != "fixed":
            matrix = multiply(matrix, rotation(vector(joint.find("axis").get("xyz")),
                                               angles.get(joint.get("name"), 0.0)))
        return multiply(self.world_frame(joint.find("parent").get("link"), angles), matrix)

    def test_runtime_constants_need_no_numerical_or_usd_packages(self):
        normal_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "scipy", "pxr"):
                raise AssertionError(f"Runtime geometry constants imported {name}")
            return normal_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=checked_import):
            geometry = load_module("foot_geometry_without_optional_packages")
        self.assertEqual(geometry.ONSHAPE_FOOT_SUPPORT_OFFSETS, GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS)

    def test_uniform_exact_hull_contract_and_source_provenance(self):
        self.assertIsInstance(GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS, tuple)
        self.assertEqual(len(GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS), 6)
        self.assertEqual(GEOMETRY.ONSHAPE_FOOT_SUPPORT_VERTEX_COUNT, 137)
        for offsets in GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS:
            self.assertIsInstance(offsets, tuple)
            self.assertEqual(len(offsets), 137)
            self.assertEqual(len(set(offsets)), 137)
            self.assertTrue(all(isinstance(point, tuple) and len(point) == 3
                                and all(math.isfinite(value) for value in point) for point in offsets))
        recorded_hash = re.search(r"Source USD SHA256: ([a-f0-9]{64})", GEOMETRY.__doc__).group(1)
        self.assertEqual(recorded_hash, hashlib.sha256((ASSET_DIR / "onshape_20260928.usd").read_bytes()).hexdigest())

    def test_every_anchor_belongs_to_actual_foot_and_excludes_tibia_servo(self):
        for anchors, foot_points in zip(GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS, self.foot_points):
            for anchor in anchors:
                error = min(max(abs(a-b) for a, b in zip(anchor, point)) for point in foot_points)
                self.assertLess(error, 3e-7)
            # The separate tibia servo mesh extends up to z=+0.00362 m.
            # Every foot-only witness is below z=-0.01272 m in the link frame.
            self.assertLess(max(point[2] for point in anchors), -0.012719)

    def test_hull_preserves_oriented_support_extrema_in_all_six_link_frames(self):
        directions = [direction for direction in itertools.product((-1.0, 0.0, 1.0), repeat=3)
                      if any(direction)]
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(64):
            z = 1.0 - 2.0 * (index + 0.5) / 64
            radius = math.sqrt(1.0 - z*z)
            directions.append((radius * math.cos(index * golden_angle),
                               radius * math.sin(index * golden_angle), z))
        for anchors, foot_points in zip(GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS, self.foot_points):
            for direction in directions:
                lowest = lambda points: min(sum(p*d for p, d in zip(point, direction)) for point in points)
                self.assertAlmostEqual(lowest(anchors), lowest(foot_points), delta=5e-7)

    def test_rotated_heel_support_fixes_the_twelve_mm_toe_false_negative(self):
        for angle in (0.0, math.radians(0.9), 0.03, 0.1):
            angles = {"leg_0_femur_joint": angle, "leg_0_tibia_joint": angle}
            tibia = self.world_frame("leg_0_tibia", angles)
            toe = self.world_frame("leg_0_foot", angles)[2][3]
            foot_ground = min(transform(tibia, point)[2] for point in self.foot_points[0])
            witness_ground = min(transform(tibia, point)[2]
                                 for point in GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS[0])
            self.assertAlmostEqual(witness_ground, foot_ground, delta=3e-7)
            self.assertLessEqual(abs(witness_ground - foot_ground), 0.004)
            if angle == 0.0:
                self.assertAlmostEqual(toe - foot_ground, 0.008, delta=2e-6)
                self.assertLess(toe - foot_ground, 0.012)
            else:
                self.assertGreater(toe - foot_ground, 0.012)


@unittest.skipIf(Usd is None, "optional pxr USD runtime unavailable")
class ActualUsdFootGeometryTests(unittest.TestCase):
    def test_actual_usd_mesh_matches_every_cached_tibia_frame_anchor(self):
        stage = Usd.Stage.Open(str(ASSET_DIR / "onshape_20260928.usd"))
        root = stage.GetDefaultPrim().GetPath()
        cache = UsdGeom.XformCache()
        for leg, anchors in enumerate(GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS):
            body = stage.GetPrimAtPath(root.AppendChild(f"leg_{leg}_tibia"))
            mesh_prim = stage.GetPrimAtPath(body.GetPath().AppendPath(GEOMETRY.ONSHAPE_FOOT_COLLISION_MESH))
            self.assertTrue(mesh_prim.IsA(UsdGeom.Mesh))
            matrix = cache.GetLocalToWorldTransform(mesh_prim) * cache.GetLocalToWorldTransform(body).GetInverse()
            raw = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            self.assertEqual(len(raw), 6312)
            points = {tuple(matrix.Transform(Gf.Vec3d(point))) for point in raw}
            for anchor in anchors:
                error = min(max(abs(a-b) for a, b in zip(anchor, point)) for point in points)
                self.assertLess(error, 1e-12)

    @unittest.skipIf(ConvexHull is None, "optional SciPy hull reproducibility check unavailable")
    def test_exact_hull_is_reproducible_from_shared_raw_usd_mesh(self):
        stage = Usd.Stage.Open(str(ASSET_DIR / "onshape_20260928.usd"))
        root = stage.GetDefaultPrim().GetPath()
        previous_raw = None
        for leg in range(6):
            path = root.AppendPath(f"leg_{leg}_tibia/{GEOMETRY.ONSHAPE_FOOT_COLLISION_MESH}")
            raw = np.unique(np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(), dtype=float), axis=0)
            self.assertEqual(len(raw), 1048)
            if previous_raw is not None:
                np.testing.assert_array_equal(raw, previous_raw)
            previous_raw = raw
            hull = raw[ConvexHull(raw).vertices]
            self.assertEqual(len(hull), 137)
            self.assertEqual(set(map(tuple, hull)), set(GEOMETRY._FOOT_HULL_POINTS_M))


if __name__ == "__main__":
    unittest.main()
