"""Verify body/joint ground-clearance witnesses against actual CAD colliders."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT / "assets/spiderbot_variants/onshape_20260928"
MODULE = PROJECT / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_non_distal_geometry.py"
GENERATOR = PROJECT / "scripts/generate_non_distal_collision_geometry.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEOMETRY = load(MODULE, "non_distal_geometry_under_test")
# Independent Rodrigues/URDF transform helpers already validated by the foot
# asset suite; the generator separately uses the expanded Euler matrix formula.
FOOT_HELPERS = load(Path(__file__).with_name("test_joint_angle_foot_geometry.py"), "cad_geometry_helpers")


class NonDistalCollisionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FOOT_HELPERS.FootGeometryTests.setUpClass()
        cls.urdf = FOOT_HELPERS.FootGeometryTests.urdf
        cls.foot_helper = FOOT_HELPERS.FootGeometryTests()
        cls.raw_meshes = {filename: FOOT_HELPERS.stl_points(ASSET_DIR / filename)
                          for filename in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_MESH_SHA256}

    def test_runtime_needs_no_assets_or_numerical_dependencies(self):
        normal_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "scipy", "pxr", "pathlib", "xml"):
                raise AssertionError(f"Runtime collision geometry imported {name}")
            return normal_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=checked_import):
            actual = load(MODULE, "non_distal_geometry_without_dependencies")
        self.assertEqual(actual.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS,
                         GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS)

    def test_asset_provenance_and_all_nineteen_rigid_bodies(self):
        self.assertEqual(GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_URDF_SHA256,
                         hashlib.sha256((ASSET_DIR / "robot.training.urdf").read_bytes()).hexdigest())
        self.assertEqual(GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_USD_SHA256,
                         hashlib.sha256((ASSET_DIR / "onshape_20260928.usd").read_bytes()).hexdigest())
        for filename, expected_hash in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_MESH_SHA256.items():
            self.assertEqual(hashlib.sha256((ASSET_DIR / filename).read_bytes()).hexdigest(), expected_hash)
        expected_bodies = {"base"} | {f"leg_{leg}_{part}" for leg in range(6)
                                     for part in ("hip", "femur", "tibia")}
        self.assertEqual(set(GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS), expected_bodies)
        self.assertNotIn("assets/part_1__8.stl", GEOMETRY._MESH_HULL_POINTS_M)
        for points in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS.values():
            self.assertIsInstance(points, tuple)
            self.assertTrue(points)
            self.assertEqual(len(points), len(set(points)))
            self.assertTrue(all(isinstance(point, tuple) and len(point) == 3
                                and all(math.isfinite(value) for value in point) for point in points))

    def test_each_source_hull_preserves_actual_mesh_support_extrema(self):
        directions = [p for p in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(p)]
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(64):
            z = 1.0 - 2.0 * (index + 0.5) / 64
            radius = math.sqrt(1.0 - z*z)
            directions.append((radius * math.cos(index * golden_angle),
                               radius * math.sin(index * golden_angle), z))
        for filename, hull in GEOMETRY._MESH_HULL_POINTS_M.items():
            raw = self.raw_meshes[filename]
            self.assertTrue(set(hull) <= set(raw), filename)
            self.assertLessEqual(len(hull), len(raw), filename)
            for dx, dy, dz in directions:
                lowest = lambda points: min(dx*x + dy*y + dz*z for x, y, z in points)
                self.assertAlmostEqual(lowest(raw), lowest(hull), delta=1e-12, msg=filename)

    def test_every_actual_collider_has_its_correct_link_frame_witnesses(self):
        collision_count = 0
        for body, actual in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS.items():
            link = self.urdf.find(f"link[@name='{body}']")
            expected, source_names = set(), []
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                filename = mesh.get("filename")
                if body.endswith("_tibia") and filename == "assets/part_1__8.stl":
                    continue
                collision_count += 1
                source_names.append(filename)
                matrix = FOOT_HELPERS.origin(collision.find("origin"))
                scale = FOOT_HELPERS.vector(mesh.get("scale", "1 1 1"))
                expected.update(FOOT_HELPERS.transform(matrix, tuple(x*s for x, s in zip(point, scale)))
                                for point in GEOMETRY._MESH_HULL_POINTS_M[filename])
            self.assertCountEqual(source_names, [name for name, _ in GEOMETRY._BODY_COLLISIONS[body]])
            # The independent matrix implementations differ only in rounding.
            self.assertEqual(len(expected), len(actual), body)
            for found, wanted in zip(sorted(actual), sorted(expected)):
                self.assertLess(max(abs(a-b) for a, b in zip(found, wanted)), 1e-14, body)
        self.assertEqual(collision_count, 62)

    def test_joint_first_posture_is_detected_while_knee_and_tibia_are_clear(self):
        angles = {"leg_0_femur_joint": -1.5, "leg_0_tibia_joint": -1.5}
        pitch = math.radians(10.0)
        root_height = 0.06854638435431198
        root_transform = FOOT_HELPERS.rotation((0.0, 1.0, 0.0), pitch)

        def world_z(body, points):
            frame = FOOT_HELPERS.multiply(root_transform, self.foot_helper.world_frame(body, angles))
            return min(FOOT_HELPERS.transform(frame, point)[2] + root_height for point in points)

        offsets = GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS
        hip_clearance = world_z("leg_0_hip", offsets["leg_0_hip"])
        femur_clearance = world_z("leg_0_femur", offsets["leg_0_femur"])
        servo_clearance = world_z("leg_0_tibia", offsets["leg_0_tibia"])
        foot_clearance = world_z("leg_0_tibia", FOOT_HELPERS.GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS[0])
        knee_clearance = world_z("leg_0_tibia", ((0.0, 0.0, 0.0),))
        self.assertAlmostEqual(hip_clearance, 0.0, delta=1e-12)
        self.assertAlmostEqual(femur_clearance, 0.0295159544, delta=1e-10)
        self.assertAlmostEqual(servo_clearance, 0.0344872722, delta=1e-10)
        self.assertAlmostEqual(foot_clearance, 0.0310946620, delta=1e-9)
        self.assertAlmostEqual(knee_clearance, 0.0485208349, delta=1e-10)
        self.assertGreater(world_z("base", offsets["base"]), 0.008)
        self.assertGreater(knee_clearance, 0.030)  # Historical hinge proxy misses the contact.
        self.assertLess(hip_clearance, 0.008)  # Collision witnesses see it immediately.
        self.assertGreater(root_height, 0.060)
        self.assertLess(-math.cos(pitch), -0.35)  # Existing fall guards also miss it.

    def test_distal_reset_posture_clears_every_non_distal_collider(self):
        angles = {f"leg_{leg}_{part}_joint": -0.3 for leg in range(6) for part in ("femur", "tibia")}
        frames = {body: self.foot_helper.world_frame(body, angles)
                  for body in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS}
        foot_geometry = FOOT_HELPERS.GEOMETRY
        distal_z = [min(FOOT_HELPERS.transform(frames[f"leg_{leg}_tibia"], point)[2]
                        for point in foot_geometry.ONSHAPE_DISTAL_FOOT_SUPPORT_OFFSETS[leg]) for leg in range(6)]
        root_height = -min(distal_z)
        self.assertAlmostEqual(root_height, 0.1924481510, delta=1e-10)
        non_distal = min(FOOT_HELPERS.transform(frames[body], point)[2] + root_height
                         for body, points in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS.items()
                         for point in points)
        proximal_foot = min(FOOT_HELPERS.transform(frames[f"leg_{leg}_tibia"], point)[2] + root_height
                            for leg in range(6) for point in foot_geometry.ONSHAPE_PROXIMAL_FOOT_OFFSETS[leg])
        self.assertGreater(non_distal, 0.058)
        self.assertGreater(proximal_foot, 0.028)

    def recorded_false_positive(self):
        fixture = json.loads((Path(__file__).with_name("fixtures")
                              / "recorded_v8_distal_false_positive.json").read_text(encoding="utf-8"))
        states = fixture["body_state"]

        def world_height(name, point):
            index = states["names"].index(name)
            w, x, y, z = states["quaternions_wxyz"][index]
            px, py, pz = point
            uv = (y*pz - z*py, z*px - x*pz, x*py - y*px)
            uuv_z = x*uv[1] - y*uv[0]
            return states["positions_w_m"][index][2] + pz + 2*(w*uv[2] + uuv_z)

        return fixture, world_height

    def test_recorded_failure_keeps_all_non_distal_meshes_above_the_floor(self):
        fixture, world_height = self.recorded_false_positive()
        foot = FOOT_HELPERS.GEOMETRY
        history = fixture["net_contact_force_history"]
        self.assertEqual(fixture["provenance"]["episode_time_s"], 20.72)
        self.assertEqual(fixture["provenance"]["behavior_profile"], "revised_v8")
        self.assertEqual(fixture["reported_support"]["invalid_legs"], [False]*5 + [True])
        for name, points in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS.items():
            self.assertGreater(min(world_height(name, point) for point in points), 0.015)
            sensor = history["body_names"].index(name)
            forces = [sample[sensor] for sample in history["force_w_n"]]
            if name.endswith("_tibia"):
                self.assertGreater(min(force[2] for force in forces), 3.0)
            else:
                self.assertTrue(all(force == [0.0, 0.0, 0.0] for force in forces))
        proximal = []
        for leg in range(6):
            name = f"leg_{leg}_tibia"
            distal_gap = min(world_height(name, point) for point in foot.ONSHAPE_DISTAL_FOOT_SUPPORT_OFFSETS[leg])
            proximal.append(min(world_height(name, point) for point in foot.ONSHAPE_PROXIMAL_FOOT_OFFSETS[leg]))
            self.assertLess(abs(distal_gap), 0.000020)
        self.assertAlmostEqual(proximal[5], 0.0078830823969, delta=1e-12)
        self.assertGreater(min(proximal[:5]), 0.008)
        self.assertGreater(proximal[5], 0.001)
        self.assertLess(proximal[5], 0.008)  # Old clearance margin alone caused the termination.
        servo_gap = min(world_height("leg_5_tibia", point)
                        for point in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS["leg_5_tibia"])
        self.assertAlmostEqual(servo_gap, 0.0153231522, delta=1e-8)

    @unittest.skipIf(FOOT_HELPERS.Usd is None, "optional pxr USD runtime unavailable")
    def test_recorded_leg5_shaft_clearance_is_confirmed_by_actual_usd_vertices(self):
        from pxr import Gf, Usd, UsdGeom

        _, world_height = self.recorded_false_positive()
        stage = Usd.Stage.Open(str(ASSET_DIR / "onshape_20260928.usd"))
        body = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild("leg_5_tibia"))
        cache = UsdGeom.XformCache()
        minimums = {}
        for mesh_prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not mesh_prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(mesh_prim.GetPath()):
                continue
            matrix = cache.GetLocalToWorldTransform(mesh_prim) * cache.GetLocalToWorldTransform(body).GetInverse()
            points = [tuple(matrix.Transform(Gf.Vec3d(point)))
                      for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()]
            minimums[mesh_prim.GetParent().GetName()] = min(world_height("leg_5_tibia", point) for point in points)
        self.assertAlmostEqual(minimums["part_1"], 0.0153231521897, delta=1e-12)
        self.assertAlmostEqual(minimums["part_1__8"], -0.0000124717159, delta=1e-12)
        # The foot's minimum is the distal toe. Its middle/proximal witnesses
        # are 7.883 mm higher, rather than a second physical grounding point.
        proximal_gap = min(world_height("leg_5_tibia", point)
                           for point in FOOT_HELPERS.GEOMETRY.ONSHAPE_PROXIMAL_FOOT_OFFSETS[5])
        self.assertGreater(proximal_gap - minimums["part_1__8"], 0.0078)

    @unittest.skipIf(FOOT_HELPERS.Usd is None, "optional pxr USD runtime unavailable")
    def test_actual_usd_collision_meshes_agree_in_every_link_frame(self):
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(ASSET_DIR / "onshape_20260928.usd"))
        root = stage.GetDefaultPrim().GetPath()
        cache = UsdGeom.XformCache()
        collision_count = 0
        tolerance = 1e-6  # URDF decimal export vs USD transform rounding.
        neighbors = tuple(itertools.product((-1, 0, 1), repeat=3))
        directions = [p for p in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(p)]
        for name, anchors in GEOMETRY.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS.items():
            body = stage.GetPrimAtPath(root.AppendChild(name))
            self.assertTrue(body.IsValid(), name)
            points = set()
            # The asset instances its mesh pieces; ordinary PrimRange omits
            # those descendants and would accidentally validate empty geometry.
            for mesh_prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not mesh_prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(mesh_prim.GetPath()):
                    continue
                if name.endswith("_tibia") and mesh_prim.GetParent().GetName() == "part_1__8":
                    continue
                collision_count += 1
                matrix = cache.GetLocalToWorldTransform(mesh_prim) * cache.GetLocalToWorldTransform(body).GetInverse()
                points.update(tuple(matrix.Transform(Gf.Vec3d(point)))
                              for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get())
            self.assertTrue(points, name)
            buckets = {}
            for point in points:
                cell = tuple(math.floor(value / tolerance) for value in point)
                buckets.setdefault(cell, []).append(point)
            for anchor in anchors:
                cell = tuple(math.floor(value / tolerance) for value in anchor)
                error = min((max(abs(a-b) for a, b in zip(anchor, point))
                             for dx, dy, dz in neighbors
                             for point in buckets.get((cell[0]+dx, cell[1]+dy, cell[2]+dz), ())),
                            default=math.inf)
                self.assertLess(error, tolerance, name)
            for dx, dy, dz in directions:
                lowest = lambda values: min(dx*x + dy*y + dz*z for x, y, z in values)
                self.assertAlmostEqual(lowest(points), lowest(anchors), delta=3*tolerance, msg=name)
        self.assertEqual(collision_count, 62)

    @unittest.skipIf(FOOT_HELPERS.ConvexHull is None, "optional SciPy generator check unavailable")
    def test_saved_module_is_reproducible(self):
        generator = load(GENERATOR, "non_distal_geometry_generator_under_test")
        self.assertEqual(generator.render_module(ASSET_DIR), MODULE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
