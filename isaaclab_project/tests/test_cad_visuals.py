"""Check viewport geometry against real USD assets, without Isaac/PhysX/GPU.

The asset tests cover Onshape instance proxies, fixed-merged feet, the historical
CAD geometry, world/link transforms and reversible original visibility. They do
not exercise Hydra rendering or the live Direct GPU transport.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import unittest

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/cad_visuals.py"
if Usd is not None:
    SPEC = importlib.util.spec_from_file_location("dksh_cad_visuals", MODULE)
    HELPER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(HELPER)


@unittest.skipIf(Usd is None, "USD core is required; no Isaac Sim or GPU is needed")
class CadVisualCopiesTests(unittest.TestCase):
    def scene(self, asset="onshape_20260928/onshape_20260928.usd", transformed_parent=False):
        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/World/env_0")
        if transformed_parent:
            transform = Gf.Matrix4d(1)
            transform.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 35))
            transform.SetTranslateOnly(Gf.Vec3d(3, -2, 1))
            parent.MakeMatrixXform().Set(transform)
        robot = stage.DefinePrim("/World/env_0/Robot")
        robot.GetReferences().AddReference(str((ROOT / "assets/spiderbot_variants" / asset).resolve()))
        names = [prim.GetName() for prim in Usd.PrimRange(robot) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        return stage, robot, names

    def assert_matrix(self, actual, expected):
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(actual[row][column], expected[row][column], places=8)

    def create(self, stage, names):
        return HELPER.LinkVisualCopies(stage, "/World/env_0/Robot", names, "/World/env_0/RobotDisplay")

    def test_active_onshape_meshes_and_fixed_merged_feet_are_copied_exactly(self):
        stage, _, names = self.scene()
        copies = self.create(stage, names)
        self.assertEqual(len(names), 19)
        self.assertEqual(len(copies.geometry_pairs), 68)
        cache = UsdGeom.XformCache()
        counts = {name: 0 for name in names}
        for source_path, target_path, body_name in copies.geometry_pairs:
            source, target = stage.GetPrimAtPath(source_path), stage.GetPrimAtPath(target_path)
            self.assertTrue(source.IsInstanceProxy())  # Onshape visuals are instanced.
            self.assertEqual(UsdGeom.Mesh(source).GetPointsAttr().Get(), UsdGeom.Mesh(target).GetPointsAttr().Get())
            self.assertEqual(UsdGeom.Mesh(source).GetFaceVertexIndicesAttr().Get(), UsdGeom.Mesh(target).GetFaceVertexIndicesAttr().Get())
            self.assert_matrix(cache.GetLocalToWorldTransform(source), cache.GetLocalToWorldTransform(target))
            self.assertEqual(UsdGeom.Imageable(source).ComputeVisibility(), UsdGeom.Tokens.invisible)
            self.assertEqual(UsdGeom.Imageable(target).ComputeVisibility(), UsdGeom.Tokens.inherited)
            counts[body_name] += 1
        # Each tibia includes its fixed-merged foot mesh. A generic old tibia
        # reference would have only one mesh and cannot pass this comparison.
        self.assertTrue(all(counts[f"leg_{leg}_tibia"] == 2 for leg in range(6)))

    def test_historical_cad_uses_its_own_active_geometry(self):
        stage, _, names = self.scene("spiderbot_6leg/spiderbot_6leg.usd")
        copies = self.create(stage, names)
        self.assertEqual(len(copies.geometry_pairs), 19)
        for source_path, target_path, _ in copies.geometry_pairs:
            self.assertEqual(UsdGeom.Mesh(stage.GetPrimAtPath(source_path)).GetPointsAttr().Get(),
                             UsdGeom.Mesh(stage.GetPrimAtPath(target_path)).GetPointsAttr().Get())

    def test_current_onshape_envelope_and_reset_clearance_cover_actual_geometry(self):
        stage, robot, _ = self.scene()
        cache = UsdGeom.XformCache()
        lower = [math.inf] * 3
        upper = [-math.inf] * 3
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
                continue
            transform = cache.GetLocalToWorldTransform(prim)
            # USD's C++ mesh extent plugin transforms the real points; rotating
            # an already axis-aligned box would overestimate this footprint.
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
                UsdGeom.Boundable(prim), Usd.TimeCode.Default(), transform,
            )
            for axis in range(3):
                lower[axis] = min(lower[axis], extent[0][axis])
                upper[axis] = max(upper[axis], extent[1][axis])

        # Resolve numeric config values without importing an Isaac runtime.
        config_tree = ast.parse(MODULE.with_name("joint_angle_env_cfg.py").read_text(encoding="utf-8"))
        classes = {node.name: node for node in config_tree.body if isinstance(node, ast.ClassDef)}
        def class_setting(name, field):
            for node in classes[name].body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == field for target in node.targets):
                    return ast.literal_eval(node.value)
            for base in classes[name].bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    try:
                        return class_setting(base.id, field)
                    except KeyError:
                        pass
            raise KeyError(field)

        width = class_setting("MG996RHipCad6JointAngleEnvCfg", "robot_width")
        height = class_setting("MG996RHipCad6JointAngleEnvCfg", "robot_height")
        for function_name in ("_cad6_joint_angle_robot", "_mg996r_hip_cad6_robot"):
            definition = next(node for node in config_tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
            position = next(ast.literal_eval(node.value) for node in ast.walk(definition)
                            if isinstance(node, ast.Assign) and any(ast.unparse(target) == "robot.init_state.pos" for target in node.targets))
            self.assertGreater(position[2] + lower[2], 0.01)
        self.assertGreaterEqual(width, max(upper[0] - lower[0], upper[1] - lower[1]))

        payload_tree = ast.parse(MODULE.with_name("payload_mass.py").read_text(encoding="utf-8"))
        payload_values = {node.targets[0].id: ast.literal_eval(node.value) for node in payload_tree.body
                          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id in ("L2_PAYLOAD_SIZE_M", "L2_MOUNT_POSITION_B")}
        enclosure_top = payload_values["L2_MOUNT_POSITION_B"][2] + payload_values["L2_PAYLOAD_SIZE_M"][2]
        self.assertGreaterEqual(height, max(upper[2], enclosure_top) - lower[2])

    def test_world_pose_updates_keep_visual_offsets_and_do_not_move_physics(self):
        stage, _, names = self.scene(transformed_parent=True)
        cache = UsdGeom.XformCache()
        before = {name: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/env_0/Robot/{name}")) for name in names}
        copies = self.create(stage, names)
        source_locals = {str(target): cache.GetLocalToWorldTransform(stage.GetPrimAtPath(source)) * before[name].GetInverse()
                         for source, target, name in copies.geometry_pairs}
        positions = [[index * 0.07, 0.2, 0.4] for index in range(len(names))]
        quaternions = [[math.cos(0.2), 0, math.sin(0.2), 0] for _ in names]
        copies.update_world_poses(positions, quaternions)
        cache.Clear()
        for name in names:
            self.assert_matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/env_0/Robot/{name}")), before[name])
        for _, target_path, name in copies.geometry_pairs:
            index = names.index(name)
            desired = Gf.Matrix4d(1)
            desired.SetRotate(Gf.Quatd(quaternions[index][0], Gf.Vec3d(*quaternions[index][1:])))
            desired.SetTranslateOnly(Gf.Vec3d(*positions[index]))
            self.assert_matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target_path)), source_locals[str(target_path)] * desired)

    def test_copies_have_no_physics_and_collision_guides_are_untouched(self):
        stage, robot, names = self.scene()
        payload = UsdGeom.Cube.Define(stage, robot.GetPath().AppendPath("base/LidarPayload"))
        UsdPhysics.CollisionAPI.Apply(payload.GetPrim())  # A rendered AND collidable enclosure.
        payload.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.2, 0.3)])
        colliders = {prim.GetPath(): tuple(prim.GetAppliedSchemas()) for prim in Usd.PrimRange(robot)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)}
        guides = {prim.GetPath(): UsdGeom.Imageable(prim).ComputeVisibility()
                  for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
                  if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide}
        copies = self.create(stage, names)
        self.assertTrue(any(str(source).endswith("/LidarPayload") for source, _, _ in copies.geometry_pairs))
        for prim in Usd.PrimRange(stage.GetPrimAtPath(copies.display_path)):
            self.assertFalse(any("Physics" in api or "Physx" in api for api in prim.GetAppliedSchemas()))
            self.assertFalse(any(prop.GetName().startswith(("physics:", "physx")) for prop in prim.GetProperties()))
            self.assertFalse(prim.HasAuthoredReferences())
            self.assertFalse(prim.IsInstance())
        for path, apis in colliders.items():
            self.assertEqual(tuple(stage.GetPrimAtPath(path).GetAppliedSchemas()), apis)
        for path, visibility in guides.items():
            self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility(), visibility)

    def test_restore_preserves_hidden_parts_and_unset_visibility_opinions(self):
        stage, _, names = self.scene()
        hidden = stage.GetPrimAtPath("/World/env_0/Robot/leg_0_tibia/visuals")
        UsdGeom.Imageable(hidden).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        visible = stage.GetPrimAtPath("/World/env_0/Robot/leg_1_tibia/visuals")
        layer = stage.GetEditTarget().GetLayer()
        self.assertIsNone(layer.GetAttributeAtPath(visible.GetPath().AppendProperty("visibility")))
        copies = self.create(stage, names)
        for _, target, name in copies.geometry_pairs:
            if name == "leg_0_tibia":
                self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath(target)).ComputeVisibility(), UsdGeom.Tokens.invisible)
        copies.restore()
        self.assertFalse(stage.GetPrimAtPath(copies.display_path))
        self.assertEqual(UsdGeom.Imageable(hidden).ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertEqual(UsdGeom.Imageable(visible).ComputeVisibility(), UsdGeom.Tokens.inherited)
        self.assertIsNone(layer.GetAttributeAtPath(visible.GetPath().AppendProperty("visibility")))
        self.assertIsNotNone(layer.GetAttributeAtPath(hidden.GetPath().AppendProperty("visibility")))
        copies.restore()  # Repeated cleanup is safe.

    def test_material_subsets_and_primvar_interpolation_are_preserved(self):
        stage = Usd.Stage.CreateInMemory()
        robot = UsdGeom.Xform.Define(stage, "/World/env_0/Robot").GetPrim()
        body = UsdGeom.Xform.Define(stage, robot.GetPath().AppendChild("base")).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("visual"))
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        colors = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("color", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        colors.Set([Gf.Vec3f(1, 0, 0)] * 3)
        material = UsdShade.Material.Define(stage, robot.GetPath().AppendPath("Looks/Red"))
        UsdShade.MaterialBindingAPI.Apply(body).Bind(material)
        subset = UsdGeom.Subset.Define(stage, mesh.GetPath().AppendChild("red_faces"))
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr([0])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
        copies = self.create(stage, ["base"])
        target = stage.GetPrimAtPath(copies.geometry_pairs[0][1])
        self.assertEqual(UsdGeom.PrimvarsAPI(target).GetPrimvar("color").GetInterpolation(), UsdGeom.Tokens.vertex)
        self.assertEqual(UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()[0].GetPath(), material.GetPath())
        copied_subset = stage.GetPrimAtPath(target.GetPath().AppendChild("red_faces"))
        self.assertEqual(UsdGeom.Subset(copied_subset).GetIndicesAttr().Get(), [0])
        self.assertEqual(UsdShade.MaterialBindingAPI(copied_subset).ComputeBoundMaterial()[0].GetPath(), material.GetPath())

    def test_invalid_body_mapping_and_existing_display_do_not_change_source(self):
        stage, _, names = self.scene()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.create(stage, names + ["missing_body"])
        self.assertFalse(stage.GetPrimAtPath("/World/env_0/RobotDisplay"))
        original = UsdGeom.Imageable(stage.GetPrimAtPath("/World/env_0/Robot/base/visuals"))
        self.assertEqual(original.ComputeVisibility(), UsdGeom.Tokens.inherited)
        display = UsdGeom.Xform.Define(stage, "/World/env_0/RobotDisplay")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create(stage, names)
        self.assertTrue(display.GetPrim())
        self.assertEqual(original.ComputeVisibility(), UsdGeom.Tokens.inherited)


if __name__ == "__main__":
    unittest.main()
