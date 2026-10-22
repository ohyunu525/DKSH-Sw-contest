"""Physics-free viewport copies of the active USD robot's actual link visuals.

Isaac's Direct GPU path need not update USD link transforms. Read visual
geometry once in each link's local frame, then drive only a separate display
hierarchy from simulated link poses. Instance-proxy meshes and fixed-joint
merged feet are included. Collision guides are left available for debugging.
"""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _graphics_attribute(name: str) -> bool:
    return not (name.startswith(("physics:", "physx", "xformOp:"))
                or name in ("xformOpOrder", "visibility", "purpose"))


def _copy_graphics_properties(source, target) -> None:
    """Copy resolved geometry and shading only, never schemas or references."""
    for attribute in source.GetAttributes():
        if not _graphics_attribute(attribute.GetName()):
            continue
        value = attribute.Get()
        samples = attribute.GetTimeSamples()
        if value is None and not samples:
            continue
        copied = target.CreateAttribute(
            attribute.GetName(), attribute.GetTypeName(),
            custom=attribute.IsCustom(), variability=attribute.GetVariability(),
        )
        if value is not None:
            copied.Set(value)
        for time in samples:
            copied.Set(attribute.Get(time), time)
        for key, metadata in attribute.GetAllAuthoredMetadata().items():
            # Type/custom/variability came from CreateAttribute. The remaining
            # metadata includes primvar interpolation and normal interpolation.
            if key not in ("typeName", "custom", "variability", "default", "timeSamples"):
                copied.SetMetadata(key, metadata)

    # Bind the material actually resolved through visual ancestors/instances.
    # Materials stay on the untouched source robot; no physics relationship is
    # copied into the display. A direct binding is also needed on face subsets.
    for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
        material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial(purpose)
        if material:
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material, materialPurpose=purpose)


class LinkVisualCopies:
    """Own a reversible display hierarchy and its source-visibility overrides."""

    def __init__(self, stage, robot_path: str, body_names, display_path: str):
        self.stage = stage
        self.display_path = Sdf.Path(display_path)
        robot = stage.GetPrimAtPath(robot_path)
        if not robot:
            raise ValueError(f"Robot prim does not exist: {robot_path}")
        if self.display_path.HasPrefix(robot.GetPath()):
            raise ValueError("Display hierarchy must be outside the physics robot")
        if stage.GetPrimAtPath(self.display_path):
            raise ValueError(f"Display prim already exists: {display_path}")
        self.body_names = tuple(body_names)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Body names must be unique")
        bodies = {}
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in self.body_names:
                if prim.GetName() in bodies:
                    raise ValueError(f"Ambiguous robot body: {prim.GetName()}")
                bodies[prim.GetName()] = prim
        missing = set(self.body_names) - bodies.keys()
        if missing:
            raise ValueError(f"Robot rigid bodies are missing: {sorted(missing)}")

        self._edit_target = stage.GetEditTarget()
        self._backup = Sdf.Layer.CreateAnonymous("cad-visual-visibility.usda")
        self._hidden = {}
        self._pose_ops = []
        self.geometry_pairs = []
        self._active = True
        body_paths = {prim.GetPath() for prim in bodies.values()}
        cache = UsdGeom.XformCache()
        try:
            UsdGeom.Xform.Define(stage, self.display_path)
            parent = stage.GetPrimAtPath(self.display_path.GetParentPath())
            parent_inverse = cache.GetLocalToWorldTransform(parent).GetInverse()
            for body_name in self.body_names:
                body = bodies[body_name]
                body_world = cache.GetLocalToWorldTransform(body)
                body_inverse = body_world.GetInverse()
                copied_body = UsdGeom.Xform.Define(stage, self.display_path.AppendChild(body_name))
                op = copied_body.MakeMatrixXform()
                op.Set(body_world * parent_inverse)
                self._pose_ops.append(op)
                for source in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if not source.IsA(UsdGeom.Gprim):
                        continue
                    owner = source
                    while owner and owner.GetPath() not in body_paths:
                        owner = owner.GetParent()
                    if not owner or owner != body:
                        continue
                    imageable = UsdGeom.Imageable(source)
                    if imageable.ComputePurpose() == UsdGeom.Tokens.guide:
                        continue
                    target_path = copied_body.GetPath().AppendChild(f"geometry_{len(self.geometry_pairs)}")
                    target = stage.DefinePrim(target_path, source.GetTypeName())
                    _copy_graphics_properties(source, target)
                    # Do not copy source xformOps twice. This single matrix
                    # includes every visual/instance/fixed-foot local transform.
                    local = cache.GetLocalToWorldTransform(source) * body_inverse
                    UsdGeom.Xformable(target).MakeMatrixXform().Set(local)
                    copied_image = UsdGeom.Imageable(target)
                    copied_image.CreateVisibilityAttr(imageable.ComputeVisibility())
                    copied_image.CreatePurposeAttr(imageable.ComputePurpose())
                    for subset in Usd.PrimRange(source, Usd.TraverseInstanceProxies()):
                        if subset.IsA(UsdGeom.Subset):
                            relative = subset.GetPath().MakeRelativePath(source.GetPath())
                            copied_subset = stage.DefinePrim(target_path.AppendPath(relative), subset.GetTypeName())
                            _copy_graphics_properties(subset, copied_subset)
                    self.geometry_pairs.append((source.GetPath(), target_path, body_name))
            if not self.geometry_pairs:
                raise ValueError("The active robot has no renderable visual geometry")
            # Hide only render geometry, not rigid bodies or guide colliders.
            # A visual instance must be hidden at its editable instance root.
            for source_path, _, _ in self.geometry_pairs:
                source = stage.GetPrimAtPath(source_path)
                while source.IsInstanceProxy():
                    source = source.GetParent()
                self._hide_original(source)
        except BaseException:
            self.restore()
            raise

    def _hide_original(self, prim):
        if prim.GetPath() in self._hidden:
            return
        attribute_path = prim.GetPath().AppendProperty("visibility")
        spec_path = self._edit_target.MapToSpecPath(attribute_path)
        layer = self._edit_target.GetLayer()
        had_spec = bool(layer.GetAttributeAtPath(spec_path))
        if had_spec:
            Sdf.CreatePrimInLayer(self._backup, spec_path.GetPrimPath())
            Sdf.CopySpec(layer, spec_path, self._backup, spec_path)
        self._hidden[prim.GetPath()] = (spec_path, had_spec)
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    def update_world_poses(self, positions, quaternions) -> None:
        """Apply link-frame positions and wxyz quaternions in body_names order."""
        if not self._active:
            raise RuntimeError("The display hierarchy has been restored")
        if len(positions) != len(self._pose_ops) or len(quaternions) != len(self._pose_ops):
            raise ValueError("One world position and quaternion is required for each body")
        if any(len(p) != 3 for p in positions) or any(len(q) != 4 for q in quaternions):
            raise ValueError("Positions must be XYZ and quaternions must be WXYZ")
        parent = self.stage.GetPrimAtPath(self.display_path.GetParentPath())
        parent_inverse = UsdGeom.XformCache().GetLocalToWorldTransform(parent).GetInverse()
        for op, position, quaternion in zip(self._pose_ops, positions, quaternions):
            transform = Gf.Matrix4d(1.0)
            transform.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
            transform.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
            op.Set(transform * parent_inverse)

    def restore(self) -> None:
        """Remove our display and restore exact preexisting visibility opinions."""
        if not self._active:
            return
        with Usd.EditContext(self.stage, self._edit_target):
            layer = self._edit_target.GetLayer()
            for path, (spec_path, had_spec) in self._hidden.items():
                if had_spec:
                    Sdf.CopySpec(self._backup, spec_path, layer, spec_path)
                else:
                    prim = self.stage.GetPrimAtPath(path)
                    if prim:
                        prim.RemoveProperty("visibility")
            self.stage.RemovePrim(self.display_path)
        self._active = False
