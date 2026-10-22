"""Six capped digital foot switches, with isolated cap collision bodies.

Cap radius/mass, activation force and debounce are explicit unmeasured defaults.
The cap is rigidly attached: button travel/compliance is not simulated.
"""
import math
import torch


class ButtonBank:
    """Hysteresis and elapsed-time debounce; only binary values reach the actor."""

    def __init__(self, shape, device, press_force, release_force, debounce_s):
        if not (math.isfinite(press_force) and math.isfinite(release_force)
                and 0 <= release_force < press_force):
            raise ValueError('Require 0 <= release_force < press_force')
        if not math.isfinite(debounce_s) or debounce_s < 0:
            raise ValueError('debounce_s must be finite and nonnegative')
        self.press_force, self.release_force, self.debounce_s = press_force, release_force, debounce_s
        self.state = torch.zeros(shape, dtype=torch.bool, device=device)
        self.elapsed = torch.zeros(shape, device=device)

    def update(self, force_magnitude, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if not torch.isfinite(force_magnitude).all():
            raise ValueError('Non-finite foot cap contact force')
        candidate = torch.where(self.state, force_magnitude > self.release_force,
                                force_magnitude >= self.press_force)
        changed = candidate != self.state
        self.elapsed = torch.where(changed, self.elapsed + dt, 0)
        committed = changed & (self.elapsed >= self.debounce_s - 1e-7)
        self.state = torch.where(committed, candidate, self.state)
        self.elapsed = torch.where(committed, 0, self.elapsed)
        return self.state.float()

    def reset(self, env_ids):
        self.state[env_ids] = False
        self.elapsed[env_ids] = 0


def add_button_caps(stage, robot_path, radius_m, mass_kg):
    """Attach six finite-mass fixed bodies so shin contact cannot press a switch.

    Each sphere centre is the CAD foot point. This provisional rounded cap
    protrudes by its radius; its real offset, material and mass need measurement.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    if not all(math.isfinite(v) and v > 0 for v in (radius_m, mass_kg)):
        raise ValueError('Cap radius and mass must be finite and positive')
    robot = stage.GetPrimAtPath(robot_path)
    for prim in list(Usd.PrimRange(robot)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    cache = UsdGeom.XformCache()
    root_inverse = cache.GetLocalToWorldTransform(robot).GetInverse()
    for leg in range(6):
        tibia_path = f'{robot_path}/leg_{leg}_tibia'
        foot = stage.GetPrimAtPath(f'{tibia_path}/foot_frame')
        tibia = stage.GetPrimAtPath(tibia_path)
        if not foot or not tibia:
            raise ValueError(f'Missing foot frame for leg {leg}')
        foot_w = cache.GetLocalToWorldTransform(foot).ExtractTranslation()
        cap_in_robot = root_inverse.Transform(foot_w)
        cap_in_tibia = cache.GetLocalToWorldTransform(tibia).GetInverse().Transform(foot_w)
        relative = cache.GetLocalToWorldTransform(tibia) * root_inverse
        rotation = relative.ExtractRotationQuat()
        orient = Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))
        cap_path = f'{robot_path}/leg_{leg}_button_cap'
        cap = UsdGeom.Sphere.Define(stage, cap_path)
        cap.CreateRadiusAttr(radius_m)
        cap.AddTranslateOp().Set(cap_in_robot)
        cap.AddOrientOp().Set(orient)
        cap.CreateDisplayColorAttr([(0.15, 0.6, 0.9)])
        UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(cap.GetPrim()).CreateThresholdAttr(0.0)
        mass = UsdPhysics.MassAPI.Apply(cap.GetPrim())
        mass.CreateMassAttr(mass_kg)
        inertia = 0.4 * mass_kg * radius_m ** 2
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        joint = UsdPhysics.FixedJoint.Define(stage, f'{robot_path}/leg_{leg}_button_fixed')
        joint.CreateBody0Rel().SetTargets([tibia_path])
        joint.CreateBody1Rel().SetTargets([cap_path])
        joint.CreateLocalPos0Attr(cap_in_tibia)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
