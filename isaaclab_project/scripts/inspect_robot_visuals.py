"""Inspect the runtime USD visual state of the spawned CAD spiderbot."""

import argparse
from pathlib import Path

from environment_cli import add_environment_args, apply_environment_cfg
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect spawned robot visual prims.")
parser.add_argument("--task", default="Isaac-DKSH-Spider-Navigation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")
add_environment_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from pxr import Gf, Usd, UsdGeom, UsdShade

import dksh_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def text(value) -> str:
    """Use an explicit marker for absent USD attributes."""
    return "<unset>" if value is None else str(value)


def main() -> None:
    visual_asset = Path(__file__).resolve().parents[1] / "assets" / "spiderbot_variants" / "spiderbot_6leg" / "visuals.usdc"
    visual_stage = Usd.Stage.Open(str(visual_asset))
    print(
        f"ROBOT_VISUAL_ASSET default_prim={visual_stage.GetDefaultPrim().GetPath()} "
        f"roots={[(str(prim.GetPath()), prim.GetTypeName()) for prim in visual_stage.GetPseudoRoot().GetChildren()]}",
        flush=True,
    )
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_environment_cfg(cfg, args_cli.environment, args_cli.difficulty, args_cli.seed)
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        robot_path = "/World/envs/env_0/Robot"
        root = stage.GetPrimAtPath(robot_path)
        print(f"ROBOT_VISUAL_ROOT valid={root.IsValid()} type={root.GetTypeName()}", flush=True)
        if not root.IsValid():
            return
        print(f"ROBOT_ROOT_ATTRIBUTES {[str(attr.GetName()) for attr in root.GetAttributes()]}", flush=True)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
        )
        world_bound = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
        print(
            f"ROBOT_WORLD_BOUND min={world_bound.GetMin()} max={world_bound.GetMax()}",
            flush=True,
        )
        print(f"ROBOT_PHYSICS_ROOT {base_env._robot.data.root_pos_w[0].tolist()}", flush=True)
        try:
            display_root = "/World/envs/env_0/RobotDisplayCheck"
            stage.DefinePrim(display_root, "Xform")
            mesh_for_body = {"base": "base", "hip": "hip", "femur": "femur", "tibia": "tibia"}
            for index, body_name in enumerate(base_env._robot.body_names):
                mesh_name = next(kind for kind in mesh_for_body if body_name == "base" or body_name.endswith(kind))
                prim = stage.DefinePrim(f"{display_root}/{body_name}", "Mesh")
                prim.GetReferences().AddReference(str(visual_asset), f"/{mesh_for_body[mesh_name]}")
                xformable = UsdGeom.Xformable(prim)
                transform_op = xformable.MakeMatrixXform()
                position = base_env._robot.data.body_pos_w[0, index].tolist()
                quaternion = base_env._robot.data.body_quat_w[0, index].tolist()
                transform = Gf.Matrix4d(1.0)
                transform.SetRotate(Gf.Quatd(quaternion[0], Gf.Vec3d(quaternion[1], quaternion[2], quaternion[3])))
                transform.SetTranslateOnly(Gf.Vec3d(*position))
                transform_op.Set(transform)
            bbox_cache.Clear()
            proxy_bound = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(display_root)).ComputeAlignedBox()
            print(
                f"ROBOT_PROXY_PASS min={proxy_bound.GetMin()} max={proxy_bound.GetMax()}",
                flush=True,
            )
        except Exception as error:
            print(f"ROBOT_PROXY_ERROR {type(error).__name__}: {error}", flush=True)
        position = base_env._robot.data.root_pos_w[0].tolist()
        quaternion = base_env._robot.data.root_quat_w[0].tolist()
        root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
        root.GetAttribute("xformOp:orient").Set(
            Gf.Quatd(quaternion[0], Gf.Vec3d(quaternion[1], quaternion[2], quaternion[3]))
        )
        bbox_cache.Clear()
        synced_bound = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
        print(
            f"ROBOT_WORLD_BOUND_MANUAL_SYNC min={synced_bound.GetMin()} max={synced_bound.GetMax()}",
            flush=True,
        )
        env.step(torch.zeros((args_cli.num_envs, cfg.action_space), device=base_env.device))
        base_env.sim.render()
        world_bound = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
        print(
            f"ROBOT_WORLD_BOUND_AFTER_STEP min={world_bound.GetMin()} max={world_bound.GetMax()}",
            flush=True,
        )
        print(f"ROBOT_PHYSICS_ROOT_AFTER_STEP {base_env._robot.data.root_pos_w[0].tolist()}", flush=True)

        imageable_count = 0
        mesh_count = 0
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Imageable):
                continue
            imageable_count += 1
            imageable = UsdGeom.Imageable(prim)
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
            # Only print all meshes plus any explicitly hidden non-mesh visual prims.
            authored_visibility = imageable.GetVisibilityAttr().Get()
            computed_visibility = imageable.ComputeVisibility()
            purpose = imageable.ComputePurpose()
            if prim.IsA(UsdGeom.Mesh) or authored_visibility == UsdGeom.Tokens.invisible:
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                opacity = "<no_material>"
                if material:
                    surface_shader = material.ComputeSurfaceSource()
                    if surface_shader:
                        opacity_input = surface_shader.GetInput("opacity")
                        opacity = text(opacity_input.Get() if opacity_input else None)
                print(
                    "ROBOT_VISUAL "
                    f"path={prim.GetPath()} type={prim.GetTypeName()} "
                    f"authored_visibility={text(authored_visibility)} "
                    f"computed_visibility={computed_visibility} purpose={purpose} opacity={opacity}",
                    flush=True,
                )
        print(
            f"ROBOT_VISUAL_SUMMARY imageables={imageable_count} meshes={mesh_count}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
