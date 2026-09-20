"""Finite physical preflight; no policy training or modifications to checkpoints."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", default="")
parser.add_argument("--task", default="Isaac-DKSH-MG90S-Walk-Direct-v0",
                    choices=["Isaac-DKSH-MG90S-Walk-Direct-v0", "Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0",
                             "Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0",
                             "Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1000 or args.num_envs < 1:
    parser.error("Use at least 1000 steps (20 seconds) and one environment")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import dksh_isaaclab
from isaaclab_tasks.utils import parse_env_cfg
from dksh_isaaclab.tasks.direct.spider_navigation.mg90s_env_cfg import MG90S_STALL_TORQUE, MG90S_NO_LOAD_SPEED


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Inspect one uninterrupted 20 second trajectory in each phase.
    cfg.episode_length_s = (args.steps + 10) * cfg.sim.dt * cfg.decimation
    env = gym.make(args.task, cfg=cfg)
    robot_env = env.unwrapped
    robot = robot_env._robot
    results = {}
    try:
        assert robot.num_joints == cfg.action_space
        assert robot.num_bodies == cfg.action_space + 1
        print("MG90S_PREFLIGHT_PHYSICS", {
            "task": args.task, "usd": cfg.robot.spawn.usd_path,
            "joints": robot.num_joints, "bodies": robot.num_bodies,
            "mass_kg": float(robot.root_physx_view.get_masses()[0].sum()),
            "effort_cap_nm": float(robot.actuators["mg90s_4v8"].effort_limit.max()),
            "no_load_speed_rad_s": MG90S_NO_LOAD_SPEED,
        }, flush=True)
        gait_lift = cfg.gait_lift
        if hasattr(cfg, "command_forward_min"):
            phases = (
                ("stand", 0.0, 0.0, 0.0, 0.0),
                ("forward", cfg.command_speed_min, 0.0, 0.0, gait_lift),
                ("lateral", 0.0, cfg.command_speed_min, 0.0, gait_lift),
                ("yaw", 0.0, 0.0, 0.04, gait_lift),
            )
        else:
            phases = (
                ("stand", 0.0, 0.0, 0.0, 0.0),
                ("wave", cfg.command_speed_min, 0.0, 0.0, gait_lift),
            )
        for label, forward, lateral, yaw, lift in phases:
            robot_env.cfg.command_speed_min = forward
            robot_env.cfg.command_speed_max = forward
            robot_env.cfg.gait_lift = lift
            if hasattr(robot_env.cfg, "command_forward_min"):
                robot_env.cfg.command_forward_min = forward
                robot_env.cfg.command_forward_max = forward
                robot_env.cfg.command_lateral_min = lateral
                robot_env.cfg.command_lateral_max = lateral
                robot_env.cfg.command_yaw_min = yaw
                robot_env.cfg.command_yaw_max = yaw
                robot_env.cfg.command_stand_probability = 0.0
            obs, _ = env.reset()
            initial = robot.data.root_pos_w.clone()
            counts_before = robot_env.completed.copy()
            min_height = 10.
            max_torque = 0.
            forward_speed_sum = lateral_speed_sum = yaw_rate_sum = 0.
            saturated_sum = 0.
            for step in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(torch.zeros((args.num_envs, cfg.action_space), device=robot_env.device))
                if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite physics or reward")
                assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
                min_height = min(min_height, float(robot.data.root_pos_w[:, 2].min()))
                max_torque = max(max_torque, float(robot.data.applied_torque.abs().max()))
                forward_speed_sum += float(robot.data.root_lin_vel_b[:, 0].mean())
                lateral_speed_sum += float(robot.data.root_lin_vel_b[:, 1].mean())
                yaw_rate_sum += float(robot.data.root_ang_vel_b[:, 2].mean())
                saturated_sum += float((robot.data.applied_torque.abs() > 0.70 * MG90S_STALL_TORQUE).float().mean())
            result = {
                "steps": args.steps, "envs": args.num_envs,
                "falls": robot_env.completed["fallen"] - counts_before["fallen"],
                "out_of_bounds": robot_env.completed["out_of_bounds"] - counts_before["out_of_bounds"],
                "min_height_m": min_height, "max_abs_torque_nm": max_torque,
                "mean_forward_speed_mps": forward_speed_sum / args.steps,
                "mean_lateral_speed_mps": lateral_speed_sum / args.steps,
                "mean_yaw_rate_rps": yaw_rate_sum / args.steps,
                "mean_displacement_x_m": float((robot.data.root_pos_w[:, 0] - initial[:, 0]).mean()),
                "mean_displacement_y_m": float((robot.data.root_pos_w[:, 1] - initial[:, 1]).mean()),
                "near_torque_cap_fraction": saturated_sum / args.steps,
            }
            results[label] = result
            print("MG90S_PREFLIGHT_PHASE " + json.dumps({label: result}), flush=True)
        # Viability gate, not a claim of learned gait success.
        passed = all(r["falls"] == 0 and r["out_of_bounds"] == 0 for r in results.values())
        if "wave" in results:
            passed &= results["wave"]["mean_displacement_x_m"] > 0.005
        else:
            passed &= results["forward"]["mean_displacement_x_m"] > 0.005
            passed &= results["lateral"]["mean_displacement_y_m"] > 0.002
            passed &= results["yaw"]["mean_yaw_rate_rps"] > 0.005
        passed &= all(r["max_abs_torque_nm"] <= 0.75 * MG90S_STALL_TORQUE + 1e-5 for r in results.values())
        results["passed"] = passed
        if args.output:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not passed:
            raise RuntimeError("MG90S physical preflight did not pass; inspect measurements before training")
        print("MG90S_PREFLIGHT_PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
