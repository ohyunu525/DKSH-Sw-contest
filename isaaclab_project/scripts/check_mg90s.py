"""Finite physical preflight; no policy training or modifications to checkpoints."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", default="")
parser.add_argument("--timeseries", default="", help="JSONL per control step; defaults beside --output")
parser.add_argument("--friction", type=float, choices=(0.4, 0.6, 0.8), default=0.6)
parser.add_argument("--physics_dt", type=float, choices=(0.0025, 0.005, 0.01), default=0.005)
parser.add_argument("--root_height", type=float, default=None, help="Optional body-height assumption in metres")
parser.add_argument("--simple_collisions", action="store_true", help="CAD bounding boxes and 5 mm assumed foot pads")
parser.add_argument("--self_collisions", action="store_true", help="Enable body/leg collisions for diagnostic comparison")
parser.add_argument("--yaw_rate", type=float, choices=(-0.04, 0.04), default=0.04,
                    help="Signed CAD6 yaw command; verify both turn directions with the same speed gate")
parser.add_argument("--task", default="Isaac-DKSH-MG90S-Walk-Direct-v0",
                    choices=["Isaac-DKSH-MG90S-Walk-Direct-v0", "Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0",
                             "Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0",
                             "Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1000 or args.num_envs < 1:
    parser.error("Use at least 1000 steps (20 seconds) and one environment")
if args.root_height is not None and args.root_height <= 0:
    parser.error("Root height must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import dksh_isaaclab
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat
from dksh_isaaclab.tasks.direct.spider_navigation.mg90s_env_cfg import (
    MG90S_NO_LOAD_SPEED,
    RC920DMG_NO_LOAD_SPEED,
)
from dksh_isaaclab.tasks.direct.spider_navigation.hexapod_diagnostics import (
    SlipTimer, foot_offsets, foot_kinematics, heading_increment,
)


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(0.020 / args.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    cfg.ground_material.static_friction = args.friction
    cfg.ground_material.dynamic_friction = args.friction
    cfg.simple_collisions = args.simple_collisions
    cfg.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions
    if args.root_height is not None:
        cfg.robot.init_state.pos = (0.0, 0.0, args.root_height)
    if args.simple_collisions and cfg.foot_contact_sensor is None:
        raise ValueError("Simple CAD collisions require the CAD6 task")
    # Inspect one uninterrupted 20 second trajectory in each phase.
    cfg.episode_length_s = (args.steps + 10) * cfg.sim.dt * cfg.decimation
    env = gym.make(args.task, cfg=cfg)
    robot_env = env.unwrapped
    robot = robot_env._robot
    results = {}
    with ExitStack() as resources:
        resources.callback(env.close)
        trace_path = Path(args.timeseries) if args.timeseries else (Path(args.output).with_suffix('.jsonl') if args.output else None)
        trace = None
        if trace_path:
            trace_path = trace_path.resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace = resources.enter_context(trace_path.open('w', encoding='utf-8'))
        contact = getattr(robot_env, '_foot_contact', None)
        if contact is not None:
            expected_names = [f'leg_{i}_tibia' for i in range(6)]
            sensor_order = [contact.body_names.index(name) for name in expected_names]
            foot_body_ids = [robot.body_names.index(name) for name in expected_names]
            offsets = torch.tensor(foot_offsets(), device=robot_env.device, dtype=torch.float32)
        conditions = {
            'task': args.task, 'usd': cfg.robot.spawn.usd_path,
            'observation_contract': cfg.observation_contract,
            'physics_dt_s': cfg.sim.dt, 'control_dt_s': robot_env.step_dt,
            'static_friction': args.friction, 'dynamic_friction': args.friction,
            'restitution': 0.0, 'friction_combine_mode': 'average',
            'initial_root_height_m': cfg.robot.init_state.pos[2],
            'self_collisions': args.self_collisions,
            'collision_model': 'cad_part_boxes_and_assumed_soles' if args.simple_collisions else 'cad_mesh_convex_hulls',
            'assumed_foot_radius_m': cfg.assumed_foot_radius_m if args.simple_collisions else None,
            'assumed_pad_center_in_foot_frame_m': [0.0, 0.0, cfg.assumed_foot_radius_m] if args.simple_collisions else None,
            'timeseries': str(trace_path) if trace_path else None,
            'sample_period_s': robot_env.step_dt,
            'yaw_command_rad_s': args.yaw_rate if hasattr(cfg, 'command_forward_min') else None,
            'yaw_gate_metric': 'unwrapped_world_heading_change_per_elapsed_time',
            'contact_scope': 'tibia_and_attached_colliders' if contact is not None else None,
            'contact_force_kind': 'normal_only_no_tangential_force',
            'foot_velocity_reference': 'world_point_relative_to_body_center_of_mass',
            'torque_sample_kind': 'last_physics_substep_per_control_step',
            'foot_order': [f'leg_{i}' for i in range(6)] if contact is not None else None,
            'slip_duration_s': 0.060, 'slip_speed_threshold_mps': 0.01,
            'mass_kg': float(robot.root_physx_view.get_masses()[0].sum()),
        }
        if trace:
            trace.write(json.dumps({'kind': 'conditions', **conditions}) + '\n')
        assert robot.num_joints == cfg.action_space
        assert robot.num_bodies == cfg.action_space + 1
        print("MG90S_PREFLIGHT_PHYSICS", {
            "task": args.task, "usd": cfg.robot.spawn.usd_path,
            "joints": robot.num_joints, "bodies": robot.num_bodies,
            "mass_kg": float(robot.root_physx_view.get_masses()[0].sum()),
            "effort_caps_nm": {
                name: float(actuator.effort_limit.max())
                for name, actuator in robot.actuators.items()
            },
            "no_load_speeds_rad_s": {
                "mg90s_4v8_hip": MG90S_NO_LOAD_SPEED,
                "rc920dmg_5v_leg": RC920DMG_NO_LOAD_SPEED,
            },
        }, flush=True)
        gait_lift = cfg.gait_lift
        if hasattr(cfg, "command_forward_min"):
            phases = (
                ("stand", 0.0, 0.0, 0.0, 0.0),
                ("forward", cfg.command_speed_min, 0.0, 0.0, gait_lift),
                ("lateral", 0.0, cfg.command_speed_min, 0.0, gait_lift),
                ("yaw", 0.0, 0.0, args.yaw_rate, gait_lift),
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
            previous_heading = euler_xyz_from_quat(robot.data.root_quat_w)[2]
            heading_change = torch.zeros(args.num_envs, device=robot_env.device)
            counts_before = robot_env.completed.copy()
            min_height = 10.
            max_torque = 0.
            max_effort_ratio = 0.
            forward_speed_sum = lateral_speed_sum = yaw_rate_sum = 0.
            saturated_sum = 0.
            torque_peak = torch.zeros(robot.num_joints, device=robot_env.device)
            torque_dwell = torch.zeros((args.num_envs, robot.num_joints), device=robot_env.device)
            speed_error_sum = 0.0
            slip_time = torch.zeros((args.num_envs, 6), device=robot_env.device) if contact else None
            slip_timer = SlipTimer((args.num_envs, 6), robot_env.device) if contact else None
            distance = torch.zeros(args.num_envs, device=robot_env.device)
            previous_position = robot.data.root_pos_w[:, :2].clone()
            first_fall_distance = [None] * args.num_envs
            for step in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(torch.zeros((args.num_envs, cfg.action_space), device=robot_env.device))
                if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite physics or reward")
                assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
                min_height = min(min_height, float(robot.data.root_pos_w[:, 2].min()))
                max_torque = max(max_torque, float(robot.data.applied_torque.abs().max()))
                effort_ratio = robot.data.applied_torque.abs() / robot.data.joint_effort_limits
                torque_peak = torch.maximum(torque_peak, robot.data.applied_torque.abs().amax(0))
                torque_dwell += (effort_ratio >= 0.80) * robot_env.step_dt
                max_effort_ratio = max(max_effort_ratio, float(effort_ratio.max()))
                forward_speed_sum += float(robot.data.root_lin_vel_b[:, 0].mean())
                lateral_speed_sum += float(robot.data.root_lin_vel_b[:, 1].mean())
                yaw_rate_sum += float(robot.data.root_ang_vel_b[:, 2].mean())
                saturated_sum += float((effort_ratio > 0.70).float().mean())
                velocity = robot.data.root_lin_vel_b
                command = robot_env._commands
                speed_error_sum += float(torch.linalg.vector_norm(velocity[:, :2] - command[:, :2], dim=-1).mean())
                current_position = robot.data.root_pos_w[:, :2]
                # Automatic resets teleport the base; exclude those intervals.
                resets = terminated | truncated
                heading = euler_xyz_from_quat(robot.data.root_quat_w)[2]
                heading_delta = heading_increment(heading, previous_heading)
                heading_change += torch.where(resets, torch.zeros_like(heading_delta), heading_delta)
                previous_heading = heading.clone()
                increment = torch.linalg.vector_norm(current_position - previous_position, dim=-1)
                distance += torch.where(resets, torch.zeros_like(increment), increment)
                for index in terminated.nonzero().flatten().tolist():
                    if first_fall_distance[index] is None and robot_env._fallen[index]:
                        first_fall_distance[index] = float(distance[index])
                previous_position = current_position.clone()
                foot_values = None
                if contact is not None:
                    force = contact.data.net_forces_w[:, sensor_order]
                    foot_position, foot_velocity = foot_kinematics(robot, foot_body_ids, offsets)
                    touching = force[..., 2] > cfg.foot_contact_sensor.force_threshold
                    touching[resets] = False
                    slip = slip_timer.update(touching, foot_velocity[..., :2].norm(dim=-1), robot_env.step_dt)
                    slip_timer.elapsed[resets] = 0.0
                    slip_time += slip * robot_env.step_dt
                    foot_values = {'frame_position_w_m': foot_position.cpu().tolist(),
                                   'frame_velocity_w_mps': foot_velocity.cpu().tolist(),
                                   'normal_contact_force_w_n': force.cpu().tolist(),
                                   'slip_qualified': slip.cpu().tolist()}
                if trace:
                    trace.write(json.dumps({
                        'kind': 'sample', 'phase': label, 'time_s': (step + 1) * robot_env.step_dt,
                        'applied_torque_nm': robot.data.applied_torque.cpu().tolist(),
                        'joint_position_rad': robot.data.joint_pos.cpu().tolist(),
                        'joint_velocity_rad_s': robot.data.joint_vel.cpu().tolist(),
                        'joint_target_rad': robot_env._targets.cpu().tolist(),
                        'effort_ratio': effort_ratio.cpu().tolist(),
                        'command_b': command.cpu().tolist(),
                        'velocity_b_mps': velocity.cpu().tolist(),
                        'yaw_rate_rps': robot.data.root_ang_vel_b[:, 2].cpu().tolist(),
                        'root_position_w_m': robot.data.root_pos_w.cpu().tolist(),
                        'root_quaternion_wxyz': robot.data.root_quat_w.cpu().tolist(),
                        'world_heading_delta_rad': heading_delta.cpu().tolist(),
                        'terminated': terminated.cpu().tolist(), 'truncated': truncated.cpu().tolist(),
                        'feet': foot_values,
                    }) + '\n')
            result = {
                "steps": args.steps, "envs": args.num_envs,
                "falls": robot_env.completed["fallen"] - counts_before["fallen"],
                "out_of_bounds": robot_env.completed["out_of_bounds"] - counts_before["out_of_bounds"],
                "min_height_m": min_height, "max_abs_torque_nm": max_torque,
                "max_effort_ratio": max_effort_ratio,
                "mean_forward_speed_mps": forward_speed_sum / args.steps,
                "mean_lateral_speed_mps": lateral_speed_sum / args.steps,
                "mean_yaw_rate_rps": yaw_rate_sum / args.steps,
                "world_heading_change_rad": heading_change.cpu().tolist(),
                "mean_world_heading_rate_rps": float(heading_change.mean()) / (args.steps * robot_env.step_dt),
                "mean_displacement_x_m": float((robot.data.root_pos_w[:, 0] - initial[:, 0]).mean()),
                "mean_displacement_y_m": float((robot.data.root_pos_w[:, 1] - initial[:, 1]).mean()),
                "near_torque_cap_fraction": saturated_sum / args.steps,
                "mean_planar_speed_error_mps": speed_error_sum / args.steps,
                "joint_names": robot.joint_names,
                "joint_effort_caps_nm": robot.data.joint_effort_limits[0].cpu().tolist(),
                "joint_peak_abs_torque_nm": torque_peak.cpu().tolist(),
                "joint_peak_effort_pct": (torque_peak / robot.data.joint_effort_limits[0] * 100).cpu().tolist(),
                "joint_mean_time_above_80pct_ms": (torque_dwell.mean(0) * 1000).cpu().tolist(),
                "joint_max_time_above_80pct_ms": (torque_dwell.amax(0) * 1000).cpu().tolist(),
                "qualified_slip_time_ms": (slip_time * 1000).cpu().tolist() if slip_time is not None else None,
                "distance_to_first_fall_m": first_fall_distance,
                "path_length_m": distance.cpu().tolist(),
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
            yaw_sign = 1.0 if args.yaw_rate > 0 else -1.0
            # Last-substep body omega can alias servo oscillation and is not
            # net world heading. Keep it in diagnostics, but gate the actual
            # orientation change over the trajectory using the same threshold.
            passed &= results["yaw"]["mean_world_heading_rate_rps"] * yaw_sign > 0.005
        passed &= all(r["max_effort_ratio"] <= 1.0001 for r in results.values())
        results["passed"] = passed
        failures = []
        for label, result in results.items():
            if label == 'passed':
                continue
            if result['falls'] or result['out_of_bounds']:
                failures.append(f'{label}: fall or out of bounds')
            if result['max_effort_ratio'] > 1.0001:
                failures.append(f'{label}: effort limit exceeded')
        if 'wave' in results:
            if results['wave']['mean_displacement_x_m'] <= 0.005:
                failures.append('wave: forward displacement below 5 mm')
        else:
            if results['forward']['mean_displacement_x_m'] <= 0.005:
                failures.append('forward: displacement below 5 mm')
            if results['lateral']['mean_displacement_y_m'] <= 0.002:
                failures.append('lateral: displacement below 2 mm')
            if results['yaw']['mean_world_heading_rate_rps'] * yaw_sign <= 0.005:
                failures.append('yaw: command-aligned world heading rate below 0.005 rad/s')
        results['failure_reasons'] = failures
        results["conditions"] = conditions
        if args.output:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not passed:
            print('MG90S_PREFLIGHT_FAIL ' + json.dumps(failures), flush=True)
            raise RuntimeError("MG90S physical preflight did not pass; inspect measurements before training")
        print("MG90S_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
