"""Train, evaluate or play a policy that directly chooses every joint angle."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from environment_cli import add_environment_args, apply_environment_cfg, experiment_for_environment
from joint_angle_checkpoint import (
    ACTION_SEMANTICS, SCHEMA_CONTRACTS, expected_cad_asset_sha256,
    require_v2_task, resolve_checkpoint, validate_checkpoint, validate_control_contract,
)
from joint_angle_runner_utils import (
    EvaluationAccumulator, PolicyStdConstraint, configure_evaluation, effective_seed,
)
from cad_readiness import assert_cad_ready
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("train", "evaluate", "play"), required=True)
parser.add_argument("--task", default="Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2")
parser.add_argument("--checkpoint", default="")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--output", default="", help="Optional JSON evaluation report")
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--allow_provisional_cad", action="store_true")
add_environment_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.max_iterations, args.steps) < 1:
    parser.error("Environment, iteration and step counts must be positive")
try:
    require_v2_task(args.task)
except ValueError as error:
    parser.error(str(error))
args.seed = effective_seed(args.mode, args.seed)
if args.checkpoint:
    validate_checkpoint(args.checkpoint, args.task, args.environment)
cad_report = None
if args.mode == "train" and expected_cad_asset_sha256(args.task) is not None:
    cad_report = assert_cad_ready(allow_provisional=args.allow_provisional_cad)
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import dksh_isaaclab  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def experiment_name(task, environment):
    return experiment_for_environment("dksh_" + task.removeprefix("Isaac-DKSH-").lower().replace("-", "_"), environment)


def train(env, runner, cfg, agent, log, checkpoint):
    base = env.unwrapped
    from dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control import exploration_std
    saturation_probability = getattr(cfg, "max_initial_action_saturation", 0.01)
    std_cap = exploration_std(base._angle_limits, dt=base.step_dt, rate_limit=cfg.target_rate_limit,
                              smoothing=cfg.target_smoothing, saturation_probability=saturation_probability)
    metadata = {
        "task": args.task, "action_semantics": ACTION_SEMANTICS,
        "cad_asset_sha256": expected_cad_asset_sha256(args.task),
        "cad_readiness": cad_report,
        **SCHEMA_CONTRACTS,
        "mass_assumption": getattr(cfg, "mass_assumption", None),
        "environment": args.environment, "difficulty": args.difficulty,
        "seed": cfg.seed, "actions": cfg.action_space, "observations": cfg.observation_space,
        "environments": base.num_envs,
        "joint_names": [base._robot.joint_names[i] for i in base._joint_ids],
        "angle_limits_rad": base._angle_limits[0].cpu().tolist(),
        "target_rate_limit_rad_s": cfg.target_rate_limit, "target_smoothing": cfg.target_smoothing,
        "servo_velocity_limits_rad_s": list(base._servo_velocity_limits[0].cpu().tolist())
            if base._servo_velocity_limits.shape[-1] > 1
            else float(base._servo_velocity_limits[0, 0].cpu()),
        "actuator_contract": getattr(cfg, "actuator_contract", None),
        "servo_model": getattr(cfg, "servo_model", None),
        "servo_reference_voltage_v": getattr(cfg, "servo_reference_voltage_v", None),
        "servo_stall_torque_nm": getattr(cfg, "servo_stall_torque_nm", None),
        "servo_effort_limit_nm": getattr(cfg, "servo_effort_limit_nm", None),
        "hip_servo_model": getattr(cfg, "hip_servo_model", None),
        "leg_servo_model": getattr(cfg, "leg_servo_model", None),
        "hip_servo_reference_voltage_v": getattr(cfg, "hip_servo_reference_voltage_v", None),
        "leg_servo_reference_voltage_v": getattr(cfg, "leg_servo_reference_voltage_v", None),
        "hip_servo_stall_torque_nm": getattr(cfg, "hip_servo_stall_torque_nm", None),
        "leg_servo_stall_torque_nm": getattr(cfg, "leg_servo_stall_torque_nm", None),
        "hip_servo_effort_limit_nm": getattr(cfg, "hip_servo_effort_limit_nm", None),
        "leg_servo_effort_limit_nm": getattr(cfg, "leg_servo_effort_limit_nm", None),
        "supply_voltage": getattr(cfg, "supply_voltage", None),
        "known_servo_mass_per_leg_kg": getattr(cfg, "known_servo_mass_per_leg_kg", None),
        "control_frequency_hz": 1.0 / base.step_dt,
        "exploration_std_cap": std_cap,
        "max_initial_action_saturation": saturation_probability,
        "source_checkpoint": str(checkpoint) if checkpoint else None,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "additional_iterations": args.max_iterations, "status": "initializing",
    }
    path = log / "run_metadata.json"
    std_constraint = None
    try:
        if checkpoint:
            runner.load(str(checkpoint))
            runner.current_learning_iteration += 1
        else:
            # Start near a static standing pose with free trainable outputs.
            # This is a one-time weight initialization, never a gait/angle offset.
            output = [layer for layer in runner.alg.policy.actor if isinstance(layer, torch.nn.Linear)][-1]
            from dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control import normalize_angles
            initial = normalize_angles(base._robot.data.default_joint_pos[:, base._joint_ids], base._angle_limits)[0]
            with torch.no_grad():
                output.weight.zero_()
                output.bias.copy_(initial)
        std_constraint = PolicyStdConstraint(
            runner.alg.policy, runner.alg.optimizer, std_cap, initialize=checkpoint is None,
        )
        metadata["exploration_std_initial"] = std_constraint.values()
        metadata.update(status="running", start_iteration=runner.current_learning_iteration)
        write_json(path, metadata)
        dump_yaml(str(log / "params/env.yaml"), cfg)
        dump_yaml(str(log / "params/agent.yaml"), agent)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        final = log / f"model_{runner.current_learning_iteration}.pt"
        if not final.is_file():
            raise RuntimeError("Training returned without its final checkpoint")
        metadata.update(status="completed", final_checkpoint=final.name,
                        final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                        final_iteration=runner.current_learning_iteration, completed_episodes=base.completed.copy(),
                        exploration_std_final=std_constraint.values())
        write_json(path, metadata)
        print("DKSH_JOINT_ANGLE_TRAINING_COMPLETE " + str(final), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(path, metadata)
        raise
    finally:
        if std_constraint is not None:
            std_constraint.close()


def evaluate_or_play(env, runner, cfg, checkpoint):
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation, _ = env.get_observations()
    base = env.unwrapped
    accumulator = EvaluationAccumulator(base.num_envs)
    steps = 0
    with torch.inference_mode():
        while app.is_running() and (args.mode == "play" or steps < args.steps):
            started = time.perf_counter()
            actions = policy(observation)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy produced non-finite joint-angle actions")
            observation, rewards, dones, _ = env.step(actions)
            if not torch.isfinite(observation).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("Non-finite observation or reward")
            if args.mode == "evaluate":
                accumulator.update(base.step_metrics, dones, base.step_dt)
            steps += 1
            if args.mode == "play" and not args.headless:
                time.sleep(max(0, base.step_dt - (time.perf_counter() - started)))
    if args.mode == "evaluate":
        if steps != args.steps:
            raise RuntimeError("Evaluation ended before the requested number of steps")
        scenario_distance = (float(torch.linalg.vector_norm(torch.tensor(base._scenario.layout.goal[:2])
                             - torch.tensor(base._scenario.layout.spawn[:2])))
                             if base._scenario is not None else float(cfg.evaluation_goal_distance))
        report = {
            "checkpoint": str(checkpoint), "task": args.task, "action_semantics": ACTION_SEMANTICS,
            "environment": args.environment, "difficulty": args.difficulty, "seed": cfg.seed,
            "environments": args.num_envs, "steps": steps,
            "fixed_scenario_distance_m": scenario_distance,
            "goal_radius_m": cfg.goal_radius, "episode_length_s": cfg.episode_length_s,
            **accumulator.report(target_speed=cfg.target_speed,
                                 minimum_speed_fraction=cfg.evaluation_min_speed_fraction, dt=base.step_dt),
        }
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        # Completion is distinct from demonstrating successful locomotion.
        print("DKSH_JOINT_ANGLE_EVAL_COMPLETE locomotion_pass=" + str(report["locomotion_pass"]), flush=True)
    else:
        print("DKSH_JOINT_ANGLE_PLAY_COMPLETE", flush=True)


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    apply_environment_cfg(cfg, args.environment, args.difficulty, args.seed)
    if args.mode == "evaluate":
        configure_evaluation(cfg, args.environment)
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent.device = cfg.sim.device
    agent.seed = cfg.seed
    agent.max_iterations = args.max_iterations
    agent.experiment_name = experiment_name(args.task, args.environment)
    root = Path(__file__).resolve().parents[2] / "logs/rsl_rl" / agent.experiment_name
    checkpoint = (resolve_checkpoint(args.checkpoint, root, args.task, args.environment)
                  if args.checkpoint or args.mode != "train" else None)
    log = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") if args.mode == "train" else None
    if log:
        log.mkdir(parents=True, exist_ok=False)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        base = env.unwrapped
        from dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control import exploration_std
        std_cap = exploration_std(base._angle_limits, dt=base.step_dt, rate_limit=cfg.target_rate_limit,
                                  smoothing=cfg.target_smoothing,
                                  saturation_probability=getattr(cfg, "max_initial_action_saturation", 0.01))
        agent.policy.init_noise_std = std_cap
        if checkpoint:
            metadata = validate_checkpoint(checkpoint, args.task, args.environment)
            validate_control_contract(metadata, [base._robot.joint_names[i] for i in base._joint_ids], base._angle_limits[0],
                                      target_rate_limit=cfg.target_rate_limit, target_smoothing=cfg.target_smoothing,
                                      control_frequency_hz=1 / base.step_dt, exploration_std_cap=std_cap)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log) if log else None, device=agent.device)
        if args.mode == "train":
            train(env, runner, cfg, agent, log, checkpoint)
        else:
            evaluate_or_play(env, runner, cfg, checkpoint)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
