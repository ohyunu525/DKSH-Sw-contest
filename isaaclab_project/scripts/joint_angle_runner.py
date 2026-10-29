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
    ACTION_SEMANTICS, BEHAVIOR_PROFILES, EXPLORATION_PROFILE, SCHEMA_CONTRACTS,
    behavior_profile_from_metadata, expected_cad_asset_sha256,
    require_v2_task, resolve_checkpoint, validate_checkpoint, validate_control_contract,
)
from joint_angle_runner_utils import (
    EvaluationAccumulator, PolicyStdConstraint, configure_evaluation, effective_seed, validate_exploration_bounds,
    add_course_args, resolve_course_args, apply_course_cfg, course_configuration,
    CurriculumCheckpointState, heldout_course_report,
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
parser.add_argument("--initial_noise_std", type=float, default=None)
parser.add_argument("--min_noise_std", type=float, default=None)
parser.add_argument("--max_noise_std", type=float, default=None)
parser.add_argument("--reset_exploration_std", action="store_true",
                    help="On resume, reset only policy std and its optimizer state before collecting rollouts")
parser.add_argument("--behavior_profile", choices=tuple(BEHAVIOR_PROFILES), default=None,
                    help="CAD6 training defaults revised_v7; other tasks default revised_v3; evaluation infers the saved profile")
parser.add_argument("--allow_behavior_transfer", action="store_true",
                    help="Explicitly allow training a known v2 reward/contact profile with another profile")
add_environment_args(parser)
add_course_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.max_iterations, args.steps) < 1:
    parser.error("Environment, iteration and step counts must be positive")
try:
    require_v2_task(args.task)
    if args.mode != "train" and (any(value is not None for value in
                                     (args.initial_noise_std, args.min_noise_std, args.max_noise_std))
                                or args.reset_exploration_std or args.allow_behavior_transfer):
        raise ValueError("Noise std overrides, exploration reset and behavior transfer are training-only options")
    if args.mode == "train":
        default_profile = "revised_v7" if "MG996R-Hip-CAD6" in args.task else "revised_v3"
        args.behavior_profile = args.behavior_profile or default_profile
        if args.behavior_profile in ("revised_v4", "revised_v5", "revised_v6", "revised_v7") and "MG996R-Hip-CAD6" not in args.task:
            raise ValueError(f"{args.behavior_profile} toe-button contacts require the MG996R CAD6 toe geometry")
        validate_exploration_bounds(args.initial_noise_std if args.initial_noise_std is not None else 0.08,
                                    args.min_noise_std if args.min_noise_std is not None else 0.03,
                                    args.max_noise_std if args.max_noise_std is not None else 0.30)
except ValueError as error:
    parser.error(str(error))
args.seed = effective_seed(args.mode, args.seed)
try:
    resolve_course_args(args)
except ValueError as error:
    parser.error(str(error))
checkpoint_compatibility = ({"expected_reward_contract": BEHAVIOR_PROFILES[args.behavior_profile]["reward_contract"],
                             "expected_contact_contract": BEHAVIOR_PROFILES[args.behavior_profile]["contact_contract"],
                             "allow_behavior_transfer": args.allow_behavior_transfer}
                            if args.mode == "train" else {})
if args.checkpoint:
    saved_metadata = validate_checkpoint(args.checkpoint, args.task, args.environment, **checkpoint_compatibility)
    if args.environment == "curriculum" and args.mode != "train" and behavior_profile_from_metadata(saved_metadata) != "revised_v7":
        parser.error("curriculum evaluation requires a revised_v7 distal-support checkpoint")
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


def toe_button_sensor_metadata(cfg):
    offsets = getattr(cfg, "toe_button_offsets", None)
    if offsets is None:
        return None
    return {
        "model": "virtual_toe_button_force_hysteresis_v1",
        "count": len(offsets),
        "offsets_in_tibia_link_frame_m": [[float(value) for value in point] for point in offsets],
        "press_force_threshold_n": float(cfg.toe_button_press_force_threshold),
        "release_force_threshold_n": float(cfg.toe_button_release_force_threshold),
        "toe_surface_gap_tolerance_m": float(cfg.toe_button_height_tolerance),
        "physical_switch_geometry_in_cad": False,
        "physical_switch_stroke_modelled": False,
        "signal_source": "upward tibia ContactSensor force plus CAD distal-toe witness proximity",
    }


def experiment_name(task, environment):
    return experiment_for_environment("dksh_" + task.removeprefix("Isaac-DKSH-").lower().replace("-", "_"), environment)


def train(env, runner, cfg, agent, log, checkpoint, source_metadata=None):
    base = env.unwrapped
    from dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control import exploration_std
    saturation_probability = cfg.exploration_reference_saturation_probability
    saturation_reference = exploration_std(base._angle_limits, dt=base.step_dt, rate_limit=cfg.target_rate_limit,
                                            smoothing=cfg.target_smoothing, saturation_probability=saturation_probability)
    source_bounds = ({"initial": source_metadata.get("exploration_initial_std"),
                      "minimum": source_metadata.get("exploration_min_std"),
                      "maximum": source_metadata.get("exploration_max_std", source_metadata.get("exploration_std_cap"))}
                     if source_metadata else None)
    current_bounds = {"initial": cfg.exploration_initial_std, "minimum": cfg.exploration_min_std,
                      "maximum": cfg.exploration_max_std}
    behavior_contracts = BEHAVIOR_PROFILES[cfg.behavior_profile]
    metadata = {
        "task": args.task, "action_semantics": ACTION_SEMANTICS,
        "cad_asset_sha256": expected_cad_asset_sha256(args.task),
        "cad_readiness": cad_report,
        **SCHEMA_CONTRACTS,
        **behavior_contracts,
        "behavior_profile": cfg.behavior_profile,
        "source_behavior_profile": behavior_profile_from_metadata(source_metadata) if source_metadata else None,
        "source_reward_contract": source_metadata.get("reward_contract") if source_metadata else None,
        "source_contact_contract": source_metadata.get("contact_contract") if source_metadata else None,
        "behavior_transfer_requested": args.allow_behavior_transfer,
        "behavior_transfer_applied": bool(source_metadata and any(source_metadata[name] != value
                                          for name, value in behavior_contracts.items())),
        "mass_assumption": getattr(cfg, "mass_assumption", None),
        "toe_button_sensor": toe_button_sensor_metadata(cfg),
        "environment": args.environment, "difficulty": args.difficulty,
        "seed": cfg.seed, "actions": cfg.action_space, "observations": cfg.observation_space,
        "environments": base.num_envs,
        "course_training": course_configuration(cfg, args.environment),
        "joint_names": [base._robot.joint_names[i] for i in base._joint_ids],
        "angle_limits_rad": base._angle_limits[0].cpu().tolist(),
        "target_rate_limit_rad_s": cfg.target_rate_limit, "target_smoothing": cfg.target_smoothing,
        "target_speed_mps": cfg.target_speed,
        "speed_tracking_sigma_mps": cfg.speed_tracking_sigma,
        "progress_reward_scale": cfg.progress_reward_scale,
        "speed_cost_scale": cfg.speed_cost_scale,
        "knee_below_body_approach_fraction": getattr(cfg, "knee_below_body_approach_fraction", None),
        "knee_below_body_penalty_scale": getattr(cfg, "knee_below_body_penalty_scale", None),
        "body_bottom_offset_z_m": getattr(cfg, "body_bottom_offset_z_m", None),
        "distal_contact_height_tolerance": getattr(cfg, "distal_contact_height_tolerance", None),
        "proximal_contact_height_tolerance": getattr(cfg, "proximal_contact_height_tolerance", None),
        "knee_ground_clearance_m": getattr(cfg, "knee_ground_clearance_m", None),
        "knee_ground_clearance_cost_scale": getattr(cfg, "knee_ground_clearance_cost_scale", None),
        "non_distal_support_cost_scale": getattr(cfg, "non_distal_support_cost_scale", None),
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
        "exploration_profile": EXPLORATION_PROFILE,
        "exploration_initial_std": cfg.exploration_initial_std,
        "exploration_min_std": cfg.exploration_min_std,
        "exploration_max_std": cfg.exploration_max_std,
        "exploration_std_cap": cfg.exploration_max_std,
        "initial_saturation_reference_std": saturation_reference,
        "exploration_reference_saturation_probability": saturation_probability,
        "source_exploration_bounds": source_bounds,
        "exploration_bounds_changed": bool(source_bounds and source_bounds != current_bounds),
        "exploration_std_reset_requested": args.reset_exploration_std,
        "exploration_std_reset_applied": bool(checkpoint and args.reset_exploration_std),
        "source_checkpoint": str(checkpoint) if checkpoint else None,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "additional_iterations": args.max_iterations, "status": "initializing",
    }
    path = log / "run_metadata.json"
    std_constraint = None
    course_state_hook = None
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
        if args.environment == "curriculum":
            course_state_hook = CurriculumCheckpointState(runner, base._scenario, metadata["course_training"])
            metadata["course_state_restore"] = course_state_hook.restore(checkpoint) if checkpoint else "new_curriculum"
            if metadata["course_state_restore"] == "restored_course_sequence_and_curriculum":
                env.reset()  # Resume the course sequence with a fresh physical episode.
            metadata["curriculum_initial"] = base._scenario.statistics()
        metadata["learning_rate_before"] = float(runner.alg.optimizer.param_groups[0]["lr"])
        metadata["optimizer_learning_rates_before"] = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        std_constraint = PolicyStdConstraint(
            runner.alg.policy, runner.alg.optimizer, initial=cfg.exploration_initial_std,
            minimum=cfg.exploration_min_std, maximum=cfg.exploration_max_std,
            initialize=checkpoint is None or args.reset_exploration_std,
            reset_optimizer_state=bool(checkpoint and args.reset_exploration_std),
        )
        metadata["exploration_std_before"] = std_constraint.before_values
        metadata["exploration_std_after"] = std_constraint.after_values
        metadata["exploration_std_optimizer_state_cleared"] = std_constraint.optimizer_state_cleared
        metadata["learning_rate_after"] = float(runner.alg.optimizer.param_groups[0]["lr"])
        metadata["optimizer_learning_rates_after"] = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        metadata["exploration_std_initial"] = std_constraint.values()
        metadata.update(status="running", start_iteration=runner.current_learning_iteration)
        write_json(path, metadata)
        dump_yaml(str(log / "params/env.yaml"), cfg)
        dump_yaml(str(log / "params/agent.yaml"), agent)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        final = log / f"model_{runner.current_learning_iteration}.pt"
        if not final.is_file():
            raise RuntimeError("Training returned without its final checkpoint")
        if course_state_hook is not None:
            metadata["curriculum_final"] = base._scenario.statistics()
            metadata["course_state_sidecar"] = str(course_state_hook.sidecar(final))
        metadata.update(status="completed", final_checkpoint=final.name,
                        final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                        final_iteration=runner.current_learning_iteration, completed_episodes=base.completed.copy(),
                        exploration_std_final=std_constraint.values(),
                        learning_rate_final=float(runner.alg.optimizer.param_groups[0]["lr"]),
                        optimizer_learning_rates_final=[float(group["lr"]) for group in runner.alg.optimizer.param_groups])
        write_json(path, metadata)
        print("DKSH_JOINT_ANGLE_TRAINING_COMPLETE " + str(final), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(path, metadata)
        raise
    finally:
        if course_state_hook is not None:
            course_state_hook.close()
        if std_constraint is not None:
            std_constraint.close()


def evaluate_or_play(env, runner, cfg, checkpoint, source_metadata):
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation, _ = env.get_observations()
    base = env.unwrapped
    accumulator = EvaluationAccumulator(base.num_envs)
    if args.environment == "curriculum":
        base.completed_course_records.clear()
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
                if args.environment == "curriculum":
                    accumulator.update(base.step_metrics, dones, base.step_dt,
                                       course_records=base.completed_course_records)
                    base.completed_course_records.clear()
                else:
                    accumulator.update(base.step_metrics, dones, base.step_dt)
            steps += 1
            if args.mode == "play" and not args.headless:
                time.sleep(max(0, base.step_dt - (time.perf_counter() - started)))
    if args.mode == "evaluate":
        if steps != args.steps:
            raise RuntimeError("Evaluation ended before the requested number of steps")
        scenario_distance = (None if args.environment == "curriculum" else
                             float(torch.linalg.vector_norm(torch.tensor(base._scenario.layout.goal[:2])
                             - torch.tensor(base._scenario.layout.spawn[:2])))
                             if base._scenario is not None else float(cfg.evaluation_goal_distance))
        report = {
            "checkpoint": str(checkpoint), "task": args.task, "action_semantics": ACTION_SEMANTICS,
            "environment": args.environment, "difficulty": args.difficulty, "seed": cfg.seed,
            "behavior_profile": cfg.behavior_profile,
            "reward_contract": cfg.reward_contract, "contact_contract": cfg.contact_contract,
            "source_behavior_profile": behavior_profile_from_metadata(source_metadata),
            "source_reward_contract": source_metadata["reward_contract"],
            "source_contact_contract": source_metadata["contact_contract"],
            "toe_button_sensor": toe_button_sensor_metadata(cfg),
            "environments": args.num_envs, "steps": steps,
            "fixed_scenario_distance_m": scenario_distance,
            "goal_radius_m": cfg.goal_radius, "episode_length_s": cfg.episode_length_s,
            **accumulator.report(target_speed=cfg.target_speed,
                                 minimum_speed_fraction=cfg.evaluation_min_speed_fraction, dt=base.step_dt),
        }
        if args.environment == "curriculum":
            report["course_evaluation"] = course_configuration(cfg, args.environment)
            report.update(heldout_course_report(accumulator.completed, target_speed=cfg.target_speed,
                                               minimum_speed_fraction=cfg.evaluation_min_speed_fraction))
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        # Completion is distinct from demonstrating successful locomotion.
        print("DKSH_JOINT_ANGLE_EVAL_COMPLETE locomotion_pass=" + str(report["locomotion_pass"])
              + (" heldout_course_pass=" + str(report["heldout_course_pass"]) if args.environment == "curriculum" else ""), flush=True)
    else:
        print("DKSH_JOINT_ANGLE_PLAY_COMPLETE", flush=True)


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    apply_environment_cfg(cfg, args.environment, args.difficulty, args.seed)
    apply_course_cfg(cfg, args)
    for argument, attribute in (("initial_noise_std", "exploration_initial_std"),
                                 ("min_noise_std", "exploration_min_std"),
                                 ("max_noise_std", "exploration_max_std")):
        if getattr(args, argument) is not None:
            setattr(cfg, attribute, getattr(args, argument))
    validate_exploration_bounds(cfg.exploration_initial_std, cfg.exploration_min_std, cfg.exploration_max_std)
    if args.mode == "evaluate":
        configure_evaluation(cfg, args.environment)
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent.device = cfg.sim.device
    agent.seed = cfg.seed
    agent.max_iterations = args.max_iterations
    agent.experiment_name = experiment_name(args.task, args.environment)
    root = Path(__file__).resolve().parents[2] / "logs/rsl_rl" / agent.experiment_name
    checkpoint = (resolve_checkpoint(args.checkpoint, root, args.task, args.environment, **checkpoint_compatibility)
                  if args.checkpoint or args.mode != "train" else None)
    source_metadata = (validate_checkpoint(checkpoint, args.task, args.environment, **checkpoint_compatibility)
                       if checkpoint else None)
    cfg.behavior_profile = args.behavior_profile or (behavior_profile_from_metadata(source_metadata)
                                                    if source_metadata else "revised_v3")
    for name, value in BEHAVIOR_PROFILES[cfg.behavior_profile].items():
        setattr(cfg, name, value)
    if args.environment == "curriculum" and cfg.behavior_profile != "revised_v7":
        raise ValueError("curriculum requires revised_v7 distal-support behavior")
    agent.policy.init_noise_std = cfg.exploration_initial_std
    log = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") if args.mode == "train" else None
    if log:
        log.mkdir(parents=True, exist_ok=False)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        base = env.unwrapped
        if checkpoint:
            validate_control_contract(source_metadata, [base._robot.joint_names[i] for i in base._joint_ids], base._angle_limits[0],
                                      target_rate_limit=cfg.target_rate_limit, target_smoothing=cfg.target_smoothing,
                                      control_frequency_hz=1 / base.step_dt)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log) if log else None, device=agent.device)
        if args.mode == "train":
            train(env, runner, cfg, agent, log, checkpoint, source_metadata)
        else:
            evaluate_or_play(env, runner, cfg, checkpoint, source_metadata)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
