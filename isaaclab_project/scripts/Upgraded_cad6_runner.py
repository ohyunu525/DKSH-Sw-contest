"""Upgraded CAD6 baseline verification, asymmetric PPO training and evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
import traceback

from environment_cli import add_environment_args, apply_environment_cfg
from cad_readiness import assert_cad_ready
from tripod_residual_runner import write_json


UPGRADED_TASK = "Upgraded-CAD6-PMTG-v0"
UPGRADED_ROOT = Path(__file__).resolve().parents[2]
UPGRADED_DEFAULT_BASELINE = UPGRADED_ROOT / "outputs/Upgraded_baseline.json"


def Upgraded_parse_args(app_launcher):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "baseline", "train", "evaluate", "play"), required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline_report", type=Path, default=UPGRADED_DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow_provisional_cad", action="store_true")
    add_environment_args(parser)
    app_launcher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations) < 1 or args.steps is not None and args.steps < 1:
        parser.error("Counts must be positive")
    if args.environment == "curriculum":
        parser.error("Select an individual terrain preset for Upgraded training")
    if args.mode == "baseline" and (args.checkpoint or args.environment != "flat"):
        parser.error("The Upgraded baseline runs the zero-action controller on flat ground")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("Evaluation requires an Upgraded checkpoint. Use baseline to assess the controller")
    if args.mode == "check" and args.checkpoint:
        parser.error("The check mode runs the zero-action controller")
    if args.mode == "train" and args.steps is not None:
        parser.error("Use --max_iterations for training")
    if args.output and args.mode not in ("baseline", "evaluate", "check"):
        parser.error("--output requires baseline, evaluate or check mode")
    if args.output and not args.output.name.startswith("Upgraded"):
        parser.error("Generated report names must start with Upgraded")
    if args.output and args.checkpoint and args.output.resolve() == args.checkpoint.resolve():
        parser.error("Report and checkpoint paths must differ")
    if args.mode == "baseline":
        args.output = args.output or args.baseline_report
        if not args.output.name.startswith("Upgraded"):
            parser.error("Baseline report names must start with Upgraded")
    if args.mode == "evaluate":
        args.output = args.output or UPGRADED_ROOT / "outputs/Upgraded_evaluation.json"
    if args.mode != "play":
        args.steps = args.steps or (64 if args.mode == "check" else 1200)
    args.seed = args.seed if args.seed is not None else (42 if args.mode == "train" else 211)
    return args


def Upgraded_contract(base):
    from joint_angle_checkpoint import expected_cad_asset_sha256

    cfg = base.cfg
    folder = UPGRADED_ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"
    sources = ("Upgraded_cad6_control.py", "Upgraded_cad6_env.py", "joint_angle_env.py",
               "joint_angle_control.py", "joint_angle_env_cfg.py", "joint_angle_contact.py",
               "joint_angle_foot_geometry.py", "joint_angle_non_distal_geometry.py")
    hashes = {name: hashlib.sha256((folder / name).read_text(encoding="utf-8").encode()).hexdigest()
              for name in sources}
    return {
        "task": UPGRADED_TASK, "cad_sha256": expected_cad_asset_sha256(UPGRADED_TASK), "source_sha256": hashes,
        "usd_sha256": hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest(),
        **{name: getattr(cfg, name) for name in (
            "action_semantics", "observation_contract", "control_contract", "reward_contract", "contact_contract",
            "actuator_contract", "contact_force_contract", "target_rate_limit", "target_smoothing",
            "target_filter_interval_s", "history_frames", "gait_period", "gait_duty", "gait_lift",
            "foothold_offset_limit", "ik_clearance", "target_speed", "imu_angular_noise", "imu_gravity_noise")},
        "actor_observations": cfg.observation_space, "critic_observations": cfg.state_space, "actions": 18,
        "joint_names": [base._robot.joint_names[i] for i in base._joint_ids],
        "angle_limits_rad": base._angle_limits[0].cpu().tolist(), "policy_dt": base.step_dt,
    }


def Upgraded_validate_baseline(path, contract):
    report = json.loads(path.read_text(encoding="utf-8"))
    rejection = report.get("Upgraded_ik_rejection_fraction")
    valid_rejection = (isinstance(rejection, (int, float)) and math.isfinite(rejection) and 0 <= rejection <= 0.001)
    if (report.get("Upgraded_contract") != contract or report.get("mode") != "baseline"
            or report.get("environment") != "flat" or report.get("locomotion_pass") is not True
            or not valid_rejection):
        raise ValueError("A passing Upgraded flat baseline report for this controller is required before fresh PPO training")


def Upgraded_validate_checkpoint(path, contract):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    infos = state.get("infos") if isinstance(state, dict) else None
    saved = infos.get("Upgraded_contract") if isinstance(infos, dict) else None
    if saved != contract:
        raise ValueError("Checkpoint CAD, controller, observations or action contract differs from this Upgraded task")


def Upgraded_evaluate(args, app, env, runner, contract):
    import torch
    from joint_angle_runner_utils import EvaluationAccumulator

    base = env.unwrapped
    if runner is not None:
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
    else:
        zero = torch.zeros((base.num_envs, 18), device=base.device)
        policy = lambda _obs: zero
    accumulator = EvaluationAccumulator(base.num_envs)
    observation, _ = env.get_observations()
    rejections, maximum_ik_error, steps = 0.0, 0.0, 0
    with torch.inference_mode():
        while app.is_running() and (args.steps is None or steps < args.steps):
            start = time.perf_counter()
            observation, reward, done, extras = env.step(policy(observation))
            critic = extras["observations"]["critic"]
            if not all(torch.isfinite(value).all() for value in (observation, critic, reward)):
                raise RuntimeError("Non-finite Upgraded observation or reward")
            accumulator.update(base.step_metrics, done, base.step_dt)
            rejections += float(base.step_metrics["Upgraded_ik_rejection_fraction"].mean())
            maximum_ik_error = max(maximum_ik_error, float(base.step_metrics["Upgraded_ik_error_m"].max()))
            steps += 1
            if steps % 100 == 0:
                print(f"Upgraded {args.mode} step={steps} completed={base.completed}", flush=True)
            if args.mode == "play" and not args.headless:
                time.sleep(max(0, base.step_dt - (time.perf_counter() - start)))
    if args.mode != "play" and steps != args.steps:
        raise RuntimeError("Upgraded evaluation stopped before its requested duration")
    if steps == 0:
        raise RuntimeError("Upgraded evaluation completed zero steps")
    report = {
        "Upgraded_contract": contract, "mode": args.mode, "environment": args.environment,
        "seed": args.seed, "steps": steps, "environments": base.num_envs,
        "Upgraded_ik_rejection_fraction": rejections / steps, "Upgraded_max_ik_error_m": maximum_ik_error,
        **accumulator.report(target_speed=base.cfg.target_speed, minimum_speed_fraction=base.cfg.evaluation_min_speed_fraction,
                             dt=base.step_dt, require_strict_contact=True),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if args.mode == "baseline" and (not report["locomotion_pass"] or rejections / steps > 0.001):
        raise RuntimeError(f"Upgraded baseline has not passed. Inspect {args.output} and its contact log")
    print(f"Upgraded_{args.mode.upper()}_COMPLETE", flush=True)


def Upgraded_train(args, env, runner, cfg, agent, log, contract, cad_report):
    import torch
    from isaaclab.utils.io import dump_yaml
    from joint_angle_runner_utils import PolicyStdConstraint, initialize_absolute_actor_output

    if args.checkpoint:
        runner.load(str(args.checkpoint))
        runner.current_learning_iteration += 1
    else:
        initialize_absolute_actor_output(runner.alg.policy.actor, torch.zeros(18, device=env.device), weight_std=0)
    metadata = {"Upgraded_contract": contract, "status": "running", "seed": cfg.seed,
                "environment": args.environment, "difficulty": args.difficulty, "cad_readiness": cad_report,
                "source_checkpoint": str(args.checkpoint) if args.checkpoint else None}
    destination = log / "Upgraded_run_metadata.json"
    constraint = PolicyStdConstraint(runner.alg.policy, runner.alg.optimizer, initial=0.12, minimum=0.04,
                                     maximum=0.25, initialize=args.checkpoint is None)
    try:
        write_json(destination, metadata)
        dump_yaml(str(log / "Upgraded_env.yaml"), cfg)
        dump_yaml(str(log / "Upgraded_agent.yaml"), agent)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        checkpoint = log / f"Upgraded_model_{runner.current_learning_iteration}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Upgraded PPO finished without its final checkpoint")
        metadata.update(status="completed", final_checkpoint=str(checkpoint), completed_episodes=env.unwrapped.completed.copy())
        write_json(destination, metadata)
        print(f"Upgraded_TRAIN_COMPLETE {checkpoint}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(destination, metadata)
        raise
    finally:
        constraint.close()


def Upgraded_run(args, app, cad_report):
    import gymnasium as gym
    import dksh_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from joint_angle_runner_utils import configure_evaluation

    cfg = parse_env_cfg(UPGRADED_TASK, device=args.device, num_envs=args.num_envs, use_fabric=not args.mode == "play")
    apply_environment_cfg(cfg, args.environment, args.difficulty, args.seed)
    if args.mode != "train":
        configure_evaluation(cfg, args.environment)
    log = UPGRADED_ROOT / "logs/rsl_rl" / f"Upgraded_cad6_{args.environment}" / datetime.now().strftime("Upgraded_%Y-%m-%d_%H-%M-%S_%f")
    if args.mode == "train":
        log.mkdir(parents=True, exist_ok=False)
        cfg.contact_diagnostics_path = str(log / "Upgraded_contact_failures.jsonl")
    elif args.output:
        cfg.contact_diagnostics_path = str(args.output.with_name(args.output.stem + "_contacts.jsonl"))
    raw = gym.make(UPGRADED_TASK, cfg=cfg)
    try:
        base = raw.unwrapped
        contract = Upgraded_contract(base)
        if args.checkpoint:
            Upgraded_validate_checkpoint(args.checkpoint, contract)
        elif args.mode == "train":
            Upgraded_validate_baseline(args.baseline_report, contract)
        agent = load_cfg_from_registry(UPGRADED_TASK, "rsl_rl_cfg_entry_point")
        agent.device, agent.seed = cfg.sim.device, cfg.seed
        agent.max_iterations = args.max_iterations
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        if args.mode == "train":
            class UpgradedRunner(OnPolicyRunner):
                def save(self, path, infos=None):
                    destination = Path(path).with_name("Upgraded_" + Path(path).name)
                    with NamedTemporaryFile(prefix="Upgraded_pending_", dir=destination.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                    try:
                        super().save(str(temporary), dict(infos or {}, Upgraded_contract=contract))
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)

            runner = UpgradedRunner(env, agent.to_dict(), log_dir=str(log), device=agent.device)
            Upgraded_train(args, env, runner, cfg, agent, log, contract, cad_report)
        else:
            runner = OnPolicyRunner(env, agent.to_dict(), device=agent.device) if args.checkpoint else None
            Upgraded_evaluate(args, app, env, runner, contract)
    finally:
        raw.close()


def Upgraded_main():
    app = None
    try:
        from isaaclab.app import AppLauncher

        args = Upgraded_parse_args(AppLauncher)
        cad_report = assert_cad_ready(allow_provisional=args.allow_provisional_cad)
        app = AppLauncher(args).app
        Upgraded_run(args, app, cad_report)
    except Exception:
        # Print before Kit shutdown, which can terminate Python before its normal traceback.
        print("Upgraded_FAILED", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1
    finally:
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(Upgraded_main())
