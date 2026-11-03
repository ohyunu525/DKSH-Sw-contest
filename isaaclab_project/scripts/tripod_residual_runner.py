"""Search for a CAD6 tripod gait, train PPO corrections, or evaluate either policy."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from cad_readiness import assert_cad_ready
from environment_cli import add_environment_args, apply_environment_cfg, seed_value


TASK = "Isaac-DKSH-Spider-CAD6-Tripod-Residual-Direct-v0"
EXPERIMENT = "dksh_cad6_tripod_residual"
PROFILE_SCHEMA = "cad6_ga_tripod_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(app_launcher):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "train", "evaluate", "play"), required=True)
    parser.add_argument("--profile", type=Path, help="GA profile JSON destination (search) or source")
    parser.add_argument("--checkpoint", type=Path, help="Hybrid PPO checkpoint for resume or inference")
    parser.add_argument("--output", type=Path, help="Optional JSON evaluation report")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--steps", type=int, help="Search horizon, evaluation duration, or optional play limit")
    parser.add_argument("--validation_seeds", type=seed_value, nargs="+")
    parser.add_argument("--allow_provisional_cad", action="store_true")
    add_environment_args(parser)
    app_launcher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.generations) < 1 or args.population < 4:
        parser.error("Environment, iteration and generation counts must be positive, with at least four candidates")
    if args.steps is not None and (args.steps < 1 or args.mode == "train"):
        parser.error("--steps must be positive and is available in search, evaluate and play modes")
    if args.output is not None and args.mode != "evaluate":
        parser.error("--output requires evaluate mode")
    if args.environment == "curriculum":
        parser.error("Choose an individual terrain preset for the hybrid experiment")
    if args.mode == "search":
        if args.environment != "flat" or args.checkpoint:
            parser.error("GA search requires the flat preset and starts from a fresh population")
        search_seed = 42 if args.seed is None else args.seed
        if args.validation_seeds is None:
            args.validation_seeds = [seed for seed in (73, 101, 157, 211) if seed != search_seed][:3]
        if search_seed in args.validation_seeds or len(set(args.validation_seeds)) != len(args.validation_seeds):
            parser.error("GA validation seeds must be distinct from one another and the search seed")
        args.profile = args.profile or PROJECT_ROOT / "outputs/cad6_ga_tripod.json"
        args.steps = args.steps or 120
    elif args.profile is None and args.checkpoint is None:
        parser.error("Pass --profile or --checkpoint")
    if args.mode == "evaluate":
        args.steps = args.steps or 12000
        if args.output and any(path and path.resolve() == args.output.resolve()
                               for path in (args.profile, args.checkpoint)):
            parser.error("--output must be separate from the profile and checkpoint")
    return args


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_gait_profile(data, cad_hash):
    from ga_tripod_search import validate_profile

    if (not isinstance(data, dict) or data.get("schema") != PROFILE_SCHEMA
            or data.get("task") != TASK or data.get("cad_asset_sha256") != cad_hash):
        raise ValueError("GA profile schema, task or CAD asset differs from this hybrid task; rerun search")
    return validate_profile(data.get("gait_profile"))


def read_inputs(args, cad_hash):
    import torch

    profile = json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else None
    parameters = validate_gait_profile(profile, cad_hash) if args.profile else None
    saved = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        infos = state.get("infos") if isinstance(state, dict) else None
        saved = infos.get("cad6_tripod_residual") if isinstance(infos, dict) else None
        if not isinstance(saved, dict):
            raise ValueError("Checkpoint lacks the CAD6 tripod residual contract; select a hybrid checkpoint")
        saved_parameters = validate_gait_profile(saved.get("profile"), cad_hash)
        if parameters is not None and parameters != saved_parameters:
            raise ValueError("The supplied GA profile does not match the checkpoint's reference gait")
        parameters, profile = saved_parameters, saved["profile"]
    return parameters, profile, saved


def run_training(args, base, runner, cfg, agent, log, contract, cad_report):
    import torch
    from isaaclab.utils.io import dump_yaml
    from joint_angle_runner_utils import PolicyStdConstraint, initialize_absolute_actor_output

    if args.checkpoint:
        runner.load(str(args.checkpoint))
        runner.current_learning_iteration += 1
    else:
        initial = torch.zeros(cfg.action_space, device=agent.device)
        initialize_absolute_actor_output(runner.alg.policy.actor, initial, weight_std=0.0)

    metadata = {**contract, "environment": args.environment, "difficulty": args.difficulty,
                "seed": cfg.seed, "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "cad_readiness": cad_report, "status": "running"}
    metadata_path = log / "run_metadata.json"
    constraint = PolicyStdConstraint(runner.alg.policy, runner.alg.optimizer,
                                     initial=cfg.exploration_initial_std, minimum=cfg.exploration_min_std,
                                     maximum=cfg.exploration_max_std, initialize=args.checkpoint is None)
    try:
        write_json(metadata_path, metadata)
        dump_yaml(str(log / "params/env.yaml"), cfg)
        dump_yaml(str(log / "params/agent.yaml"), agent)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        checkpoint = log / f"model_{runner.current_learning_iteration}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("PPO returned without writing its final checkpoint")
        metadata.update(status="completed", final_checkpoint=checkpoint.name, completed_episodes=base.completed.copy())
        write_json(metadata_path, metadata)
        print(f"DKSH_CAD6_HYBRID_TRAIN_COMPLETE {checkpoint}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(metadata_path, metadata)
        raise
    finally:
        constraint.close()


def evaluate_or_play(args, app, env, runner, cfg, contract):
    import torch
    from joint_angle_runner_utils import EvaluationAccumulator

    base = env.unwrapped
    evaluating = args.mode == "evaluate"
    accumulator = EvaluationAccumulator(base.num_envs) if evaluating else None
    if runner:
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
    else:
        zero = torch.zeros((base.num_envs, cfg.action_space), device=base.device)
        policy = lambda _observation: zero

    observation, _ = env.get_observations()
    steps = 0
    with torch.inference_mode():
        while app.is_running() and (args.steps is None or steps < args.steps):
            started = time.perf_counter()
            actions = policy(observation)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy produced non-finite residual actions")
            observation, reward, done, _ = env.step(actions)
            if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite observation or reward")
            if accumulator is not None:
                accumulator.update(base.step_metrics, done, base.step_dt)
            steps += 1
            if not evaluating and not args.headless:
                time.sleep(max(0.0, base.step_dt - (time.perf_counter() - started)))

    if evaluating:
        if steps != args.steps:
            raise RuntimeError("Evaluation ended before the requested number of steps")
        report = {
            **contract, "policy": "ga_plus_ppo" if runner else "ga_only",
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "environment": args.environment, "difficulty": args.difficulty, "seed": cfg.seed,
            "steps": steps, "environments": base.num_envs,
            **accumulator.report(target_speed=cfg.target_speed, minimum_speed_fraction=cfg.evaluation_min_speed_fraction,
                                 dt=base.step_dt, require_strict_contact=True),
        }
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        print(f"DKSH_CAD6_HYBRID_EVALUATE_COMPLETE locomotion_pass={report['locomotion_pass']}", flush=True)
    else:
        print("DKSH_CAD6_HYBRID_PLAY_COMPLETE", flush=True)


def run(args, app, cad_report):
    import gymnasium as gym
    import dksh_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from ga_tripod_search import search
    from joint_angle_checkpoint import expected_cad_asset_sha256, expected_stance_slip_measurement_contract
    from joint_angle_runner_utils import configure_evaluation, effective_seed

    cad_hash = expected_cad_asset_sha256(TASK)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, use_fabric=args.mode != "play")
    apply_environment_cfg(cfg, args.environment, args.difficulty, effective_seed(args.mode, args.seed))
    if args.mode in ("search", "evaluate"):
        configure_evaluation(cfg, args.environment)
    if args.mode != "search":
        cfg.gait_profile, profile, saved = read_inputs(args, cad_hash)

    raw_env = gym.make(TASK, cfg=cfg)
    try:
        base = raw_env.unwrapped
        if args.mode == "search":
            gait = search(base, seed=cfg.seed, population=args.population, generations=args.generations,
                          steps=args.steps, validation_seeds=args.validation_seeds)
            write_json(args.profile, {
                "schema": PROFILE_SCHEMA, "task": TASK, "cad_asset_sha256": cad_hash, "gait_profile": gait,
                "search": {"seed": cfg.seed, "population": args.population, "generations": args.generations,
                           "steps": args.steps, "validation_seeds": args.validation_seeds,
                           "difficulty": args.difficulty, "cad_readiness": cad_report},
            })
            print(f"DKSH_CAD6_HYBRID_SEARCH_COMPLETE {args.profile.resolve()}", flush=True)
            return

        joints = [base._robot.joint_names[i] for i in base._joint_ids]
        contract = {
            "task": TASK, "profile": profile, "action_semantics": cfg.action_semantics,
            "observation_contract": cfg.observation_contract, "actuator_contract": cfg.actuator_contract,
            "terrain_observation_contract": cfg.terrain_observation_contract,
            "behavior_profile": cfg.behavior_profile, "contact_force_contract": cfg.contact_force_contract,
            "reward_contract": cfg.reward_contract, "contact_contract": cfg.contact_contract,
            "stance_slip_measurement_contract": expected_stance_slip_measurement_contract(cfg.behavior_profile),
            "observations": cfg.observation_space, "actions": cfg.action_space,
            "residual_limit_rad": cfg.residual_limit_rad, "joint_names": joints,
            "angle_limits_rad": base._angle_limits[0].cpu().tolist(),
            "target_rate_limit_rad_s": cfg.target_rate_limit, "target_smoothing": cfg.target_smoothing,
            "target_filter_interval_s": cfg.target_filter_interval_s,
            "target_speed_mps": cfg.target_speed,
            "control_frequency_hz": 1 / base.step_dt,
        }
        if saved is not None:
            differences = [name for name, value in contract.items() if saved.get(name) != value]
            if differences:
                raise ValueError(f"Checkpoint contract differs from the current environment in {', '.join(differences)}")

        agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        agent.device, agent.seed = cfg.sim.device, cfg.seed
        agent.max_iterations = args.max_iterations
        agent.experiment_name = f"{EXPERIMENT}_{args.environment}"
        agent.policy.init_noise_std = cfg.exploration_initial_std
        agent.algorithm.schedule, agent.algorithm.learning_rate = "fixed", 1.5e-4
        agent.algorithm.gamma, agent.algorithm.lam = 0.9995 ** 5, 0.98 ** 5
        agent.num_steps_per_env = 32
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent.clip_actions)
        if args.mode == "train":
            class TripodRunner(OnPolicyRunner):
                def save(self, path, infos=None):
                    super().save(path, dict(infos or {}, cad6_tripod_residual=contract))

            log = PROJECT_ROOT / "logs/rsl_rl" / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            log.mkdir(parents=True, exist_ok=False)
            runner = TripodRunner(env, agent.to_dict(), log_dir=str(log), device=agent.device)
            run_training(args, base, runner, cfg, agent, log, contract, cad_report)
        else:
            runner = OnPolicyRunner(env, agent.to_dict(), device=agent.device) if args.checkpoint else None
            evaluate_or_play(args, app, env, runner, cfg, contract)
    finally:
        raw_env.close()


def main():
    from isaaclab.app import AppLauncher

    args = parse_args(AppLauncher)
    cad_report = (assert_cad_ready(allow_provisional=args.allow_provisional_cad)
                  if args.mode in ("search", "train") else None)
    app = AppLauncher(args).app
    try:
        run(args, app, cad_report)
    finally:
        app.close()


if __name__ == "__main__":
    main()
