"""Inspect HexapodRobot.zip, collect CAD demonstrations, clone, evaluate and refine with PPO."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from zipfile import ZipFile

from environment_cli import add_environment_args, apply_environment_cfg
from Upgraded_imitation_learning import (Upgraded_atomic, Upgraded_bc, Upgraded_json, Upgraded_load_bc,
                                         Upgraded_path, UpgradedDemonstrations)


UPGRADED_ROOT = Path(__file__).resolve().parents[2]
UPGRADED_TASK = 'Upgraded-CAD6-Imitation-v0'


def Upgraded_inspect_archive(path):
    """Read source/AST only. Never import hardware modules or extract ZIP members."""
    path = Path(path)
    members, imports = {}, set()
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if Path(entry.filename).name.lower() == 'claude.md' or not entry.filename.endswith('.py'):
                continue
            if entry.file_size > 2_000_000 or len(members) >= 100:
                raise ValueError('Unexpected archive size')
            data = archive.read(entry)
            tree = ast.parse(data.decode('utf-8-sig'))
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split('.')[0])
                elif isinstance(node, ast.Import):
                    imports.update(n.name.split('.')[0] for n in node.names)
                if isinstance(node, ast.ClassDef):
                    names.append(node.name)
            members[entry.filename] = dict(sha256=hashlib.sha256(data).hexdigest(), classes=names)
    if not all(name in members for name in ('hexapod_sdk.py', 'hexapod_stair.py', 'hexapod_stair_seq.py')):
        raise ValueError('The archive does not contain the expected HexapodRobot reference sources')
    available = {Path(n).stem for n in members}
    return dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), members=members,
                missing_local_modules=sorted(n + '.py' for n in imports if n.startswith('hexapod_') and n not in available),
                adapter='Upgraded_single_foot_cad_reference_v1', original_code_executed=False,
                original_firmware_angles_imported=False)


def Upgraded_contract(base, archive):
    folder = UPGRADED_ROOT / 'isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation'
    names = ('Upgraded_hexapod_reference.py', 'Upgraded_imitation_env.py', 'Upgraded_cad6_control.py',
             'Upgraded_cad6_env.py', 'payload_mass.py', 'spider_navigation_env_cfg.py',
             'joint_angle_env.py', 'joint_angle_env_cfg.py', 'spider_navigation_env.py',
             'joint_angle_control.py', 'joint_angle_contact.py', 'joint_angle_foot_geometry.py',
             'joint_angle_non_distal_geometry.py', 'robot_actuators.py', 'servo_specs.py')
    cfg = base.cfg
    urdf = UPGRADED_ROOT / 'isaaclab_project/assets/spiderbot_variants/onshape_20260928/robot.training.urdf'
    return dict(task=UPGRADED_TASK, archive_sha256=archive['archive_sha256'], adapter=archive['adapter'],
                source_sha256={n: hashlib.sha256((folder / n).read_text(encoding='utf-8').encode()).hexdigest() for n in names},
                urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                usd_sha256=hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest(),
                actor_observations=cfg.observation_space, critic_observations=cfg.state_space, actions=18,
                joint_names=[base._robot.joint_names[i] for i in base._joint_ids],
                angle_limits_rad=base._angle_limits[0].cpu().tolist(), policy_dt=base.step_dt,
                reference_settings=dict(cfg.reference_settings),
                actuator_settings={name: {field: getattr(actuator, field) for field in
                    ('joint_names_expr', 'stiffness', 'damping', 'armature', 'friction', 'effort_limit',
                     'effort_limit_sim', 'velocity_limit', 'velocity_limit_sim', 'saturation_effort')}
                    for name, actuator in cfg.robot.actuators.items()},
                physics_dt=cfg.sim.dt, lidar_payload_mass_kg=cfg.lidar_payload_mass_kg,
                **{n: getattr(cfg, n) for n in ('action_semantics', 'observation_contract', 'control_contract',
                    'reward_contract', 'contact_contract', 'actuator_contract', 'contact_force_contract',
                    'target_rate_limit', 'target_smoothing', 'target_filter_interval_s', 'history_frames',
                    'ik_clearance', 'target_speed', 'imu_angular_noise', 'imu_gravity_noise')})


def Upgraded_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('inspect', 'collect', 'bc', 'evaluate', 'train'))
    parser.add_argument('--archive', type=Path, default=UPGRADED_ROOT / 'HexapodRobot.zip')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--maximum_samples', type=int, default=200000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--device')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--allow_provisional_cad', action='store_true')
    add_environment_args(parser)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps, args.epochs, args.batch_size, args.maximum_samples, args.max_iterations) < 1:
        parser.error('Counts must be positive')
    if args.environment == 'curriculum':
        parser.error('Select an individual terrain preset')
    if args.mode == 'bc' and not args.dataset:
        parser.error('BC requires --dataset from a completed collect run')
    if args.mode in ('train', 'evaluate') and not args.checkpoint:
        parser.error('Use --checkpoint with an Upgraded BC or PPO checkpoint')
    if args.mode in ('inspect', 'collect', 'bc') and args.checkpoint:
        parser.error('--checkpoint is only used by train/evaluate')
    args.seed = 42 if args.seed is None else args.seed
    args.device = args.device or ('cpu' if args.mode in ('inspect', 'bc') else 'cuda:0')
    identity = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    defaults = {'inspect': 'Upgraded_archive_report.json', 'collect': f'Upgraded_demonstrations_{identity}',
                'bc': f'Upgraded_bc_{identity}.pt', 'evaluate': f'Upgraded_imitation_evaluation_{identity}.json',
                'train': f'Upgraded_imitation_ppo_{identity}'}
    args.output = Upgraded_path(args.output or UPGRADED_ROOT / 'outputs' / defaults[args.mode])
    if args.mode == 'bc' and args.output.suffix != '.pt':
        parser.error('BC output must be an Upgraded*.pt file')
    if args.mode in ('inspect', 'evaluate') and args.output.suffix != '.json':
        parser.error('Report output must be an Upgraded*.json file')
    if args.mode != 'inspect' and args.output.exists():
        parser.error('Output already exists; choose a new Upgraded output path')
    if args.checkpoint and args.output == args.checkpoint.resolve():
        parser.error('Checkpoint and output must differ')
    return args


def Upgraded_rollout(args, app, raw, contract, policy=None):
    import torch
    from joint_angle_runner_utils import EvaluationAccumulator

    base = raw.unwrapped
    collector = (UpgradedDemonstrations(args.output, contract, base.num_envs,
                                       maximum_episode_steps=base.max_episode_length + 2)
                 if args.mode == 'collect' else None)
    accumulator = EvaluationAccumulator(base.num_envs)
    obs, _ = raw.reset()
    faults, valid_steps, steps = Counter(), 0.0, 0
    max_tracking, max_torque = 0.0, 0.0
    completed = False
    try:
        with torch.inference_mode():
            while app.is_running() and steps < args.steps:
                before = obs['policy'].clone()
                action = (base.reference_actions.clone() if policy is None else policy(before).clamp(-1, 1))
                obs, reward, terminated, truncated, _ = raw.step(action)
                if not all(torch.isfinite(t).all() for t in (obs['policy'], obs['critic'], reward)):
                    raise RuntimeError('Non-finite imitation rollout')
                done = terminated | truncated
                metrics = base.step_metrics
                if collector:
                    collector.update(before, action, done, metrics, base.step_dt)
                accumulator.update(metrics, done, base.step_dt)
                valid_steps += float(metrics['Upgraded_reference_valid'].sum())
                max_tracking = max(max_tracking, float(metrics['Upgraded_tracking_error_max_rad'].max()))
                max_torque = max(max_torque, float(metrics['Upgraded_torque_abs_max_nm'].max()))
                for fault in metrics['Upgraded_reference_fault'][done].cpu().tolist():
                    faults[base._reference.fault_names[int(fault)]] += 1
                steps += 1
                if steps % 250 == 0:
                    print(f'Upgraded {args.mode} step={steps} episodes={len(accumulator.completed)} '
                          f'accepted={len(collector.records) if collector else "n/a"} '
                          f'max_tracking_rad={max_tracking:.3f} faults={dict(faults)}', flush=True)
        if steps != args.steps:
            raise RuntimeError('Simulation closed before the requested rollout finished')
        completed = True
    finally:
        manifest = collector.finish() if collector else None
        if steps:
            report = accumulator.report(target_speed=base.cfg.target_speed, minimum_speed_fraction=0.1,
                                        dt=base.step_dt, require_strict_contact=True)
            report.update(Upgraded_contract=contract, mode=args.mode, seed=args.seed,
                          environment=args.environment, difficulty=args.difficulty, requested_steps=args.steps,
                          completed_steps=steps, run_completed=completed,
                          reference_valid_fraction=valid_steps / (steps * base.num_envs),
                          reference_terminal_faults=dict(faults), maximum_tracking_error_rad=max_tracking,
                          maximum_applied_torque_nm=max_torque,
                          accepted_demonstration_episodes=manifest['accepted_episodes'] if manifest else None)
            destination = args.output / 'Upgraded_collection_report.json' if collector else args.output
            Upgraded_json(destination, report)
    if collector and manifest['accepted_episodes'] < 2:
        raise RuntimeError(f'Fewer than two usable demonstration episodes. BC was not started. Inspect {args.output}')
    print(f'Upgraded_{args.mode.upper()}_COMPLETE {args.output}', flush=True)


def Upgraded_simulation(args, app, archive):
    import gymnasium as gym
    import torch
    import dksh_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = parse_env_cfg(UPGRADED_TASK, device=args.device, num_envs=args.num_envs)
    apply_environment_cfg(cfg, args.environment, args.difficulty, args.seed)
    cfg.evaluation_mode = args.mode == 'evaluate'
    diagnostic_dir = args.output if args.mode in ('collect', 'train') else args.output.parent
    # The collector creates its new directory itself. The writer opens lazily on failure.
    cfg.contact_diagnostics_path = str(diagnostic_dir / (args.output.stem + '_contacts.jsonl'))
    raw = gym.make(UPGRADED_TASK, cfg=cfg)
    try:
        base = raw.unwrapped
        contract = Upgraded_contract(base, archive)
        if args.mode == 'collect':
            Upgraded_rollout(args, app, raw, contract)
            return
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        is_bc = state.get('kind') == 'Upgraded_BC_v1'
        if is_bc:
            actor, _ = Upgraded_load_bc(args.checkpoint, contract, base.device)
        elif state.get('infos', {}).get('Upgraded_contract') != contract:
            raise ValueError('PPO checkpoint contract mismatch')
        if args.mode == 'evaluate' and is_bc:
            Upgraded_rollout(args, app, raw, contract, actor)
            return
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from joint_angle_runner_utils import PolicyStdConstraint

        agent = load_cfg_from_registry(UPGRADED_TASK, 'rsl_rl_cfg_entry_point')
        agent.device, agent.seed = cfg.sim.device, cfg.seed
        agent.empirical_normalization = False
        agent.policy.activation = 'elu'
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)

        class UpgradedRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                destination = Path(path).with_name('Upgraded_' + Path(path).name)
                def write(temporary):
                    super(UpgradedRunner, self).save(str(temporary), dict(infos or {}, Upgraded_contract=contract))
                Upgraded_atomic(destination, write)

        if args.mode == 'train':
            args.output.mkdir(parents=True, exist_ok=False)
        runner = UpgradedRunner(env, agent.to_dict(), log_dir=str(args.output) if args.mode == 'train' else None,
                               device=agent.device)
        if is_bc:
            runner.alg.policy.actor.load_state_dict(actor.state_dict(), strict=True)
        else:
            runner.load(str(args.checkpoint), load_optimizer=args.mode == 'train')
            runner.current_learning_iteration += 1
        if args.mode == 'evaluate':
            Upgraded_rollout(args, app, raw, contract, runner.get_inference_policy(device=base.device))
            return
        metadata = dict(Upgraded_contract=contract, source_checkpoint=str(args.checkpoint), source_kind='BC' if is_bc else 'PPO',
                        seed=args.seed, environment=args.environment, difficulty=args.difficulty, status='running')
        Upgraded_json(args.output / 'Upgraded_run_metadata.json', metadata)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(args.output / 'Upgraded_env.yaml'), cfg)
        dump_yaml(str(args.output / 'Upgraded_agent.yaml'), agent)
        constraint = PolicyStdConstraint(runner.alg.policy, runner.alg.optimizer, initial=0.06,
                                         minimum=0.02, maximum=0.15, initialize=is_bc)
        try:
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            metadata['status'] = 'completed'
            metadata['final_checkpoint'] = str(args.output / f'Upgraded_model_{runner.current_learning_iteration}.pt')
            if not Path(metadata['final_checkpoint']).is_file():
                raise RuntimeError('PPO completed without a final checkpoint')
        except BaseException as error:
            metadata.update(status='failed', error=repr(error))
            raise
        finally:
            constraint.close()
            Upgraded_json(args.output / 'Upgraded_run_metadata.json', metadata)
        print(f'Upgraded_TRAIN_COMPLETE {metadata["final_checkpoint"]}', flush=True)
    finally:
        raw.close()


def Upgraded_main(argv=None):
    app = None
    try:
        args = Upgraded_args(argv)
        if args.mode == 'bc':
            report = Upgraded_bc(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size,
                                 seed=args.seed, device=args.device, maximum_samples=args.maximum_samples)
            print(f'Upgraded_BC_COMPLETE {report["checkpoint"]}', flush=True)
        else:
            archive = Upgraded_inspect_archive(args.archive)
            if args.mode == 'inspect':
                Upgraded_json(args.output, archive)
                print(json.dumps(archive, ensure_ascii=False, indent=2), flush=True)
                print(f'Upgraded_INSPECT_COMPLETE {args.output}', flush=True)
            else:
                from cad_readiness import assert_cad_ready
                assert_cad_ready(allow_provisional=args.allow_provisional_cad)
                from isaaclab.app import AppLauncher
                app = AppLauncher(headless=not args.gui, device=args.device).app
                Upgraded_simulation(args, app, archive)
        return 0
    except Exception:
        print('Upgraded_FAILED', flush=True)
        traceback.print_exc(file=sys.stdout)
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(Upgraded_main())
