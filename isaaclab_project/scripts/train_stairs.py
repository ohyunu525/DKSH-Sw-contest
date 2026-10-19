"""Train or independently evaluate the sensor-only provisional CAD6 stair task."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from cad_readiness import assert_cad_ready
from mg90s_checkpoint import validate_checkpoint
from isaaclab.app import AppLauncher

TASK = 'Isaac-DKSH-MG90S-CAD6-Stairs-Sensors-Direct-v0'
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('train', 'evaluate'), default='train')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--max_iterations', type=int, default=300)
parser.add_argument('--checkpoint', default='')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--level', type=int, choices=range(6), default=0)
parser.add_argument('--steps', type=int, default=4100, help='Evaluation: >= one full 80s episode')
parser.add_argument('--allow_provisional_cad', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed is None:
    args.seed = 43 if args.mode == 'evaluate' else 42
if args.num_envs < 1 or args.max_iterations < 1:
    parser.error('Environment and iteration counts must be positive')
if args.mode == 'evaluate' and (not args.checkpoint or args.steps < 4100):
    parser.error('Evaluation requires a checkpoint and >=4100 steps')
cad_report = assert_cad_ready(allow_provisional=args.allow_provisional_cad)
checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
if checkpoint:
    validate_checkpoint(checkpoint, observations=240, actions=18, task=TASK)
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import dksh_isaaclab
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
except BaseException:
    import sys
    import traceback
    traceback.print_exc(file=sys.__stderr__)
    app.close()
    raise


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.stair_initial_level = args.level
    cfg.stair_curriculum = args.mode == 'train'
    agent = load_cfg_from_registry(TASK, 'rsl_rl_cfg_entry_point')
    agent.seed, agent.device = args.seed, cfg.sim.device
    log = ROOT / 'logs/rsl_rl' / agent.experiment_name / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    log.mkdir(parents=True, exist_ok=False)
    metadata = {'task': TASK, 'mode': args.mode, 'seed': args.seed,
                'started_at': datetime.now().astimezone().isoformat(), 'status': 'initializing',
                'observations': cfg.observation_space, 'actions': cfg.action_space,
                'observation_contract': cfg.observation_contract, 'hardware_deployable': False,
                'actuator_contract': cfg.actuator_contract,
                'hip_servo_model': cfg.hip_servo_model, 'leg_servo_model': cfg.leg_servo_model,
                'hip_reference_voltage_v': cfg.hip_servo_reference_voltage_v,
                'leg_reference_voltage_v': cfg.leg_servo_reference_voltage_v,
                'known_servo_mass_per_leg_kg': cfg.known_servo_mass_per_leg_kg,
                'mass_assumption': cfg.mass_assumption,
                'servo_zero_pwm_calibrated': False,
                'sensor_assumptions': cfg.sensor_assumptions, 'cad_source': cad_report,
                'initial_level': args.level, 'num_envs': args.num_envs,
                'physics_dt_s': cfg.sim.dt, 'control_dt_s': cfg.sim.dt * cfg.decimation,
                'additional_iterations': args.max_iterations if args.mode == 'train' else 0,
                'source_checkpoint': str(checkpoint) if checkpoint else None,
                'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None}
    record = log / 'run_metadata.json'
    record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    env = None
    try:
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent.clip_actions)
        base = env.unwrapped
        metadata['mass_kg'] = float(base._robot.root_physx_view.get_masses()[0].sum())
        metadata['actuator_limits'] = {
            name: {'effort_limit_nm': float(actuator.effort_limit.max()),
                   'velocity_limit_rad_s': float(actuator.velocity_limit.max())}
            for name, actuator in base._robot.actuators.items()
        }
        hip = base._robot.actuators['mg996r_4v8_hip']
        from dksh_isaaclab.tasks.direct.spider_navigation.servo_specs import (
            MG996R_STALL_TORQUE_NM, MG996R_NO_LOAD_SPEED_RAD_S, PROVISIONAL_EFFORT_DERATING,
        )
        if not torch.allclose(hip.effort_limit, torch.full_like(hip.effort_limit, PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM)):
            raise RuntimeError('MG996R hip effort cap was not applied')
        if not torch.allclose(hip.velocity_limit, torch.full_like(hip.velocity_limit, MG996R_NO_LOAD_SPEED_RAD_S)):
            raise RuntimeError('MG996R hip speed cap was not applied')
        hip_ids = base._joint_ids[::3]
        if not torch.allclose(base._robot.data.joint_effort_limits[:, hip_ids], hip.effort_limit):
            raise RuntimeError('PhysX hip effort caps do not match MG996R actuator model')
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log), device=agent.device)
        if checkpoint:
            runner.load(str(checkpoint), load_optimizer=args.mode == 'train')
            if args.mode == 'train':
                runner.current_learning_iteration += 1
        metadata.update(status='running', start_iteration=runner.current_learning_iteration)
        record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        dump_yaml(str(log / 'params/env.yaml'), cfg)
        dump_yaml(str(log / 'params/agent.yaml'), agent)
        snapshot = log / 'source_snapshot'
        snapshot.mkdir()
        task_dir = ROOT / 'isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation'
        for source in [*task_dir.glob('mg90s*.py'), task_dir / 'stair_sensing.py',
                       task_dir / 'onshape_cad6_gait.py', task_dir / 'payload_mass.py',
                       task_dir / 'servo_specs.py', task_dir / '__init__.py',
                       ROOT / 'isaaclab_project/assets/spiderbot_variants/cad_source.json',
                       Path(__file__), Path(__file__).with_name('mg90s_checkpoint.py')]:
            shutil.copy2(source, snapshot / source.name)
        print('STAIRS_RUN_STARTED ' + str(log), flush=True)
        if args.mode == 'train':
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            final = log / f'model_{runner.current_learning_iteration}.pt'
            if not final.is_file():
                raise RuntimeError('Training returned without final checkpoint')
            metadata.update(final_checkpoint=final.name, final_iteration=runner.current_learning_iteration,
                            final_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
            # Finishing PPO is NOT proof that any staircase was climbed.
            metadata['stair_climbing_verified'] = False
        else:
            policy = runner.get_inference_policy(device=base.device)
            obs, _ = env.get_observations()
            counts = base.completed.copy()
            max_effort, max_x, min_clearance = 0.0, 0.0, float('inf')
            from dksh_isaaclab.tasks.direct.spider_navigation import stair_sensing as sensing
            heights = torch.tensor(sensing.HEIGHTS_M, device=base.device)[base._levels]
            with torch.inference_mode():
                for _ in range(args.steps):
                    actions = policy(obs)
                    if actions.shape != (args.num_envs, 18) or not torch.isfinite(actions).all():
                        raise RuntimeError('Invalid policy action')
                    obs, reward, _, _ = env.step(actions)
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        raise RuntimeError('Non-finite sensor observation/reward')
                    pos = base._robot.data.root_pos_w - base._episode_origins
                    max_x = max(max_x, float(pos[:, 0].max()))
                    min_clearance = min(min_clearance, float((pos[:, 2] - sensing.terrain_height(pos[:, 0], heights)).min()))
                    max_effort = max(max_effort, float((base._robot.data.applied_torque.abs() / base._robot.data.joint_effort_limits).max()))
            attempts = sum(base.stair_stats['attempts'])
            successes = sum(base.stair_stats['successes'])
            # Avoid small-sample/no-episode passes; evaluate many episodes before hardware.
            passed = attempts >= args.num_envs and successes / max(attempts, 1) >= 0.8
            falls = base.completed['fallen'] - counts['fallen']
            escapes = base.completed['out_of_bounds'] - counts['out_of_bounds']
            passed = passed and falls == 0 and escapes == 0 and max_effort <= 1.0001
            metadata['evaluation'] = {'passed': passed, 'step_height_m': sensing.HEIGHTS_M[args.level],
                                      'attempts': attempts, 'successes': successes, 'falls': falls,
                                      'out_of_bounds': escapes, 'max_progress_x_m': max_x,
                                      'min_body_clearance_m': min_clearance, 'max_effort_ratio': max_effort}
            metadata['stair_climbing_verified'] = passed
        metadata.update(status='completed', stair_stats=base.stair_stats, completed_episodes=base.completed,
                        finished_at=datetime.now().astimezone().isoformat())
        record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('STAIRS_RUN_COMPLETE ' + json.dumps(metadata), flush=True)
        if args.mode == 'evaluate' and not metadata['evaluation']['passed']:
            raise RuntimeError('Stair climbing evaluation failed; training completion is not success')
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import sys
        import traceback
        traceback.print_exc(file=sys.__stderr__)
        raise
    finally:
        app.close()
