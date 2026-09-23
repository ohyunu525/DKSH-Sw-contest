"""Train/resume the verified CAD6 MG90S task, with explicit completion records."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from mg90s_checkpoint import validate_checkpoint
from cad6_asset_provenance import cad6_asset_hashes
from isaaclab.app import AppLauncher

TASKS = (
    'Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0',
    'Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0',
    'Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0',
    'Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v1',
    'Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v2',
    'Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v3',
)


def checkpoint_observations(task: str) -> int:
    """Select the observation contract before opening Isaac Sim."""
    return 69 if task.endswith(('-v2', '-v3')) and '-Velocity-Direct-' in task else 68


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--max_iterations', type=int, default=2000, help='Additional iterations when resuming')
parser.add_argument('--checkpoint', default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task', choices=TASKS, default=TASKS[0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.max_iterations < 1:
    parser.error('Environment and iteration counts must be positive')
checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
if checkpoint:
    validate_checkpoint(checkpoint, observations=checkpoint_observations(args.task), task=args.task)
asset_hashes = cad6_asset_hashes() if args.task.endswith('-Velocity-Direct-v3') else None
app = AppLauncher(args).app

import gymnasium as gym
import torch
import dksh_isaaclab
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    agent.seed = args.seed
    agent.device = cfg.sim.device
    agent.max_iterations = args.max_iterations
    agent.resume = checkpoint is not None
    if checkpoint:
        agent.load_checkpoint = str(checkpoint)
    log = ROOT / 'logs/rsl_rl' / agent.experiment_name / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f_cad6_4v8')
    log.mkdir(parents=True, exist_ok=False)
    metadata = {'task': args.task, 'legs': 6, 'actions': int(cfg.action_space),
                'observations': int(cfg.observation_space), 'seed': args.seed,
                'environments': args.num_envs, 'additional_iterations': args.max_iterations,
                'started_at': datetime.now().astimezone().isoformat(), 'status': 'initializing',
                'source_checkpoint': str(checkpoint) if checkpoint else None,
                'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None}
    if asset_hashes is not None:
        metadata['asset_sha256'] = asset_hashes
    metadata_path = log / 'run_metadata.json'
    env = None
    try:
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
        base = env.unwrapped
        assert base._robot.num_joints == 18 and base._robot.num_bodies == 19
        metadata['usd'] = str(Path(cfg.robot.spawn.usd_path).relative_to(ROOT))
        metadata['mass_kg'] = float(base._robot.root_physx_view.get_masses()[0].sum())
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log), device=agent.device)
        runner.add_git_repo_to_log(__file__)
        if checkpoint:
            runner.load(str(checkpoint))
            runner.current_learning_iteration += 1
        metadata['start_iteration'] = runner.current_learning_iteration
        metadata['status'] = 'running'
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        dump_yaml(str(log / 'params/env.yaml'), cfg)
        dump_yaml(str(log / 'params/agent.yaml'), agent)
        snapshot = log / 'source_snapshot'
        snapshot.mkdir()
        task_dir = ROOT / 'isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation'
        for source in [*task_dir.glob('mg90s*.py'), task_dir / 'cad6_wave_gait.py',
                       Path(__file__), Path(__file__).with_name('mg90s_checkpoint.py'),
                       Path(__file__).with_name('cad6_asset_provenance.py'), ROOT / 'run_mg90s.ps1']:
            shutil.copy2(source, snapshot / source.name)
        print('CAD6_TRAINING_STARTED ' + json.dumps({'log': str(log), **metadata}), flush=True)
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
        final = log / f'model_{runner.current_learning_iteration}.pt'
        assert final.is_file()
        metadata.update(status='completed', final_checkpoint=final.name,
                        final_iteration=runner.current_learning_iteration,
                        final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                        completed_episodes=base.completed.copy(),
                        finished_at=datetime.now().astimezone().isoformat())
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('MG90S_CAD6_TRAINING_COMPLETE ' + str(final), flush=True)
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
