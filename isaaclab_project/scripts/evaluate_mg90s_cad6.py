"""Evaluate a completed six-leg MG90S policy with direct locomotion metrics."""
import argparse
import json
from pathlib import Path

from mg90s_checkpoint import validate_checkpoint
from isaaclab.app import AppLauncher

TASKS = ('Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0', 'Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0')
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=3000)
parser.add_argument('--output', default='')
parser.add_argument('--task', choices=TASKS, default=TASKS[0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1000:
    parser.error('Use at least one environment and 1,000 steps.')
checkpoint = Path(args.checkpoint).resolve()
validate_checkpoint(checkpoint, task=args.task)
app = AppLauncher(args).app

import gymnasium as gym
import torch
import dksh_isaaclab
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    agent.device = cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        base = env.unwrapped
        assert base._robot.num_joints == 18 and base._robot.num_bodies == 19
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        obs, _ = env.get_observations()
        counts = base.completed.copy()
        speed_sum = error_sum = torque_near_sum = 0.0
        min_height, max_torque = float('inf'), 0.0
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = policy(obs)
                if actions.shape != (args.num_envs, 18) or not torch.isfinite(actions).all():
                    raise RuntimeError('Policy emitted non-finite or incorrectly shaped actions')
                obs, reward, _, _ = env.step(actions)
                if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Non-finite observation or reward')
                velocity = base._robot.data.root_lin_vel_b[:, 0]
                speed_sum += float(velocity.mean())
                error_sum += float((velocity - base._speed).abs().mean())
                min_height = min(min_height, float(base._robot.data.root_pos_w[:, 2].min()))
                torque = base._robot.data.applied_torque.abs()
                max_torque = max(max_torque, float(torque.max()))
                torque_near_sum += float((torque > 0.70 * 0.1765197).float().mean())
        result = {
            'task': args.task, 'checkpoint': str(checkpoint), 'steps': args.steps, 'envs': args.num_envs,
            'mean_forward_speed_mps': speed_sum / args.steps,
            'mean_speed_error_mps': error_sum / args.steps,
            'min_base_height_m': min_height, 'max_abs_torque_nm': max_torque,
            'near_torque_cap_fraction': torque_near_sum / args.steps,
            'episodes': base.completed['episodes'] - counts['episodes'],
            'falls': base.completed['fallen'] - counts['fallen'],
            'out_of_bounds': base.completed['out_of_bounds'] - counts['out_of_bounds'],
            'timeouts': base.completed['time_out'] - counts['time_out'],
        }
        result['passed'] = (result['falls'] == 0 and result['out_of_bounds'] == 0
                            and result['mean_forward_speed_mps'] >= 0.0025
                            and result['mean_speed_error_mps'] <= 0.004
                            and result['max_abs_torque_nm'] <= 0.1323998)
        if args.output:
            path = Path(args.output).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('MG90S_CAD6_EVALUATION ' + json.dumps(result), flush=True)
        if not result['passed']:
            raise RuntimeError('CAD6 evaluation criteria not met')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
