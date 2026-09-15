"""Preview a selectable course without requiring a trained checkpoint."""

import argparse
import time

from environment_cli import add_environment_args, apply_environment_cfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DKSH-Spider-Navigation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000, help="Finite preview length (default: 60 simulated seconds).")
add_environment_args(parser)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("--steps and --num_envs must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch

import dksh_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_environment_cfg(cfg, args.environment, args.difficulty, args.seed)
    env = gym.make(args.task, cfg=cfg)
    try:
        env.reset()
        actions = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.unwrapped.device)
        print(f"Preview: {args.environment}, difficulty={args.difficulty}. Robot holds its initial joint targets.")
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                start = time.perf_counter()
                env.step(actions)
                if not args.headless:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        print("DKSH_ISAACLAB_PREVIEW_PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
