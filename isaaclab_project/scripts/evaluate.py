"""Load a DKSH RSL-RL checkpoint and run a finite headless evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained DKSH spiderbot policy.")
parser.add_argument("--task", default="Isaac-DKSH-Spider-Navigation-Direct-v0")
parser.add_argument("--checkpoint", default="", help="Checkpoint path; defaults to the newest local model_*.pt.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import dksh_isaaclab  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _resolve_checkpoint() -> Path:
    if args_cli.checkpoint:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    log_root = Path.cwd() / "logs" / "rsl_rl" / "dksh_spider_navigation"
    candidates = list(log_root.glob("**/model_*.pt"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    bundled = Path(__file__).resolve().parents[1] / "checkpoints" / "balance_baseline.pt"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"No checkpoint found below {log_root}, and the bundled baseline is missing.")


def main() -> None:
    checkpoint = _resolve_checkpoint()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
        observation, _ = wrapped_env.get_observations()
        base_env = wrapped_env.unwrapped
        total_reward = torch.zeros(args_cli.num_envs, device=wrapped_env.unwrapped.device)
        reset_count = 0
        goal_count = 0
        fall_count = 0
        timeout_count = 0
        planar_speed_sum = 0.0
        goal_distance_sum = 0.0

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(observation)
                observation, rewards, dones, extras = wrapped_env.step(actions)
                if not torch.isfinite(observation).all() or not torch.isfinite(rewards).all():
                    raise RuntimeError("Non-finite policy output or reward encountered during evaluation.")
                total_reward += rewards
                step_reset_count = torch.count_nonzero(dones).item()
                reset_count += step_reset_count
                planar_speed_sum += (
                    torch.linalg.vector_norm(base_env._robot.data.root_lin_vel_w[:, :2], dim=1).mean().item()
                )
                goal_distance_sum += base_env._goal_data()[0].mean().item()
                if step_reset_count:
                    log = extras.get("log", {})
                    goal_count += int(log.get("Episode_Termination/goal", 0))
                    fall_count += int(log.get("Episode_Termination/fallen", 0))
                    timeout_count += int(log.get("Episode_Termination/time_out", 0))

        print(
            f"DKSH_ISAACLAB_EVAL_PASS checkpoint={checkpoint} envs={args_cli.num_envs} "
            f"steps={args_cli.steps} mean_reward={total_reward.mean().item():.6f} resets={reset_count} "
            f"goals={goal_count} falls={fall_count} timeouts={timeout_count} "
            f"mean_speed={planar_speed_sum / args_cli.steps:.6f} "
            f"mean_goal_distance={goal_distance_sum / args_cli.steps:.6f}",
            flush=True,
        )
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
