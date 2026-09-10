"""Run a finite end-to-end test of the custom spiderbot environment."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the DKSH spiderbot task.")
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dksh_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """Create, reset, and step the environment with bounded random actions."""
    task_name = "Isaac-DKSH-Spider-Navigation-Direct-v0"
    cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(task_name, cfg=cfg)
    try:
        observation, _ = env.reset()
        base_env = env.unwrapped
        policy_obs = observation["policy"]
        assert policy_obs.shape == (args_cli.num_envs, 84), policy_obs.shape
        assert base_env._robot.num_joints == 24, base_env._robot.num_joints
        assert base_env._robot.num_bodies == 25, base_env._robot.num_bodies
        reset_count = 0
        reward_sum = 0.0
        for _ in range(args_cli.steps):
            actions = 0.1 * (2.0 * torch.rand((args_cli.num_envs, 24), device=env.unwrapped.device) - 1.0)
            observation, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(observation["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(base_env._robot.data.root_state_w).all()
            assert torch.isfinite(base_env._robot.data.joint_pos).all()
            assert terminated.shape == (args_cli.num_envs,)
            assert truncated.shape == (args_cli.num_envs,)
            reset_count += torch.count_nonzero(terminated | truncated).item()
            reward_sum += rewards.sum().item()
        print(
            f"DKSH_ISAACLAB_SMOKE_PASS envs={args_cli.num_envs} steps={args_cli.steps} "
            f"obs={tuple(observation['policy'].shape)} joints={base_env._robot.num_joints} "
            f"bodies={base_env._robot.num_bodies} resets={reset_count} "
            f"mean_reward={reward_sum / (args_cli.num_envs * args_cli.steps):.6f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
