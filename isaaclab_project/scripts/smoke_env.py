"""Run a finite end-to-end test of the custom spiderbot environment."""

import argparse
from pathlib import Path

from environment_cli import add_environment_args, apply_environment_cfg
from contact_diagnostics_cli import (
    add_contact_diagnostics_args, apply_contact_diagnostics_cfg, validate_contact_diagnostics_args,
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the DKSH spiderbot task.")
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument('--task', default='Isaac-DKSH-Spider-Navigation-Direct-v0')
add_environment_args(parser)
add_contact_diagnostics_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps < 1 or args_cli.num_envs < 1:
    parser.error("--steps and --num_envs must be positive")
if args_cli.task.endswith("-JointAngles-v1"):
    parser.error("JointAngles-v1 is retired; retrain and use the v2 absolute-angle task")
try:
    validate_contact_diagnostics_args(args_cli)
except ValueError as error:
    parser.error(str(error))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dksh_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from joint_angle_runner_utils import validate_observation_batch


def main() -> None:
    """Create, reset, and step the environment with bounded random actions."""
    task_name = args_cli.task
    cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    apply_environment_cfg(cfg, args_cli.environment, args_cli.difficulty, args_cli.seed)
    diagnostic_path = apply_contact_diagnostics_cfg(
        cfg, args_cli, project_root=Path(__file__).resolve().parents[2], mode="check",
    )
    if diagnostic_path is not None:
        print(f"DKSH_CONTACT_DIAGNOSTICS path={diagnostic_path} pairs={cfg.contact_diagnostics_pairs}", flush=True)
    env = gym.make(task_name, cfg=cfg)
    try:
        observation, _ = env.reset()
        base_env = env.unwrapped
        policy_obs = observation["policy"]
        dof = cfg.action_space
        validate_observation_batch(observation, num_envs=args_cli.num_envs,
                                   actor_size=cfg.observation_space, critic_size=cfg.state_space)
        assert base_env.num_states == cfg.state_space
        assert base_env._robot.num_joints == dof, base_env._robot.num_joints
        assert base_env._robot.num_bodies == dof + 1, base_env._robot.num_bodies
        base_index = base_env._robot.body_names.index("base")
        expected_base_mass = torch.full_like(
            base_env._robot.data.default_mass[:, base_index], base_env._base_mass_with_lidar
        )
        torch.testing.assert_close(base_env._robot.data.default_mass[:, base_index], expected_base_mass)
        reset_count = 0
        reward_sum = 0.0
        scenario = getattr(base_env, "_scenario", None)
        initial_floor = scenario.floor_height.clone() if scenario is not None else None
        floor_displacement = 0.0
        drops_seen = 0
        falling_motion_seen = False
        for _ in range(args_cli.steps):
            actions = 0.1 * (2.0 * torch.rand((args_cli.num_envs, dof), device=env.unwrapped.device) - 1.0)
            direct_angles = getattr(cfg, "action_semantics", None) == "absolute_joint_angles_v2"
            if direct_angles:
                from dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control import normalize_angles
                # Exercise motor control near a static reset pose during this
                # simulator smoke test, without executing a reference gait.
                standing = normalize_angles(base_env._robot.data.default_joint_pos[:, base_env._joint_ids], base_env._angle_limits)
                actions = standing + 0.2 * actions
            observation, rewards, terminated, truncated, _ = env.step(actions)
            validate_observation_batch(observation, num_envs=args_cli.num_envs,
                                       actor_size=cfg.observation_space, critic_size=cfg.state_space)
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(base_env._robot.data.root_state_w).all()
            assert torch.isfinite(base_env._robot.data.joint_pos).all()
            assert terminated.shape == (args_cli.num_envs,)
            assert truncated.shape == (args_cli.num_envs,)
            if direct_angles:
                targets = base_env.joint_angle_targets_rad
                assert torch.isfinite(targets).all()
                assert (targets >= base_env._angle_limits[..., 0] - 1e-6).all()
                assert (targets <= base_env._angle_limits[..., 1] + 1e-6).all()
                assert (base_env.step_metrics["target_rate_rad_s"] <= cfg.target_rate_limit + 1e-5).all()
                assert base_env._contact_state()[0].shape == (args_cli.num_envs, dof // 3)
            reset_count += torch.count_nonzero(terminated | truncated).item()
            reward_sum += rewards.sum().item()
            if scenario is not None:
                assert scenario.observations().shape == (args_cli.num_envs, 48)
                floor_displacement = max(
                    floor_displacement, (scenario.floor_height - initial_floor).abs().max().item()
                )
                if scenario.platform is not None:
                    assert torch.isfinite(scenario.platform.data.root_state_w).all()
                drops_seen = max(drops_seen, scenario.drop_count.sum().item())
                for debris in scenario.debris:
                    assert torch.isfinite(debris.data.root_state_w).all()
                    falling_motion_seen |= bool((debris.data.root_lin_vel_w[:, 2] < -0.5).any())
        if scenario is not None:
            if scenario.platform is not None and args_cli.steps >= 100:
                assert floor_displacement > 0.001, "Floor did not move"
            if scenario.debris and args_cli.steps >= 100:
                assert drops_seen > 0, "No falling debris was released"
                assert falling_motion_seen, "Released debris did not fall under gravity"
            # A subset reset must not disturb neighboring hazard timers or bodies.
            if args_cli.num_envs > 1:
                other_times = scenario.time[1:].clone()
                other_debris = [d.data.root_state_w[1:].clone() for d in scenario.debris]
                base_env._reset_idx(torch.tensor([0], device=base_env.device))
                assert torch.equal(scenario.time[1:], other_times)
                for debris, previous in zip(scenario.debris, other_debris):
                    assert torch.equal(debris.data.root_state_w[1:], previous)
                assert scenario.drop_count[0] == 0
        print(
            f"DKSH_ISAACLAB_SMOKE_PASS environment={args_cli.environment} difficulty={args_cli.difficulty} "
            f"envs={args_cli.num_envs} steps={args_cli.steps} drops={drops_seen} "
            f"obs={tuple(observation['policy'].shape)} joints={base_env._robot.num_joints} "
            f"bodies={base_env._robot.num_bodies} base_mass={base_env._base_mass_with_lidar:.3f}kg "
            f"resets={reset_count} "
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
