"""Shared environment options and compatibility with the bundled Isaac Lab runners.

This module deliberately imports no Isaac Sim modules, so invalid input can be
reported before the simulator starts. Registry factories are evaluated lazily by
Isaac Lab after AppLauncher has initialized the runtime.
"""

from __future__ import annotations

import argparse
import importlib
import math


ENVIRONMENTS = ("flat", "narrow", "vibrating", "falling_debris", "rough", "mixed")


def difficulty_value(value: str) -> float:
    difficulty = float(value)
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise argparse.ArgumentTypeError("difficulty must be a finite number between 0 and 1")
    return difficulty


def seed_value(value: str) -> int:
    seed = int(value)
    if not 0 <= seed <= 2**31 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2147483647")
    return seed


def add_environment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="flat", help="Training environment preset.")
    parser.add_argument("--difficulty", type=difficulty_value, default=0.5, help="Environment difficulty, from 0 to 1.")
    parser.add_argument("--seed", type=seed_value, default=None, help="Seed for reproducible environment randomization.")


def experiment_for_environment(experiment_name: str, environment: str) -> str:
    if environment != "flat" and not experiment_name.endswith("_obstacles"):
        return f"{experiment_name}_obstacles"
    return experiment_name


def apply_environment_cfg(cfg, environment: str, difficulty: float, seed: int | None = None):
    cfg.environment_preset = environment
    cfg.environment_difficulty = difficulty
    if seed is not None:
        cfg.seed = seed
    return cfg


def _instantiate(entry_point):
    if isinstance(entry_point, str):
        module_name, attribute_name = entry_point.split(":")
        entry_point = getattr(importlib.import_module(module_name), attribute_name)
    return entry_point()


def configure_runner(argv: list[str], *, training: bool) -> list[str]:
    """Consume DKSH options, register lazy config factories, and retain runner flags.

    Isaac Lab 2.1's trainer supports Hydra but its player does not. Both runners
    also parse --experiment_name without applying it. These factories give both
    paths the same effective settings without modifying the Isaac Lab checkout.
    """
    import gymnasium as gym

    parser = argparse.ArgumentParser(add_help=False)
    add_environment_args(parser)
    parser.add_argument("--task", default="Isaac-DKSH-Spider-Navigation-Direct-v0")
    parser.add_argument("--experiment_name", default=None)

    # Accept the same env.* spelling as the trainer on the player as well.
    aliases = {
        "env.environment_preset": "--environment",
        "env.environment_difficulty": "--difficulty",
        "env.seed": "--seed",
        "agent.experiment_name": "--experiment_name",
    }
    normalized = []
    for arg in argv:
        name, separator, value = arg.partition("=")
        normalized.append(f"{aliases[name]}={value}" if separator and name in aliases else arg)
    options, remaining = parser.parse_known_args(normalized)
    if options.environment != "flat" and "--use_pretrained_checkpoint" in remaining:
        parser.error("non-flat environments require an obstacle checkpoint; omit --use_pretrained_checkpoint")

    spec = gym.spec(options.task)
    env_entry_point = spec.kwargs["env_cfg_entry_point"]
    agent_entry_point = spec.kwargs["rsl_rl_cfg_entry_point"]

    def environment_factory():
        return apply_environment_cfg(
            _instantiate(env_entry_point), options.environment, options.difficulty, options.seed
        )

    def agent_factory():
        cfg = _instantiate(agent_entry_point)
        cfg.experiment_name = experiment_for_environment(
            options.experiment_name or cfg.experiment_name, options.environment
        )
        if options.seed is not None:
            cfg.seed = options.seed
        return cfg

    spec.kwargs["env_cfg_entry_point"] = environment_factory
    spec.kwargs["rsl_rl_cfg_entry_point"] = agent_factory
    remaining.append(f"--task={options.task}")
    # The factories above already apply the requested environment, difficulty and
    # seed.  Do not re-inject them as Hydra `env.*` overrides: Isaac Lab 2.1's
    # trainer consumes the root config before our factory fields exist, which
    # rejects those overrides on Windows.
    return remaining
