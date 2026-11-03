"""Search bounded CAD6 tripod profiles using parallel flat-ground rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

import torch


def _profile_contract():
    # The Isaac Lab task package requires the simulator to be launched first.
    from dksh_isaaclab.tasks.direct.spider_navigation.onshape_cad6_gait import (
        DEFAULT_GAIT_PROFILE,
        GAIT_PROFILE_BOUNDS,
        GAIT_PROFILE_KEYS,
    )

    return GAIT_PROFILE_KEYS, GAIT_PROFILE_BOUNDS, DEFAULT_GAIT_PROFILE


def validate_profile(profile: Mapping[str, float]) -> tuple[float, ...]:
    """Read a saved profile in the environment's column order."""
    keys, bounds, _ = _profile_contract()
    if not isinstance(profile, Mapping) or set(profile) != set(keys):
        raise ValueError(f"A gait profile must contain exactly {', '.join(keys)}")
    values = []
    for key, (low, high) in zip(keys, bounds):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not low <= value <= high:
            raise ValueError(f"Gait profile {key} must be between {low} and {high}, got {value!r}")
        values.append(float(value))
    return tuple(values)


def _rollout(env, profiles: torch.Tensor, *, steps: int, seed: int) -> torch.Tensor:
    """Return failure, failure time, signed progress speed, and stance slip per lane."""
    env.set_gait_profiles(profiles.to(env.device))
    env.reset(seed=seed)
    actions = torch.zeros((env.num_envs, 18), device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(active)
    elapsed = torch.zeros(env.num_envs, device=env.device)
    progress = torch.zeros_like(elapsed)
    slip = torch.zeros_like(elapsed)

    for step in range(steps):
        _, reward, terminated, truncated, _ = env.step(actions)
        # Isaac Lab resets finished lanes in step(). These metrics describe the
        # frame before reset, so count the terminal frame and mask later ones.
        metrics = env.step_metrics
        speed = metrics["step_progress_speed_mps"]
        stance_slip = metrics["mean_stance_slip_speed_mps"]
        finite = torch.isfinite(speed) & torch.isfinite(stance_slip) & torch.isfinite(reward)
        reached_goal = metrics["goal"] > 0
        failure = (
            ~finite
            | (metrics["strict_contact_failure"] > 0)
            | (terminated & ~reached_goal)
        )
        if step < steps - 1:
            failure |= truncated & ~reached_goal
        measured = active & finite
        progress += torch.where(measured, speed, 0.0)
        slip += torch.where(measured, stance_slip, 0.0)
        elapsed += active
        failed |= active & failure
        active &= ~(terminated | truncated | failure)
        if not active.any():
            break

    elapsed = elapsed.clamp_min(1)
    return torch.stack((failed.float(), elapsed * failed, progress / elapsed, slip / elapsed), dim=-1).cpu()


def _evaluate(env, profiles: torch.Tensor, *, steps: int, seed: int, lane_offset: int) -> torch.Tensor:
    """Reuse lanes across batches and rotate assignments across resets."""
    order = torch.arange(len(profiles)).roll(lane_offset % len(profiles))
    metrics = torch.empty((len(profiles), 4))
    for indices in order.split(env.num_envs):
        batch = profiles[indices]
        if len(batch) < env.num_envs:
            batch = torch.cat((batch, batch[-1:].expand(env.num_envs - len(batch), -1)))
        result = _rollout(env, batch, steps=steps, seed=seed)
        metrics[indices] = result[:len(indices)]
    return metrics


def _rank(metrics: torch.Tensor, target_speed: float) -> list[int]:
    """Rank stable gait first, then speed error plus stance slip."""
    # Shape: [reset seeds, candidates, metrics]. Delay failures to guide the
    # population even when no candidate can complete its first rollout.
    failures = metrics[..., 0].sum(dim=0)
    failure_time = metrics[..., 1].sum(dim=0)
    cost = ((metrics[..., 2] - target_speed).abs() + 0.5 * metrics[..., 3]).mean(dim=0)
    return sorted(
        range(len(cost)),
        key=lambda index: (failures[index].item(), -failure_time[index].item(), cost[index].item()),
    )


@torch.inference_mode()
def search(
    env,
    *,
    seed: int,
    population: int,
    generations: int,
    steps: int,
    validation_seeds: Sequence[int],
) -> dict[str, float]:
    """Evolve tripod parameters and return an elite that walks on held-out resets."""
    if population < 4 or generations < 1 or steps < 1:
        raise ValueError("GA search needs population >= 4, generations >= 1, and steps >= 1")
    if (not validation_seeds or len(set(validation_seeds)) != len(validation_seeds)
            or seed in validation_seeds):
        raise ValueError("Use distinct validation seeds that are separate from the search seed")
    if any(not 0 <= value < 2**32 for value in (seed, *validation_seeds)):
        raise ValueError("Search and validation seeds must be in [0, 2**32)")
    if env.cfg.environment_preset != "flat" or not env.cfg.evaluation_mode:
        raise ValueError("GA search requires the flat environment in evaluation mode")
    target_speed = env.cfg.target_speed
    if not math.isfinite(target_speed) or target_speed <= 0:
        raise ValueError("GA search requires a positive target speed")

    keys, bounds, default = _profile_contract()
    minimum_steps = math.ceil(3 * bounds[0][1] / env.step_dt)
    if not minimum_steps <= steps < env.max_episode_length:
        raise ValueError(
            f"GA search needs {minimum_steps}–{env.max_episode_length - 1} steps "
            "to cover at least three gait cycles within one episode"
        )
    lower, upper = torch.tensor(bounds).T
    span = upper - lower
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = torch.rand((population, len(keys)), generator=generator)
    candidates[0] = (torch.tensor(default) - lower) / span
    elite_count = max(1, population // 8)
    archive = {}

    # Flat evaluation has a fixed heading and v12 has no joint reset noise
    # by default. Give the held-out resets small, reproducible differences.
    reset_noise = env.cfg.joint_reset_noise
    env.cfg.joint_reset_noise = max(reset_noise, 0.005)
    try:
        for generation in range(generations):
            profiles = lower + span * candidates
            metrics = _evaluate(env, profiles, steps=steps, seed=seed, lane_offset=generation)
            ranked = _rank(metrics.unsqueeze(0), target_speed)
            failed, _, speed, slip = metrics[ranked[0]].tolist()
            print(
                f"GA generation {generation + 1}/{generations} | "
                f"failed={bool(failed)} | progress={speed:.4f} m/s | slip={slip:.4f} m/s",
                flush=True,
            )
            for row in profiles[ranked[:elite_count]].tolist():
                archive[tuple(row)] = None
            if generation == generations - 1:
                break
            elites = candidates[ranked[:elite_count]]
            parents = candidates[ranked[:population // 2]]
            child_count = population - elite_count - 1
            first, second = parents[torch.randint(len(parents), (2, child_count), generator=generator)]
            blend = torch.rand(first.shape, generator=generator)
            mutation = 0.10 * torch.randn(first.shape, generator=generator)
            children = (torch.lerp(first, second, blend) + mutation).clamp(0, 1)
            immigrant = torch.rand((1, len(keys)), generator=generator)
            candidates = torch.cat((elites, children, immigrant))

        finalists = torch.tensor(list(archive))
        held_out = torch.stack([
            _evaluate(env, finalists, steps=steps, seed=value, lane_offset=offset)
            for offset, value in enumerate(validation_seeds)
        ])
    finally:
        env.cfg.joint_reset_noise = reset_noise

    ranked = _rank(held_out, target_speed)
    minimum_speed = 0.20 * target_speed
    for index in ranked:
        trials = held_out[:, index]
        if not trials[:, 0].any() and (trials[:, 2] >= minimum_speed).all():
            print(f"GA validation passed | worst progress={trials[:, 2].min().item():.4f} m/s", flush=True)
            # The environment uses float32. Clamp to the exact saved-profile
            # bounds so rounding at a limit can never invalidate the JSON.
            return {
                key: min(high, max(low, value))
                for key, value, (low, high) in zip(keys, finalists[index].tolist(), bounds)
            }
    best = held_out[:, ranked[0]]
    raise RuntimeError(
        f"No feasible GA gait on {len(validation_seeds)} validation resets. "
        f"Best ranked gait had {int(best[:, 0].sum())} failures and "
        f"{best[:, 2].min().item():.4f} m/s worst progress speed "
        f"(required {minimum_speed:.4f} m/s on every reset)."
    )
