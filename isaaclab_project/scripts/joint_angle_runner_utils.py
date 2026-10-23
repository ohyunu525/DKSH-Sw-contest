"""CPU-testable exploration constraints and completed-episode evaluation."""

from __future__ import annotations

import math

import torch


def effective_seed(mode, seed):
    return seed if seed is not None else (73 if mode == "evaluate" else 42)


def configure_evaluation(cfg, environment):
    """Evaluate a meaningful fixed course, independent of short training goals."""
    cfg.evaluation_mode = True
    cfg.goal_radius = 0.08
    distance = float(cfg.evaluation_goal_distance)
    if not math.isfinite(distance) or distance <= cfg.goal_radius:
        raise ValueError("Evaluation goal distance must exceed the goal radius")
    cfg.goal_min_distance = cfg.goal_max_distance = distance
    cfg.episode_length_s = float(cfg.evaluation_episode_length_s)
    if environment != "flat":
        # Preserve the real terrain course rather than substituting a short goal.
        cfg.episode_length_s = max(240.0, cfg.episode_length_s)
    return cfg


def validate_observation_batch(observation, *, num_envs, actor_size, critic_size):
    """Support actor-only v2 and the separate legacy asymmetric critic contract."""
    actor = observation["policy"]
    if actor.shape != (num_envs, actor_size) or not torch.isfinite(actor).all():
        raise AssertionError("Invalid actor observation batch")
    if critic_size:
        critic = observation["critic"]
        if critic.shape != (num_envs, critic_size) or not torch.isfinite(critic).all():
            raise AssertionError("Invalid critic observation batch")
        torch.testing.assert_close(critic[:, :actor_size], actor)


def validate_exploration_bounds(initial, minimum, maximum):
    """Exploration is an optimizer setting, independent of physical rate limits."""
    values = (initial, minimum, maximum)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Exploration standard deviations must be finite and positive")
    if not minimum <= initial <= maximum:
        raise ValueError("Exploration standard deviations must satisfy minimum <= initial <= maximum")
    return tuple(float(value) for value in values)


class PolicyStdConstraint:
    """Project learned Normal standard deviation after actual optimizer updates.

    The policy samples and evaluates log probabilities with the same parameter;
    samples themselves are never post-processed to impose exploration bounds.
    Both RSL-RL's direct ``std`` and logarithmic ``log_std`` are supported.
    """

    def __init__(self, policy, optimizer, *, initial, minimum, maximum, initialize=False,
                 reset_optimizer_state=False):
        self.initial, self.minimum, self.maximum = validate_exploration_bounds(initial, minimum, maximum)
        self.parameter = getattr(policy, "std", None)
        self.logarithmic = False
        if not isinstance(self.parameter, torch.nn.Parameter):
            self.parameter = getattr(policy, "log_std", None)
            self.logarithmic = True
        if not isinstance(self.parameter, torch.nn.Parameter):
            raise TypeError("Policy must expose a trainable std or log_std parameter")
        if not hasattr(optimizer, "register_step_post_hook"):
            raise TypeError("Optimizer must support register_step_post_hook to constrain exploration")
        if reset_optimizer_state and not initialize:
            raise ValueError("Resetting exploration optimizer state requires initializing the std parameter")
        self.before_values = self.values()
        self.optimizer_state_cleared = False
        if initialize:
            with torch.no_grad():
                self.parameter.fill_(math.log(self.initial) if self.logarithmic else self.initial)
            if reset_optimizer_state:
                self.optimizer_state_cleared = optimizer.state.pop(self.parameter, None) is not None
        self.clamp()
        self.after_values = self.values()
        self.handle = optimizer.register_step_post_hook(self._after_step)

    def _after_step(self, optimizer, args, kwargs):
        self.clamp()

    def clamp(self):
        with torch.no_grad():
            if not torch.isfinite(self.parameter).all():
                raise ValueError("Policy exploration standard deviation became non-finite")
            lower, upper = self.minimum, self.maximum
            if self.logarithmic:
                lower, upper = math.log(lower), math.log(upper)
            self.parameter.clamp_(lower, upper)

    def values(self):
        value = self.parameter.detach()
        return (value.exp() if self.logarithmic else value).cpu().tolist()

    def close(self):
        self.handle.remove()


class EvaluationAccumulator:
    """Count goals and sustained progress only after an actual episode ends."""

    METRICS = (
        "mechanical_power_w", "progress_m", "travel_distance_m", "forward_speed_mps",
        "body_contact", "joint_limit_cost", "target_rate_rad_s", "target_rate_saturation",
        "torque_saturation", "tracking_error_rad", "feet_contact", "goal", "fallen", "out_of_bounds",
    )
    OPTIONAL_METRICS = ("step_progress_speed_mps", "target_excursion_rad", "joint_excursion_rad",
                        "joint_velocity_rms_rad_s", "mean_foot_clearance_m")

    def __init__(self, num_envs):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.steps = 0
        self.duration = torch.zeros(num_envs, dtype=torch.float64)
        self.progress = torch.zeros_like(self.duration)
        self.travel = torch.zeros_like(self.duration)
        self.energy = torch.zeros_like(self.duration)
        self.completed = []
        self.totals = {name: 0.0 for name in self.METRICS}
        self.max_target_rate = 0.0
        self.optional_totals = {}
        self.optional_samples = {}

    def update(self, metrics, dones, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Evaluation dt must be finite and positive")
        values = {}
        for name in self.METRICS:
            tensor = torch.as_tensor(metrics[name]).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if tensor.shape != (self.num_envs,) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid pre-reset metric {name}")
            values[name] = tensor
            self.totals[name] += float(tensor.sum())
        for name in self.OPTIONAL_METRICS:
            if name not in metrics:
                continue
            tensor = torch.as_tensor(metrics[name]).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if tensor.shape != (self.num_envs,) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid pre-reset metric {name}")
            self.optional_totals[name] = self.optional_totals.get(name, 0.0) + float(tensor.sum())
            self.optional_samples[name] = self.optional_samples.get(name, 0) + self.num_envs
        done = torch.as_tensor(dones, dtype=torch.bool).reshape(-1).cpu()
        if done.shape != (self.num_envs,):
            raise ValueError("Episode done flags must contain one value per environment")
        self.duration += dt
        self.progress += values["progress_m"]
        self.travel += values["travel_distance_m"]
        self.energy += values["mechanical_power_w"] * dt
        self.max_target_rate = max(self.max_target_rate, float(values["target_rate_rad_s"].max()))
        self.steps += 1
        for index in done.nonzero(as_tuple=False).flatten().tolist():
            # Failures take precedence; a terminal observation cannot manufacture a goal.
            outcome = ("fallen" if values["fallen"][index] > 0 else
                       "out_of_bounds" if values["out_of_bounds"][index] > 0 else
                       "goal" if values["goal"][index] > 0 else "time_out")
            duration = float(self.duration[index])
            progress = float(self.progress[index])
            self.completed.append({
                "environment_index": index, "outcome": outcome, "duration_s": duration,
                "signed_progress_m": progress, "travel_distance_m": float(self.travel[index]),
                "mechanical_energy_j": float(self.energy[index]),
                "sustained_progress_speed_mps": progress / duration,
            })
            self.duration[index] = self.progress[index] = self.travel[index] = self.energy[index] = 0

    def report(self, *, target_speed, minimum_speed_fraction, dt):
        if self.steps < 1:
            raise ValueError("Evaluation requires at least one control step")
        counts = {name: 0 for name in ("episodes", "goal", "fallen", "out_of_bounds", "time_out")}
        counts["episodes"] = len(self.completed)
        for episode in self.completed:
            counts[episode["outcome"]] += 1
        completed_duration = sum(episode["duration_s"] for episode in self.completed)
        completed_progress = sum(episode["signed_progress_m"] for episode in self.completed)
        sustained = completed_progress / completed_duration if completed_duration > 0 else 0.0
        success_rate = counts["goal"] / max(1, counts["episodes"])
        successful_environments = {episode["environment_index"] for episode in self.completed if episode["outcome"] == "goal"}
        minimum_speed = target_speed * minimum_speed_fraction
        samples = self.steps * self.num_envs
        energy = self.totals["mechanical_power_w"] * dt
        net_progress = self.totals["progress_m"]
        passed = (counts["goal"] >= self.num_envs and len(successful_environments) == self.num_envs and success_rate >= 0.8
                  and counts["fallen"] == 0 and counts["out_of_bounds"] == 0
                  and sustained >= minimum_speed)
        reasons = []
        if counts["goal"] < self.num_envs:
            reasons.append("fewer completed successful episodes than environments")
        if len(successful_environments) != self.num_envs:
            reasons.append("not every environment completed a successful episode")
        if success_rate < 0.8:
            reasons.append("completed-episode success rate below 0.8")
        if counts["fallen"] or counts["out_of_bounds"]:
            reasons.append("falls or out-of-bounds episodes occurred")
        if sustained < minimum_speed:
            reasons.append("completed-episode sustained progress speed is below the minimum")
        return {
            "episodes": counts, "success_rate": success_rate,
            "successful_environment_count": len(successful_environments),
            "completed_episode_records": self.completed,
            "incomplete_episode_count": int((self.duration > 0).sum()),
            "completed_progress_m": completed_progress,
            "completed_duration_s": completed_duration,
            "sustained_progress_speed_mps": sustained,
            "minimum_sustained_progress_speed_mps": minimum_speed,
            "forward_speed_mps": self.totals["forward_speed_mps"] / samples,
            "signed_progress_m": net_progress, "travel_distance_m": self.totals["travel_distance_m"],
            "mechanical_energy_j": energy,
            "energy_per_progress_j_m": energy / net_progress if net_progress > 1e-6 else None,
            "body_contact_fraction": self.totals["body_contact"] / samples,
            "joint_limit_cost": self.totals["joint_limit_cost"] / samples,
            "max_target_rate_rad_s": self.max_target_rate,
            "target_rate_saturation_fraction": self.totals["target_rate_saturation"] / samples,
            "torque_saturation_fraction": self.totals["torque_saturation"] / samples,
            "tracking_error_rad": self.totals["tracking_error_rad"] / samples,
            "mean_support_feet": self.totals["feet_contact"] / samples,
            **{name: total / self.optional_samples[name] for name, total in self.optional_totals.items()},
            "locomotion_pass": passed, "failure_reasons": reasons,
        }
