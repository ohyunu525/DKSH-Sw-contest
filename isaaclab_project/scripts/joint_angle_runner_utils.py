"""CPU-testable exploration constraints and completed-episode evaluation."""

from __future__ import annotations

import math
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


COURSE_CONTRACT = "mixed_randomized_courses_v1"
COURSE_FAMILIES = ("flat", "rough", "stairs", "narrow")


def add_course_args(parser):
    """Direct-runner options; None defaults distinguish explicit fixed-course overrides."""
    parser.add_argument("--course_seed", type=int, default=None)
    parser.add_argument("--course_split", choices=("train", "holdout"), default=None)
    parser.add_argument("--flat_fraction", type=float, default=None)
    parser.add_argument("--curriculum_window", type=int, default=None)
    parser.add_argument("--curriculum_min_episodes", type=int, default=None)
    parser.add_argument("--curriculum_promote_threshold", type=float, default=None)
    parser.add_argument("--curriculum_demote_threshold", type=float, default=None)
    parser.add_argument("--curriculum_step", type=float, default=None)
    parser.add_argument("--freeze_curriculum", action="store_true")


def resolve_course_args(args):
    """Reject invalid or leaking course settings before launching the simulator."""
    names = ("course_seed", "course_split", "flat_fraction", "curriculum_window", "curriculum_min_episodes",
             "curriculum_promote_threshold", "curriculum_demote_threshold", "curriculum_step")
    specified = any(getattr(args, name) is not None for name in names) or args.freeze_curriculum
    if args.environment != "curriculum":
        if specified:
            raise ValueError("Course randomization and curriculum options require --environment=curriculum")
        return args
    if "MG996R-Hip-CAD6" not in args.task:
        raise ValueError("curriculum requires the MG996R-Hip-CAD6 direct-angle task and distal-support behavior")
    if args.behavior_profile not in (None, "revised_v7", "revised_v8", "revised_v9"):
        raise ValueError("curriculum requires revised_v7, revised_v8 or revised_v9 distal-support behavior")
    if args.freeze_curriculum and args.mode != "train":
        raise ValueError("freeze_curriculum is training-only; evaluation is always frozen")
    split = "train" if args.mode == "train" else "holdout"
    if args.course_split is not None and args.course_split != split:
        raise ValueError(f"{args.mode} curriculum requires --course_split={split}")
    args.course_split = split
    args.course_seed = args.seed if args.course_seed is None else args.course_seed
    if not 0 <= args.course_seed <= 2**31 - 1:
        raise ValueError("course_seed must be between 0 and 2147483647")
    defaults = {"flat_fraction": 0.25, "curriculum_window": 20, "curriculum_min_episodes": 20,
                "curriculum_promote_threshold": 0.8, "curriculum_demote_threshold": 0.3, "curriculum_step": 0.1}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not math.isfinite(args.flat_fraction) or not 0 < args.flat_fraction < 1:
        raise ValueError("flat_fraction must be finite and strictly between 0 and 1")
    if args.curriculum_window < 1 or not 1 <= args.curriculum_min_episodes <= args.curriculum_window:
        raise ValueError("curriculum_min_episodes must be between 1 and curriculum_window")
    low, high = args.curriculum_demote_threshold, args.curriculum_promote_threshold
    if not all(math.isfinite(value) for value in (low, high)) or not 0 <= low < high <= 1:
        raise ValueError("curriculum thresholds must satisfy 0 <= demote < promote <= 1")
    if not math.isfinite(args.curriculum_step) or not 0 < args.curriculum_step <= 1:
        raise ValueError("curriculum_step must be finite and in (0, 1]")
    return args


def apply_course_cfg(cfg, args):
    if args.environment != "curriculum":
        return cfg
    for name in ("course_seed", "course_split", "curriculum_window", "curriculum_min_episodes",
                 "curriculum_promote_threshold", "curriculum_demote_threshold", "curriculum_step"):
        setattr(cfg, name, getattr(args, name))
    cfg.curriculum_flat_fraction = args.flat_fraction
    cfg.curriculum_enabled = args.mode == "train" and not args.freeze_curriculum
    return cfg


def course_configuration(cfg, environment):
    if environment != "curriculum":
        return None
    names = ("course_seed", "course_split", "curriculum_enabled", "curriculum_flat_fraction", "curriculum_window",
             "curriculum_min_episodes", "curriculum_promote_threshold", "curriculum_demote_threshold", "curriculum_step")
    return {"contract": COURSE_CONTRACT, "initial_difficulty": cfg.environment_difficulty,
            "num_envs": getattr(getattr(cfg, "scene", None), "num_envs", None),
            "robot_width": getattr(cfg, "robot_width", None), "robot_height": getattr(cfg, "robot_height", None),
            **{name: getattr(cfg, name) for name in names}}


class CurriculumCheckpointState:
    """Save course state beside the exact model checkpoint, never beside a different iteration."""

    def __init__(self, runner, runtime, configuration):
        self.runner, self.runtime, self.configuration = runner, runtime, configuration
        self.original_save = runner.save
        runner.save = self.save

    @staticmethod
    def sidecar(path):
        return Path(str(path) + ".courses.json")

    def save(self, path, *args, **kwargs):
        result = self.original_save(path, *args, **kwargs)
        model = Path(path)
        payload = {"configuration": self.configuration, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                   "state": self.runtime.state_dict(), "resume_scope": "course_sequence_and_curriculum_not_physics_state"}
        sidecar = self.sidecar(model)
        temporary = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=sidecar.parent, prefix=sidecar.name + ".",
                                    suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(sidecar)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return result

    def restore(self, path):
        sidecar = self.sidecar(path)
        if not sidecar.is_file():
            return "fresh_curriculum_no_checkpoint_sidecar"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("configuration") != self.configuration:
            return "fresh_curriculum_configuration_changed"
        if payload.get("model_sha256") != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("Curriculum state sidecar does not match its model checkpoint")
        self.runtime.load_state_dict(payload["state"])
        return "restored_course_sequence_and_curriculum"

    def close(self):
        self.runner.save = self.original_save


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


def _contact_limits(max_body_contact_fraction, max_non_distal_support_fraction,
                    max_non_distal_collision_fraction):
    limits = {
        "body_contact_fraction": max_body_contact_fraction,
        "non_distal_support_fraction": max_non_distal_support_fraction,
        "non_distal_collision_fraction": max_non_distal_collision_fraction,
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in limits.values()):
        raise ValueError("Contact fraction limits must be finite values between zero and one")
    return limits


def _contact_failure_reasons(records, limits, *, require_strict_contact):
    """Check each episode so clean runs cannot dilute a bad one."""
    reasons = []
    for record in records:
        failure = record.get("strict_contact_failure_fraction", 0.0)
        if (isinstance(failure, bool) or not isinstance(failure, (int, float))
                or not math.isfinite(failure) or not 0 <= failure <= 1 or failure > 0):
            reasons.append("strict contact failure occurred or its evidence is invalid")
            break
    for name, limit in limits.items():
        missing = False
        violated = False
        for record in records:
            coverage = record.get("contact_diagnostic_coverage", {}).get(
                name, 1.0 if name in record else 0.0
            )
            if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
                    or not math.isfinite(coverage) or not 0 <= coverage <= 1.0 + 1e-9):
                missing = True
                continue
            if name not in record or coverage < 1.0 - 1e-9:
                if name == "body_contact_fraction" or require_strict_contact:
                    missing = True
            if name not in record:
                continue
            value = record[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                missing = True
            elif value > limit:
                violated = True
        if missing:
            reasons.append(f"evaluated episodes lack complete valid {name} evidence")
        if violated:
            reasons.append(f"an evaluated episode exceeds {name} limit {limit:g}")
    return reasons


class EvaluationAccumulator:
    """Count goals and sustained progress only after an actual episode ends."""

    METRICS = (
        "mechanical_power_w", "progress_m", "travel_distance_m", "forward_speed_mps",
        "body_contact", "joint_limit_cost", "target_rate_rad_s", "target_rate_saturation",
        "torque_saturation", "tracking_error_rad", "feet_contact", "goal", "fallen", "out_of_bounds",
    )
    OPTIONAL_METRICS = ("step_progress_speed_mps", "target_excursion_rad", "joint_excursion_rad",
                        "joint_velocity_rms_rad_s", "mean_foot_clearance_m",
                        "foot_slip_m2_s2",
                        "distal_support_fraction", "non_distal_support_fraction", "non_distal_collision_fraction",
                        "strict_contact_failure", "min_non_distal_clearance_m",
                        "mean_knee_ground_clearance_m", "knee_ground_clearance_violation_fraction",
                        "knees_below_body_fraction", "approach_low_knee_time_fraction")

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
        self.min_non_distal_clearance_m = None
        self.episode_diagnostics = {name: torch.zeros_like(self.duration) for name in
                                    ("body_contact_fraction", "non_distal_support_fraction",
                                     "non_distal_collision_fraction", "strict_contact_failure_fraction", "foot_slip_m2_s2")}
        self.contact_diagnostic_duration = {name: torch.zeros_like(self.duration) for name in
                                            ("non_distal_support_fraction", "non_distal_collision_fraction")}

    def update(self, metrics, dones, dt, *, course_records=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Evaluation dt must be finite and positive")
        values = {}
        for name in self.METRICS:
            tensor = torch.as_tensor(metrics[name]).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if tensor.shape != (self.num_envs,) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid pre-reset metric {name}")
            if name == "body_contact" and ((tensor < 0) | (tensor > 1)).any():
                raise ValueError(f"Invalid pre-reset contact fraction {name}")
            values[name] = tensor
            self.totals[name] += float(tensor.sum())
        for name in self.OPTIONAL_METRICS:
            if name not in metrics:
                continue
            tensor = torch.as_tensor(metrics[name]).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if tensor.shape != (self.num_envs,) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid pre-reset metric {name}")
            if (name in self.contact_diagnostic_duration or name == "strict_contact_failure") and ((tensor < 0) | (tensor > 1)).any():
                raise ValueError(f"Invalid pre-reset contact fraction {name}")
            values[name] = tensor
            if name == "min_non_distal_clearance_m":
                minimum = float(tensor.min())
                self.min_non_distal_clearance_m = (minimum if self.min_non_distal_clearance_m is None
                                                 else min(self.min_non_distal_clearance_m, minimum))
                continue
            self.optional_totals[name] = self.optional_totals.get(name, 0.0) + float(tensor.sum())
            self.optional_samples[name] = self.optional_samples.get(name, 0) + self.num_envs
        done = torch.as_tensor(dones, dtype=torch.bool).reshape(-1).cpu()
        if done.shape != (self.num_envs,):
            raise ValueError("Episode done flags must contain one value per environment")
        course_by_env = None
        if course_records is not None:
            records = list(course_records)
            course_by_env = {record["environment_index"]: record for record in records}
            if len(course_by_env) != len(records) or set(course_by_env) != set(done.nonzero(as_tuple=False).flatten().tolist()):
                raise ValueError("Completed course metadata must match each terminal environment exactly once")
        self.episode_diagnostics["body_contact_fraction"] += values["body_contact"] * dt
        if "strict_contact_failure" in values:
            self.episode_diagnostics["strict_contact_failure_fraction"] += values["strict_contact_failure"] * dt
        for name in ("non_distal_support_fraction", "non_distal_collision_fraction", "foot_slip_m2_s2"):
            if name in values:
                self.episode_diagnostics[name] += values[name] * dt
                if name in self.contact_diagnostic_duration:
                    self.contact_diagnostic_duration[name] += dt
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
            record = {
                "environment_index": index, "outcome": outcome, "duration_s": duration,
                "signed_progress_m": progress, "travel_distance_m": float(self.travel[index]),
                "mechanical_energy_j": float(self.energy[index]),
                "sustained_progress_speed_mps": progress / duration,
                **{name: float(value[index]) / duration for name, value in self.episode_diagnostics.items()},
                "contact_diagnostic_coverage": {
                    name: float(value[index]) / duration for name, value in self.contact_diagnostic_duration.items()
                },
            }
            if course_by_env is not None:
                course = course_by_env[index]
                if course.get("outcome", outcome) != outcome:
                    raise ValueError("Course snapshot and evaluation outcome disagree")
                record.update({name: course[name] for name in
                               ("family", "difficulty", "course_id", "course_seed", "split")})
            self.completed.append(record)
            for value in self.episode_diagnostics.values():
                value[index] = 0
            for value in self.contact_diagnostic_duration.values():
                value[index] = 0
            self.duration[index] = self.progress[index] = self.travel[index] = self.energy[index] = 0

    def report(self, *, target_speed, minimum_speed_fraction, dt, require_strict_contact=False,
               max_body_contact_fraction=0.0, max_non_distal_support_fraction=0.0,
               max_non_distal_collision_fraction=0.0):
        if self.steps < 1:
            raise ValueError("Evaluation requires at least one control step")
        contact_limits = _contact_limits(max_body_contact_fraction, max_non_distal_support_fraction,
                                         max_non_distal_collision_fraction)
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
        pending_contact_records = [
            {**{name: float(value[index]) / float(self.duration[index])
                for name, value in self.episode_diagnostics.items()},
             "contact_diagnostic_coverage": {
                 name: float(value[index]) / float(self.duration[index])
                 for name, value in self.contact_diagnostic_duration.items()
             }}
            for index in (self.duration > 0).nonzero(as_tuple=False).flatten().tolist()
        ]
        reasons.extend(_contact_failure_reasons(self.completed + pending_contact_records, contact_limits,
                                                require_strict_contact=require_strict_contact))
        passed = passed and not reasons
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
            **{("strict_contact_failure_fraction" if name == "strict_contact_failure" else name):
               total / self.optional_samples[name] for name, total in self.optional_totals.items()},
            **({"min_non_distal_clearance_m": self.min_non_distal_clearance_m}
               if self.min_non_distal_clearance_m is not None else {}),
            "require_strict_contact": require_strict_contact,
            **{f"max_{name}": limit for name, limit in contact_limits.items()},
            "locomotion_pass": passed, "failure_reasons": reasons,
        }


def heldout_course_report(completed, *, target_speed, minimum_speed_fraction, minimum_unique_courses=2,
                          max_non_distal_support_fraction=0.0, max_body_contact_fraction=0.0,
                          max_non_distal_collision_fraction=0.0, require_strict_contact=False):
    """Require independent held-out evidence for every family, including flat replay."""
    records = list(completed)
    contact_limits = _contact_limits(max_body_contact_fraction, max_non_distal_support_fraction,
                                     max_non_distal_collision_fraction)
    expected = set(COURSE_FAMILIES)
    if any(record.get("family") not in expected or record.get("split") != "holdout" for record in records):
        raise ValueError("Held-out evaluation requires recognized families from the holdout split")
    groups = {}
    by_family = {}
    reasons = []
    minimum_speed = target_speed * minimum_speed_fraction

    def summarize(rows):
        duration = sum(row["duration_s"] for row in rows)
        progress = sum(row["signed_progress_m"] for row in rows)
        goals = sum(row["outcome"] == "goal" for row in rows)
        falls = sum(row["outcome"] == "fallen" for row in rows)
        out_of_bounds = sum(row["outcome"] == "out_of_bounds" for row in rows)
        unique = len({row["course_id"] for row in rows})
        non_distal = (sum(row.get("non_distal_support_fraction", 0) * row["duration_s"] for row in rows)
                      / duration if duration else 0)
        body_contact = (sum(row.get("body_contact_fraction", 0) * row["duration_s"] for row in rows)
                        / duration if duration else 0)
        non_distal_collision = (sum(row.get("non_distal_collision_fraction", 0) * row["duration_s"] for row in rows)
                                / duration if duration else 0)
        contact_reasons = _contact_failure_reasons(rows, contact_limits,
                                                   require_strict_contact=require_strict_contact)
        success_rate = goals / max(1, len(rows))
        speed = progress / duration if duration else 0
        return {"episodes": len(rows), "goal": goals, "fallen": falls, "out_of_bounds": out_of_bounds,
                "unique_courses": unique, "success_rate": success_rate, "sustained_progress_speed_mps": speed,
                "non_distal_support_fraction": non_distal,
                "body_contact_fraction": body_contact, "non_distal_collision_fraction": non_distal_collision,
                "contact_failure_reasons": contact_reasons,
                "pass": bool(unique >= minimum_unique_courses and success_rate >= 0.8 and not falls
                             and not out_of_bounds and speed >= minimum_speed
                             and not contact_reasons)}

    for family in COURSE_FAMILIES:
        rows = [row for row in records if row["family"] == family]
        by_family[family] = summarize(rows)
        if not by_family[family]["pass"]:
            reasons.append(f"{family}: insufficient course coverage, success, speed or clean contact evidence")
        for difficulty in sorted({row["difficulty"] for row in rows}):
            key = f"{family}@{float(difficulty):.6g}"
            groups[key] = summarize([row for row in rows if row["difficulty"] == difficulty])
    return {"course_contract": COURSE_CONTRACT, "course_split": "holdout", "course_families": by_family,
            "course_family_difficulty_groups": groups, "minimum_unique_courses_per_family": minimum_unique_courses,
            "require_strict_contact": require_strict_contact,
            **{f"max_{name}": limit for name, limit in contact_limits.items()},
            "heldout_course_pass": not reasons, "heldout_failure_reasons": reasons}
