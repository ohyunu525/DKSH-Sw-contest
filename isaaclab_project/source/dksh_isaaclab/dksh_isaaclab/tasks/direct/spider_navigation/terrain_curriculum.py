"""Reproducible fixed-shape courses and success-based terrain difficulty.

This module uses only the standard library. All courses share one immutable
collider pool: episode resets move boxes, never resize active PhysX shapes.
Coordinates are local to an environment origin and all dimensions are metres.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import random

from .scenario_layout import BoxSpec


COURSE_CONTRACT = "mixed_randomized_courses_v1"
FAMILIES = ("flat", "rough", "stairs", "narrow")
OBSTACLE_FAMILIES = FAMILIES[1:]
DEFAULT_FLAT_FRACTION = 0.25
DEFAULT_WINDOW = 20
DEFAULT_MIN_EPISODES = 20
DEFAULT_PROMOTE_THRESHOLD = 0.8
DEFAULT_DEMOTE_THRESHOLD = 0.3
DEFAULT_DIFFICULTY_STEP = 0.1

_POOL_DEFINITIONS = (
    ("wall_left", (5.6, 0.16, 0.65), "wall"),
    ("wall_right", (5.6, 0.16, 0.65), "wall"),
    ("rough_0", (0.28, 1.6, 0.4), "rough"),
    ("rough_1", (0.28, 1.6, 0.4), "rough"),
    ("rough_2", (0.28, 1.6, 0.4), "rough"),
    ("stair_0", (0.55, 1.6, 0.4), "stairs"),
    ("stair_1", (0.55, 1.6, 0.4), "stairs"),
    ("stair_2", (2.55, 1.6, 0.4), "stairs"),
    ("ceiling", (1.2, 1.8, 0.1), "ceiling"),
)
POOL_BOX_SPECS = tuple(
    BoxSpec(name, (0.0, 0.0, -3.0 - 0.05 * index), size, kind)
    for index, (name, size, kind) in enumerate(_POOL_DEFINITIONS)
)


@dataclass(frozen=True)
class SampledCourse:
    family: str
    difficulty: float
    boxes: tuple[BoxSpec, ...]
    spawn: tuple[float, float, float]
    goal: tuple[float, float, float]
    corridor_width: float
    course_id: str
    course_seed: int
    split: str


def _unit_interval(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return float(value)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def sample_course(*, seed: int, split: str, env_id: int, episode: int,
                  family: str, difficulty: float, robot_width: float = 0.56,
                  robot_height: float = 0.24) -> SampledCourse:
    """Sample one independent course with unchanged pool shapes.

    Train/holdout random streams have separately hashed namespaces and disjoint
    signed-int64 seed domains. Sampling depends on the individual environment's
    episode counter, so another environment resetting cannot alter its course.
    Geometry remains axis aligned, including contiguous ascending stair treads.
    Unused pool boxes are parked entirely below the global ground plane.
    """
    seed = _nonnegative_integer(seed, "seed")
    env_id = _nonnegative_integer(env_id, "env_id")
    episode = _nonnegative_integer(episode, "episode")
    difficulty = _unit_interval(difficulty, "difficulty")
    if split not in ("train", "holdout"):
        raise ValueError("split must be train or holdout")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    for value, name, maximum in ((robot_width, "robot_width", 1.2), (robot_height, "robot_height", 0.5)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be finite, positive and no greater than {maximum}")

    material = json.dumps((COURSE_CONTRACT, split, seed, env_id, episode, family), separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    course_seed = int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)
    if split == "holdout":
        course_seed |= 1 << 62
    rng = random.Random(course_seed)
    boxes = {box.name: box for box in POOL_BOX_SPECS}

    def place(name, x, y, bottom=None, top=None):
        box = boxes[name]
        if (bottom is None) == (top is None):
            raise ValueError("Specify exactly one box bottom or top")
        z = bottom + box.size[2] / 2 if bottom is not None else top - box.size[2] / 2
        boxes[name] = BoxSpec(box.name, (float(x), float(y), float(z)), box.size, box.kind)

    if family == "narrow":
        # A strict standing-width margin remains, including sampled lateral poses.
        factor = 1.38 - 0.28 * difficulty + rng.uniform(-0.015, 0.015)
        corridor_width = min(1.6, max(1.08 * robot_width, robot_width * factor))
    else:
        corridor_width = rng.uniform(max(1.08 * robot_width, 1.35), 1.6)
    for name, sign in (("wall_left", 1), ("wall_right", -1)):
        place(name, 0.0, sign * (corridor_width / 2 + boxes[name].size[1] / 2), bottom=0.0)

    goal_z = 0.0
    if family == "rough":
        # Ordered obstacles have independent spacing jitter and broad safe aprons.
        for index, nominal in enumerate((-0.7, 0.0, 0.7)):
            height = robot_height * (0.06 + 0.17 * difficulty) * rng.uniform(0.8, 1.15)
            place(f"rough_{index}", nominal + rng.uniform(-0.10, 0.10), 0.0, top=height)
    elif family == "stairs":
        shift = rng.uniform(-0.10, 0.10)
        riser = robot_height * (0.025 + 0.10 * difficulty) * rng.uniform(0.8, 1.15)
        # Fixed lengths share edges; moving the group preserves continuous support.
        for index, center in enumerate((-0.575, -0.025, 1.525)):
            place(f"stair_{index}", center + shift, 0.0, top=(index + 1) * riser)
        goal_z = 3 * riser
    elif family == "narrow":
        ceiling_clearance = robot_height + (0.12 - 0.07 * difficulty) * rng.uniform(0.9, 1.1)
        place("ceiling", rng.uniform(-0.15, 0.15), 0.0, bottom=ceiling_clearance)

    # Lateral changes cannot spawn legs through walls, even at maximum difficulty.
    lateral = min(0.08, (corridor_width - robot_width) * 0.20)
    spawn = (-1.6, rng.uniform(-lateral, lateral), 0.0)
    goal = (1.6, rng.uniform(-lateral, lateral), goal_z)
    ordered = tuple(boxes[box.name] for box in POOL_BOX_SPECS)
    fingerprint = json.dumps(
        (COURSE_CONTRACT, split, course_seed, family, difficulty, spawn, goal, corridor_width,
         tuple((box.name, box.position, box.size, box.kind) for box in ordered)),
        separators=(",", ":"), allow_nan=False,
    )
    course_id = split + ":" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return SampledCourse(family, difficulty, ordered, spawn, goal, corridor_width,
                         course_id, course_seed, split)


class CurriculumController:
    """Per-family rolling success rates; evaluation and incomplete episodes freeze it.

    Outcomes from a previous difficulty are ignored after a level change. This
    avoids promoting a new level using easier episodes that finished late in
    asynchronous vector environments. A flat success never advances obstacles.
    """

    def __init__(self, initial_difficulty=0.0, *, window=DEFAULT_WINDOW,
                 min_episodes=DEFAULT_MIN_EPISODES,
                 promote_threshold=DEFAULT_PROMOTE_THRESHOLD,
                 demote_threshold=DEFAULT_DEMOTE_THRESHOLD,
                 step=DEFAULT_DIFFICULTY_STEP, frozen=False):
        initial = _unit_interval(initial_difficulty, "initial_difficulty")
        window = _nonnegative_integer(window, "window")
        minimum = _nonnegative_integer(min_episodes, "min_episodes")
        if not 1 <= minimum <= window:
            raise ValueError("Require 1 <= min_episodes <= window")
        promote = _unit_interval(promote_threshold, "promote_threshold")
        demote = _unit_interval(demote_threshold, "demote_threshold")
        step = _unit_interval(step, "step")
        if not demote < promote or step == 0:
            raise ValueError("Require demote_threshold < promote_threshold and positive step")
        if not isinstance(frozen, bool):
            raise ValueError("frozen must be boolean")
        self.config = {"initial_difficulty": initial, "window": window, "min_episodes": minimum,
                       "promote_threshold": promote, "demote_threshold": demote, "step": step}
        self.frozen = frozen
        self.levels = {family: initial for family in OBSTACLE_FAMILIES}
        self.history = {family: deque(maxlen=window) for family in OBSTACLE_FAMILIES}
        self.completed = {family: 0 for family in OBSTACLE_FAMILIES}
        self.promotions = {family: 0 for family in OBSTACLE_FAMILIES}
        self.demotions = {family: 0 for family in OBSTACLE_FAMILIES}

    def difficulty_for(self, family):
        if family not in FAMILIES:
            raise ValueError("Unknown terrain family")
        return 0.0 if family == "flat" else self.levels[family]

    def record(self, family, success, *, completed=True, difficulty=None):
        if family not in FAMILIES:
            raise ValueError("Unknown terrain family")
        if not isinstance(success, bool) or not isinstance(completed, bool):
            raise ValueError("success and completed must be boolean")
        if difficulty is not None:
            difficulty = _unit_interval(difficulty, "difficulty")
        if self.frozen or not completed or family == "flat":
            return False
        if difficulty is not None and not math.isclose(difficulty, self.levels[family], abs_tol=1e-9, rel_tol=0):
            return False
        self.completed[family] += 1
        history = self.history[family]
        history.append(success)
        if len(history) < self.config["min_episodes"]:
            return False
        rate = sum(history) / len(history)
        previous = self.levels[family]
        if rate >= self.config["promote_threshold"]:
            proposed = min(1.0, round(previous + self.config["step"], 10))
        elif rate <= self.config["demote_threshold"]:
            proposed = max(0.0, round(previous - self.config["step"], 10))
        else:
            return False
        if proposed == previous:
            return False
        self.levels[family] = proposed
        self.promotions[family] += int(proposed > previous)
        self.demotions[family] += int(proposed < previous)
        history.clear()
        return True

    def statistics(self):
        values = {}
        for family in OBSTACLE_FAMILIES:
            history = self.history[family]
            values.update({f"difficulty/{family}": self.levels[family],
                           f"success_rate/{family}": sum(history) / len(history) if history else 0.0,
                           f"window_episodes/{family}": len(history),
                           f"completed/{family}": self.completed[family],
                           f"promotions/{family}": self.promotions[family],
                           f"demotions/{family}": self.demotions[family]})
        return values

    def state_dict(self):
        return {"contract": COURSE_CONTRACT, "config": self.config.copy(),
                "levels": self.levels.copy(),
                "history": {family: list(values) for family, values in self.history.items()},
                "completed": self.completed.copy(), "promotions": self.promotions.copy(),
                "demotions": self.demotions.copy()}

    def load_state_dict(self, state):
        """Validate before mutation; preserve the caller's evaluation freeze flag."""
        if not isinstance(state, dict) or state.get("contract") != COURSE_CONTRACT or state.get("config") != self.config:
            raise ValueError("Curriculum state contract or configuration differs")
        expected = set(OBSTACLE_FAMILIES)
        for key in ("levels", "history", "completed", "promotions", "demotions"):
            if not isinstance(state.get(key), dict) or set(state[key]) != expected:
                raise ValueError(f"Curriculum state requires exactly the obstacle families in {key}")
        levels = {family: _unit_interval(value, "saved difficulty") for family, value in state["levels"].items()}
        history = {}
        for family, values in state["history"].items():
            if (not isinstance(values, list) or len(values) > self.config["window"]
                    or any(not isinstance(value, bool) for value in values)):
                raise ValueError("Curriculum state has invalid success history")
            history[family] = deque(values, maxlen=self.config["window"])
        counters = {key: {family: _nonnegative_integer(value, key) for family, value in state[key].items()}
                    for key in ("completed", "promotions", "demotions")}
        if any(counters["completed"][family] < len(history[family]) for family in OBSTACLE_FAMILIES):
            raise ValueError("Curriculum completion count cannot be below its history length")
        self.levels, self.history = levels, history
        for key, values in counters.items():
            setattr(self, key, values)
