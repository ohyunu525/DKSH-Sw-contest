"""Simulator-independent geometry for selectable spiderbot training courses.

All positions are relative to an environment origin and all dimensions are in
metres. Box positions describe their centres; spawn and goal heights describe
the supporting surface, to which the robot's standing root height is added.
Dynamic floors and debris are instantiated by the environment, not this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


ENVIRONMENT_PRESETS = ("flat", "narrow", "vibrating", "falling_debris", "rough", "mixed")
COURSE_LENGTH = 5.6
VIBRATING_SURFACE_HEIGHT = 0.08


@dataclass(frozen=True)
class BoxSpec:
    """One fixed cuboid with a unique, USD-safe name."""

    name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    kind: str = "obstacle"


@dataclass(frozen=True)
class ScenarioLayout:
    """Course geometry and dynamic-hazard switches consumed by the simulator."""

    preset: str
    difficulty: float
    boxes: tuple[BoxSpec, ...]
    spawn: tuple[float, float, float]
    goal: tuple[float, float, float]
    corridor_width: float
    vibration_active: bool
    debris_active: bool


def validate_environment(preset: str, difficulty: float) -> None:
    """Raise ``ValueError`` for an unsupported preset or invalid difficulty."""
    if preset not in ENVIRONMENT_PRESETS:
        raise ValueError(
            f"Unknown environment preset {preset!r}; choose from {', '.join(ENVIRONMENT_PRESETS)}."
        )
    if (
        isinstance(difficulty, bool)
        or not isinstance(difficulty, Real)
        or not math.isfinite(difficulty)
        or not 0.0 <= difficulty <= 1.0
    ):
        raise ValueError("Environment difficulty must be a finite number between 0 and 1.")


def build_layout(
    preset: str,
    difficulty: float,
    robot_width: float = 0.62,
    robot_height: float = 0.28,
) -> ScenarioLayout:
    """Build a repeatable course with safe standing areas at either end.

    ``robot_width`` must include the standing legs and collision boxes, rather
    than just the body. The primitive robot's configured stance needs about
    0.69 m; callers supply their robot-specific envelope. The default remains
    available for smaller robots. Footprints wider than 1.2 m cannot safely
    occupy the clear standing areas at the course endpoints and are rejected.

    Narrow corridors leave 4--19% of the robot width free on each side, and a
    short overhead lintel adds a low-clearance section. Rough courses contain
    three full-width shallow steps. Mixed courses combine the narrow corridor,
    vibration and falling debris, with an open top so debris reaches the robot.
    """
    validate_environment(preset, difficulty)
    for name, value, maximum in (("robot_width", robot_width, 1.2), ("robot_height", robot_height, 2.0)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0.0 < value <= maximum
        ):
            raise ValueError(f"{name} must be a finite positive number no greater than {maximum} m.")

    difficulty = float(difficulty)
    vibration_active = preset in ("vibrating", "mixed")
    debris_active = preset in ("falling_debris", "mixed")
    narrow = preset in ("narrow", "mixed")
    floor_z = VIBRATING_SURFACE_HEIGHT if vibration_active else 0.0
    corridor_width = robot_width * (1.38 - 0.30 * difficulty) if narrow else max(1.6, 2.0 * robot_width)
    boxes: list[BoxSpec] = []

    if preset != "flat":
        # Side walls extend beyond both standing areas; the simulator also
        # terminates attempts to leave the course, so hazards cannot be bypassed.
        wall_thickness = 0.16
        wall_height = robot_height + 0.24
        wall_y = (corridor_width + wall_thickness) / 2.0
        for name, sign in (("wall_left", 1.0), ("wall_right", -1.0)):
            boxes.append(
                BoxSpec(
                    name=name,
                    position=(0.0, sign * wall_y, floor_z + wall_height / 2.0),
                    size=(COURSE_LENGTH, wall_thickness, wall_height),
                    kind="wall",
                )
            )

    if preset == "narrow":
        ceiling_clearance = robot_height + 0.12 - 0.08 * difficulty
        ceiling_thickness = 0.10
        boxes.append(
            BoxSpec(
                name="low_ceiling",
                position=(0.0, 0.0, floor_z + ceiling_clearance + ceiling_thickness / 2.0),
                size=(1.2, corridor_width + 0.32, ceiling_thickness),
                kind="ceiling",
            )
        )

    if preset == "rough":
        # Scale steps to the robot height so CAD and primitive robots both get
        # traversable terrain. Endpoints remain flat and free of obstacles.
        step_height = robot_height * (0.06 + 0.17 * difficulty)
        for index, (x, fraction) in enumerate(((-0.8, 0.65), (0.0, 1.0), (0.8, 0.8))):
            height = step_height * fraction
            boxes.append(
                BoxSpec(
                    name=f"rough_step_{index}",
                    position=(x, 0.0, floor_z + height / 2.0),
                    size=(0.28, corridor_width, height),
                    kind="rough",
                )
            )

    return ScenarioLayout(
        preset=preset,
        difficulty=difficulty,
        boxes=tuple(boxes),
        spawn=(-1.6, 0.0, floor_z),
        goal=(1.6, 0.0, floor_z),
        corridor_width=float(corridor_width),
        vibration_active=vibration_active,
        debris_active=debris_active,
    )
