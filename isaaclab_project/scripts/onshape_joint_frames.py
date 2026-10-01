"""Canonical joint names, zero pose, and axes for the current Onshape hexapod.

The Onshape Mate angle is an editor coordinate, not a robot joint coordinate.  This
module therefore maps Mate names to stable policy names while defining joint zero
from the saved assembly pose and rebuilding every axis in a canonical frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SPEC = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "spiderbot_variants"
    / "onshape_joint_frames.json"
)

JOINT_ORDER = ("coxa", "femur", "tibia")
POLICY_JOINT_ORDER = ("hip", "femur", "tibia")


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("onshape_joint_frames.json schema_version must be 1")
    if len(data["legs"]) != 6:
        raise ValueError("joint frame specification must contain six legs")
    return data


def joint_records(path: Path = DEFAULT_SPEC) -> tuple[dict[str, Any], ...]:
    """Return the 18 joints in policy order: leg_0..5, hip/femur/tibia."""

    data = load_spec(path)
    convention = data["joint_convention"]
    records: list[dict[str, Any]] = []
    for leg in sorted(data["legs"], key=lambda item: item["canonical_index"]):
        index = int(leg["canonical_index"])
        onshape_leg = str(leg["onshape_leg"])
        for onshape_kind, policy_kind in zip(JOINT_ORDER, POLICY_JOINT_ORDER):
            lower_deg, upper_deg = convention[onshape_kind]["position_limit_degrees"]
            records.append(
                {
                    "canonical_name": f"leg_{index}_{policy_kind}_joint",
                    "onshape_mate": f"dof_{onshape_leg}_{onshape_kind}",
                    "axis_in_joint_frame": tuple(
                        float(value)
                        for value in convention[onshape_kind]["axis_in_joint_frame"]
                    ),
                    "zero_position_radians": 0.0,
                    "lower_limit_radians": math.radians(float(lower_deg)),
                    "upper_limit_radians": math.radians(float(upper_deg)),
                    "azimuth_degrees": float(leg["azimuth_degrees"]),
                }
            )
    return tuple(records)


def canonical_name_for_mate(mate_name: str, path: Path = DEFAULT_SPEC) -> str:
    """Translate a named Onshape revolute Mate to the stable policy joint name."""

    matches = {
        record["onshape_mate"]: record["canonical_name"]
        for record in joint_records(path)
    }
    try:
        return matches[mate_name]
    except KeyError as exc:
        raise KeyError(f"unknown Onshape joint Mate: {mate_name}") from exc


def validate_spec(path: Path = DEFAULT_SPEC) -> tuple[str, ...]:
    """Return validation errors; an empty tuple means the convention is usable."""

    data = load_spec(path)
    records = joint_records(path)
    errors: list[str] = []
    names = [record["canonical_name"] for record in records]
    mates = [record["onshape_mate"] for record in records]
    if len(records) != 18:
        errors.append("expected 18 canonical joints")
    if len(set(names)) != len(names):
        errors.append("canonical joint names are not unique")
    if len(set(mates)) != len(mates):
        errors.append("Onshape Mate names are not unique")
    if data["zero_pose"].get("raw_onshape_mate_angles_are_joint_zero") is not False:
        errors.append("raw Onshape Mate angles must not define joint zero")
    if data["import_policy"].get("rebuild_axes_from_canonical_convention") is not True:
        errors.append("importer must rebuild canonical joint axes")
    for kind in JOINT_ORDER:
        limits = data["joint_convention"][kind].get("position_limit_degrees")
        if not isinstance(limits, list) or len(limits) != 2:
            errors.append(f"{kind} must define lower and upper position limits")
            continue
        lower, upper = (float(value) for value in limits)
        if not math.isclose(lower, -90.0) or not math.isclose(upper, 90.0):
            errors.append(f"{kind} position limit must be -90 to +90 degrees")
        if not math.isclose(upper - lower, 180.0):
            errors.append(f"{kind} position range must be 180 degrees")
    return tuple(errors)


def main() -> int:
    errors = validate_spec()
    if errors:
        for error in errors:
            print(f"JOINT_FRAME_INVALID {error}")
        return 2
    for record in joint_records():
        print(
            f"{record['onshape_mate']} -> {record['canonical_name']} "
            f"axis={record['axis_in_joint_frame']} q0=0 "
            f"limit=[{record['lower_limit_radians']:.6f}, "
            f"{record['upper_limit_radians']:.6f}] rad"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
