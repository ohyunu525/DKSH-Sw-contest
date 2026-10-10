"""Normalize an Onshape-to-Robot hexapod export for Isaac Lab training.

The exporter preserves CAD geometry, origins, and inertials but emits
Onshape-specific names, raw Mate axis signs, and placeholder actuator/limit
values. This tool keeps the CAD transforms intact while giving the 18 joints
the project's stable names, canonical rotation directions, and +/-90 degree
software limits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET


JOINT_KIND_TO_POLICY = {"coxa": "hip", "femur": "femur", "tibia": "tibia"}
MG90S_EFFORT_NM = 0.13239
MG90S_VELOCITY_RAD_S = math.radians(600.0)
RC920DMG_EFFORT_NM = 1.39745
RC920DMG_VELOCITY_RAD_S = math.radians(60.0) / 0.16


def onshape_leg_index(onshape_leg: int) -> int:
    """Map visual Onshape L1..L6 order to policy leg_5..leg_0."""

    if not 1 <= onshape_leg <= 6:
        raise ValueError(f"expected an Onshape leg from 1 through 6, got L{onshape_leg}")
    return 6 - onshape_leg


def source_link_leg(name: str) -> int | None:
    """Return the Onshape leg number for a coxa/femur/tibia source link."""

    for kind in ("coxa", "femur", "tibia"):
        if name == kind:
            return 1
        if name.startswith(f"{kind}_") and name[len(kind) + 1 :].isdigit():
            return int(name[len(kind) + 1 :])
    return None


def canonical_link_name(name: str) -> str:
    if name == "body":
        return "base"
    for kind in ("coxa", "femur", "tibia"):
        leg = source_link_leg(name)
        if leg is not None and (name == kind or name.startswith(f"{kind}_")):
            return f"leg_{onshape_leg_index(leg)}_{JOINT_KIND_TO_POLICY[kind]}"
    return name


def _joint_limits(kind: str) -> tuple[float, float]:
    if kind == "coxa":
        return MG90S_EFFORT_NM, MG90S_VELOCITY_RAD_S
    return RC920DMG_EFFORT_NM, RC920DMG_VELOCITY_RAD_S


def normalize_tree(tree: ET.ElementTree) -> None:
    root = tree.getroot()
    root.set("name", "dksh_onshape_hexapod_20260928")

    for link in root.findall("link"):
        link.set("name", canonical_link_name(link.attrib["name"]))
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename and filename.startswith("package://assets/"):
                mesh.set("filename", "assets/" + filename.removeprefix("package://assets/"))

    fixed_foot_names: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} has no parent or child")
        source_parent = parent.attrib["link"]
        source_child = child.attrib["link"]
        parent.set("link", canonical_link_name(source_parent))

        if joint.attrib.get("type") == "fixed" and source_link_leg(source_parent) is not None:
            leg = onshape_leg_index(source_link_leg(source_parent) or 1)
            fixed_foot_names[source_child] = f"leg_{leg}_foot"
            child.set("link", fixed_foot_names[source_child])
            joint.set("name", f"leg_{leg}_foot_fixed")
        else:
            child.set("link", canonical_link_name(source_child))

        source_name = joint.attrib.get("name", "")
        if joint.attrib.get("type") != "revolute":
            continue
        pieces = source_name.split("_")
        if len(pieces) != 2 or not pieces[0].startswith("L") or not pieces[0][1:].isdigit():
            raise ValueError(f"unexpected Onshape revolute joint name: {source_name}")
        onshape_leg = int(pieces[0][1:])
        source_kind = pieces[1]
        if source_kind not in JOINT_KIND_TO_POLICY:
            raise ValueError(f"unexpected Onshape joint kind: {source_kind}")
        policy_kind = JOINT_KIND_TO_POLICY[source_kind]
        joint.set("name", f"leg_{onshape_leg_index(onshape_leg)}_{policy_kind}_joint")
        # The raw Onshape-to-Robot coxa axis is clockwise viewed from +Z and
        # the tibia axis initially lowers the foot.  Flip those two signs so
        # every normalized asset obeys the project convention.
        if source_kind in {"coxa", "tibia"}:
            axis = joint.find("axis")
            if axis is None:
                raise ValueError(f"joint {source_name} has no axis")
            axis.set("xyz", "0 0 -1")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        effort, velocity = _joint_limits(source_kind)
        limit.attrib.update(
            effort=f"{effort:.8f}",
            velocity=f"{velocity:.8f}",
            lower=f"{-math.pi / 2:.16f}",
            upper=f"{math.pi / 2:.16f}",
        )

    for link in root.findall("link"):
        original_name = link.attrib["name"]
        if original_name in fixed_foot_names:
            link.set("name", fixed_foot_names[original_name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Onshape-to-Robot source URDF")
    parser.add_argument("output", type=Path, help="normalized output URDF")
    args = parser.parse_args()
    tree = ET.parse(args.input)
    normalize_tree(tree)
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"ONSHAPE_URDF_NORMALIZED {args.output.resolve()}")


if __name__ == "__main__":
    main()
