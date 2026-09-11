"""Generate a deterministic primitive URDF for the DKSH spiderbot."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path


_URDF_VERSION = "2"


def _inertial(mass: float, ixx: float, iyy: float, izz: float, center_x: float = 0.0) -> str:
    return f"""
    <inertial>
      <origin xyz="{center_x:.6f} 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>
    </inertial>"""


def _box_link(name: str, length: float, width: float, height: float, mass: float, color: str) -> str:
    ixx = mass * (width * width + height * height) / 12.0
    iyy = mass * (length * length + height * height) / 12.0
    izz = mass * (length * length + width * width) / 12.0
    center = length / 2.0
    return f"""
  <link name="{name}">
{_inertial(mass, ixx, iyy, izz, center)}
    <visual>
      <origin xyz="{center:.6f} 0 0" rpy="0 0 0"/>
      <geometry><box size="{length:.6f} {width:.6f} {height:.6f}"/></geometry>
      <material name="{color}"/>
    </visual>
    <collision>
      <origin xyz="{center:.6f} 0 0" rpy="0 0 0"/>
      <geometry><box size="{length:.6f} {width:.6f} {height:.6f}"/></geometry>
    </collision>
  </link>"""


def _joint(name: str, parent: str, child: str, xyz: tuple[float, float, float], yaw: float, axis: str) -> str:
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="0 0 {yaw:.8f}"/>
    <axis xyz="{axis}"/>
    <limit lower="{-math.pi / 2:.8f}" upper="{math.pi / 2:.8f}" effort="0.980665" velocity="6.981317"/>
    <dynamics damping="0.05" friction="0.01"/>
  </joint>"""


def build_spiderbot_urdf() -> str:
    """Return the complete 24-DoF robot description."""
    sections = [
        '<?xml version="1.0"?>',
        '<robot name="dksh_spiderbot">',
        '  <material name="body"><color rgba="0.08 0.10 0.14 1"/></material>',
        '  <material name="hip"><color rgba="0.10 0.35 0.85 1"/></material>',
        '  <material name="femur"><color rgba="0.15 0.55 0.95 1"/></material>',
        '  <material name="tibia"><color rgba="0.90 0.38 0.08 1"/></material>',
        """
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.500000"/>
      <inertia ixx="0.00216667" ixy="0" ixz="0" iyy="0.00390000" iyz="0" izz="0.00576667"/>
    </inertial>
    <visual>
      <geometry><box size="0.30 0.22 0.06"/></geometry>
      <material name="body"/>
    </visual>
    <collision><geometry><box size="0.30 0.22 0.06"/></geometry></collision>
  </link>""",
    ]

    hip_length = 0.08617
    femur_length = 0.100
    tibia_length = 0.120
    attachment_radius = 0.145
    for leg_index in range(8):
        angle = 2.0 * math.pi * leg_index / 8.0
        prefix = f"leg_{leg_index}"
        hip = f"{prefix}_hip"
        femur = f"{prefix}_femur"
        tibia = f"{prefix}_tibia"
        attach = (attachment_radius * math.cos(angle), attachment_radius * math.sin(angle), 0.0)

        sections.append(_box_link(hip, hip_length, 0.030, 0.030, 0.100, "hip"))
        sections.append(_box_link(femur, femur_length, 0.025, 0.025, 0.100, "femur"))
        sections.append(_box_link(tibia, tibia_length, 0.022, 0.022, 0.10227, "tibia"))
        sections.append(_joint(f"{prefix}_hip_joint", "base", hip, attach, angle, "0 0 1"))
        sections.append(_joint(f"{prefix}_femur_joint", hip, femur, (hip_length, 0.0, 0.0), 0.0, "0 1 0"))
        sections.append(_joint(f"{prefix}_tibia_joint", femur, tibia, (femur_length, 0.0, 0.0), 0.0, "0 1 0"))

    sections.append("</robot>")
    return "\n".join(sections) + "\n"


def ensure_spiderbot_urdf() -> Path:
    """Write the generated URDF to a stable cache and return its path."""
    output_dir = Path(tempfile.gettempdir()) / "dksh_isaaclab" / _URDF_VERSION
    output_path = output_dir / "spiderbot.urdf"
    expected = build_spiderbot_urdf()
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != expected:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(expected, encoding="utf-8")
    return output_path
