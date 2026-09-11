"""Fast structural tests for the generated DKSH spiderbot URDF."""

from __future__ import annotations

import importlib.util
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dksh_isaaclab"
    / "dksh_isaaclab"
    / "assets"
    / "urdf_generator.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_urdf_generator", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class SpiderbotUrdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.xml = GENERATOR.build_spiderbot_urdf()
        self.root = ET.fromstring(self.xml)

    def test_structure_is_eight_legs_and_twenty_four_dof(self) -> None:
        self.assertEqual(self.root.attrib["name"], "dksh_spiderbot")
        links = self.root.findall("link")
        joints = self.root.findall("joint")
        self.assertEqual(len(links), 25)
        self.assertEqual(len(joints), 24)
        self.assertEqual({joint.attrib["type"] for joint in joints}, {"revolute"})
        for leg_index in range(8):
            names = {joint.attrib["name"] for joint in joints}
            self.assertIn(f"leg_{leg_index}_hip_joint", names)
            self.assertIn(f"leg_{leg_index}_femur_joint", names)
            self.assertIn(f"leg_{leg_index}_tibia_joint", names)

    def test_measured_mass_and_servo_limits_are_preserved(self) -> None:
        masses = [float(node.attrib["value"]) for node in self.root.findall("./link/inertial/mass")]
        self.assertAlmostEqual(sum(masses), 0.5 + 8 * 0.30227, places=6)
        for joint in self.root.findall("joint"):
            limit = joint.find("limit")
            assert limit is not None
            self.assertAlmostEqual(float(limit.attrib["effort"]), 0.980665, places=6)
            self.assertAlmostEqual(float(limit.attrib["velocity"]), math.radians(400.0), places=6)
            self.assertAlmostEqual(float(limit.attrib["lower"]), -math.pi / 2, places=6)
            self.assertAlmostEqual(float(limit.attrib["upper"]), math.pi / 2, places=6)

    def test_link_lengths_match_the_robot_profile(self) -> None:
        expected = {"hip": 0.08617, "femur": 0.100, "tibia": 0.120}
        links = {link.attrib["name"]: link for link in self.root.findall("link")}
        for leg_index in range(8):
            for segment, expected_length in expected.items():
                size = links[f"leg_{leg_index}_{segment}"].find("./visual/geometry/box")
                assert size is not None
                length = float(size.attrib["size"].split()[0])
                self.assertAlmostEqual(length, expected_length, places=6)

    def test_output_and_cache_are_deterministic(self) -> None:
        self.assertEqual(self.xml, GENERATOR.build_spiderbot_urdf())
        cached_path = GENERATOR.ensure_spiderbot_urdf()
        self.assertEqual(cached_path.read_text(encoding="utf-8"), self.xml)


if __name__ == "__main__":
    unittest.main()
