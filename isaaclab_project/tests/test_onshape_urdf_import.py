"""Regression checks for the normalized current-Onshape URDF asset."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "spiderbot_variants" / "onshape_20260928"
NORMALIZER_PATH = ROOT / "scripts" / "normalize_onshape_urdf.py"
SPEC = importlib.util.spec_from_file_location("dksh_normalize_onshape_urdf", NORMALIZER_PATH)
assert SPEC is not None and SPEC.loader is not None
NORMALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NORMALIZER)


class OnshapeUrdfImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ET.parse(ASSET / "robot.training.urdf").getroot()
        self.joints = self.root.findall("joint")
        self.revolute = [joint for joint in self.joints if joint.attrib["type"] == "revolute"]

    def test_export_has_complete_normalized_six_leg_structure(self) -> None:
        self.assertEqual(len(self.root.findall("link")), 25)
        self.assertEqual(len(self.joints), 24)
        self.assertEqual(len(self.revolute), 18)
        names = {joint.attrib["name"] for joint in self.revolute}
        for leg in range(6):
            for kind in ("hip", "femur", "tibia"):
                self.assertIn(f"leg_{leg}_{kind}_joint", names)

    def test_normalized_limits_and_actuator_caps_are_safe(self) -> None:
        for joint in self.revolute:
            limit = joint.find("limit")
            assert limit is not None
            self.assertAlmostEqual(float(limit.attrib["lower"]), -math.pi / 2)
            self.assertAlmostEqual(float(limit.attrib["upper"]), math.pi / 2)
            if "_hip_joint" in joint.attrib["name"]:
                self.assertAlmostEqual(float(limit.attrib["effort"]), NORMALIZER.MG90S_EFFORT_NM)
                self.assertAlmostEqual(float(limit.attrib["velocity"]), NORMALIZER.MG90S_VELOCITY_RAD_S)
            else:
                self.assertAlmostEqual(float(limit.attrib["effort"]), NORMALIZER.RC920DMG_EFFORT_NM)
                self.assertAlmostEqual(float(limit.attrib["velocity"]), NORMALIZER.RC920DMG_VELOCITY_RAD_S)

    def test_axis_signs_match_project_rotation_convention(self) -> None:
        for joint in self.revolute:
            name = joint.attrib["name"]
            axis = joint.find("axis")
            assert axis is not None
            expected = "0 0 -1" if name.endswith(("_hip_joint", "_tibia_joint")) else "0 0 1"
            self.assertEqual(axis.attrib["xyz"], expected)

    def test_all_normalized_mesh_references_are_present(self) -> None:
        meshes = self.root.findall(".//mesh")
        self.assertGreater(len(meshes), 0)
        for mesh in meshes:
            filename = mesh.attrib["filename"]
            self.assertFalse(filename.startswith("package://"))
            self.assertTrue((ASSET / filename).is_file(), filename)

    def test_exported_mass_matches_the_onshape_measurement(self) -> None:
        masses = [float(link.find("./inertial/mass").attrib["value"]) for link in self.root.findall("link")]
        self.assertAlmostEqual(sum(masses), 1.7923392, places=6)


if __name__ == "__main__":
    unittest.main()
