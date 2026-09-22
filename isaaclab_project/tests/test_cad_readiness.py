"""Tests for the Onshape-to-training readiness gate."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts/cad_readiness.py"
SPEC = importlib.util.spec_from_file_location("dksh_cad_readiness", MODULE)
assert SPEC is not None and SPEC.loader is not None
CAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAD)


class CadReadinessTests(unittest.TestCase):
    def test_current_onshape_design_is_explicitly_provisional(self) -> None:
        report = CAD.readiness_report()
        self.assertFalse(report["ready"])
        self.assertEqual(report["lifecycle"], "in_progress")
        self.assertIn("joint limits are incomplete", report["blockers"])
        self.assertIn("MG90S/RC920DMG layout is not verified in CAD", report["blockers"])

    def test_complete_release_evidence_is_accepted(self) -> None:
        data = copy.deepcopy(CAD.load_manifest())
        data["source"]["lifecycle"] = "released_for_simulation"
        data["release_evidence"] = {
            "step_export": {
                "path": "cad/hexapod.step",
                "sha256": "0" * 64,
                "parts_preserved": True,
            },
            "joint_spec": {
                "path": "cad/joints.json",
                "axes_and_centers_complete": True,
                "zero_pose_complete": True,
                "limits_complete": True,
            },
            "mass_properties": {
                "path": "cad/mass_properties.json",
                "complete": True,
                "motor_inclusion_labeled": True,
            },
            "servo_layout": {
                "verified_in_cad": True,
                "performance_source": "cad/servo_bench.json",
                "operating_point_verified": True,
            },
        }
        self.assertEqual(CAD.collect_blockers(data), [])

    def test_provisional_override_is_recorded_but_not_called_ready(self) -> None:
        report = CAD.assert_cad_ready(allow_provisional=True)
        self.assertFalse(report["ready"])
        self.assertTrue(report["provisional_override"])

    def test_training_gate_rejects_current_manifest(self) -> None:
        with self.assertRaises(CAD.CadNotReadyError):
            CAD.assert_cad_ready()


if __name__ == "__main__":
    unittest.main()
