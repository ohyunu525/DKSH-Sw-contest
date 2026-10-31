"""Pre-reset evidence keeps transient pairs and stays bounded on disk."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta

import torch


MODULE = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_contact_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("joint_angle_contact_diagnostics_under_test", MODULE)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class DiagnosticWriterTests(unittest.TestCase):
    def test_no_file_until_event_and_snapshots_append_with_context_and_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run" / "contact_failures.jsonl"
            writer = DIAGNOSTICS.ContactDiagnosticWriter(path, context={"profile": "revised_v8"})
            self.assertFalse(path.exists())
            self.assertEqual(writer.record_batch([]), 0)
            self.assertFalse(path.exists())
            self.assertEqual(writer.record_batch([{"environment_id": 2, "force": [1., 2., 3.]}]), 1)
            first = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["environment_id"], 2)
            self.assertEqual(first["context"], {"profile": "revised_v8"})
            self.assertEqual(first["schema_version"], 1)
            self.assertTrue(first["timestamp_utc"].endswith("Z"))
            timestamp = datetime.fromisoformat(first["timestamp_utc"].replace("Z", "+00:00"))
            self.assertEqual(timestamp.utcoffset(), timedelta(0))
            writer.record_batch([{"environment_id": 3}])
            self.assertEqual([json.loads(line)["environment_id"] for line in path.read_text().splitlines()], [2, 3])

    def test_selection_is_stable_and_counts_record_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DIAGNOSTICS.ContactDiagnosticWriter(Path(directory) / "evidence.jsonl", max_records=3, max_per_step=2)
            selected = writer.select(torch.tensor([8, 1, 4, 3]))
            self.assertEqual(selected, [1, 3])
            self.assertEqual(writer.dropped_records, 2)
            writer.record_batch([{"environment_id": index} for index in selected])
            selected = writer.select([9, 5, 2])
            self.assertEqual(selected, [2])
            writer.record_batch([{"environment_id": index} for index in selected])
            self.assertEqual(writer.written_records, 3)
            self.assertEqual(writer.dropped_records, 4)
            self.assertEqual(writer.remaining_records, 0)
            self.assertEqual(writer.select([7]), [])
            self.assertEqual(writer.dropped_records, 5)

    def test_direct_batches_and_existing_files_cannot_bypass_run_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            writer = DIAGNOSTICS.ContactDiagnosticWriter(path, max_records=2, max_per_step=1)
            writer.record_batch([{"index": 0}, {"index": 1}])
            self.assertEqual(writer.written_records, 1)
            self.assertEqual(writer.dropped_records, 1)
            resumed = DIAGNOSTICS.ContactDiagnosticWriter(path, max_records=2)
            self.assertEqual(resumed.written_records, 1)
            resumed.record_batch([{"index": 2}, {"index": 3}])
            self.assertEqual(resumed.written_records, 2)
            self.assertEqual(resumed.dropped_records, 1)
            self.assertEqual([json.loads(line)["index"] for line in path.read_text().splitlines()], [0, 2])

    def test_invalid_measurements_are_explicit_nulls_not_invalid_json_or_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            writer = DIAGNOSTICS.ContactDiagnosticWriter(path)
            writer.record_batch([{"force": torch.tensor([math.nan, math.inf, -math.inf, 4.]),
                                  "gap": math.nan}])
            parsed = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertEqual(parsed["force"], [None, None, None, 4.])
            self.assertIsNone(parsed["gap"])
            self.assertEqual(parsed["nonfinite_fields"], ["record.force[0]", "record.force[1]", "record.force[2]", "record.gap"])

    def test_append_separates_complete_record_without_newline_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_text('{"environment_id":1}', encoding="utf-8")
            writer = DIAGNOSTICS.ContactDiagnosticWriter(path)
            writer.record_batch([{"environment_id": 2}])
            self.assertEqual([json.loads(line)["environment_id"] for line in path.read_text().splitlines()], [1, 2])
            path.write_text('{"environment_id":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete JSON"):
                DIAGNOSTICS.ContactDiagnosticWriter(path)
            self.assertEqual(path.read_text(), '{"environment_id":')

    def test_bad_selected_record_does_not_write_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            writer = DIAGNOSTICS.ContactDiagnosticWriter(path)
            with self.assertRaises(TypeError):
                writer.record_batch([{"environment_id": 1}, {"unsupported": object()}])
            self.assertFalse(path.exists())
            self.assertEqual(writer.written_records, 0)

    def test_input_validation(self):
        for key in ("max_records", "max_per_step"):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    DIAGNOSTICS.ContactDiagnosticWriter("unused.jsonl", **{key: value})
        writer = DIAGNOSTICS.ContactDiagnosticWriter("unused.jsonl")
        for values in ([1, 1], [-1], [True], [1.5], torch.ones((1, 1), dtype=torch.long)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                writer.select(values)


class ContactPairHistoryTests(unittest.TestCase):
    def history(self, *, length=4):
        return DIAGNOSTICS.ContactPairHistory(
            2, ["base", "leg_1_femur"],
            ["/World/ground", "/World/envs/env_.*/Robot/base"], length,
        )

    def test_earlier_substep_pair_survives_a_zero_final_matrix(self):
        history = self.history()
        history.begin_step()
        forces = torch.zeros((2, 2, 2, 3))
        forces[1, 1, 1] = torch.tensor([-6., 0., 0.])
        history.capture(forces, 0)
        forces.zero_()
        for substep in (1, 2, 3):
            history.capture(forces, substep)
        snapshot = history.snapshot(1, force_threshold=1.)
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["physics_substeps_captured"], [0, 1, 2, 3])
        self.assertFalse(snapshot["contact_point_available"])
        self.assertEqual(len(snapshot["pairs"]), 1)
        pair = snapshot["pairs"][0]
        self.assertEqual(pair["source_name"], "leg_1_femur")
        self.assertEqual(pair["source_path"], "/World/envs/env_1/Robot/leg_1_femur")
        self.assertEqual(pair["target_path"], "/World/envs/env_1/Robot/base")
        self.assertEqual(pair["peak_force_norm_n"], 6.)
        self.assertEqual(pair["samples"][0]["force_w_n"], [-6., 0., 0.])
        self.assertEqual(pair["samples"][-1]["force_w_n"], [0., 0., 0.])
        self.assertEqual(history.snapshot(0)["pairs"], [])

    def test_one_sensor_per_source_matrix_shape_preserves_exact_mapping(self):
        history = self.history(length=1)
        matrices = [torch.zeros((2, 1, 2, 3)) for _ in range(2)]
        matrices[0][0, 0, 0, 2] = 2.
        matrices[1][0, 0, 1, 0] = -3.
        history.capture(matrices, 0)
        pairs = history.snapshot(0)["pairs"]
        self.assertEqual([(pair["source_name"], pair["target_path"]) for pair in pairs],
                         [("base", "/World/ground"), ("leg_1_femur", "/World/envs/env_0/Robot/base")])
        self.assertEqual([pair["peak_force_norm_n"] for pair in pairs], [2., 3.])

    def test_empty_pair_metadata_preserves_configured_sources_and_targets_without_claiming_no_contact(self):
        history = self.history(length=1)
        history.capture(torch.zeros((2, 2, 2, 3)), 0)
        snapshot = history.snapshot(1)
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["pairs"], [])
        self.assertEqual(snapshot["source_names"], ["base", "leg_1_femur"])
        self.assertEqual(snapshot["filter_target_paths"],
                         ["/World/ground", "/World/envs/env_1/Robot/base"])
        self.assertFalse(snapshot["empty_pairs_prove_no_contact"])

    def test_new_step_never_reuses_stale_previous_contact(self):
        history = self.history()
        history.capture(torch.ones((2, 2, 2, 3)), 3)
        self.assertFalse(history.snapshot(0)["complete"])
        history.begin_step()
        self.assertEqual(history.snapshot(0)["status"], "not_captured")
        self.assertEqual(history.snapshot(0)["pairs"], [])
        history.capture(torch.zeros((2, 2, 2, 3)), 0)
        self.assertEqual(history.snapshot(0)["physics_substeps_captured"], [0])
        self.assertEqual(history.snapshot(0)["pairs"], [])

    def test_nonfinite_pair_is_preserved_even_when_finite_force_is_below_threshold(self):
        history = self.history(length=1)
        forces = torch.zeros((2, 2, 2, 3))
        forces[0, 1, 1, 0] = math.nan
        history.capture(forces, 0)
        pairs = history.snapshot(0, force_threshold=100.)["pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["has_nonfinite"])
        self.assertIsNone(pairs[0]["peak_force_norm_n"])
        self.assertEqual(pairs[0]["samples"][0]["force_w_n"], [None, 0., 0.])

    def test_target_count_or_ambiguous_body_filters_are_rejected(self):
        history = self.history()
        with self.assertRaisesRegex(ValueError, "matching source and target"):
            history.capture(torch.zeros((2, 2, 3, 3)), 0)
        with self.assertRaisesRegex(ValueError, "shape"):
            history.capture([torch.zeros((2, 2, 2, 3)), torch.zeros((2, 1, 2, 3))], 0)
        with self.assertRaisesRegex(ValueError, "body wildcards"):
            DIAGNOSTICS.ContactPairHistory(1, ["base"], ["/World/envs/env_.*/Robot/.*"], 4)

    def test_duplicate_substeps_invalid_values_and_dtype_changes_are_rejected(self):
        history = self.history()
        forces = torch.zeros((2, 2, 2, 3))
        for substep in (-1, 4, True, 0.5):
            with self.subTest(substep=substep), self.assertRaises(ValueError):
                history.capture(forces, substep)
        history.capture(forces, 0)
        with self.assertRaisesRegex(ValueError, "twice"):
            history.capture(forces, 0)
        with self.assertRaisesRegex(ValueError, "dtype"):
            history.capture(forces.double(), 1)
        with self.assertRaisesRegex(ValueError, "floating point"):
            history.capture(forces.long(), 1)
        for environment_id in (-1, 2, True, 0.5):
            with self.subTest(environment_id=environment_id), self.assertRaises(ValueError):
                history.snapshot(environment_id)
        for threshold in (-1, math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                history.snapshot(0, threshold)


if __name__ == "__main__":
    unittest.main()
