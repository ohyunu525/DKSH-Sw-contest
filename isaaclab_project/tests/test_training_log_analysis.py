"""Read-only diagnostics against real TensorBoard event files and edge cases."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_joint_angle_training.py"
SPEC = importlib.util.spec_from_file_location("_training_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def sample(step, value, wall_time=None):
    return NS(step=step, value=value, wall_time=float(step) if wall_time is None else wall_time)


class TrainingLogAnalysisTests(unittest.TestCase):
    def metadata(self):
        return {"behavior_profile": "revised_v12", "target_speed_mps": 0.05,
                "control_frequency_hz": 10.0, "status": "running", "episode_length_s": 30.0,
                "completed_episodes": {"episodes": 12, "goal": 1, "fallen": 11}}

    def test_old_local_profile_is_identified_without_claiming_remote_result(self):
        metadata = {**self.metadata(), "behavior_profile": "revised_v9"}
        scalars = {"Metrics/step_progress_speed_mps": [sample(step, 0.0002) for step in range(100, 130)],
                   "Episode_Termination/fallen": [sample(129, 0.5)]}
        report = ANALYSIS.summarize_run("old_run", metadata, scalars, window=20,
                                        expected_profile="revised_v12")
        codes = {warning["code"] for warning in report["warnings"]}
        self.assertEqual(codes, {"profile_mismatch", "near_stationary"})
        self.assertEqual(report["latest_logged_step"], 129)
        self.assertAlmostEqual(report["logged_speed_fraction_of_target"], 0.004)
        self.assertEqual(report["completed_episodes_at_metadata_write"]["fallen"], 11)
        self.assertIn("not_policy_evaluation", report["scope"])

    def test_resume_duplicates_use_latest_write_and_window_respects_steps(self):
        events = [sample(10, 1, 11), sample(11, 2, 12), sample(10, 3, 15), sample(9, 50, 20)]
        report = ANALYSIS.summarize_run("run", self.metadata(), {"Policy/mean_noise_std": events}, window=2)
        metric = report["metrics"]["Policy/mean_noise_std"]
        self.assertEqual((metric["first_step"], metric["last_step"], metric["samples"]), (10, 11, 2))
        self.assertEqual((metric["mean"], metric["last"]), (2.5, 2.0))

    def test_nonfinite_values_are_reported_without_breaking_strict_json(self):
        report = ANALYSIS.summarize_run("run", self.metadata(), {
            "Loss/value_function": [sample(0, math.inf), sample(1, math.nan), sample(2, 3.0)],
        })
        self.assertEqual(len(report["warnings"]), 2)
        self.assertEqual(report["metrics"]["Loss/value_function"]["mean"], 3.0)
        json.dumps(report, allow_nan=False)

    def test_wall_time_axes_are_not_misreported_as_training_iterations(self):
        report = ANALYSIS.summarize_run("run", self.metadata(), {
            "Metrics/step_progress_speed_mps": [sample(99, 0.0001)],
            "Train/mean_reward/time": [sample(20035, -40)],
            "Train/mean_episode_length/time": [sample(20035, 300)],
        })
        self.assertEqual(report["latest_logged_training_iteration"], 99)
        self.assertEqual(report["latest_logged_step"], 99)

    def test_cost_comparison_requires_matching_units_and_iteration_windows(self):
        rates = {"Metrics/progress_reward_per_s": [sample(20, 0.01)],
                 "Metrics/speed_reward_per_s": [sample(20, 0.02)],
                 "Metrics/stance_slip_reward_per_s": [sample(20, -0.06)],
                 "Episode_Reward/foot_slip": [sample(20, -1000)]}
        report = ANALYSIS.summarize_run("run", self.metadata(), rates)
        self.assertAlmostEqual(report["slip_cost_to_locomotion_reward_ratio"], 2)
        self.assertIn("slip_cost_exceeds_locomotion_reward", {w["code"] for w in report["warnings"]})
        rates["Metrics/stance_slip_reward_per_s"] = [sample(21, -0.06)]
        report = ANALYSIS.summarize_run("run", self.metadata(), rates)
        self.assertIsNone(report["slip_cost_to_locomotion_reward_ratio"])
        self.assertIn("unaligned_reward_windows", {w["code"] for w in report["warnings"]})

    def test_latest_run_uses_run_timestamp_instead_of_copy_modification_time(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("2026-11-02_01-00-00", "2026-11-01_23-00-00"):
                path = root / name
                path.mkdir()
                (path / "run_metadata.json").write_text("{}")
            self.assertEqual(ANALYSIS.latest_run(root).name, "2026-11-02_01-00-00")
        for window in (0, -1, True, 1.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                ANALYSIS.summarize_run("run", self.metadata(), {}, window=window)

    def test_cli_reads_real_tensorboard_events_and_writes_a_reviewable_report(self):
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            self.skipTest("TensorBoard optional outside the training Python environment")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_metadata.json").write_text(json.dumps(self.metadata()), encoding="utf-8")
            writer = EventFileWriter(str(root))
            try:
                for step in (100, 101, 102):
                    writer.add_event(Event(wall_time=1793500000 + step, step=step, summary=Summary(value=[
                        Summary.Value(tag="Metrics/step_progress_speed_mps", simple_value=0.02),
                        Summary.Value(tag="Metrics/stance_slip_reward_per_s", simple_value=-0.006),
                    ])))
                writer.flush()
            finally:
                writer.close()
            output = root / "analysis.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = ANALYSIS.main(["--run", str(root), "--window", "2", "--output", str(output),
                                        "--expected-profile", "revised_v12"])
            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["behavior_profile"], "revised_v12")
            self.assertEqual(report["latest_logged_step"], 102)
            self.assertAlmostEqual(report["logged_speed_fraction_of_target"], 0.4, places=6)
            self.assertEqual(report["metrics"]["Metrics/step_progress_speed_mps"]["samples"], 2)
            self.assertEqual(report["warnings"], [])


if __name__ == "__main__":
    unittest.main()
