"""Completed-course evaluation and checkpoint sidecars without Isaac Sim."""

import argparse
import ast
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("course_runner_utils_test", SCRIPTS / "joint_angle_runner_utils.py")
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)


class CourseArgumentsTests(unittest.TestCase):
    def arguments(self, *argv, mode="train", environment="curriculum", profile="revised_v7"):
        parser = argparse.ArgumentParser()
        UTILS.add_course_args(parser)
        args = parser.parse_args(argv)
        args.mode, args.environment, args.behavior_profile = mode, environment, profile
        args.task = "Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2"
        args.seed = 73 if mode == "evaluate" else 42
        return UTILS.resolve_course_args(args)

    def test_course_configuration_freezes_holdout_but_keeps_training_adaptive(self):
        cfg = NS(environment_difficulty=0)
        training = self.arguments()
        UTILS.apply_course_cfg(cfg, training)
        self.assertTrue(cfg.curriculum_enabled)
        self.assertEqual((cfg.course_split, cfg.course_seed, cfg.curriculum_flat_fraction), ("train", 42, 0.25))
        metadata = UTILS.course_configuration(cfg, "curriculum")
        self.assertEqual(metadata["contract"], "mixed_randomized_courses_v1")
        self.assertEqual(metadata["initial_difficulty"], 0)
        evaluation = self.arguments(mode="evaluate", profile=None)
        UTILS.apply_course_cfg(cfg, evaluation)
        self.assertFalse(cfg.curriculum_enabled)
        self.assertEqual((cfg.course_split, cfg.course_seed), ("holdout", 73))

    def test_strict_distal_profile_is_allowed_for_curriculum(self):
        for profile in ("revised_v7", "revised_v8", "revised_v9"):
            with self.subTest(profile=profile):
                self.assertEqual(self.arguments(profile=profile).behavior_profile, profile)

    def test_invalid_randomization_and_windows_are_rejected(self):
        options = (("--course_seed=-1",), ("--flat_fraction=nan",), ("--flat_fraction=1",),
                   ("--curriculum_min_episodes=21",), ("--curriculum_window=0",),
                   ("--curriculum_step=0",), ("--curriculum_demote_threshold=.9",),
                   ("--curriculum_promote_threshold=inf",), ("--course_split=holdout",))
        for argv in options:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.arguments(*argv)
        with self.assertRaises(ValueError):
            self.arguments("--freeze_curriculum", mode="evaluate")
        with self.assertRaises(ValueError):
            self.arguments("--course_split=train", mode="evaluate")

    def test_fixed_course_options_and_metadata_remain_untouched(self):
        args = self.arguments(environment="flat")
        cfg = NS()
        self.assertIs(UTILS.apply_course_cfg(cfg, args), cfg)
        self.assertEqual(vars(cfg), {})
        self.assertIsNone(UTILS.course_configuration(cfg, "flat"))
        with self.assertRaises(ValueError):
            self.arguments("--course_seed=7", environment="stairs")


def metrics(count, *, goal=1):
    values = {name: torch.zeros(count) for name in UTILS.EvaluationAccumulator.METRICS}
    values["goal"][:] = goal
    values["progress_m"][:] = 0.2
    values["travel_distance_m"][:] = 0.2
    values["forward_speed_mps"][:] = 0.2
    values["non_distal_support_fraction"] = torch.zeros(count)
    values["foot_slip_m2_s2"] = torch.full((count,), .02)
    return values


class CourseEvaluationTests(unittest.TestCase):
    def records(self, episode):
        return [{"environment_index": index, "family": family, "difficulty": .3, "course_id": f"{family}-{episode}",
                 "course_seed": 100 + episode, "split": "holdout", "outcome": "goal"}
                for index, family in enumerate(UTILS.COURSE_FAMILIES)]

    def completed(self):
        accumulator = UTILS.EvaluationAccumulator(4)
        for episode in range(2):
            accumulator.update(metrics(4), torch.ones(4, dtype=torch.bool), 1, course_records=self.records(episode))
        return accumulator.completed

    def report(self, rows):
        return UTILS.heldout_course_report(rows, target_speed=.2, minimum_speed_fraction=.5)

    def test_terminal_course_is_matched_by_env_before_autoreset_and_partial_state_is_preserved(self):
        accumulator = UTILS.EvaluationAccumulator(4)
        old_course = self.records(0)
        accumulator.update(metrics(4), torch.tensor([False, True, False, True]), 1,
                           course_records=[old_course[3], old_course[1]])
        self.assertEqual([row["course_id"] for row in accumulator.completed], ["rough-0", "narrow-0"])
        torch.testing.assert_close(accumulator.duration, torch.tensor([1., 0., 1., 0.], dtype=torch.float64))
        accumulator.update(metrics(4), torch.tensor([True, False, True, False]), 1,
                           course_records=[old_course[0], old_course[2]])
        self.assertEqual(accumulator.completed[2]["duration_s"], 2)
        self.assertAlmostEqual(accumulator.completed[2]["non_distal_support_fraction"], 0)
        self.assertAlmostEqual(accumulator.completed[2]["foot_slip_m2_s2"], .02, places=6)

    def test_every_family_needs_two_distinct_completed_holdout_courses(self):
        rows = self.completed()
        report = self.report(rows)
        self.assertTrue(report["heldout_course_pass"])
        self.assertEqual(set(report["course_families"]), set(UTILS.COURSE_FAMILIES))
        for family in UTILS.COURSE_FAMILIES:
            self.assertEqual(report["course_families"][family]["unique_courses"], 2)
            self.assertIn(f"{family}@0.3", report["course_family_difficulty_groups"])
        for subset in (rows[:4], [row for row in rows if row["family"] != "stairs"], []):
            self.assertFalse(self.report(subset)["heldout_course_pass"])
        repeated = copy.deepcopy(rows)
        for row in repeated:
            row["course_id"] = row["family"]
        self.assertFalse(self.report(repeated)["heldout_course_pass"])

    def test_fast_flat_episodes_cannot_hide_failed_obstacles_or_knee_walking(self):
        rows = self.completed()
        rows[1]["outcome"] = "fallen"
        report = self.report(rows)
        self.assertFalse(report["heldout_course_pass"])
        self.assertFalse(report["course_families"]["rough"]["pass"])
        rows = self.completed()
        rows[2]["non_distal_support_fraction"] = .3
        self.assertFalse(self.report(rows)["heldout_course_pass"])
        rows = self.completed()
        rows[3]["signed_progress_m"] = 0
        rows[7]["signed_progress_m"] = 0
        self.assertFalse(self.report(rows)["heldout_course_pass"])

    def test_femur_contacts_cannot_pass_heldout_courses_after_clean_goal_stance(self):
        rows = self.completed()
        for row in rows:
            row["body_contact_fraction"] = .9
        report = self.report(rows)
        self.assertFalse(report["heldout_course_pass"])
        self.assertTrue(all(not family["pass"] for family in report["course_families"].values()))

    def test_one_dirty_course_is_not_diluted_by_many_clean_courses(self):
        for metric in ("body_contact_fraction", "non_distal_support_fraction", "non_distal_collision_fraction"):
            with self.subTest(metric=metric):
                rows = self.completed()
                rows[0][metric] = .1
                for index in range(20):
                    clean = copy.deepcopy(rows[4])
                    clean["course_id"] = f"flat-extra-{index}"
                    rows.append(clean)
                report = UTILS.heldout_course_report(rows, target_speed=.2, minimum_speed_fraction=.5,
                                                    **{f"max_{metric}": .05})
                self.assertFalse(report["heldout_course_pass"])
                self.assertFalse(report["course_families"]["flat"]["pass"])

    def test_strict_heldout_evidence_fails_closed_when_contact_metrics_are_missing(self):
        rows = self.completed()
        report = UTILS.heldout_course_report(rows, target_speed=.2, minimum_speed_fraction=.5,
                                            require_strict_contact=True)
        self.assertFalse(report["heldout_course_pass"])
        accumulator = UTILS.EvaluationAccumulator(4)
        for episode in range(2):
            sample = metrics(4)
            sample["non_distal_collision_fraction"] = torch.zeros(4)
            accumulator.update(sample, torch.ones(4, dtype=torch.bool), 1,
                               course_records=self.records(episode))
        complete = accumulator.completed
        self.assertTrue(UTILS.heldout_course_report(complete, target_speed=.2, minimum_speed_fraction=.5,
                                                  require_strict_contact=True)["heldout_course_pass"])
        del complete[0]["body_contact_fraction"]
        self.assertFalse(UTILS.heldout_course_report(complete, target_speed=.2, minimum_speed_fraction=.5,
                                                   require_strict_contact=True)["heldout_course_pass"])
    def test_training_split_and_incomplete_or_duplicate_terminal_metadata_are_rejected(self):
        rows = self.completed()
        rows[0]["split"] = "train"
        with self.assertRaises(ValueError):
            self.report(rows)
        for records in ([], self.records(0) * 2, self.records(0)[:3]):
            accumulator = UTILS.EvaluationAccumulator(4)
            with self.assertRaises(ValueError):
                accumulator.update(metrics(4), torch.ones(4, dtype=torch.bool), 1, course_records=records)

    def test_actual_evaluation_runner_consumes_terminal_snapshots_and_exports_holdout_groups(self):
        source = SCRIPTS / "joint_angle_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("evaluate_or_play", "toe_button_sensor_metadata")]
        cfg = NS(environment_difficulty=.3, target_speed=.2, evaluation_min_speed_fraction=.5,
                 goal_radius=.08, episode_length_s=240, behavior_profile="revised_v7",
                 reward_contract="signed_progress_huber_speed_distal_support_v1",
                 contact_contract="cad6_distal_support_toe_button_v1", seed=73, evaluation_goal_distance=1)
        course_args = CourseArgumentsTests().arguments(mode="evaluate", profile=None)
        UTILS.apply_course_cfg(cfg, course_args)
        base = NS(num_envs=4, device="cpu", step_dt=1, _scenario=NS(), completed_course_records=[])
        step = 0

        def transition(actions):
            nonlocal step
            base.step_metrics = metrics(4)
            base.completed_course_records.extend(self.records(step))
            step += 1
            return torch.zeros((4, 140)), torch.zeros(4), torch.ones(4, dtype=torch.bool), {}

        env = NS(unwrapped=base, get_observations=lambda: (torch.zeros((4, 140)), {}), step=transition)
        runner = NS(load=Mock(), get_inference_policy=lambda device: lambda observation: torch.zeros((4, 18)))
        write = Mock()
        namespace = {"torch": torch, "time": __import__("time"), "json": json,
                     "EvaluationAccumulator": UTILS.EvaluationAccumulator, "heldout_course_report": UTILS.heldout_course_report,
                     "course_configuration": UTILS.course_configuration, "write_json": write,
                     "args": NS(mode="evaluate", environment="curriculum", steps=2, task="test", difficulty=.3,
                                num_envs=4, output="holdout.json", headless=True),
                     "app": NS(is_running=lambda: True), "ACTION_SEMANTICS": "absolute_joint_angles_v2",
                     "behavior_profile_from_metadata": lambda metadata: "revised_v7"}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        source_metadata = {"reward_contract": cfg.reward_contract, "contact_contract": cfg.contact_contract}
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["evaluate_or_play"](env, runner, cfg, "model.pt", source_metadata)
        report = write.call_args.args[1]
        self.assertTrue(report["heldout_course_pass"])
        self.assertIsNone(report["fixed_scenario_distance_m"])
        self.assertEqual(len(report["completed_episode_records"]), 8)
        self.assertEqual(report["completed_episode_records"][0]["course_id"], "flat-0")
        self.assertFalse(report["course_evaluation"]["curriculum_enabled"])
        self.assertEqual(base.completed_course_records, [])


class CurriculumSidecarTests(unittest.TestCase):
    def test_interrupted_sidecar_replacement_preserves_previous_json_and_cleans_temporary_file(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10.pt"
            runtime = NS(state_dict=lambda: {"episode_counters": [1]})
            runner = NS(save=lambda path: Path(path).write_bytes(b"saved model"))
            hook = UTILS.CurriculumCheckpointState(runner, runtime, {"seed": 42})
            try:
                runner.save(checkpoint)
                original = hook.sidecar(checkpoint).read_bytes()
                with patch.object(Path, "replace", side_effect=OSError("interrupted replace")):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        runner.save(checkpoint)
                self.assertEqual(hook.sidecar(checkpoint).read_bytes(), original)
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])
                self.assertEqual(json.loads(original)["state"], {"episode_counters": [1]})
            finally:
                hook.close()

    def test_sidecar_is_bound_to_checkpoint_hash_and_restores_matching_configuration(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10.pt"
            state = {"episode_counters": [2, 4], "controller": {"rough": .3}}
            runtime = NS(state_dict=lambda: copy.deepcopy(state), load_state_dict=lambda saved: state.update(saved))
            runner = NS(save=lambda path: Path(path).write_bytes(b"model iteration ten"))
            original = runner.save
            hook = UTILS.CurriculumCheckpointState(runner, runtime, {"seed": 42})
            try:
                self.assertEqual(hook.restore(checkpoint), "fresh_curriculum_no_checkpoint_sidecar")
                runner.save(checkpoint)
                payload = json.loads(hook.sidecar(checkpoint).read_text())
                self.assertEqual(payload["state"], state)
                state["episode_counters"] = [0, 0]
                self.assertEqual(hook.restore(checkpoint), "restored_course_sequence_and_curriculum")
                self.assertEqual(state["episode_counters"], [2, 4])
                hook.configuration = {"seed": 73}
                self.assertEqual(hook.restore(checkpoint), "fresh_curriculum_configuration_changed")
                hook.configuration = {"seed": 42}
                checkpoint.write_bytes(b"different model")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    hook.restore(checkpoint)
            finally:
                hook.close()
            self.assertIs(runner.save, original)


if __name__ == "__main__":
    unittest.main()
