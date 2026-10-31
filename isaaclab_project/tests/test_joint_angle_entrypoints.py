"""Entry routing and retired-task rejection before a simulator can start."""

import contextlib
import ast
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import types
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class JointAngleEntrypointTests(unittest.TestCase):
    def modules(self):
        environment_cli = types.ModuleType("environment_cli")
        environment_cli.configure_runner = Mock()
        environment_cli.add_environment_args = lambda parser: None
        environment_cli.apply_environment_cfg = Mock()
        environment_cli.experiment_for_environment = Mock()
        isaac = types.ModuleType("isaaclab")
        app = types.ModuleType("isaaclab.app")
        app.AppLauncher = Mock(side_effect=AssertionError("Simulator must not start in an entry routing test"))
        app.AppLauncher.add_app_launcher_args = lambda parser: None
        diagnostic_spec = importlib.util.spec_from_file_location("contact_diagnostics_cli", SCRIPTS / "contact_diagnostics_cli.py")
        diagnostics = importlib.util.module_from_spec(diagnostic_spec)
        diagnostic_spec.loader.exec_module(diagnostics)
        return {"dksh_isaaclab": types.ModuleType("dksh_isaaclab"), "environment_cli": environment_cli,
                "isaaclab": isaac, "isaaclab.app": app, "contact_diagnostics_cli": diagnostics}

    def execute(self, name):
        path = SCRIPTS / name
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {"__file__": str(path), "__name__": "_entry_test_"})

    def test_train_play_and_evaluate_default_to_guarded_v2_runner(self):
        for name, mode in (("train.py", "train"), ("play.py", "play"), ("evaluate.py", "evaluate")):
            with self.subTest(entry=name), patch.dict(sys.modules, self.modules()), patch.object(sys, "argv", [name]), patch.object(runpy, "run_path") as delegate:
                if mode == "evaluate":
                    with self.assertRaises(SystemExit) as exit_result:
                        self.execute(name)
                    self.assertEqual(exit_result.exception.code, 0)
                else:
                    self.execute(name)
                delegate.assert_called_once()
                self.assertEqual(Path(delegate.call_args.args[0]).name, "joint_angle_runner.py")
                self.assertIn(f"--mode={mode}", sys.argv)
                self.assertIn("--task=Isaac-DKSH-Spider-JointAngles-v2", sys.argv)

    def test_retired_v1_is_rejected_before_any_runner_or_simulator_starts(self):
        for name in ("train.py", "play.py", "evaluate.py", "preview_env.py", "smoke_env.py"):
            modules = self.modules()
            with self.subTest(entry=name), patch.dict(sys.modules, modules), patch.object(sys, "argv", [name, "--task=Isaac-DKSH-Spider-JointAngles-v1"]), patch.object(runpy, "run_path") as delegate, contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as exit_result:
                    self.execute(name)
                self.assertEqual(exit_result.exception.code, 2)
                self.assertIn("retrain", error.getvalue())
                delegate.assert_not_called()
                modules["isaaclab.app"].AppLauncher.assert_not_called()

    def runner_arguments(self, arguments):
        modules = self.modules()
        def environment_arguments(parser):
            parser.add_argument("--seed", type=int, default=None)
            parser.add_argument("--environment", default="flat")
            parser.add_argument("--difficulty", type=float, default=0.5)
        modules["environment_cli"].add_environment_args = environment_arguments
        modules["cad_readiness"] = types.ModuleType("cad_readiness")
        modules["cad_readiness"].assert_cad_ready = Mock()
        for name in ("joint_angle_checkpoint", "joint_angle_runner_utils", "contact_diagnostics_cli"):
            spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
            modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modules[name])
        source = SCRIPTS / "joint_angle_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        prefix = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets):
                break
            prefix.append(node)
        namespace = {"__file__": str(source), "__name__": "_runner_argument_test_"}
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", [source.name, "--task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2", *arguments]):
            exec(compile(ast.Module(body=prefix, type_ignores=[]), str(source), "exec"), namespace)
        return namespace

    def test_runner_defaults_cad6_to_strict_support_in_every_mode(self):
        for mode in ("train", "evaluate", "play"):
            with self.subTest(mode=mode):
                namespace = self.runner_arguments([f"--mode={mode}"])
                self.assertEqual(namespace["args"].behavior_profile, "revised_v8")
                self.assertEqual(namespace["checkpoint_compatibility"]["expected_reward_contract"],
                                 "signed_progress_huber_speed_strict_support_v1")
                self.assertEqual(namespace["checkpoint_compatibility"]["expected_contact_contract"],
                                 "cad6_strict_distal_all_links_v1")
                self.assertFalse(namespace["checkpoint_compatibility"]["allow_behavior_transfer"])
                if mode == "evaluate":
                    self.assertEqual(namespace["args"].seed, 73)
        legacy = self.runner_arguments(["--mode=train", "--behavior_profile=legacy_v2", "--reset_exploration_std"])
        self.assertEqual(legacy["checkpoint_compatibility"]["expected_reward_contract"], "signed_progress_sustained_speed_v2")
        self.assertTrue(legacy["args"].reset_exploration_std)

    def test_contact_diagnostics_options_are_parsed_and_validated_before_simulator(self):
        namespace = self.runner_arguments(["--mode=train", "--contact_diagnostics",
                                           "--contact_diagnostics_output=C:/diagnostic log/failure.jsonl",
                                           "--contact_diagnostics_max_records=8", "--contact_diagnostics_per_step=2"])
        self.assertTrue(namespace["args"].contact_diagnostics)
        self.assertEqual(namespace["args"].contact_diagnostics_output, "C:/diagnostic log/failure.jsonl")
        self.assertEqual((namespace["args"].contact_diagnostics_max_records, namespace["args"].contact_diagnostics_per_step), (8, 2))
        for value in ("--contact_diagnostics_max_records=0", "--contact_diagnostics_per_step=-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.runner_arguments(["--mode=train", value])

    def create_checkpoint(self, root, profile, *, task="Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2", saved_profile=None):
        spec = importlib.util.spec_from_file_location("entrypoint_checkpoint_fixture", SCRIPTS / "joint_angle_checkpoint.py")
        checkpoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checkpoint)
        observations, actions = checkpoint.expected_dimensions(task, "flat")
        path = Path(root) / "model_100.pt"
        torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(4, observations),
                                        "actor.6.weight": torch.zeros(actions, 4),
                                        "std": torch.full((actions,), 0.02)}}, path)
        metadata = {"task": task, "observations": observations, "actions": actions,
                    "action_semantics": checkpoint.ACTION_SEMANTICS,
                    "actuator_contract": checkpoint.expected_actuator_contract(task),
                    "cad_asset_sha256": checkpoint.expected_cad_asset_sha256(task),
                    **checkpoint.SCHEMA_CONTRACTS, **checkpoint.BEHAVIOR_PROFILES[profile],
                    "behavior_profile": saved_profile or profile,
                    "target_rate_limit_rad_s": 1.5, "target_smoothing": 0.35,
                    "control_frequency_hz": 50, "exploration_std_cap": 0.03}
        (path.parent / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def test_old_cad6_checkpoints_do_not_silently_restore_historical_behavior(self):
        with TemporaryDirectory() as root:
            for profile, saved_profile in (("revised_v7", None), ("revised_v7_com", "revised_v7")):
                path = self.create_checkpoint(root, profile, saved_profile=saved_profile)
                for mode in ("evaluate", "play"):
                    with self.subTest(profile=profile, mode=mode):
                        with self.assertRaisesRegex(ValueError, f"--behavior_profile={profile}.*historical comparison"):
                            self.runner_arguments([f"--mode={mode}", f"--checkpoint={path}"])
                        selected = self.runner_arguments([f"--mode={mode}", f"--checkpoint={path}",
                                                          f"--behavior_profile={profile}"])
                        self.assertEqual(selected["args"].behavior_profile, profile)
                        self.assertEqual((selected["saved_metadata"]["observations"],
                                          selected["saved_metadata"]["actions"]), (140, 18))

    def test_old_cad6_training_requires_explicit_behavior_transfer(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "revised_v7")
            arguments = ["--mode=train", f"--checkpoint={path}"]
            with self.assertRaisesRegex(ValueError, "allow_behavior_transfer"):
                self.runner_arguments(arguments)
            transferred = self.runner_arguments([*arguments, "--allow_behavior_transfer"])
            self.assertEqual(transferred["args"].behavior_profile, "revised_v8")
            self.assertTrue(transferred["checkpoint_compatibility"]["allow_behavior_transfer"])
            self.assertEqual(transferred["saved_metadata"]["behavior_profile"], "revised_v7")

    def test_strict_cad6_checkpoint_replays_with_default_guards(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "revised_v8")
            for mode in ("evaluate", "play"):
                with self.subTest(mode=mode):
                    selected = self.runner_arguments([f"--mode={mode}", f"--checkpoint={path}"])
                    self.assertEqual(selected["args"].behavior_profile, "revised_v8")
                    self.assertEqual(selected["saved_metadata"]["behavior_profile"], "revised_v8")

    def test_other_robots_preserve_saved_evaluation_profile_inference(self):
        task = "Isaac-DKSH-Spider-JointAngles-v2"
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "legacy_v2", task=task)
            for mode in ("evaluate", "play"):
                with self.subTest(mode=mode):
                    selected = self.runner_arguments([f"--mode={mode}", f"--task={task}", f"--checkpoint={path}"])
                    self.assertIsNone(selected["args"].behavior_profile)
                    self.assertEqual(selected["checkpoint_compatibility"], {})
                    self.assertEqual(selected["saved_metadata"]["behavior_profile"], "legacy_v2")

    def test_previous_toe_button_profiles_remain_explicitly_selectable(self):
        for profile in ("revised_v4", "revised_v5", "revised_v6"):
            with self.subTest(profile=profile):
                train = self.runner_arguments(["--mode=train", f"--behavior_profile={profile}"])
                self.assertEqual(train["args"].behavior_profile, profile)
                self.assertEqual(train["checkpoint_compatibility"]["expected_contact_contract"],
                                 "cad6_toe_button_hysteresis_v1")

    def test_distal_support_rejects_tasks_without_onshape_cad6_geometry_before_simulator(self):
        for profile in ("revised_v7", "revised_v8"):
            with self.subTest(profile=profile), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                self.runner_arguments(["--mode=train", "--task=Isaac-DKSH-Spider-CAD6-JointAngles-v2",
                                       f"--behavior_profile={profile}"])
            self.assertEqual(result.exception.code, 2)

    def test_runner_rejects_nontraining_noise_options_and_invalid_bounds_before_simulator(self):
        for arguments in (["--mode=evaluate", "--initial_noise_std=.08"], ["--mode=play", "--reset_exploration_std"],
                          ["--mode=evaluate", "--allow_behavior_transfer"], ["--mode=train", "--initial_noise_std=.01"],
                          ["--mode=train", "--max_noise_std=nan"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                self.runner_arguments(arguments)
            self.assertEqual(result.exception.code, 2)

    def test_curriculum_defaults_strict_support_and_disjoint_frozen_evaluation(self):
        train = self.runner_arguments(["--mode=train", "--environment=curriculum", "--difficulty=0"])["args"]
        self.assertEqual((train.behavior_profile, train.course_seed, train.course_split, train.flat_fraction),
                         ("revised_v8", 42, "train", 0.25))
        evaluation = self.runner_arguments(["--mode=evaluate", "--environment=curriculum"])["args"]
        self.assertEqual((evaluation.course_seed, evaluation.course_split), (73, "holdout"))
        self.assertEqual(evaluation.behavior_profile, "revised_v8")
        historical = self.runner_arguments(["--mode=train", "--environment=curriculum",
                                            "--behavior_profile=revised_v7"])["args"]
        self.assertEqual(historical.behavior_profile, "revised_v7")

    def test_curriculum_invalid_profiles_splits_and_settings_fail_before_simulator(self):
        for arguments in (["--mode=train", "--environment=curriculum", "--behavior_profile=revised_v6"],
                          ["--mode=train", "--environment=curriculum", "--course_split=holdout"],
                          ["--mode=evaluate", "--environment=curriculum", "--course_split=train"],
                          ["--mode=train", "--environment=curriculum", "--flat_fraction=0"],
                          ["--mode=train", "--environment=curriculum", "--curriculum_window=5"],
                          ["--mode=train", "--environment=curriculum", "--curriculum_promote_threshold=nan"],
                          ["--mode=train", "--course_seed=7"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                self.runner_arguments(arguments)
            self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
