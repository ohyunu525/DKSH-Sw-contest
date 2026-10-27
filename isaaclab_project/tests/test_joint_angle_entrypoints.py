"""Entry routing and retired-task rejection before a simulator can start."""

import contextlib
import ast
import importlib.util
import io
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch


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
        return {"dksh_isaaclab": types.ModuleType("dksh_isaaclab"), "environment_cli": environment_cli,
                "isaaclab": isaac, "isaaclab.app": app}

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
        for name in ("joint_angle_checkpoint", "joint_angle_runner_utils"):
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

    def test_runner_defaults_cad6_training_to_toe_button_profile_and_infers_evaluation_profile(self):
        train = self.runner_arguments(["--mode=train"])
        self.assertEqual(train["args"].behavior_profile, "revised_v4")
        self.assertEqual(train["checkpoint_compatibility"]["expected_contact_contract"], "cad6_toe_button_hysteresis_v1")
        evaluation = self.runner_arguments(["--mode=evaluate"])
        self.assertIsNone(evaluation["args"].behavior_profile)
        self.assertEqual(evaluation["checkpoint_compatibility"], {})
        self.assertEqual(evaluation["args"].seed, 73)
        legacy = self.runner_arguments(["--mode=train", "--behavior_profile=legacy_v2", "--reset_exploration_std"])
        self.assertEqual(legacy["checkpoint_compatibility"]["expected_reward_contract"], "signed_progress_sustained_speed_v2")
        self.assertTrue(legacy["args"].reset_exploration_std)

    def test_runner_rejects_nontraining_noise_options_and_invalid_bounds_before_simulator(self):
        for arguments in (["--mode=evaluate", "--initial_noise_std=.08"], ["--mode=play", "--reset_exploration_std"],
                          ["--mode=evaluate", "--allow_behavior_transfer"], ["--mode=train", "--initial_noise_std=.01"],
                          ["--mode=train", "--max_noise_std=nan"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                self.runner_arguments(arguments)
            self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
