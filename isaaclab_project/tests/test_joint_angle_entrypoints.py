"""Entry routing and retired-task rejection before a simulator can start."""

import contextlib
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


if __name__ == "__main__":
    unittest.main()
