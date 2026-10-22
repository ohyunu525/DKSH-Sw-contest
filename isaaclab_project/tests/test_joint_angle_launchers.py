"""Run Windows launchers against a stub batch file; no Isaac process starts."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell launcher test")
class JointAngleLauncherTests(unittest.TestCase):
    def setUp(self):
        cache = (ROOT / ".uv-cache/joint-angle-launcher-tests").resolve()
        if not cache.is_relative_to(ROOT.resolve()):
            raise RuntimeError("Launcher fixture must remain in the workspace")
        cache.mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(prefix="fixture with spaces ", dir=cache)
        self.fixture = Path(self.directory.name)
        for name in ("run_isaaclab.ps1", "run_mg90s.ps1"):
            shutil.copy2(ROOT / name, self.fixture / name)
        (self.fixture / "IsaacLab").mkdir()
        lines = [
            "@echo off", "echo STUB_ARGS %*",
            'if "%DKSH_V2_TEST_NATIVE_FAIL%"=="1" exit /b 2',
            'if "%DKSH_V2_TEST_NO_MARKER%"=="1" exit /b 0',
            "echo DKSH_JOINT_ANGLE_TRAINING_COMPLETE STUB",
            "echo DKSH_JOINT_ANGLE_EVAL_COMPLETE STUB",
            "echo DKSH_JOINT_ANGLE_PLAY_COMPLETE STUB",
            "echo DKSH_ISAACLAB_SMOKE_PASS STUB",
            "echo MG90S_CAD6_TRAINING_COMPLETE STUB", "exit /b 0",
        ]
        (self.fixture / "IsaacLab/isaaclab.bat").write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))

    def tearDown(self):
        self.directory.cleanup()

    def launch(self, script, *arguments, extra_env=None):
        environment = os.environ.copy()
        environment.update(extra_env or {})
        return subprocess.run([POWERSHELL, "-NoProfile", "-File", str(self.fixture / script), *arguments],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=environment, timeout=30)

    def test_direct_default_routes_to_v2_and_preserves_explicit_paths_and_seed(self):
        result = self.launch("run_mg90s.ps1", "-Mode", "train", "-NumEnvs", "4", "-MaxIterations", "2",
                             "-Checkpoint", "C:/model path/model.pt", "-Seed", "19")
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ("--task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2", "--mode=train",
                         "--num_envs=4", "--max_iterations=2", "C:/model path/model.pt", "--seed=19"):
            self.assertIn(expected, result.stdout)

    def test_evaluation_default_does_not_override_runner_independent_seed(self):
        result = self.launch("run_mg90s.ps1", "-Mode", "evaluate", "-NumEnvs", "4", "-Steps", "12000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode=evaluate", result.stdout)
        self.assertNotIn("--seed=", result.stdout)  # Python's evaluation default is 73.

    def test_gui_choice_and_legacy_v0_path_are_preserved(self):
        gui = self.launch("run_mg90s.ps1", "-Mode", "train", "-Gui", "-NumEnvs", "4")
        self.assertEqual(gui.returncode, 0, gui.stderr)
        self.assertNotIn("--headless", gui.stdout)
        legacy = self.launch("run_mg90s.ps1", "-Mode", "train", "-ControlMode", "legacy", "-NumEnvs", "4")
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertIn("--task=Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0", legacy.stdout)

    def test_native_failure_or_missing_completion_marker_is_not_accepted(self):
        for environment in ({"DKSH_V2_TEST_NATIVE_FAIL": "1"}, {"DKSH_V2_TEST_NO_MARKER": "1"}):
            with self.subTest(environment=environment):
                result = self.launch("run_isaaclab.ps1", "-Mode", "train", extra_env=environment)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
