"""Exercise the durable Windows controller with stub launchers, never Isaac/GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell controller")
class V9ValidationJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="v9 fixture's spaces ")
        self.root = Path(self.directory.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/run_v9_validation_job.ps1", self.root / "scripts")
        self.job = self.root / "job with spaces"
        self.experiment = self.root / "logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2"
        self.experiment.mkdir(parents=True)
        self.metadata = {
            "status": "completed", "behavior_profile": "revised_v9",
            "reward_contract": "signed_progress_huber_speed_strict_support_v2",
            "contact_contract": "cad6_strict_distal_all_links_v2", "environments": 4,
            "additional_iterations": 10, "non_distal_contact_height_tolerance_m": 0.001,
            "non_distal_ground_clearance_m": 0.008, "source_checkpoint": None,
            "final_checkpoint": "model_9.pt", "completed_episodes": {"episodes": 1},
            "contact_diagnostics": {"path": str(self.root / "contacts.jsonl")},
        }
        (self.root / "metadata-template.json").write_text(json.dumps(self.metadata))
        (self.root / "run_cad6.ps1").write_text(r'''
param([string]$Mode, [string]$Environment, [int]$NumEnvs, [int]$Steps,
 [int]$MaxIterations, [string]$BehaviorProfile, [switch]$ContactDiagnostics,
 [switch]$AllowProvisionalCad)
if ($Mode -eq 'check') {
 if ($env:DKSH_V9_JOB_CASE -ne 'missing_marker') { 'DKSH_ISAACLAB_SMOKE_PASS envs=4 steps=200 resets=0' }
 exit 0
}
if ($env:DKSH_V9_JOB_CASE -ne 'stale') {
 $folder = Join-Path $PSScriptRoot 'logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2/new'
 New-Item -ItemType Directory -Path $folder -Force | Out-Null
 $meta = Get-Content (Join-Path $PSScriptRoot 'metadata-template.json') -Raw | ConvertFrom-Json
 if ($env:DKSH_V9_JOB_CASE -eq 'wrong_contract') { $meta.non_distal_contact_height_tolerance_m = 0.008 }
 $meta | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $folder 'run_metadata.json')
 'checkpoint' | Set-Content (Join-Path $folder 'model_9.pt')
}
'DKSH_JOINT_ANGLE_TRAINING_COMPLETE new'
''', encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def launch(self, case="clean", *arguments):
        environment = os.environ.copy()
        environment["DKSH_V9_JOB_CASE"] = case
        status_path = self.job / "status.json"
        previous_update = json.loads(status_path.read_text())["updated_at_utc"] if status_path.exists() else None
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(self.root / "scripts/run_v9_validation_job.ps1"), "-JobDirectory", str(self.job), *arguments],
            capture_output=True, text=True, timeout=15, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        dispatch = json.loads(result.stdout)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                status = json.loads(Path(dispatch["status_path"]).read_text(encoding="utf-8-sig"))
                if status["updated_at_utc"] != previous_update and status["status"] in {"failed", "validated", "validated_contact_review_required", "completed"}:
                    return status
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        self.fail("Fixture worker failed to finish: " + str(dispatch))

    def test_clean_fresh_run_validates_and_does_not_launch_large_by_default(self):
        status = self.launch()
        self.assertEqual(status["status"], "validated", status)
        self.assertEqual(status["contact_record_count"], 0)
        self.assertEqual(status["strict_contact_record_count"], 0)
        self.assertNotIn("large_worker_pid", status)
        self.assertTrue(Path(status["small_final_checkpoint"]).is_file())

    def test_strict_contact_is_reviewable_and_blocks_automatic_large(self):
        record = {"strict_contact_failure": True, "termination_reasons": ["strict_contact"],
                  "support": {"invalid_legs": [False, True]}, "non_distal_clearance_m": {"femur": 0.02},
                  "contact_pairs": {"pairs": [], "empty_pairs_prove_no_contact": False}}
        (self.root / "contacts.jsonl").write_text(json.dumps(record) + "\n")
        status = self.launch("strict", "-LaunchLargeTraining")
        self.assertEqual(status["status"], "validated_contact_review_required", status)
        self.assertEqual(status["strict_contact_record_count"], 1)
        self.assertNotIn("large_worker_pid", status)
        summary = json.loads(Path(status["contact_summary_path"]).read_text())
        self.assertEqual(summary[0]["support"]["invalid_legs"], [False, True])

    def test_stale_metadata_cannot_validate_new_training(self):
        old = self.experiment / "old"
        old.mkdir()
        (old / "run_metadata.json").write_text(json.dumps(self.metadata))
        status = self.launch("stale")
        self.assertEqual(status["status"], "failed", status)
        self.assertIn("exactly one new metadata", status["error"])

    def test_contract_must_separate_contact_and_clearance(self):
        status = self.launch("wrong_contract")
        self.assertEqual(status["status"], "failed", status)
        self.assertIn("0.001", status["error"])

    def test_missing_smoke_marker_stops_before_training(self):
        status = self.launch("missing_marker")
        self.assertEqual(status["status"], "failed", status)
        self.assertNotIn("small", status)

    def test_explicit_large_stage_runs_fresh_5120_100_without_pair_sensors(self):
        self.assertEqual(self.launch()["status"], "validated")
        lab = self.root / "IsaacLab"
        lab.mkdir()
        (lab / "isaaclab.bat").write_bytes((
            '@echo off\r\necho STUB_ARGS %*\r\n'
            'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0large_stub.ps1"\r\n'
            'exit /b %ERRORLEVEL%\r\n'
        ).encode("ascii"))
        (lab / "large_stub.ps1").write_text(r'''
$repo = Split-Path $PSScriptRoot -Parent
$folder = Join-Path $repo 'logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2/new-large'
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$meta = Get-Content (Join-Path $repo 'metadata-template.json') -Raw | ConvertFrom-Json
$meta.environments = 5120
$meta.additional_iterations = 100
$meta.final_checkpoint = 'model_99.pt'
$meta | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $folder 'run_metadata.json')
'checkpoint' | Set-Content (Join-Path $folder 'model_99.pt')
'DKSH_JOINT_ANGLE_TRAINING_COMPLETE new-large'
''')
        status = self.launch("clean", "-Stage", "Large")
        self.assertEqual(status["status"], "completed", status)
        self.assertEqual(status["large_exit_code"], 0)
        self.assertTrue(Path(status["large_final_checkpoint"]).is_file())
        output = Path(status["large_stdout"]).read_text(encoding="utf-16")
        for expected in ("--num_envs=5120", "--max_iterations=100", "--behavior_profile=revised_v9",
                         "--task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2", "--allow_provisional_cad", "--headless"):
            self.assertIn(expected, output)
        self.assertNotIn("--contact_diagnostics", output)
        repeat = subprocess.run(
            [POWERSHELL, "-NoProfile", "-File", str(self.root / "scripts/run_v9_validation_job.ps1"),
             "-JobDirectory", str(self.job), "-Stage", "Large"], capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn("refusing duplicate", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
