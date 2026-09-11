[CmdletBinding()]
param(
    [ValidateSet("smoke", "isaac-smoke", "train", "evaluate", "play", "viewer", "cartpole")]
    [string]$Mode = "smoke",
    [ValidateRange(1, 128)]
    [int]$NumEnvs = 4,
    [ValidateRange(1, 100000)]
    [int]$Steps = 32,
    [ValidateRange(1, 100000)]
    [int]$MaxIterations = 1000,
    [string]$Checkpoint = "",
    [ValidateSet('baseline', 'cad8', 'cad6')]
    [string]$RobotModel = 'baseline'
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"
$taskName = "Isaac-DKSH-Spider-Navigation-Direct-v0"
$experimentName = 'dksh_spider_navigation'
if ($RobotModel -ne 'baseline') {
    $cadCount = $RobotModel.Substring(3)
    $taskName = "Isaac-DKSH-Spider-CAD$cadCount-Navigation-Direct-v0"
    $experimentName = "dksh_spider_cad${cadCount}_navigation"
}

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is missing. Run .\setup_isaaclab.ps1 first."
}

Push-Location $projectRoot
try {
    switch ($Mode) {
        "smoke" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\smoke_env.py"
            $output = @(& $launcher -p $scriptPath --headless "--task=$taskName" "--num_envs=$NumEnvs" "--steps=$Steps" 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "DKSH_ISAACLAB_SMOKE_PASS")) {
                throw "The custom environment smoke test failed (native exit code $nativeExitCode)."
            }
            cmd.exe /d /c exit 0
        }
        "isaac-smoke" {
            $scriptPath = Join-Path $projectRoot "scripts\isaaclab_smoke.py"
            $output = @(& $launcher -p $scriptPath --headless "--steps=$Steps" 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "ISAAC_LAB_SMOKE_TEST_PASS")) {
                throw "Isaac Lab smoke test failed (native exit code $nativeExitCode)."
            }
            # Isaac Sim 4.5 can return 1 after a clean headless Kit shutdown.
            cmd.exe /d /c exit 0
        }
        "train" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\train.py"
            $logRoot = Join-Path $projectRoot "logs\rsl_rl\$experimentName"
            $latestCheckpointBefore = Get-ChildItem $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            & $launcher -p $scriptPath "--task=$taskName" "--num_envs=$NumEnvs" `
                "--max_iterations=$MaxIterations"
            $nativeExitCode = $LASTEXITCODE
            $latestCheckpointAfter = Get-ChildItem $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $createdCheckpoint = $null -ne $latestCheckpointAfter -and (
                $null -eq $latestCheckpointBefore -or
                $latestCheckpointAfter.LastWriteTime -gt $latestCheckpointBefore.LastWriteTime
            )
            if (-not $createdCheckpoint) {
                throw "Training failed before producing a checkpoint (native exit code $nativeExitCode)."
            }
            if ($createdCheckpoint) {
                Write-Host "Training checkpoint: $($latestCheckpointAfter.FullName)"
                cmd.exe /d /c exit 0
            }
        }
        "evaluate" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\evaluate.py"
            $arguments = @(
                "-p", $scriptPath, "--headless", "--task=$taskName",
                "--num_envs=$NumEnvs", "--steps=$Steps"
            )
            if ($Checkpoint) {
                $arguments += "--checkpoint=$Checkpoint"
            }
            $output = @(& $launcher @arguments 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "DKSH_ISAACLAB_EVAL_PASS")) {
                throw "Checkpoint evaluation failed (native exit code $nativeExitCode)."
            }
            cmd.exe /d /c exit 0
        }
        "play" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\play.py"
            $arguments = @("-p", $scriptPath, "--task=$taskName", "--num_envs=$NumEnvs")
            if ($Checkpoint) {
                $arguments += "--checkpoint=$Checkpoint"
            }
            else {
                $localCheckpoints = Get-ChildItem (Join-Path $projectRoot "logs\rsl_rl\$experimentName") `
                    -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue
                if (-not $localCheckpoints -and $RobotModel -eq 'baseline') {
                    $bundledCheckpoint = Join-Path $projectRoot "isaaclab_project\checkpoints\balance_baseline.pt"
                    if (Test-Path -LiteralPath $bundledCheckpoint) {
                        $arguments += "--checkpoint=$bundledCheckpoint"
                    }
                }
                elseif (-not $localCheckpoints) {
                    throw "No checkpoint for $RobotModel. Train this model first or provide -Checkpoint."
                }
            }
            & $launcher @arguments
            if ($LASTEXITCODE -eq 1) {
                # Isaac Sim 4.5 commonly reports 1 after the user closes a healthy Kit window.
                cmd.exe /d /c exit 0
            }
            elseif ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab playback exited with code $LASTEXITCODE."
            }
        }
        "viewer" {
            $scriptPath = Join-Path $isaacLabRoot "scripts\tutorials\00_sim\create_empty.py"
            & $launcher -p $scriptPath
            if ($LASTEXITCODE -eq 1) {
                # Isaac Sim 4.5 commonly reports 1 after the user closes a healthy Kit window.
                cmd.exe /d /c exit 0
            }
            elseif ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab viewer exited with code $LASTEXITCODE."
            }
        }
        "cartpole" {
            $scriptPath = Join-Path $isaacLabRoot "scripts\reinforcement_learning\rsl_rl\train.py"
            & $launcher -p $scriptPath `
                "--task=Isaac-Cartpole-Direct-v0" --headless "--num_envs=$NumEnvs"
            if ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab Cartpole training exited with code $LASTEXITCODE."
            }
        }
    }
}
finally {
    Pop-Location
}
