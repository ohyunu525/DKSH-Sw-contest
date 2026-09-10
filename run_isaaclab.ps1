[CmdletBinding()]
param(
    [ValidateSet("smoke", "train", "evaluate", "play")]
    [string]$Mode = "smoke",
    [ValidateRange(1, 128)]
    [int]$NumEnvs = 4,
    [ValidateRange(1, 100000)]
    [int]$Steps = 32,
    [ValidateRange(1, 100000)]
    [int]$MaxIterations = 1000,
    [string]$Checkpoint = ""
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"
$taskName = "Isaac-DKSH-Spider-Navigation-Direct-v0"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is missing. Run .\setup_isaaclab.ps1 first."
}

Push-Location $projectRoot
try {
    switch ($Mode) {
        "smoke" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\smoke_env.py"
            $output = @(& $launcher -p $scriptPath --headless "--num_envs=$NumEnvs" "--steps=$Steps" 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "DKSH_ISAACLAB_SMOKE_PASS")) {
                throw "The custom environment smoke test failed (native exit code $nativeExitCode)."
            }
            cmd.exe /d /c exit 0
        }
        "train" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\train.py"
            $logRoot = Join-Path $projectRoot "logs\rsl_rl\dksh_spider_navigation"
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
                $localCheckpoints = Get-ChildItem (Join-Path $projectRoot "logs\rsl_rl\dksh_spider_navigation") `
                    -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue
                if (-not $localCheckpoints) {
                    $bundledCheckpoint = Join-Path $projectRoot "isaaclab_project\checkpoints\balance_baseline.pt"
                    if (Test-Path -LiteralPath $bundledCheckpoint) {
                        $arguments += "--checkpoint=$bundledCheckpoint"
                    }
                }
            }
            & $launcher @arguments
            if ($LASTEXITCODE -eq 1) {
                # Isaac Sim 4.5 commonly reports 1 after the user closes a healthy Kit window.
                cmd.exe /d /c exit 0
            }
        }
    }
}
finally {
    Pop-Location
}
