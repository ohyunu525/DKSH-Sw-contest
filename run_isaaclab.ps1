[CmdletBinding()]
param(
    [ValidateSet("smoke", "isaac-smoke", "train", "evaluate", "play", "preview", "viewer", "cartpole")]
    [string]$Mode = "smoke",
    [ValidateRange(1, 128)]
    [int]$NumEnvs = 4,
    [ValidateRange(1, 100000)]
    [int]$Steps = 32,
    [ValidateRange(1, 100000)]
    [int]$MaxIterations = 1000,
    [string]$Checkpoint = "",
    [ValidateSet('baseline', 'cad8', 'cad6')]
    [string]$RobotModel = 'baseline',
    [ValidateSet('flat', 'narrow', 'vibrating', 'falling_debris', 'rough', 'mixed')]
    [string]$Environment = 'flat',
    [ValidateRange(0.0, 1.0)]
    [double]$Difficulty = 0.5,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42
)

$projectRoot = $PSScriptRoot
$Environment = $Environment.ToLowerInvariant()
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"
$taskName = "Isaac-DKSH-Spider-Navigation-Direct-v0"
$experimentName = 'dksh_spider_navigation'
if ($RobotModel -ne 'baseline') {
    $cadCount = $RobotModel.Substring(3)
    $taskName = "Isaac-DKSH-Spider-CAD$cadCount-Navigation-Direct-v0"
    $experimentName = "dksh_spider_cad${cadCount}_navigation"
}
if ($Environment -ne 'flat') {
    $experimentName += '_obstacles'
}
$difficultyText = $Difficulty.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$environmentArguments = @("--environment=$Environment", "--difficulty=$difficultyText")
$runnerEnvironmentArguments = @(
    "--environment=$Environment", "--difficulty=$difficultyText",
    "--experiment_name=$experimentName"
)
if ($PSBoundParameters.ContainsKey('Seed')) {
    $environmentArguments += "--seed=$Seed"
    $runnerEnvironmentArguments += "--seed=$Seed"
}

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is missing. Run .\setup_isaaclab.ps1 first."
}

Push-Location $projectRoot
try {
    switch ($Mode) {
        "smoke" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\smoke_env.py"
            $output = @(& $launcher -p $scriptPath --headless "--task=$taskName" "--num_envs=$NumEnvs" "--steps=$Steps" @environmentArguments 2>&1)
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
            # Pass a single argument array to the batch launcher.  Inline splatting
            # after positional arguments can lose the value part of Hydra overrides
            # on Windows PowerShell, producing e.g. `env.environment_preset` without
            # its `=rough` value.  Training is intentionally headless on this host.
            $arguments = @(
                "-p", $scriptPath, "--headless", "--task=$taskName",
                "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations"
            ) + $runnerEnvironmentArguments
            $trainingCompleted = $false
            & $launcher @arguments 2>&1 | ForEach-Object {
                if ([string]$_ -match '^DKSH_ISAACLAB_TRAIN_COMPLETE\s*$') {
                    $trainingCompleted = $true
                }
                Write-Output $_
            }
            $nativeExitCode = $LASTEXITCODE
            $latestCheckpointAfter = Get-ChildItem $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $createdCheckpoint = $null -ne $latestCheckpointAfter -and (
                $null -eq $latestCheckpointBefore -or
                $latestCheckpointAfter.LastWriteTime -gt $latestCheckpointBefore.LastWriteTime
            )
            if (-not $trainingCompleted -or -not $createdCheckpoint -or $nativeExitCode -notin @(0, 1)) {
                throw "Training did not complete cleanly or produce a new checkpoint (native exit code $nativeExitCode, completion marker $trainingCompleted)."
            }
            Write-Host "Training checkpoint: $($latestCheckpointAfter.FullName)"
            # Isaac Sim 4.5 can return 1 after the completed trainer shuts Kit down.
            cmd.exe /d /c exit 0
        }
        "evaluate" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\evaluate.py"
            $arguments = @(
                "-p", $scriptPath, "--headless", "--task=$taskName",
                "--num_envs=$NumEnvs", "--steps=$Steps"
            ) + $environmentArguments
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
            # GUI playback must use USD I/O.  Fabric is faster but does not keep
            # the USD viewport synchronized with the simulated articulation.
            $arguments = @("-p", $scriptPath, "--disable_fabric", "--task=$taskName", "--num_envs=$NumEnvs") + $runnerEnvironmentArguments
            if ($Checkpoint) {
                $arguments += "--checkpoint=$Checkpoint"
            }
            else {
                $localCheckpoints = Get-ChildItem (Join-Path $projectRoot "logs\rsl_rl\$experimentName") `
                    -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue
                if (-not $localCheckpoints -and $RobotModel -eq 'baseline' -and $Environment -eq 'flat') {
                    $bundledCheckpoint = Join-Path $projectRoot "isaaclab_project\checkpoints\balance_baseline.pt"
                    if (Test-Path -LiteralPath $bundledCheckpoint) {
                        $arguments += "--checkpoint=$bundledCheckpoint"
                    }
                }
                elseif (-not $localCheckpoints) {
                    throw "No compatible checkpoint for $RobotModel / $Environment. Use -Mode preview to inspect the environment, train first, or provide -Checkpoint."
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
        "preview" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\preview_env.py"
            $arguments = @("-p", $scriptPath, "--task=$taskName", "--num_envs=$NumEnvs") + $environmentArguments
            if ($PSBoundParameters.ContainsKey('Steps')) {
                $arguments += "--steps=$Steps"
            }
            $previewPassed = $false
            & $launcher @arguments 2>&1 | ForEach-Object {
                if ([string]$_ -match '^DKSH_ISAACLAB_PREVIEW_PASS\s*$') {
                    $previewPassed = $true
                }
                Write-Output $_
            }
            $nativeExitCode = $LASTEXITCODE
            if (-not $previewPassed -or $nativeExitCode -notin @(0, 1)) {
                throw "Isaac Lab environment preview failed (native exit code $nativeExitCode)."
            }
            # Normalize Isaac Sim 4.5's shutdown code only after successful playback.
            cmd.exe /d /c exit 0
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
