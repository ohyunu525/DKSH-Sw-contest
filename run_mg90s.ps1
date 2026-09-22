[CmdletBinding()]
param(
    [ValidateSet('train', 'check', 'play')][string]$Mode = 'train',
    [ValidateRange(1, 128)][int]$NumEnvs = 32,
    [ValidateRange(1, 100000)][int]$MaxIterations = 2000,
    [string]$Checkpoint = '',
    [ValidateSet('cad6', 'legacy8')][string]$RobotModel = 'cad6',
    [ValidateSet('normal', 'sprint', 'velocity')][string]$SpeedProfile = 'velocity',
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
$taskName = 'Isaac-DKSH-MG90S-Walk-Direct-v0'
$observationCount = 86
$actionCount = 24
if ($RobotModel -eq 'cad6') {
    $taskName = 'Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0'
    $observationCount = 68
    $actionCount = 18
}
if ($RobotModel -eq 'cad6' -and $SpeedProfile -eq 'sprint') {
    $taskName = 'Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0'
}
if ($RobotModel -eq 'cad6' -and $SpeedProfile -eq 'velocity') {
    $taskName = 'Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0'
}
if ($RobotModel -eq 'legacy8' -and $SpeedProfile -eq 'velocity') {
    throw 'The velocity command profile is available only for the CAD6 robot.'
}
Push-Location $PSScriptRoot
try {
    if ($Checkpoint) {
        $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint -ErrorAction Stop).Path
        $guard = Join-Path $PSScriptRoot 'isaaclab_project\scripts\mg90s_checkpoint.py'
        # The IsaacLab batch wrapper can return 1 even for a successful plain Python check.
        $pythonLauncher = Join-Path $PSScriptRoot 'IsaacLab\_isaac_sim\python.bat'
        & $pythonLauncher $guard "--checkpoint=$checkpointPath" "--observations=$observationCount" "--actions=$actionCount" "--task=$taskName"
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint is incompatible with the selected MG90S task.' }
    }
    if ($Mode -eq 'check') {
        $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\check_mg90s.py'
        $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", '--steps=1000', "--output=logs/mg90s_${RobotModel}_${SpeedProfile}_preflight_${NumEnvs}.json")
    }
    elseif ($Mode -eq 'play') {
        if (-not $Checkpoint) { throw 'Supply -Checkpoint with a compatible MG90S model.' }
        if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint missing: $Checkpoint" }
        $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\play.py'
        $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
        $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", "--checkpoint=$checkpointPath", '--real-time', '--disable_fabric')
    }
    else {
        if ($RobotModel -eq 'cad6') {
            $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\train_mg90s_cad6.py'
            $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations", '--seed=42')
            if ($Checkpoint) { $arguments += "--checkpoint=$checkpointPath" }
            if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
        }
        else {
            if ($Checkpoint) { throw 'The legacy eight-leg launcher only supports fresh training.' }
            $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\train.py'
            $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations", '--seed=42')
        }
    }
    if (-not $Gui -and $Mode -ne 'play') { $arguments += '--headless' }
    if ($Mode -eq 'check') {
        $preflightPassed = $false
        & $launcher @arguments 2>&1 | ForEach-Object {
            if ([string]$_ -eq 'MG90S_PREFLIGHT_PASS') { $preflightPassed = $true }
            Write-Output $_
        }
        $exitCode = $LASTEXITCODE
        # Kit 4.5 reports 1 on some clean shutdowns. Require the physical pass marker.
        if (-not $preflightPassed -or $exitCode -notin @(0, 1)) {
            throw "MG90S physical preflight failed (exit code $exitCode)."
        }
    }
    elseif ($Mode -eq 'train' -and $RobotModel -eq 'cad6') {
        $trainingCompleted = $false
        & $launcher @arguments 2>&1 | ForEach-Object {
            if ([string]$_ -like 'MG90S_CAD6_TRAINING_COMPLETE *') { $trainingCompleted = $true }
            Write-Output $_
        }
        $exitCode = $LASTEXITCODE
        if (-not $trainingCompleted -or $exitCode -notin @(0, 1)) {
            throw "CAD6 training did not complete (exit code $exitCode). Check the run metadata and logs."
        }
        $exitCode = 0
    }
    else {
        & $launcher @arguments
        $exitCode = $LASTEXITCODE
    }
    if ($Mode -ne 'check' -and $exitCode -ne 0) {
        # Preserve errors; a checkpoint alone does not prove the requested run finished.
        throw "MG90S $Mode process exited with code $exitCode. Check its output."
    }
}
finally {
    Pop-Location
}
