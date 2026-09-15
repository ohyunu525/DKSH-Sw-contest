[CmdletBinding()]
param(
    [ValidateSet('train', 'check', 'play')][string]$Mode = 'train',
    [ValidateRange(1, 128)][int]$NumEnvs = 32,
    [ValidateRange(1, 100000)][int]$MaxIterations = 2000,
    [string]$Checkpoint = '',
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
$taskName = 'Isaac-DKSH-MG90S-Walk-Direct-v0'
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'check') {
        $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\check_mg90s.py'
        $arguments = @('-p', $script, "--num_envs=$NumEnvs", '--steps=1000', '--output=logs/mg90s_preflight.json')
    }
    elseif ($Mode -eq 'play') {
        if (-not $Checkpoint) { throw 'Supply -Checkpoint with an MG90S 86-observation model.' }
        if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint missing: $Checkpoint" }
        $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\play.py'
        $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
        $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", "--checkpoint=$checkpointPath", '--real-time', '--disable_fabric')
    }
    else {
        if ($Checkpoint) { throw 'This launcher starts a fresh MG90S policy. Old navigation checkpoints are incompatible.' }
        $script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\train.py'
        $arguments = @('-p', $script, "--task=$taskName", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations", '--seed=42')
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
