[CmdletBinding()]
param(
    [ValidateSet('train', 'evaluate')][string]$Mode = 'train',
    [ValidateRange(1, 64)][int]$NumEnvs = 16,
    [ValidateRange(1, 100000)][int]$MaxIterations = 300,
    [string]$Checkpoint = '',
    [ValidateRange(0, 8)][int]$Level = 0,
    [ValidateRange(0.05, 2.0)][double]$TreadM = 0.25,
    [ValidateRange(0.05, 2.0)][double]$StairWidthM = 1.2,
    [ValidateSet('legacy_v1', 'staged_v1', 'staged_15cm_v2')][string]$Strategy = 'legacy_v1',
    [ValidateRange(4100, 1000000)][int]$Steps = 4100,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [switch]$AllowProvisionalCad,
    [switch]$AllowCurriculumTransfer,
    [switch]$Gui
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
$script = Join-Path $PSScriptRoot 'isaaclab_project\scripts\train_stairs.py'
if ($Mode -eq 'evaluate' -and -not $Checkpoint) { throw 'Evaluation requires -Checkpoint.' }
if ($Checkpoint) { $Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path }
if ($Mode -eq 'evaluate' -and -not $PSBoundParameters.ContainsKey('Seed')) { $Seed = 43 }
$arguments = @('-p', $script, "--mode=$Mode", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations",
               "--level=$Level", "--steps=$Steps", "--seed=$Seed",
               "--tread_m=$($TreadM.ToString([cultureinfo]::InvariantCulture))",
               "--stair_width_m=$($StairWidthM.ToString([cultureinfo]::InvariantCulture))",
               "--strategy=$Strategy")
if ($Checkpoint) { $arguments += "--checkpoint=$Checkpoint" }
if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
if ($AllowCurriculumTransfer) { $arguments += '--allow_curriculum_transfer' }
if (-not $Gui) { $arguments += '--headless' }
Push-Location $PSScriptRoot
try {
    $record = $null
    & $launcher @arguments 2>&1 | ForEach-Object {
        $line = [string]$_
        if ($line.StartsWith('STAIRS_RUN_COMPLETE ')) {
            $record = $line.Substring('STAIRS_RUN_COMPLETE '.Length) | ConvertFrom-Json
        }
        Write-Output $_
    }
    $nativeExitCode = $LASTEXITCODE
    # Kit 4.5 may return 1 on shutdown. Require structured completion evidence.
    if ($null -eq $record -or $record.status -ne 'completed' -or $nativeExitCode -notin @(0, 1)) {
        throw "Stair $Mode did not complete; inspect run_metadata.json (exit $nativeExitCode)."
    }
    if ($Mode -eq 'evaluate' -and -not $record.evaluation.passed) {
        throw 'Stair climbing evaluation FAILED. A saved model is not proof of climbing.'
    }
    if ($Mode -eq 'train') { Write-Output 'PPO completed; independent stair evaluation is still required.' }
    # The Isaac wrapper can exit 1 after writing a valid completion record.
    # Do not leak that accepted native exit code to callers of this script.
    $global:LASTEXITCODE = 0
}
finally { Pop-Location }
