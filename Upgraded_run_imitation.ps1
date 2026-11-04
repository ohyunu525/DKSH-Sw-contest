[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('inspect', 'collect', 'bc', 'evaluate', 'train')][string]$Mode,
    [ValidateRange(1, 4096)][int]$NumEnvs = 4,
    [ValidateRange(1, 1000000)][int]$Steps = 6000,
    [ValidateRange(1, 100000)][int]$Epochs = 100,
    [ValidateRange(1, 100000)][int]$MaxIterations = 1000,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateSet('flat', 'stairs', 'rough', 'narrow', 'vibrating', 'falling_debris', 'mixed')][string]$Environment = 'flat',
    [ValidateRange(0, 1)][double]$Difficulty = 0,
    [string]$Archive = 'HexapodRobot.zip',
    [string]$SupplementArchive = 'HexapodRobot2.zip',
    [string]$Dataset = '',
    [string]$Checkpoint = '',
    [string]$Output = '',
    [string]$Device = '',
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'isaaclab_project\scripts\Upgraded_imitation_runner.py'
$arguments = @($scriptPath, "--mode=$Mode", "--archive=$Archive", "--supplement_archive=$SupplementArchive", "--num_envs=$NumEnvs", "--steps=$Steps",
               "--epochs=$Epochs", "--max_iterations=$MaxIterations", "--seed=$Seed", "--environment=$Environment",
               "--difficulty=$($Difficulty.ToString([System.Globalization.CultureInfo]::InvariantCulture))")
if ($Dataset) { $arguments += "--dataset=$Dataset" }
if ($Checkpoint) { $arguments += "--checkpoint=$Checkpoint" }
if ($Output) { $arguments += "--output=$Output" }
if ($Device) { $arguments += "--device=$Device" }
if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
if ($Gui) { $arguments += '--gui' }
$executable = 'python'
if ($Mode -notin @('inspect', 'bc')) {
    $executable = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
    if (-not (Test-Path -LiteralPath $executable)) { throw 'Isaac Lab launcher is missing.' }
    $arguments = @('-p') + $arguments
}
Push-Location $PSScriptRoot
try {
    $logFolder = Join-Path $PSScriptRoot 'outputs'
    [void](New-Item -ItemType Directory -Path $logFolder -Force)
    $logPath = Join-Path $logFolder ("Upgraded_imitation_{0}_{1}.log" -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'))
    $marker = "Upgraded_$($Mode.ToUpperInvariant())_COMPLETE"
    $completed = $false
    $failed = $false
    $ErrorActionPreference = 'Continue'
    & $executable @arguments 2>&1 | ForEach-Object {
        $line = [string]$_
        if ($line -like "$marker*") { $completed = $true }
        if ($line -eq 'Upgraded_FAILED') { $failed = $true }
        $line | Tee-Object -FilePath $logPath -Append
    }
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($failed -or -not $completed -or $nativeExitCode -notin @(0, 1)) {
        throw "Upgraded imitation $Mode failed (exit $nativeExitCode). Diagnostic log: $logPath"
    }
    Write-Host "Upgraded log saved to $logPath"
    cmd.exe /d /c exit 0
}
finally {
    $ErrorActionPreference = 'Stop'
    Pop-Location
}
