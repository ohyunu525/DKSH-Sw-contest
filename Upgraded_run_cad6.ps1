[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('check', 'baseline', 'train', 'evaluate', 'play')][string]$Mode,
    [ValidateRange(1, 4096)][int]$NumEnvs = 4,
    [ValidateRange(1, 100000)][int]$MaxIterations = 2000,
    [ValidateRange(1, 100000)][int]$Steps,
    [ValidateRange(0, 2147483647)][int]$Seed,
    [ValidateSet('flat', 'narrow', 'vibrating', 'falling_debris', 'rough', 'mixed', 'stairs')][string]$Environment = 'flat',
    [ValidateRange(0, 1)][double]$Difficulty = 0,
    [string]$Checkpoint = '',
    [string]$BaselineReport = 'outputs/Upgraded_baseline.json',
    [string]$Output = '',
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
$scriptPath = Join-Path $PSScriptRoot 'isaaclab_project\scripts\Upgraded_cad6_runner.py'
if (-not (Test-Path -LiteralPath $launcher)) { throw 'Isaac Lab is missing. Run .\setup_isaaclab.ps1 first.' }
$difficultyText = $Difficulty.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$arguments = @('-p', $scriptPath, "--mode=$Mode", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations",
               "--environment=$Environment", "--difficulty=$difficultyText", "--baseline_report=$BaselineReport")
if ($Checkpoint) { $arguments += "--checkpoint=$Checkpoint" }
if ($Output) { $arguments += "--output=$Output" }
if ($PSBoundParameters.ContainsKey('Steps')) { $arguments += "--steps=$Steps" }
if ($PSBoundParameters.ContainsKey('Seed')) { $arguments += "--seed=$Seed" }
if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
if ($Mode -ne 'play' -and -not $Gui) { $arguments += '--headless' }

Push-Location $PSScriptRoot
try {
    $logFolder = Join-Path $PSScriptRoot 'outputs'
    [void](New-Item -ItemType Directory -Path $logFolder -Force)
    $logPath = Join-Path $logFolder ("Upgraded_{0}_{1}.log" -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'))
    $marker = "Upgraded_$($Mode.ToUpperInvariant())_COMPLETE"
    $completed = $false
    $failed = $false
    # Windows PowerShell 5.1 wraps native stderr as ErrorRecord objects.
    # Keep those diagnostics in the log and decide using explicit markers.
    $ErrorActionPreference = 'Continue'
    & $launcher @arguments 2>&1 | ForEach-Object {
        $line = [string]$_
        if ($line -like "$marker*") { $completed = $true }
        if ($line -eq 'Upgraded_FAILED') { $failed = $true }
        $line | Tee-Object -FilePath $logPath -Append
    }
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($failed -or -not $completed -or $nativeExitCode -notin @(0, 1)) {
        throw "Upgraded $Mode failed (native exit $nativeExitCode). Full diagnostic log: $logPath"
    }
    Write-Host "Upgraded log saved to $logPath"
    cmd.exe /d /c exit 0
}
finally {
    $ErrorActionPreference = 'Stop'
    Pop-Location
}
