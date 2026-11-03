[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('search', 'train', 'evaluate', 'play')][string]$Mode,
    [ValidateRange(1, 4096)][int]$NumEnvs = 4,
    [ValidateRange(1, 100000)][int]$MaxIterations = 2000,
    [ValidateRange(4, 100000)][int]$Population = 24,
    [ValidateRange(1, 100000)][int]$Generations = 12,
    [ValidateRange(1, 100000)][int]$Steps,
    [ValidateRange(0, 2147483647)][int]$Seed,
    [ValidateRange(0, 2147483647)][int[]]$ValidationSeeds,
    [ValidateSet('flat', 'narrow', 'vibrating', 'falling_debris', 'rough', 'mixed', 'stairs')][string]$Environment = 'flat',
    [ValidateRange(0, 1)][double]$Difficulty = 0.5,
    [string]$Profile = '',
    [string]$Checkpoint = '',
    [string]$Output = '',
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'IsaacLab\isaaclab.bat'
if (-not (Test-Path -LiteralPath $launcher)) {
    throw 'Isaac Lab is missing. Run .\setup_isaaclab.ps1 first.'
}

$scriptPath = Join-Path $PSScriptRoot 'isaaclab_project\scripts\tripod_residual_runner.py'
$difficultyText = $Difficulty.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$arguments = @('-p', $scriptPath, "--mode=$Mode", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations",
               "--environment=$Environment", "--difficulty=$difficultyText", "--population=$Population", "--generations=$Generations")
if ($Profile) { $arguments += "--profile=$Profile" }
if ($Checkpoint) { $arguments += "--checkpoint=$Checkpoint" }
if ($Output) { $arguments += "--output=$Output" }
if ($PSBoundParameters.ContainsKey('Steps')) { $arguments += "--steps=$Steps" }
if ($PSBoundParameters.ContainsKey('Seed')) { $arguments += "--seed=$Seed" }
if ($ValidationSeeds) { $arguments += @('--validation_seeds') + $ValidationSeeds }
if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
if ($Mode -ne 'play' -and -not $Gui) { $arguments += '--headless' }

Push-Location $PSScriptRoot
try {
    $marker = "DKSH_CAD6_HYBRID_$($Mode.ToUpperInvariant())_COMPLETE"
    $completed = $false
    & $launcher @arguments 2>&1 | ForEach-Object {
        if ([string]$_ -like "$marker*") { $completed = $true }
        Write-Output $_
    }
    $nativeExitCode = $LASTEXITCODE
    if (-not $completed -or $nativeExitCode -notin @(0, 1)) {
        throw "CAD6 hybrid $Mode did not complete (native exit code $nativeExitCode). Verify Isaac Sim and Isaac Lab are installed with .\setup_isaaclab.ps1."
    }
    cmd.exe /d /c exit 0
}
finally {
    Pop-Location
}
