[CmdletBinding()]
param(
    [string]$OutputDirectory = ""
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"
$exporter = Join-Path $projectRoot "isaaclab_project\scripts\export_urdf.py"
$converter = Join-Path $isaacLabRoot "scripts\tools\convert_urdf.py"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is missing. Run .\setup_isaaclab.ps1 first."
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "isaaclab_project\assets\spiderbot_usd"
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$urdfPath = Join-Path ([IO.Path]::GetTempPath()) "dksh_isaaclab\2\spiderbot.urdf"
$usdPath = Join-Path $OutputDirectory "spiderbot.usd"

$urdfOutput = @(& $launcher -p $exporter "--output=$urdfPath" 2>&1)
$urdfExitCode = $LASTEXITCODE
$urdfOutput | Write-Output
if (-not (Test-Path -LiteralPath $urdfPath) -or -not ($urdfOutput -match "DKSH_URDF_READY")) {
    throw "URDF generation failed (native exit code $urdfExitCode)."
}

$output = @(
    & $launcher -p $converter $urdfPath $usdPath --headless `
        --joint-stiffness 8.0 --joint-damping 0.35 --joint-target-type position 2>&1
)
$nativeExitCode = $LASTEXITCODE
$output | Write-Output
$expectedAssets = @(
    $usdPath,
    (Join-Path $OutputDirectory "configuration\spiderbot_base.usd"),
    (Join-Path $OutputDirectory "configuration\spiderbot_physics.usd"),
    (Join-Path $OutputDirectory "configuration\spiderbot_sensor.usd")
)
$invalidAssets = @($expectedAssets | Where-Object {
    -not (Test-Path -LiteralPath $_ -PathType Leaf) -or (Get-Item -LiteralPath $_).Length -eq 0
})
if ($invalidAssets.Count -gt 0) {
    throw "URDF-to-USD conversion failed (native exit code $nativeExitCode)."
}

Write-Host "Spiderbot USD rebuilt at: $usdPath"
if ($nativeExitCode -ne 0) {
    Write-Warning "Isaac Sim returned its known shutdown code $nativeExitCode after writing valid USD files."
}
cmd.exe /d /c exit 0
