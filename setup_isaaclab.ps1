[CmdletBinding()]
param(
    [string]$IsaacSimPath = "C:\isaac-sim-standalone-4.5.0-windows-x86_64"
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$simPython = Join-Path $IsaacSimPath "python.bat"
$simLink = Join-Path $isaacLabRoot "_isaac_sim"

if (-not (Test-Path -LiteralPath $isaacLabRoot)) {
    throw "IsaacLab is missing. Run: git submodule update --init --recursive"
}

if (-not (Test-Path -LiteralPath $simPython)) {
    throw "Isaac Sim 4.5 was not found at '$IsaacSimPath'. Install it first or pass -IsaacSimPath with its absolute path."
}

if (-not (Test-Path -LiteralPath $simLink)) {
    New-Item -ItemType Junction -Path $simLink -Target $IsaacSimPath | Out-Null
}

Push-Location $isaacLabRoot
try {
    # Do not use `isaaclab.bat -i`: it also installs Isaac Lab Mimic, which is
    # unnecessary for this project and can conflict with the bundled Python.
    & $simPython -m pip install --editable ".\source\isaaclab"
    & $simPython -m pip install --editable ".\source\isaaclab_assets"
    & $simPython -m pip install --editable ".\source\isaaclab_tasks" --no-deps
    & $simPython -m pip install --editable ".\source\isaaclab_rl[rsl_rl]"

    # Isaac Lab 2.1.0 leaves Warp unconstrained. These versions match Isaac Sim 4.5.
    & $simPython -m pip install --upgrade --force-reinstall "numpy==1.26.4" "warp-lang==1.5.0"

    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab dependency installation failed with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Isaac Lab setup complete. Verify it with: .\run_isaaclab.ps1"
