[CmdletBinding()]
param(
    [string]$IsaacSimPath = "C:\isaac-sim-standalone-4.5.0-windows-x86_64"
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$simPython = Join-Path $IsaacSimPath "python.bat"
$simLink = Join-Path $isaacLabRoot "_isaac_sim"
$extensionRoot = Join-Path $projectRoot "isaaclab_project\source\dksh_isaaclab"

function Install-IsaacPythonPackage {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & $simPython -m pip install @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Sim Python package installation failed: pip install $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $isaacLabRoot "isaaclab.bat"))) {
    throw "IsaacLab is missing. Run: git submodule update --init --recursive"
}
if (-not (Test-Path -LiteralPath $simPython)) {
    throw "Isaac Sim 4.5 was not found at '$IsaacSimPath'. Pass its absolute directory with -IsaacSimPath."
}
if (-not (Test-Path -LiteralPath $simLink)) {
    New-Item -ItemType Junction -Path $simLink -Target $IsaacSimPath | Out-Null
}
else {
    $existingTarget = (Get-Item -LiteralPath $simLink).Target
    $requestedTarget = [IO.Path]::GetFullPath($IsaacSimPath).TrimEnd('\')
    if (-not $existingTarget -or [IO.Path]::GetFullPath($existingTarget).TrimEnd('\') -ne $requestedTarget) {
        throw "'$simLink' already exists but does not point to '$requestedTarget'. Remove or correct it first."
    }
}

Push-Location $isaacLabRoot
try {
    # Do not use `isaaclab.bat -i`: it also installs Isaac Lab Mimic, which is
    # unnecessary for this project and can conflict with the bundled Python.
    # Isaac Lab 2.1.0 leaves Warp unconstrained, so pin versions compatible with Isaac Sim 4.5.
    Install-IsaacPythonPackage @("--upgrade", "numpy==1.26.4", "warp-lang==1.5.0")
    Install-IsaacPythonPackage @("--editable", ".\source\isaaclab")
    Install-IsaacPythonPackage @("--editable", ".\source\isaaclab_assets")
    Install-IsaacPythonPackage @("--editable", ".\source\isaaclab_tasks", "--no-deps")
    Install-IsaacPythonPackage @("--editable", ".\source\isaaclab_rl[rsl_rl]")
    Install-IsaacPythonPackage @("--editable", $extensionRoot)
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Run: .\run_isaaclab.ps1 -Mode smoke"
