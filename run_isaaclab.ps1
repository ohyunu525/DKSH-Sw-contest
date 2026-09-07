[CmdletBinding()]
param(
    [ValidateSet("smoke", "viewer", "cartpole")]
    [string]$Mode = "smoke",

    [ValidateRange(1, 10000)]
    [int]$Steps = 64,

    [ValidateRange(1, 128)]
    [int]$NumEnvs = 32
)

$projectRoot = $PSScriptRoot
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is not installed at '$isaacLabRoot'. Run setup_isaaclab.ps1 first."
}

Push-Location $isaacLabRoot
try {
    $allowKnownSmokeShutdownCode = $false
    switch ($Mode) {
        "smoke" {
            $scriptPath = Join-Path $projectRoot "scripts\isaaclab_smoke.py"
            $smokeOutput = @(& $launcher -p $scriptPath --headless "--steps=$Steps" 2>&1)
            $smokeExitCode = $LASTEXITCODE
            $smokeOutput | Write-Output

            if (-not ($smokeOutput -match "ISAAC_LAB_SMOKE_TEST_PASS")) {
                throw "Isaac Lab smoke test did not reach the physics-step success marker (exit code $smokeExitCode)."
            }

            # Isaac Sim 4.5 on this Windows bundle returns code 1 after a clean
            # headless Kit shutdown. The marker above confirms the simulation ran.
            $allowKnownSmokeShutdownCode = $smokeExitCode -eq 1
        }
        "viewer" {
            & $launcher -p "scripts\tutorials\00_sim\create_empty.py"
        }
        "cartpole" {
            & $launcher -p "scripts\reinforcement_learning\rsl_rl\train.py" `
                "--task=Isaac-Cartpole-Direct-v0" --headless "--num_envs=$NumEnvs"
        }
    }

    if ($LASTEXITCODE -ne 0 -and -not $allowKnownSmokeShutdownCode) {
        throw "Isaac Lab exited with code $LASTEXITCODE."
    }

    if ($allowKnownSmokeShutdownCode) {
        # Reset the native-process status so a verified smoke test succeeds for
        # callers as well as for an interactive PowerShell session.
        cmd.exe /d /c exit 0
    }
}
finally {
    Pop-Location
}
