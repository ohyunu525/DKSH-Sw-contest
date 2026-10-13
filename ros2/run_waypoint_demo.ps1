[CmdletBinding()]
param(
    [ValidateSet('Server', 'Test')][string]$Mode = 'Server',
    [string]$PythonPath = '',
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
    else {
        $pythonCommand = Get-Command python3, python -ErrorAction SilentlyContinue |
            Where-Object { $_.Source -notlike '*WindowsApps*' } | Select-Object -First 1
        if (-not $pythonCommand) { throw 'Python 3.10+ is required. Pass its executable with -PythonPath.' }
        $PythonPath = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Python executable was not found.' }
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'dksh_remote_navigation'
    if ($Mode -eq 'Test') {
        & $PythonPath -m unittest discover -s (Join-Path $PSScriptRoot 'dksh_remote_navigation/tests') -v
    } else {
        & $PythonPath (Join-Path $PSScriptRoot 'tools/waypoint_gateway.py') --host $BindAddress --port $Port
    }
    if ($LASTEXITCODE -ne 0) { throw "Waypoint $Mode failed ($LASTEXITCODE)." }
} finally { $env:PYTHONPATH = $oldPythonPath }
