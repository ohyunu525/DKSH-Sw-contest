[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Configure', 'Status', 'Disable')]
    [string]$Action = 'Status',

    [ValidateRange(1, 65535)]
    [int]$Port = 10000
)

$ErrorActionPreference = 'Stop'

function Get-TailscaleCli {
    $command = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $command = Get-Command tailscale -ErrorAction SilentlyContinue
    }

    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Tailscale\tailscale.exe')
    )
    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw 'Tailscale is not installed. Install Tailscale for Windows, sign in to the same tailnet on both machines, then run this script again from an elevated PowerShell window.'
}

function Get-TailscaleIPv4([string]$Cli) {
    $ipv4 = @(& $Cli ip -4 | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' }) | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not read Tailscale status. Make sure the Tailscale Windows service is running and this terminal is elevated.'
    }

    return $ipv4
}

$tailscale = Get-TailscaleCli
$ipv4 = Get-TailscaleIPv4 $tailscale

if ([string]::IsNullOrWhiteSpace($ipv4)) {
    throw 'This machine is not connected to a tailnet yet. Sign in through the Tailscale app, then retry.'
}

switch ($Action) {
    'Configure' {
        # The ROS endpoint remains bound to 127.0.0.1 in Docker. Serve is the only
        # network-facing listener, so the bridge is reachable only inside the tailnet.
        & $tailscale serve --bg "--tcp=$Port" "tcp://127.0.0.1:$Port"
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to configure Tailscale Serve. Confirm that Serve is permitted for this tailnet and rerun from an elevated PowerShell window.'
        }

        Write-Host "DKSH_TAILSCALE_ROS_READY: tcp://$ipv4`:$Port"
        Write-Host 'Use this Tailscale IPv4 address and port in Unity on the remote machine.'
        & $tailscale serve status
    }
    'Status' {
        Write-Host "Tailscale IPv4: $ipv4"
        Write-Host "Expected ROS TCP endpoint: tcp://127.0.0.1:$Port"
        & $tailscale serve status
    }
    'Disable' {
        & $tailscale serve "--tcp=$Port" off
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to remove the Tailscale Serve TCP forwarder.'
        }
        Write-Host "DKSH_TAILSCALE_ROS_DISABLED: tcp://$ipv4`:$Port"
    }
}
