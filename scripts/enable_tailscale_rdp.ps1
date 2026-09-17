[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script from an elevated PowerShell window (Run as administrator).'
    }
}

function Get-TailscaleCli {
    $cli = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    if ($null -ne $cli) { return $cli.Source }

    $installedCli = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
    if (Test-Path -LiteralPath $installedCli) { return $installedCli }

    throw 'Tailscale for Windows is not installed.'
}

Assert-Administrator
$tailscale = Get-TailscaleCli
$tailscaleIPv4 = @(& $tailscale ip -4 | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' }) | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($tailscaleIPv4)) {
    throw 'This computer is not signed in to a Tailscale tailnet.'
}

# Use Windows' RDP configuration API instead of force-restarting TermService.
# Restarting this shared service can fail with access denied on some Windows builds.
$rdpSetting = Get-CimInstance `
    -Namespace 'root\cimv2\TerminalServices' `
    -ClassName Win32_TerminalServiceSetting
$result = Invoke-CimMethod `
    -InputObject $rdpSetting `
    -MethodName SetAllowTSConnections `
    -Arguments @{ AllowTSConnections = 1; ModifyFirewallException = 0 }
if ($result.ReturnValue -ne 0) {
    throw "Windows refused to enable Remote Desktop (error code $($result.ReturnValue))."
}

Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name UserAuthentication -Value 1
if ((Get-Service -Name TermService).Status -ne 'Running') {
    Start-Service -Name TermService
}

# Do not expose RDP to the LAN or internet: accept only Tailscale address ranges.
$ruleName = 'DKSH Tailscale-only RDP'
Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule `
    -Name $ruleName `
    -DisplayName $ruleName `
    -Description 'Allows Remote Desktop only from Tailscale peers.' `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 3389 `
    -RemoteAddress '100.64.0.0/10', 'fd7a:115c:a1e0::/48' `
    -Profile Any | Out-Null

$listener = Get-NetTCPConnection -State Listen -LocalPort 3389 -ErrorAction SilentlyContinue
if ($null -eq $listener) {
    throw 'RDP was enabled, but port 3389 is not listening. Check the Remote Desktop Services event log for the reported service error.'
}

Write-Host "DKSH_TAILSCALE_RDP_READY: mstsc /v:$tailscaleIPv4"
Write-Host 'Sign in using a Windows account and its password; Windows Hello PIN cannot be used for Remote Desktop.'
