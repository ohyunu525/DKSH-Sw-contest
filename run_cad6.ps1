[CmdletBinding()]
param(
    [ValidateSet('train', 'check', 'evaluate', 'eval', 'play')][string]$Mode = 'train',
    [ValidateRange(1, 128)][int]$NumEnvs = 32,
    [ValidateRange(1, 100000)][int]$MaxIterations = 2000,
    [string]$Checkpoint = '',
    [ValidateSet('flat', 'narrow', 'vibrating', 'falling_debris', 'rough', 'mixed', 'stairs')][string]$Environment = 'flat',
    [ValidateRange(0, 1)][double]$Difficulty = 0.5,
    [ValidateRange(1, 100000)][int]$Steps = 12000,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [string]$Output = '',
    [double]$InitialNoiseStd,
    [double]$MinNoiseStd,
    [double]$MaxNoiseStd,
    [switch]$ResetExplorationStd,
    [ValidateSet('legacy_v2', 'revised_v3')][string]$BehaviorProfile,
    [switch]$AllowBehaviorTransfer,
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$ErrorActionPreference = 'Stop'
$directTrainingKeys = @('InitialNoiseStd', 'MinNoiseStd', 'MaxNoiseStd', 'ResetExplorationStd', 'AllowBehaviorTransfer')
if (@($directTrainingKeys | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0 -and $Mode -ne 'train') {
    throw 'Noise std overrides, exploration reset and behavior transfer require -Mode train.'
}
if ($PSBoundParameters.ContainsKey('BehaviorProfile') -and $Mode -eq 'check') {
    throw 'BehaviorProfile requires train, evaluate or play mode.'
}
$directMode = switch ($Mode) {
    'check' { 'smoke' }
    'eval' { 'evaluate' }
    default { $Mode }
}
$options = @{
    # CAD6 hardware: six MG996R hip servos and twelve RC920DMG distal servos.
    Mode = $directMode; RobotModel = 'cad6'; ServoModel = 'mg996r_hip'; ControlMode = 'direct'
    Environment = $Environment; Difficulty = $Difficulty; NumEnvs = $NumEnvs
    MaxIterations = $MaxIterations; Checkpoint = $Checkpoint; Output = $Output; Gui = $Gui
    AllowProvisionalCad = $AllowProvisionalCad
}
# Evaluation owns its independent default seed (73); training defaults to 42.
if ($PSBoundParameters.ContainsKey('Seed')) { $options.Seed = $Seed }
foreach ($key in $directTrainingKeys + @('BehaviorProfile')) {
    if ($PSBoundParameters.ContainsKey($key)) { $options[$key] = $PSBoundParameters[$key] }
}
if ($Mode -eq 'check') { $options.Steps = if ($PSBoundParameters.ContainsKey('Steps')) { $Steps } else { 1000 } }
elseif ($directMode -eq 'evaluate' -or $PSBoundParameters.ContainsKey('Steps')) { $options.Steps = $Steps }
& (Join-Path $PSScriptRoot 'run_isaaclab.ps1') @options
