[CmdletBinding()]
param(
    [ValidateSet('Validate', 'Large')][string]$Stage = 'Validate',
    [switch]$LaunchLargeTraining,
    [string]$StatusPath = '',
    [string]$JobDirectory = '',
    [switch]$Worker
)

# Starts a hidden, durable worker and returns a status-file path immediately.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$experiment = Join-Path $repo 'logs\rsl_rl\dksh_mg996r_hip_cad6_jointangles_v2'
$powershell = Join-Path $PSHOME 'powershell.exe'
if (-not (Test-Path -LiteralPath $powershell)) { $powershell = (Get-Process -Id $PID).Path }

function Get-UtcStamp { (Get-Date).ToUniversalTime().ToString('o') }
function Get-AbsolutePath([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repo $Path }
    [IO.Path]::GetFullPath($Path)
}
function Write-Status {
    $script:state.updated_at_utc = Get-UtcStamp
    $json = $script:state | ConvertTo-Json -Depth 40
    foreach ($destination in @($script:statusFile, (Join-Path $script:job 'status.json')) | Select-Object -Unique) {
        $directory = Split-Path $destination -Parent
        [IO.Directory]::CreateDirectory($directory) | Out-Null
        $temporary = Join-Path $directory ('.v9-status-' + [guid]::NewGuid().ToString('N') + '.tmp')
        [IO.File]::WriteAllText($temporary, $json, (New-Object Text.UTF8Encoding($false)))
        # Both paths are explicit files in the same verified directory; no recursive operation.
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    }
}
function Start-Worker([string]$NextStage) {
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PSCommandPath + '"'),
        '-Worker', '-Stage', $NextStage, '-JobDirectory', ('"' + $script:job + '"'),
        '-StatusPath', ('"' + $script:statusFile + '"'))
    if ($LaunchLargeTraining) { $arguments += '-LaunchLargeTraining' }
    Start-Process -FilePath $powershell -ArgumentList $arguments `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $script:job ($NextStage.ToLowerInvariant() + '-worker.stdout.log')) `
        -RedirectStandardError (Join-Path $script:job ($NextStage.ToLowerInvariant() + '-worker.stderr.log'))
}
function Get-MetadataPaths {
    if (Test-Path -LiteralPath $experiment) {
        @(Get-ChildItem -LiteralPath $experiment -Directory | ForEach-Object {
            $candidate = Join-Path $_.FullName 'run_metadata.json'
            if (Test-Path -LiteralPath $candidate) { $candidate }
        })
    }
}
function Invoke-Cad6([string]$Name, [string[]]$Arguments) {
    $stdout = Join-Path $script:job ($Name + '.stdout.log')
    $stderr = Join-Path $script:job ($Name + '.stderr.log')
    $script:state[$Name] = [ordered]@{ started_at_utc = Get-UtcStamp; stdout = $stdout; stderr = $stderr }
    Write-Status
    $wrapper = Join-Path $repo 'run_cad6.ps1'
    $quotedWrapper = '"' + $wrapper + '"'
    $process = Start-Process -FilePath $powershell -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $quotedWrapper) + $Arguments) `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $process.Handle # Keep the Windows process handle until its exit code is read.
    $script:state[$Name].pid = $process.Id
    Write-Status
    $process.WaitForExit()
    $process.Refresh()
    $script:state[$Name].exit_code = $process.ExitCode
    $script:state[$Name].finished_at_utc = Get-UtcStamp
    Write-Status
    if ($process.ExitCode -ne 0) { throw "$Name process failed with exit code $($process.ExitCode); inspect $stderr" }
    return $stdout
}
function Read-ValidatedMetadata([string]$MetadataPath, [int]$Environments, [int]$Iterations) {
    $metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
    $expected = @{
        status = 'completed'; behavior_profile = 'revised_v9';
        reward_contract = 'signed_progress_huber_speed_strict_support_v2';
        contact_contract = 'cad6_strict_distal_all_links_v2';
        environments = $Environments; additional_iterations = $Iterations
    }
    foreach ($key in $expected.Keys) {
        if ($metadata.$key -ne $expected[$key]) { throw "Unexpected metadata $key=$($metadata.$key); expected $($expected[$key])" }
    }
    if ([Math]::Abs([double]$metadata.non_distal_contact_height_tolerance_m - 0.001) -gt 1e-9 -or
        [Math]::Abs([double]$metadata.non_distal_ground_clearance_m - 0.008) -gt 1e-9) {
        throw 'Expected separate 0.001 m contact tolerance and 0.008 m clearance penalty.'
    }
    if ($metadata.source_checkpoint) { throw 'Expected fresh training, but source_checkpoint was set.' }
    $checkpoint = Get-CheckpointPath $MetadataPath $metadata.final_checkpoint
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Final checkpoint is missing: $checkpoint" }
    return $metadata
}
function Get-CheckpointPath([string]$MetadataPath, [string]$Checkpoint) {
    if (-not $Checkpoint) { throw 'Final checkpoint missing from metadata.' }
    if ([IO.Path]::IsPathRooted($Checkpoint)) { return $Checkpoint }
    return Join-Path (Split-Path $MetadataPath -Parent) $Checkpoint
}
function Get-FailureSummary([string]$Path) {
    $records = @()
    if (Test-Path -LiteralPath $Path) {
        foreach ($line in Get-Content -LiteralPath $Path) {
            if (-not $line.Trim()) { continue }
            $record = $line | ConvertFrom-Json
            $records += [ordered]@{
                environment_index = $record.environment_index; episode_index = $record.episode_index
                episode_step = $record.episode_step; termination_reasons = $record.termination_reasons
                strict_contact_failure = $record.strict_contact_failure; support = $record.support
                non_distal_clearance_m = $record.non_distal_clearance_m; contact_pairs = $record.contact_pairs
                state_sample = $record.state_sample; root_state = $record.root_state
                body_state = $record.body_state; net_contact_force_history = $record.net_contact_force_history
            }
        }
    }
    return $records
}

if ($Stage -eq 'Large' -and -not $JobDirectory) { throw '-Stage Large requires the validated -JobDirectory.' }
if (-not $JobDirectory) {
    $JobDirectory = Join-Path $repo ('logs\v9_validation\' + (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
$script:job = Get-AbsolutePath $JobDirectory
[IO.Directory]::CreateDirectory($script:job) | Out-Null
$script:statusFile = if ($StatusPath) { Get-AbsolutePath $StatusPath } else { Join-Path $script:job 'status.json' }
if ($Stage -eq 'Large' -or $Worker) {
    if (-not (Test-Path -LiteralPath $script:statusFile)) { throw "Status file is missing: $script:statusFile" }
    $oldState = Get-Content -LiteralPath $script:statusFile -Raw | ConvertFrom-Json
    $script:state = [ordered]@{}
    foreach ($property in $oldState.PSObject.Properties) { $script:state[$property.Name] = $property.Value }
} else {
    if (Test-Path -LiteralPath $script:statusFile) {
        throw "Validation job already exists; use a new -JobDirectory or review it before -Stage Large: $script:statusFile"
    }
    $script:state = [ordered]@{
        schema = 'dksh_v9_validation_job_v1'; job_directory = $script:job; status_path = $script:statusFile
        started_at_utc = Get-UtcStamp; updated_at_utc = Get-UtcStamp; status = 'starting'
        requested_large_training = [bool]$LaunchLargeTraining; launcher_pid = $PID
        note = 'A short smoke/train verifies execution and contact contracts, not learned gait quality. Empty GPU plane contact pairs do not prove absence of ground contact.'
    }
    Write-Status
}
if (-not $Worker) {
    if ($Stage -eq 'Large') {
        if (-not $script:state.small_metadata_path) { throw 'No validated small run exists in this job.' }
        Read-ValidatedMetadata $script:state.small_metadata_path 4 10 | Out-Null
        if (-not $script:state.smoke_pass) { throw 'This job did not pass smoke validation.' }
        if ($script:state.large_worker_pid -or $script:state.large_pid) { throw 'This job already launched large training; refusing duplicate.' }
    }
    $process = Start-Worker $Stage
    [ordered]@{ status = 'dispatched'; stage = $Stage; worker_pid = $process.Id; status_path = $script:statusFile; job_directory = $script:job } | ConvertTo-Json -Compress
    exit 0
}

try {
    Set-Location -LiteralPath $repo
    if ($Stage -eq 'Validate') {
        $script:state.validation_worker_pid = $PID
        $script:state.status = 'smoke_running'
        Write-Status
        $smokeLog = Invoke-Cad6 'smoke' @('-Mode', 'check', '-Environment', 'flat', '-NumEnvs', '4', '-Steps', '200', '-ContactDiagnostics', '-AllowProvisionalCad')
        if (-not (Select-String -LiteralPath $smokeLog -SimpleMatch 'DKSH_ISAACLAB_SMOKE_PASS' -Quiet)) { throw 'Smoke process did not emit its PASS marker.' }
        $script:state.smoke_pass = $true
        $before = @(Get-MetadataPaths)
        $script:state.metadata_paths_before_small = $before
        $script:state.status = 'small_training_running'
        Write-Status
        $trainLog = Invoke-Cad6 'small' @('-Mode', 'train', '-Environment', 'flat', '-NumEnvs', '4', '-MaxIterations', '10', '-BehaviorProfile', 'revised_v9', '-ContactDiagnostics', '-AllowProvisionalCad')
        if (-not (Select-String -LiteralPath $trainLog -SimpleMatch 'DKSH_JOINT_ANGLE_TRAINING_COMPLETE' -Quiet)) { throw 'Small training did not emit its completion marker.' }
        $newPaths = @(Get-MetadataPaths | Where-Object { $before -notcontains $_ })
        if ($newPaths.Count -ne 1) { throw "Expected exactly one new metadata file; found $($newPaths.Count). Concurrent runs require manual identification." }
        $metadata = Read-ValidatedMetadata $newPaths[0] 4 10
        $script:state.small_metadata_path = $newPaths[0]
        $script:state.small_run_directory = Split-Path $newPaths[0] -Parent
        $script:state.small_final_checkpoint = Get-CheckpointPath $newPaths[0] $metadata.final_checkpoint
        $script:state.small_completed_episodes = $metadata.completed_episodes
        $diagnostics = $metadata.contact_diagnostics.path
        if (-not $diagnostics) { throw 'Contact diagnostics path missing from new metadata.' }
        if (-not [IO.Path]::IsPathRooted($diagnostics)) { $diagnostics = Join-Path $repo $diagnostics }
        $script:state.contact_diagnostics_path = $diagnostics
        $records = @(Get-FailureSummary $diagnostics)
        $summaryPath = Join-Path $script:job 'contact_summary.json'
        [IO.File]::WriteAllText($summaryPath, (ConvertTo-Json -InputObject $records -Depth 40), (New-Object Text.UTF8Encoding($false)))
        $script:state.contact_summary_path = $summaryPath
        $script:state.contact_record_count = $records.Count
        $strict = @($records | Where-Object { $_.strict_contact_failure -or $_.termination_reasons -contains 'strict_contact' })
        $script:state.strict_contact_record_count = $strict.Count
        $script:state.validation_finished_at_utc = Get-UtcStamp
        $script:state.status = if ($strict.Count) { 'validated_contact_review_required' } else { 'validated' }
        $script:state.contact_review_note = 'Strict failures may be real falls of an untrained policy. Review forces, distal support, gaps and pair coverage together; a gap above 1 mm alone cannot identify a false positive.'
        Write-Status
        if (-not $LaunchLargeTraining -or $strict.Count) { exit 0 }
        $Stage = 'Large'
    }
    # Explicit -Stage Large means the caller has reviewed the diagnostic summary.
    if ($script:state.large_worker_pid -or $script:state.large_pid) { throw 'Large training was already launched for this job.' }
    Read-ValidatedMetadata $script:state.small_metadata_path 4 10 | Out-Null
    if (-not $script:state.smoke_pass) { throw 'Smoke validation is required before large training.' }
    $script:state.large_worker_pid = $PID
    $script:state.large_pid = $PID
    $script:state.large_started_at_utc = Get-UtcStamp
    $script:state.large_stdout = Join-Path $script:job 'large.stdout.log'
    $script:state.large_stderr = Join-Path $script:job 'large.stderr.log'
    $script:state.status = 'large_training_running'
    $beforeLarge = @(Get-MetadataPaths)
    Write-Status
    $launcher = Join-Path $repo 'IsaacLab\isaaclab.bat'
    if (-not (Test-Path -LiteralPath $launcher)) { throw "Isaac Lab launcher missing: $launcher" }
    $arguments = @('-p', (Join-Path $repo 'isaaclab_project\scripts\joint_angle_runner.py'), '--mode=train', '--task=Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2', '--num_envs=5120', '--max_iterations=100', '--environment=flat', '--behavior_profile=revised_v9', '--allow_provisional_cad', '--headless')
    # Invoke the native launcher directly to bypass the small wrapper's 128-environment limit.
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $launcher @arguments 1> $script:state.large_stdout 2> $script:state.large_stderr
    $largeExit = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    $script:state.large_exit_code = $largeExit
    if ($largeExit -ne 0) { throw "Large training failed with exit code $largeExit" }
    $newLarge = @(Get-MetadataPaths | Where-Object { $beforeLarge -notcontains $_ })
    if ($newLarge.Count -ne 1) { throw "Expected exactly one new large-run metadata file; found $($newLarge.Count)." }
    $largeMetadata = Read-ValidatedMetadata $newLarge[0] 5120 100
    $script:state.large_metadata_path = $newLarge[0]
    $script:state.large_final_checkpoint = Get-CheckpointPath $newLarge[0] $largeMetadata.final_checkpoint
    $script:state.large_finished_at_utc = Get-UtcStamp
    $script:state.status = 'completed'
    Write-Status
} catch {
    $script:state.status = 'failed'
    $script:state.error = $_.Exception.Message
    Write-Status
    Write-Error $_
    exit 1
}
