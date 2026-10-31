[CmdletBinding()]
param(
    [ValidateSet("smoke", "isaac-smoke", "train", "evaluate", "play", "preview", "viewer", "cartpole")]
    [string]$Mode = "smoke",
    [ValidateRange(1, 128)]
    [int]$NumEnvs = 4,
    [ValidateRange(1, 100000)]
    [int]$Steps = 32,
    [ValidateRange(1, 100000)]
    [int]$MaxIterations = 1000,
    [string]$Checkpoint = "",
    [ValidateSet('baseline', 'cad8', 'cad6')]
    [string]$RobotModel = 'baseline',
    [ValidateSet('mg996r', 'mg996r_hip')]
    [string]$ServoModel = 'mg996r',
    [ValidateSet('direct', 'legacy')]
    [string]$ControlMode = 'direct',
    [ValidateSet('flat', 'narrow', 'vibrating', 'falling_debris', 'rough', 'mixed', 'stairs', 'curriculum')]
    [string]$Environment = 'flat',
    [ValidateRange(0.0, 1.0)]
    [double]$Difficulty = 0.5,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,
    [string]$Output = '',
    [double]$InitialNoiseStd,
    [double]$MinNoiseStd,
    [double]$MaxNoiseStd,
    [switch]$ResetExplorationStd,
    [ValidateSet('legacy_v2', 'revised_v3', 'revised_v4', 'revised_v5', 'revised_v6', 'revised_v7', 'revised_v7_com', 'revised_v8')][string]$BehaviorProfile,
    [switch]$AllowBehaviorTransfer,
    [switch]$ContactDiagnostics,
    [string]$ContactDiagnosticsOutput = '',
    [ValidateRange(1, 2147483647)][int]$ContactDiagnosticsMaxRecords = 64,
    [ValidateRange(1, 2147483647)][int]$ContactDiagnosticsPerStep = 4,
    [ValidateRange(0, 2147483647)][int]$CourseSeed,
    [ValidateSet('train', 'holdout')][string]$CourseSplit,
    [double]$FlatFraction,
    [ValidateRange(1, 100000)][int]$CurriculumWindow,
    [ValidateRange(1, 100000)][int]$CurriculumMinEpisodes,
    [double]$CurriculumPromoteThreshold,
    [double]$CurriculumDemoteThreshold,
    [double]$CurriculumStep,
    [switch]$FreezeCurriculum,
    [switch]$AllowProvisionalCad,
    [switch]$Gui
)

$projectRoot = $PSScriptRoot
$noiseKeys = @('InitialNoiseStd', 'MinNoiseStd', 'MaxNoiseStd', 'ResetExplorationStd', 'AllowBehaviorTransfer')
$noiseOptionsSpecified = @($noiseKeys | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0
$contactKeys = @('ContactDiagnostics', 'ContactDiagnosticsOutput', 'ContactDiagnosticsMaxRecords', 'ContactDiagnosticsPerStep')
$contactOptionsSpecified = @($contactKeys | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0
if ($contactOptionsSpecified -and ($ControlMode -ne 'direct' -or $Mode -notin @('smoke', 'train', 'evaluate', 'play'))) {
    throw 'Contact diagnostics require direct smoke, train, evaluate or play mode.'
}
$contactArguments = @()
if ($ContactDiagnostics) { $contactArguments += '--contact_diagnostics' }
foreach ($pair in @(@('ContactDiagnosticsOutput', 'contact_diagnostics_output'),
                    @('ContactDiagnosticsMaxRecords', 'contact_diagnostics_max_records'),
                    @('ContactDiagnosticsPerStep', 'contact_diagnostics_per_step'))) {
    if ($PSBoundParameters.ContainsKey($pair[0])) { $contactArguments += "--$($pair[1])=$($PSBoundParameters[$pair[0]])" }
}
$courseKeys = @('CourseSeed', 'CourseSplit', 'FlatFraction', 'CurriculumWindow', 'CurriculumMinEpisodes',
                'CurriculumPromoteThreshold', 'CurriculumDemoteThreshold', 'CurriculumStep', 'FreezeCurriculum')
$courseOptionsSpecified = @($courseKeys | Where-Object { $PSBoundParameters.ContainsKey($_) }).Count -gt 0
if ($courseOptionsSpecified -and ($Environment -ne 'curriculum' -or $ControlMode -ne 'direct' -or $Mode -notin @('train', 'evaluate', 'play'))) {
    throw 'Course options require -Environment curriculum -ControlMode direct and train, evaluate or play mode.'
}
if ($Environment -eq 'curriculum' -and ($ControlMode -ne 'direct' -or $RobotModel -ne 'cad6' -or $ServoModel -ne 'mg996r_hip' -or
    $Mode -notin @('train', 'evaluate', 'play', 'smoke'))) {
    throw 'Curriculum requires the mixed-servo CAD6 direct-angle task.'
}
if ($Environment -eq 'curriculum') {
    if ($PSBoundParameters.ContainsKey('BehaviorProfile') -and $BehaviorProfile -notin @('revised_v7', 'revised_v8')) {
        throw 'Curriculum requires revised_v7 or revised_v8 distal-support behavior.'
    }
    $flatShare = if ($PSBoundParameters.ContainsKey('FlatFraction')) { $FlatFraction } else { 0.25 }
    $windowSize = if ($PSBoundParameters.ContainsKey('CurriculumWindow')) { $CurriculumWindow } else { 20 }
    $minimumEpisodes = if ($PSBoundParameters.ContainsKey('CurriculumMinEpisodes')) { $CurriculumMinEpisodes } else { 20 }
    $promote = if ($PSBoundParameters.ContainsKey('CurriculumPromoteThreshold')) { $CurriculumPromoteThreshold } else { 0.8 }
    $demote = if ($PSBoundParameters.ContainsKey('CurriculumDemoteThreshold')) { $CurriculumDemoteThreshold } else { 0.3 }
    $levelStep = if ($PSBoundParameters.ContainsKey('CurriculumStep')) { $CurriculumStep } else { 0.1 }
    if ([double]::IsNaN($flatShare) -or [double]::IsInfinity($flatShare) -or $flatShare -le 0 -or $flatShare -ge 1 -or
        $minimumEpisodes -gt $windowSize -or [double]::IsNaN($promote) -or [double]::IsInfinity($promote) -or
        [double]::IsNaN($demote) -or [double]::IsInfinity($demote) -or $demote -lt 0 -or $demote -ge $promote -or $promote -gt 1 -or
        [double]::IsNaN($levelStep) -or [double]::IsInfinity($levelStep) -or $levelStep -le 0 -or $levelStep -gt 1) {
        throw 'Invalid curriculum fraction, episode window, thresholds or difficulty step.'
    }
    if ($FreezeCurriculum -and $Mode -ne 'train') { throw 'FreezeCurriculum is training-only; evaluation is frozen.' }
    if ($PSBoundParameters.ContainsKey('CourseSplit') -and (($Mode -eq 'train' -and $CourseSplit -ne 'train') -or
        ($Mode -in @('evaluate', 'play') -and $CourseSplit -ne 'holdout'))) {
        throw 'Training requires the train course split; evaluation/play requires holdout.'
    }
}
if ($noiseOptionsSpecified -and ($ControlMode -ne 'direct' -or $Mode -ne 'train')) {
    throw 'Noise std overrides, exploration reset and behavior transfer require -ControlMode direct -Mode train.'
}
if ($PSBoundParameters.ContainsKey('BehaviorProfile') -and ($ControlMode -ne 'direct' -or $Mode -notin @('train', 'evaluate', 'play'))) {
    throw 'BehaviorProfile requires direct train, evaluate or play mode.'
}
if ($ControlMode -eq 'direct' -and $Mode -eq 'train') {
    $initialStd = if ($PSBoundParameters.ContainsKey('InitialNoiseStd')) { $InitialNoiseStd } else { 0.08 }
    $minimumStd = if ($PSBoundParameters.ContainsKey('MinNoiseStd')) { $MinNoiseStd } else { 0.03 }
    $maximumStd = if ($PSBoundParameters.ContainsKey('MaxNoiseStd')) { $MaxNoiseStd } else { 0.30 }
    if ([double]::IsNaN($initialStd) -or [double]::IsInfinity($initialStd) -or
        [double]::IsNaN($minimumStd) -or [double]::IsInfinity($minimumStd) -or
        [double]::IsNaN($maximumStd) -or [double]::IsInfinity($maximumStd) -or
        $minimumStd -le 0 -or $initialStd -lt $minimumStd -or $initialStd -gt $maximumStd) {
        throw 'Exploration standard deviations must be finite and satisfy 0 < minimum <= initial <= maximum.'
    }
}
$Environment = $Environment.ToLowerInvariant()
$isaacLabRoot = Join-Path $projectRoot "IsaacLab"
$launcher = Join-Path $isaacLabRoot "isaaclab.bat"
$taskName = "Isaac-DKSH-Spider-Navigation-Direct-v0"
$experimentName = 'dksh_spider_navigation_l2v1'
if ($RobotModel -ne 'baseline') {
    $cadCount = $RobotModel.Substring(3)
    $taskName = "Isaac-DKSH-Spider-CAD$cadCount-Navigation-Direct-v0"
    $experimentName = "dksh_spider_cad${cadCount}_navigation_l2v1"
}
if ($Environment -ne 'flat') {
    $experimentName += '_obstacles'
}
if ($ControlMode -eq 'direct') {
    if ($ServoModel -eq 'mg996r_hip' -and $RobotModel -ne 'cad6') {
        throw 'The mixed MG996R-hip/RC920DMG-leg actuator contract is defined for CAD6 only.'
    }
    $taskPrefix = switch ($ServoModel) {
        'mg996r_hip' { 'MG996R-Hip' }
        default { 'Spider' }
    }
    if ($RobotModel -ne 'baseline') { $taskPrefix += "-$($RobotModel.ToUpperInvariant())" }
    $taskName = "Isaac-DKSH-$taskPrefix-JointAngles-v2"
}
elseif ($ServoModel -eq 'mg996r_hip') {
    throw 'MG996R-hip/RC920DMG-leg actuators are supported by the CAD6 direct-angle task only.'
}
$difficultyText = $Difficulty.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$environmentArguments = @("--environment=$Environment", "--difficulty=$difficultyText")
$runnerEnvironmentArguments = @(
    "--environment=$Environment", "--difficulty=$difficultyText",
    "--experiment_name=$experimentName"
)
if ($PSBoundParameters.ContainsKey('Seed')) {
    $environmentArguments += "--seed=$Seed"
    $runnerEnvironmentArguments += "--seed=$Seed"
}

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab is missing. Run .\setup_isaaclab.ps1 first."
}

Push-Location $projectRoot
try {
    if ($ControlMode -eq 'direct' -and $Mode -in @('train', 'evaluate', 'play')) {
        $scriptPath = Join-Path $projectRoot 'isaaclab_project\scripts\joint_angle_runner.py'
        $arguments = @('-p', $scriptPath, "--mode=$Mode", "--task=$taskName", "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations") + $environmentArguments + $contactArguments
        if ($Checkpoint) { $arguments += "--checkpoint=$Checkpoint" }
        if ($Output) { $arguments += "--output=$Output" }
        if ($AllowProvisionalCad) { $arguments += '--allow_provisional_cad' }
        foreach ($pair in @(@('InitialNoiseStd', 'initial_noise_std'), @('MinNoiseStd', 'min_noise_std'), @('MaxNoiseStd', 'max_noise_std'))) {
            if ($PSBoundParameters.ContainsKey($pair[0])) {
                $value = $PSBoundParameters[$pair[0]].ToString([System.Globalization.CultureInfo]::InvariantCulture)
                $arguments += "--$($pair[1])=$value"
            }
        }
        if ($ResetExplorationStd) { $arguments += '--reset_exploration_std' }
        if ($PSBoundParameters.ContainsKey('BehaviorProfile')) { $arguments += "--behavior_profile=$BehaviorProfile" }
        if ($AllowBehaviorTransfer) { $arguments += '--allow_behavior_transfer' }
        foreach ($pair in @(@('CourseSeed', 'course_seed'), @('CourseSplit', 'course_split'),
                             @('FlatFraction', 'flat_fraction'), @('CurriculumWindow', 'curriculum_window'),
                             @('CurriculumMinEpisodes', 'curriculum_min_episodes'),
                             @('CurriculumPromoteThreshold', 'curriculum_promote_threshold'),
                             @('CurriculumDemoteThreshold', 'curriculum_demote_threshold'), @('CurriculumStep', 'curriculum_step'))) {
            if ($PSBoundParameters.ContainsKey($pair[0])) {
                $value = $PSBoundParameters[$pair[0]]
                if ($value -is [double]) { $value = $value.ToString([System.Globalization.CultureInfo]::InvariantCulture) }
                $arguments += "--$($pair[1])=$value"
            }
        }
        if ($FreezeCurriculum) { $arguments += '--freeze_curriculum' }
        if ($Mode -eq 'play') { $arguments += '--disable_fabric' }
        elseif (-not $Gui) { $arguments += '--headless' }
        if ($PSBoundParameters.ContainsKey('Steps')) { $arguments += "--steps=$Steps" }
        $completionMarker = switch ($Mode) {
            'train' { 'DKSH_JOINT_ANGLE_TRAINING_COMPLETE' }
            'evaluate' { 'DKSH_JOINT_ANGLE_EVAL_COMPLETE' }
            'play' { 'DKSH_JOINT_ANGLE_PLAY_COMPLETE' }
        }
        $completed = $false
        & $launcher @arguments 2>&1 | ForEach-Object {
            if ([string]$_ -like "$completionMarker*") { $completed = $true }
            Write-Output $_
        }
        $nativeExitCode = $LASTEXITCODE
        if (-not $completed -or $nativeExitCode -notin @(0, 1)) {
            throw "Absolute joint-angle $Mode did not complete (native exit code $nativeExitCode)."
        }
        cmd.exe /d /c exit 0
        return
    }
    switch ($Mode) {
        "smoke" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\smoke_env.py"
            $smokeArguments = @('-p', $scriptPath, "--task=$taskName", "--num_envs=$NumEnvs", "--steps=$Steps") + $environmentArguments + $contactArguments
            if (-not $Gui) { $smokeArguments += '--headless' }
            $output = @(& $launcher @smokeArguments 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "DKSH_ISAACLAB_SMOKE_PASS")) {
                throw "The custom environment smoke test failed (native exit code $nativeExitCode)."
            }
            cmd.exe /d /c exit 0
        }
        "isaac-smoke" {
            $scriptPath = Join-Path $projectRoot "scripts\isaaclab_smoke.py"
            $output = @(& $launcher -p $scriptPath --headless "--steps=$Steps" 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "ISAAC_LAB_SMOKE_TEST_PASS")) {
                throw "Isaac Lab smoke test failed (native exit code $nativeExitCode)."
            }
            # Isaac Sim 4.5 can return 1 after a clean headless Kit shutdown.
            cmd.exe /d /c exit 0
        }
        "train" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\train.py"
            $logRoot = Join-Path $projectRoot "logs\rsl_rl\$experimentName"
            $latestCheckpointBefore = Get-ChildItem $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            # Pass a single argument array to the batch launcher.  Inline splatting
            # after positional arguments can lose the value part of Hydra overrides
            # on Windows PowerShell, producing e.g. `env.environment_preset` without
            # its `=rough` value.  Training is intentionally headless on this host.
            $arguments = @(
                "-p", $scriptPath, "--headless", "--task=$taskName",
                "--num_envs=$NumEnvs", "--max_iterations=$MaxIterations"
            ) + $runnerEnvironmentArguments
            $trainingCompleted = $false
            & $launcher @arguments 2>&1 | ForEach-Object {
                if ([string]$_ -match '^DKSH_ISAACLAB_TRAIN_COMPLETE\s*$') {
                    $trainingCompleted = $true
                }
                Write-Output $_
            }
            $nativeExitCode = $LASTEXITCODE
            $latestCheckpointAfter = Get-ChildItem $logRoot -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $createdCheckpoint = $null -ne $latestCheckpointAfter -and (
                $null -eq $latestCheckpointBefore -or
                $latestCheckpointAfter.LastWriteTime -gt $latestCheckpointBefore.LastWriteTime
            )
            if (-not $trainingCompleted -or -not $createdCheckpoint -or $nativeExitCode -notin @(0, 1)) {
                throw "Training did not complete cleanly or produce a new checkpoint (native exit code $nativeExitCode, completion marker $trainingCompleted)."
            }
            Write-Host "Training checkpoint: $($latestCheckpointAfter.FullName)"
            # Isaac Sim 4.5 can return 1 after the completed trainer shuts Kit down.
            cmd.exe /d /c exit 0
        }
        "evaluate" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\evaluate.py"
            $arguments = @(
                "-p", $scriptPath, "--headless", "--task=$taskName",
                "--num_envs=$NumEnvs", "--steps=$Steps"
            ) + $environmentArguments
            if ($Checkpoint) {
                $arguments += "--checkpoint=$Checkpoint"
            }
            $output = @(& $launcher @arguments 2>&1)
            $nativeExitCode = $LASTEXITCODE
            $output | Write-Output
            if (-not ($output -match "DKSH_ISAACLAB_EVAL_PASS")) {
                throw "Checkpoint evaluation failed (native exit code $nativeExitCode)."
            }
            cmd.exe /d /c exit 0
        }
        "play" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\play.py"
            # GUI playback must use USD I/O.  Fabric is faster but does not keep
            # the USD viewport synchronized with the simulated articulation.
            $arguments = @("-p", $scriptPath, "--disable_fabric", "--task=$taskName", "--num_envs=$NumEnvs") + $runnerEnvironmentArguments
            if ($Checkpoint) {
                $arguments += "--checkpoint=$Checkpoint"
            }
            else {
                $localCheckpoints = Get-ChildItem (Join-Path $projectRoot "logs\rsl_rl\$experimentName") `
                    -Recurse -Filter "model_*.pt" -ErrorAction SilentlyContinue
                if (-not $localCheckpoints) {
                    throw "No compatible checkpoint for $RobotModel / $Environment. Use -Mode preview to inspect the environment, train first, or provide -Checkpoint."
                }
            }
            & $launcher @arguments
            if ($LASTEXITCODE -eq 1) {
                # Isaac Sim 4.5 commonly reports 1 after the user closes a healthy Kit window.
                cmd.exe /d /c exit 0
            }
            elseif ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab playback exited with code $LASTEXITCODE."
            }
        }
        "preview" {
            $scriptPath = Join-Path $projectRoot "isaaclab_project\scripts\preview_env.py"
            $arguments = @("-p", $scriptPath, "--task=$taskName", "--num_envs=$NumEnvs") + $environmentArguments
            if ($PSBoundParameters.ContainsKey('Steps')) {
                $arguments += "--steps=$Steps"
            }
            $previewPassed = $false
            & $launcher @arguments 2>&1 | ForEach-Object {
                if ([string]$_ -match '^DKSH_ISAACLAB_PREVIEW_PASS\s*$') {
                    $previewPassed = $true
                }
                Write-Output $_
            }
            $nativeExitCode = $LASTEXITCODE
            if (-not $previewPassed -or $nativeExitCode -notin @(0, 1)) {
                throw "Isaac Lab environment preview failed (native exit code $nativeExitCode)."
            }
            # Normalize Isaac Sim 4.5's shutdown code only after successful playback.
            cmd.exe /d /c exit 0
        }
        "viewer" {
            $scriptPath = Join-Path $isaacLabRoot "scripts\tutorials\00_sim\create_empty.py"
            & $launcher -p $scriptPath
            if ($LASTEXITCODE -eq 1) {
                # Isaac Sim 4.5 commonly reports 1 after the user closes a healthy Kit window.
                cmd.exe /d /c exit 0
            }
            elseif ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab viewer exited with code $LASTEXITCODE."
            }
        }
        "cartpole" {
            $scriptPath = Join-Path $isaacLabRoot "scripts\reinforcement_learning\rsl_rl\train.py"
            & $launcher -p $scriptPath `
                "--task=Isaac-Cartpole-Direct-v0" --headless "--num_envs=$NumEnvs"
            if ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab Cartpole training exited with code $LASTEXITCODE."
            }
        }
    }
}
finally {
    Pop-Location
}
