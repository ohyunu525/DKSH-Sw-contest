[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Build', 'Start', 'StartLidar', 'Stop', 'Status', 'Validate', 'Shell')]
    [string]$Action = 'Status'
)

$ErrorActionPreference = 'Stop'
$composeFile = Join-Path $PSScriptRoot 'compose.yaml'
$lidarComposeFile = Join-Path $PSScriptRoot 'compose.lidar.yaml'
$minimumBuildFreeGB = 25.0

function Assert-DockerReady {
    docker info --format '{{.ServerVersion}}' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Desktop is not running.'
    }
}

function Get-FreeDiskGB {
    return [math]::Round((Get-PSDrive -Name C).Free / 1GB, 2)
}

Assert-DockerReady

switch ($Action) {
    'Build' {
        $freeGB = Get-FreeDiskGB
        if ($freeGB -lt $minimumBuildFreeGB) {
            throw "ROS2 image build stopped: C: has ${freeGB} GB free; at least ${minimumBuildFreeGB} GB is required."
        }

        docker compose --file $composeFile build
        if ($LASTEXITCODE -ne 0) { throw 'ROS2 image build failed.' }
    }
    'Start' {
        docker compose --file $composeFile up --detach
        if ($LASTEXITCODE -ne 0) { throw 'ROS2 container start failed.' }
    }
    'StartLidar' {
        docker compose --file $composeFile --file $lidarComposeFile up --detach
        if ($LASTEXITCODE -ne 0) {
            throw 'ROS2 Unitree L1 container start failed. Check UNITREE_LIDAR_DEVICE and USB access.'
        }
    }
    'Stop' {
        docker compose --file $composeFile stop
        if ($LASTEXITCODE -ne 0) { throw 'ROS2 container stop failed.' }
    }
    'Status' {
        docker compose --file $composeFile ps
        docker stats dksh-ros2-bridge --no-stream 2>$null
        Write-Host ('C: free: {0} GB' -f (Get-FreeDiskGB))
    }
    'Validate' {
        docker compose --file $composeFile exec ros2 bash -c @'
set -e
source /opt/ros/jazzy/setup.bash
source /opt/dksh_ros2/install/setup.bash
ros2 pkg prefix ros_tcp_endpoint
ros2 pkg prefix frontier_exploration_ros2
ros2 pkg prefix unitree_lidar_ros2
ros2 pkg prefix dksh_l1_features
ros2 pkg prefix pointcloud_to_laserscan
ros2 pkg prefix nav2_bt_navigator
ros2 pkg prefix nav2_mppi_controller
ros2 pkg prefix nav2_smac_planner
ros2 pkg prefix nav2_collision_monitor
ros2 pkg prefix slam_toolbox
ros2 node list
'@
        if ($LASTEXITCODE -ne 0) { throw 'ROS2 package validation failed.' }
    }
    'Shell' {
        docker compose --file $composeFile exec ros2 bash -c 'source /opt/ros/jazzy/setup.bash && source /opt/dksh_ros2/install/setup.bash && exec bash'
        if ($LASTEXITCODE -ne 0) { throw 'ROS2 shell failed.' }
    }
}
