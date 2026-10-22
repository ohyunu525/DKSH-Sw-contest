# DKSH Spiderbot ROS2/Nav2 runtime

This directory is the replaceable ROS2 navigation side of the project. Unity remains the sensor/physics simulator. The image includes the Unity bridge endpoint, the ROS2 frontier candidate, the required Nav2 runtime servers and `slam_toolbox`.

Machine roles and the rule separating the RTX 3070 Isaac workstation, the 16 GB Unity/ROS integration machine and lower-spec development machines are defined in [`DEVELOPMENT_REQUIREMENTS.md`](../DEVELOPMENT_REQUIREMENTS.md).

## Hardware profile

The current development machine has a 14-core/18-thread Intel Core Ultra 5 125H, 16 GB shared system memory, Intel Arc integrated graphics and about 35 GB free on `C:` at setup time. The stack therefore uses a headless `ros-base` image and deliberately excludes Gazebo, desktop metapackages and CUDA. `slam_toolbox` brings some RViz support libraries as binary dependencies, but no RViz process is launched.

The container is capped at:

- 4 CPU cores
- 5 GB RAM, 6 GB including swap
- 256 MB shared memory
- 1024 processes

The build helper refuses a new image build when less than 25 GB remains on `C:`. It never prunes unrelated Docker data.

Measured after the first complete build on this machine:

- pre-LiDAR project image: 686,709,619 bytes (about 655 MiB); the current image also includes PCL,
  `pointcloud_to_laserscan`, and the pinned Unitree L2 SDK2 driver, so remeasure after rebuilding
- endpoint-only idle: about 106.5 MiB RAM and 0.29% Docker CPU
- all SLAM/Nav2 lifecycle nodes active against a synthetic 6 m map: about 307.3 MiB RAM and 32% Docker CPU (roughly 0.32 of one logical core)
- `C:` free after the build: about 29.3 GB

These are configuration smoke-test snapshots, not worst-case figures with a growing live SLAM map. Keep Isaac Lab stopped while Unity and Nav2 are running.

## Pinned components

- ROS2 Jazzy on Ubuntu 24.04 (`ros:jazzy-ros-base-noble`)
- Explicit Nav2 runtime servers and `slam_toolbox` from Jazzy binary packages
- Unity ROS-TCP-Endpoint `ROS2v0.7.0`
- `frontier_exploration_ros2` `v1.6.1`
- Unitree `unilidar_sdk2` commit `0e3c51f512e6b8ff60b8c32f160b412cb48445c2`

The `navigation2` and `nav2_bringup` metapackages are intentionally not installed: Jazzy's bringup package depends on Gazebo simulation packages. Only controller, planner, BT navigator, behavior, velocity smoother, collision monitor and lifecycle packages are installed. RViz and Gazebo are never launched locally.

## Commands

Run these from PowerShell:

```powershell
.\ros2\manage.ps1 Build
.\ros2\manage.ps1 Start
.\ros2\manage.ps1 Status
.\ros2\manage.ps1 Validate
```

## Unitree L2 hardware

The image pins Unitree's official [SDK2](https://github.com/unitreerobotics/unilidar_sdk2) at commit
`0e3c51f512e6b8ff60b8c32f160b412cb48445c2`. Rebuild the image after pulling this change: the old
SDK1 image does not implement L2's transport parameters. The driver publishes `/unilidar/cloud`
(`sensor_msgs/PointCloud2`) and `/unilidar/imu`. `launch/l2_hardware.launch.py` selects 3D NEGA mode
with IMU enabled: work mode 1 for UDP or 9 for serial. It groups 39 vertical scans per cloud,
approximately one 5.55 Hz horizontal sweep (216 / 5.55), rather than the SDK's 18-ring partial frame.

The projection uses the central ±0.10 m height band, 0.5° angular resolution, 0.05–30 m range and
`scan_time=1/5.55` for `/scan`. The shared encoder publishes 16 normalized nearest obstacle ranges
and 16 validity flags on `/policy/l2_features`; it accepts elevations -6° through 90° and quantizes
at 4.5 mm. The package name `dksh_l1_features` remains for import compatibility and provides the
new `l2_policy_features` executable alongside the legacy L1 offline encoder. Cloud frame IDs must
be `unilidar_lidar`. Remove robot/ground points before using features in a navigation policy,
including ground returns in the newly visible negative elevation band.

The launch owns the provisional 30 mm base-to-LiDAR transform and SDK2's documented LiDAR-to-IMU
offset. SDK2's visualization TF is remapped to `/unilidar/sdk_tf` so it cannot assign a second
parent to the same frames or create a TF cycle. Hardware mode uses wall time.

UDP is the default. Configure the dedicated LiDAR NIC with the sensor's receiver address
(factory target `192.168.1.2`), and keep the sensor on the same subnet (factory sensor address
`192.168.1.62`). This repository does not change Windows NIC settings. Docker publishes only
6201/UDP on that NIC and binds the SDK socket to `0.0.0.0` inside the container. If the sensor's
target IP/port was changed, align its target with the host port mapping before starting.

```powershell
$env:L2_HOST_IP = '192.168.1.2'
$env:L2_LIDAR_IP = '192.168.1.62'
.\ros2\manage.ps1 Build
.\ros2\manage.ps1 StartLidar
```

For TTL UART, first switch the L2 to serial mode over Ethernet with Unitree's tool/SDK, then power
cycle it as described in the [SDK2 instructions](https://github.com/unitreerobotics/unilidar_sdk2#34-running-with-serial-port).
The default serial baud rate is 4,000,000. On Linux, expose the adapter device with:

```powershell
$env:UNITREE_LIDAR_DEVICE = '/dev/ttyACM0'
.\ros2\manage.ps1 StartLidar -LidarTransport serial
```

Set `UNITREE_LIDAR_DEVICE` to the actual host device if it differs; Docker maps it to `/dev/ttyACM0`.
Docker Desktop on Windows requires the USB adapter to be attached to WSL2 (for example with
`usbipd-win`) and visible to the Docker backend. The Jazzy container build and physical packet
reception are still required verification gates; static source/configuration checks do not prove
either transport works with the connected device.

SLAM/Nav2 remains opt-in until measured hardware odometry supplies `/odom` and
`odom -> base_footprint`. Then set `$env:L2_START_NAVIGATION = 'true'` before `StartLidar`.
Do not substitute command-only dead reckoning for measured walking-robot odometry.

Useful checks after startup:

```powershell
docker exec dksh-ros2-bridge bash -c "source /opt/ros/jazzy/setup.bash && source /opt/dksh_ros2/install/setup.bash && ros2 topic hz /unilidar/cloud"
docker exec dksh-ros2-bridge bash -c "source /opt/ros/jazzy/setup.bash && source /opt/dksh_ros2/install/setup.bash && ros2 topic echo --once /scan"
docker exec dksh-ros2-bridge bash -c "source /opt/ros/jazzy/setup.bash && source /opt/dksh_ros2/install/setup.bash && ros2 topic echo --once /policy/l2_features"
```

Stop only this project's container with:

```powershell
.\ros2\manage.ps1 Stop
```

The endpoint listens on host `127.0.0.1:10000`. `ROS_DOMAIN_ID` is fixed to `42`, and DDS discovery is restricted to the container's localhost because the Unity boundary is TCP. This avoids accidental discovery of unrelated ROS2 systems and reduces multicast traffic.

## Tailscale remote Unity connection

When Unity and ROS2 run on different computers, keep the Docker endpoint local and publish it only to the tailnet with Tailscale Serve. This does not open port 10000 on the LAN or public internet.

1. Install Tailscale for Windows on both computers and sign in to the same tailnet. The ROS2 computer must allow Tailscale Serve in the tailnet policy.
2. Start the ROS2 endpoint on the ROS2 computer: `./ros2/manage.ps1 Start`.
3. In an elevated PowerShell window on that computer, run `./ros2/tailscale.ps1 Configure`. It prints the Tailscale IPv4 endpoint, for example `tcp://100.x.y.z:10000`.
4. On the Unity computer, set **Ros2UnityBridge > Ros Ip Address** to that `100.x.y.z` address, keep **Ros Tcp Port** at `10000`, then enable `Ros2UnityBridge` and `HexapodVelocityCommandAdapter`.

Run `./ros2/tailscale.ps1 Status` to inspect the forwarding rule. Use `./ros2/tailscale.ps1 Disable` to remove it. The remote Unity machine must be permitted to reach the ROS2 machine by the tailnet's access-control policy. This path forwards raw ROS TCP traffic; it is not Tailscale Funnel and is never public.

For standalone Unity exploration, leave `HexapodVelocityCommandAdapter` and `Ros2UnityBridge` disabled on the `LiDAR Sensor` object. For ROS2 validation, enable those two components in the Inspector before entering Play Mode. Enabling the bridge temporarily disables the standalone path follower and coordinator, so only one system owns motion.

Start the low-resource SLAM/Nav2 nodes inside the running container after Unity Play Mode is publishing `/clock`, `/scan`, `/odom` and `/tf`:

```powershell
docker exec -it dksh-ros2-bridge bash -c "source /opt/ros/jazzy/setup.bash && source /opt/dksh_ros2/install/setup.bash && ros2 launch /opt/dksh/launch/navigation.launch.py"
```

The launch deliberately omits waypoint follower and the standalone smoothing server. `SmacPlanner2D` performs the global grid search and its own smoothing; MPPI uses the Omni model. Velocity flow is `cmd_vel_nav` -> velocity smoother -> `cmd_vel_smoothed` -> collision monitor -> `/cmd_vel` -> Unity hexapod adapter.

## Headless smoke-test input

`tools/smoke_test_inputs.py` publishes a transient empty map and static test-only TF chain. It is never included in the production launch; it exists only to activate every lifecycle node without opening Unity, RViz or Gazebo. The verified result is `Managed nodes are active` for SLAM Toolbox, MPPI, SmacPlanner2D, behavior server, BT navigator, velocity smoother and collision monitor.

Nav2 Jazzy currently prints a misleading SmacPlanner2D inflation-layer error during startup even when the 2D planner has a valid inflation layer ([upstream issue #6380](https://github.com/ros-navigation/navigation2/issues/6380)). The project uses a conservative 0.50 m inflation radius; MPPI accepts the footprint/inflation combination and the complete lifecycle stack activates. Do not reduce the radius merely to silence or investigate that message.

## Low-resource frontier profile

`config/frontier_low_resource.yaml` starts exploration disabled, processes maps at 0.5 Hz and uses the lower-CPU greedy selector. The package is compiled in this compatibility image, but the explorer must be started only after the later Nav2 runtime provides `/map`, both costmaps, TF and `navigate_to_pose`:

```bash
ros2 launch frontier_exploration_ros2 frontier_explorer.launch.py \
  params_file:=/opt/dksh/config/frontier_low_resource.yaml \
  use_sim_time:=true
```

Do not run Unity, Isaac Sim/Isaac Lab training and this full navigation stack simultaneously on this machine. Unity plus the headless ROS2 stack is the supported local combination; reinforcement-learning training remains a separate workload.
