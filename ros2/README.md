# DKSH Spiderbot ROS2/Nav2 runtime

This directory is the replaceable ROS2 navigation side of the project. Unity remains the sensor/physics simulator. The image includes the Unity bridge endpoint, the ROS2 frontier candidate, the required Nav2 runtime servers and `slam_toolbox`.

## Hardware profile

The current development machine has a 14-core/18-thread Intel Core Ultra 5 125H, 16 GB shared system memory, Intel Arc integrated graphics and about 35 GB free on `C:` at setup time. The stack therefore uses a headless `ros-base` image and deliberately excludes Gazebo, desktop metapackages and CUDA. `slam_toolbox` brings some RViz support libraries as binary dependencies, but no RViz process is launched.

The container is capped at:

- 4 CPU cores
- 5 GB RAM, 6 GB including swap
- 256 MB shared memory
- 1024 processes

The build helper refuses a new image build when less than 25 GB remains on `C:`. It never prunes unrelated Docker data.

Measured after the first complete build on this machine:

- final project image: 686,709,619 bytes (about 655 MiB)
- endpoint-only idle: about 106.5 MiB RAM and 0.29% Docker CPU
- all SLAM/Nav2 lifecycle nodes active against a synthetic 6 m map: about 307.3 MiB RAM and 32% Docker CPU (roughly 0.32 of one logical core)
- `C:` free after the build: about 29.3 GB

These are configuration smoke-test snapshots, not worst-case figures with a growing live SLAM map. Keep Isaac Lab stopped while Unity and Nav2 are running.

## Pinned components

- ROS2 Jazzy on Ubuntu 24.04 (`ros:jazzy-ros-base-noble`)
- Explicit Nav2 runtime servers and `slam_toolbox` from Jazzy binary packages
- Unity ROS-TCP-Endpoint `ROS2v0.7.0`
- `frontier_exploration_ros2` `v1.6.1`

The `navigation2` and `nav2_bringup` metapackages are intentionally not installed: Jazzy's bringup package depends on Gazebo simulation packages. Only controller, planner, BT navigator, behavior, velocity smoother, collision monitor and lifecycle packages are installed. RViz and Gazebo are never launched locally.

## Commands

Run these from PowerShell:

```powershell
.\ros2\manage.ps1 Build
.\ros2\manage.ps1 Start
.\ros2\manage.ps1 Status
.\ros2\manage.ps1 Validate
```

Stop only this project's container with:

```powershell
.\ros2\manage.ps1 Stop
```

The endpoint listens on host `127.0.0.1:10000`. `ROS_DOMAIN_ID` is fixed to `42`, and DDS discovery is restricted to the container's localhost because the Unity boundary is TCP. This avoids accidental discovery of unrelated ROS2 systems and reduces multicast traffic.

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
