#!/bin/bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/dksh_ros2/install/setup.bash

exec "$@"
