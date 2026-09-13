ARG ROS_DISTRO=jazzy
ARG ROS_BASE_IMAGE=ros:jazzy-ros-base-noble

FROM ${ROS_BASE_IMAGE} AS builder

ARG ROS_DISTRO
ARG ROS_TCP_ENDPOINT_TAG=ROS2v0.7.0
ARG FRONTIER_EXPLORATION_TAG=v1.6.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-ament-cmake-gtest \
        ros-${ROS_DISTRO}-nav2-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dksh_ros2/src

RUN git clone --branch "${ROS_TCP_ENDPOINT_TAG}" --depth 1 \
        https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git ros_tcp_endpoint \
    && git clone --branch "${FRONTIER_EXPLORATION_TAG}" --depth 1 \
        https://github.com/mertgulerx/frontier_exploration_ros2.git frontier_exploration_ros2

WORKDIR /opt/dksh_ros2

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build \
        --merge-install \
        --packages-select ros_tcp_endpoint frontier_exploration_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && source install/setup.bash \
    && colcon test --merge-install --packages-select frontier_exploration_ros2 \
    && colcon test-result --verbose

FROM ${ROS_BASE_IMAGE} AS runtime

ARG ROS_DISTRO

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-nav2-behaviors \
        ros-${ROS_DISTRO}-nav2-bt-navigator \
        ros-${ROS_DISTRO}-nav2-collision-monitor \
        ros-${ROS_DISTRO}-nav2-controller \
        ros-${ROS_DISTRO}-nav2-lifecycle-manager \
        ros-${ROS_DISTRO}-nav2-mppi-controller \
        ros-${ROS_DISTRO}-nav2-planner \
        ros-${ROS_DISTRO}-nav2-smac-planner \
        ros-${ROS_DISTRO}-nav2-velocity-smoother \
        ros-${ROS_DISTRO}-slam-toolbox \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY --from=builder /opt/dksh_ros2/install /opt/dksh_ros2/install
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod 0755 /ros_entrypoint.sh

ENV ROS_DISTRO=${ROS_DISTRO} \
    ROS_DOMAIN_ID=42 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

EXPOSE 10000
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "run", "ros_tcp_endpoint", "default_server_endpoint", "--ros-args", "-p", "ROS_IP:=0.0.0.0", "-p", "ROS_TCP_PORT:=10000"]
