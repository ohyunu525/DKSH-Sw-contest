"""Unitree L2 SDK2 driver, planar navigation projection, and policy features."""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _integer(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def generate_launch_description():
    cloud_topic = LaunchConfiguration("cloud_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    base_frame = LaunchConfiguration("base_frame")
    lidar_frame = LaunchConfiguration("lidar_frame")
    imu_frame = LaunchConfiguration("imu_frame")

    declarations = [
        # SDK2: 2 = UDP / 1 = serial. NEGA+3D+IMU is work mode 1 (UDP)
        # or 9 (serial). Transport must already match the device configuration.
        DeclareLaunchArgument("initialize_type", default_value="2"),
        DeclareLaunchArgument("work_mode", default_value="1"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("baudrate", default_value="4000000"),
        DeclareLaunchArgument("lidar_ip", default_value="192.168.1.62"),
        DeclareLaunchArgument("lidar_port", default_value="6101"),
        DeclareLaunchArgument("local_ip", default_value="192.168.1.2"),
        DeclareLaunchArgument("local_port", default_value="6201"),
        DeclareLaunchArgument("cloud_topic", default_value="/unilidar/cloud"),
        DeclareLaunchArgument("imu_topic", default_value="/unilidar/imu"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("lidar_frame", default_value="unilidar_lidar"),
        DeclareLaunchArgument("imu_frame", default_value="unilidar_imu"),
        DeclareLaunchArgument("mount_x", default_value="0.0"),
        DeclareLaunchArgument("mount_y", default_value="0.0"),
        DeclareLaunchArgument("mount_z", default_value="0.030"),
        DeclareLaunchArgument("mount_yaw", default_value="0.0"),
        DeclareLaunchArgument("mount_pitch", default_value="0.0"),
        DeclareLaunchArgument("mount_roll", default_value="0.0"),
        # 216 vertical scans / 5.55 horizontal frames ~= 39 rings per sweep.
        DeclareLaunchArgument("cloud_scan_num", default_value="39"),
        DeclareLaunchArgument("min_height", default_value="-0.10"),
        DeclareLaunchArgument("max_height", default_value="0.10"),
        DeclareLaunchArgument("start_navigation", default_value="false"),
        DeclareLaunchArgument(
            "params_file", default_value="/opt/dksh/config/nav2_low_resource.yaml"
        ),
    ]

    lidar_driver = Node(
        package="unitree_lidar_ros2",
        executable="unitree_lidar_ros2_node",
        name="unitree_lidar_ros2_node",
        output="screen",
        # SDK2 publishes its own IMU->LiDAR TF. Isolate it to avoid duplicate
        # parents/cycles with the measured static base->LiDAR->IMU chain.
        remappings=[("/tf", "/unilidar/sdk_tf")],
        parameters=[
            {
                "initialize_type": _integer("initialize_type"),
                "work_mode": _integer("work_mode"),
                "serial_port": LaunchConfiguration("serial_port"),
                "baudrate": _integer("baudrate"),
                "lidar_ip": LaunchConfiguration("lidar_ip"),
                "lidar_port": _integer("lidar_port"),
                "local_ip": LaunchConfiguration("local_ip"),
                "local_port": _integer("local_port"),
                "use_system_timestamp": True,
                "range_min": 0.05,
                "range_max": 30.0,
                "cloud_frame": lidar_frame,
                "cloud_topic": cloud_topic,
                "cloud_scan_num": _integer("cloud_scan_num"),
                "imu_frame": imu_frame,
                "imu_topic": imu_topic,
            }
        ],
    )

    cloud_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="l2_cloud_to_scan",
        output="screen",
        remappings=[("cloud_in", cloud_topic), ("scan", scan_topic)],
        parameters=[
            {
                "target_frame": lidar_frame,
                "transform_tolerance": 0.02,
                "min_height": _float("min_height"),
                "max_height": _float("max_height"),
                "angle_min": -math.pi,
                "angle_max": math.pi,
                "angle_increment": math.radians(0.5),
                "scan_time": 1.0 / 5.55,
                "range_min": 0.05,
                "range_max": 30.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
                "concurrency_level": 1,
            }
        ],
    )

    policy_features = Node(
        package="dksh_l1_features",
        executable="l2_policy_features",
        name="l2_policy_features",
        output="screen",
        parameters=[
            {
                "input_topic": cloud_topic,
                "output_topic": "/policy/l2_features",
                "expected_frame": lidar_frame,
            }
        ],
    )

    base_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_l2_tf",
        arguments=[
            "--x", LaunchConfiguration("mount_x"),
            "--y", LaunchConfiguration("mount_y"),
            "--z", LaunchConfiguration("mount_z"),
            "--yaw", LaunchConfiguration("mount_yaw"),
            "--pitch", LaunchConfiguration("mount_pitch"),
            "--roll", LaunchConfiguration("mount_roll"),
            "--frame-id", base_frame,
            "--child-frame-id", lidar_frame,
        ],
    )
    lidar_to_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="l2_to_imu_tf",
        arguments=[
            "--x", "-0.007698",
            "--y", "-0.014655",
            "--z", "0.00667",
            "--yaw", "0.0",
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", lidar_frame,
            "--child-frame-id", imu_frame,
        ],
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource("/opt/dksh/launch/navigation.launch.py"),
        condition=IfCondition(LaunchConfiguration("start_navigation")),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "use_sim_time": "false",
        }.items(),
    )
    return LaunchDescription(
        declarations
        + [lidar_driver, cloud_to_scan, policy_features, base_to_lidar, lidar_to_imu, navigation]
    )
