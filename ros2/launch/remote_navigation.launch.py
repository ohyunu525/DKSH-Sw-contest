"""Existing Unity synthetic sensors + headless Nav2 + the narrow waypoint gateway."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    directory = Path(__file__).resolve().parent
    return LaunchDescription([
        Node(package="ros_tcp_endpoint", executable="default_server_endpoint",
             parameters=[{"ROS_IP": "0.0.0.0", "ROS_TCP_PORT": 10000}], output="screen"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(directory / "navigation.launch.py")),
                                 launch_arguments={"use_sim_time": "true"}.items()),
        ExecuteProcess(cmd=["ros2", "run", "dksh_remote_navigation", "waypoint_gateway",
                            "--backend", "nav2-sim", "--host", "0.0.0.0", "--port", "8765",
                            "--log-dir", "/opt/dksh/logs/remote_navigation"], output="screen"),
    ])
