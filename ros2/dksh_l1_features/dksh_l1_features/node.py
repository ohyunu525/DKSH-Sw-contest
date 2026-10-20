"""ROS2 adapters from Unitree PointCloud2 frames to 32-value policy inputs."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .encoder import encode_l1_obstacles, encode_l2_obstacles


def _xyz(point):
    names = getattr(getattr(point, "dtype", None), "names", None)
    if names:
        return point["x"], point["y"], point["z"]
    return point[0], point[1], point[2]


class PolicyFeaturesNode(Node):
    def __init__(self, model: str):
        super().__init__(f"{model}_policy_features")
        self._model = model
        self._encode = encode_l2_obstacles if model == "l2" else encode_l1_obstacles
        self.declare_parameter("input_topic", "/unilidar/cloud")
        self.declare_parameter("output_topic", f"/policy/{model}_features")
        self.declare_parameter("expected_frame", "unilidar_lidar")
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._expected_frame = self.get_parameter("expected_frame").value
        self._publisher = self.create_publisher(Float32MultiArray, output_topic, 10)
        self._subscription = self.create_subscription(
            PointCloud2, input_topic, self._on_cloud, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Encoding {model.upper()} obstacle frames from {input_topic} to {output_topic} in {self._expected_frame}"
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        if message.header.frame_id != self._expected_frame:
            self.get_logger().warning(
                f"Ignoring cloud frame {message.header.frame_id!r}; expected {self._expected_frame!r}",
                throttle_duration_sec=5.0,
            )
            return
        points = (
            _xyz(point)
            for point in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        features = self._encode(points)
        output = Float32MultiArray()
        output.layout.dim = [MultiArrayDimension(label=f"{self._model}_policy_features", size=32, stride=32)]
        output.data = features
        self._publisher.publish(output)


class L1PolicyFeaturesNode(PolicyFeaturesNode):
    def __init__(self):
        super().__init__("l1")


class L2PolicyFeaturesNode(PolicyFeaturesNode):
    def __init__(self):
        super().__init__("l2")


def _run(node_type, args=None):
    rclpy.init(args=args)
    node = node_type()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    _run(L1PolicyFeaturesNode, args)


def main_l2(args=None):
    _run(L2PolicyFeaturesNode, args)
