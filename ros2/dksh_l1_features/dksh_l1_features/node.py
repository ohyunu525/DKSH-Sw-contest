"""ROS2 adapter from one L1 PointCloud2 frame to the 32-value policy input."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .encoder import encode_l1_obstacles


def _xyz(point):
    names = getattr(getattr(point, "dtype", None), "names", None)
    if names:
        return point["x"], point["y"], point["z"]
    return point[0], point[1], point[2]


class L1PolicyFeaturesNode(Node):
    def __init__(self):
        super().__init__("l1_policy_features")
        self.declare_parameter("input_topic", "/unilidar/cloud")
        self.declare_parameter("output_topic", "/policy/l1_features")
        self.declare_parameter("expected_frame", "unilidar_lidar")
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._expected_frame = self.get_parameter("expected_frame").value
        self._publisher = self.create_publisher(Float32MultiArray, output_topic, 10)
        self._subscription = self.create_subscription(
            PointCloud2, input_topic, self._on_cloud, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Encoding complete L1 frames from {input_topic} to {output_topic} in {self._expected_frame}"
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
        features = encode_l1_obstacles(points)
        output = Float32MultiArray()
        output.layout.dim = [MultiArrayDimension(label="l1_policy_features", size=32, stride=32)]
        output.data = features
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = L1PolicyFeaturesNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
