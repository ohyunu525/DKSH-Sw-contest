#!/usr/bin/env python3
"""Publish minimal deterministic inputs for a headless Nav2 lifecycle smoke test.

This helper is never launched by the production launch file. It exists so the
configuration can be activated without starting Unity, RViz, or Gazebo.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class SmokeTestInputs(Node):
    def __init__(self) -> None:
        super().__init__("dksh_smoke_test_inputs")
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_publisher = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()
        self._publish_map()
        self.create_timer(1.0, self._publish_map)

    def _publish_static_transforms(self) -> None:
        now = self.get_clock().now().to_msg()
        transforms = []
        for parent, child in (
            ("map", "odom"),
            ("odom", "base_footprint"),
            ("base_footprint", "laser"),
        ):
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = parent
            transform.child_frame_id = child
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self._static_broadcaster.sendTransform(transforms)

    def _publish_map(self) -> None:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.info.map_load_time = message.header.stamp
        message.info.resolution = 0.1
        message.info.width = 60
        message.info.height = 60
        message.info.origin.position.x = -3.0
        message.info.origin.position.y = -3.0
        message.info.origin.orientation.w = 1.0
        message.data = [0] * (message.info.width * message.info.height)
        self._map_publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = SmokeTestInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
