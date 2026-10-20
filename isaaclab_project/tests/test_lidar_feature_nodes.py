"""Exercise L2 ROS callbacks without requiring ROS2 on the development host."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2/dksh_l1_features"))


class FakeNode:
    def __init__(self, name):
        self.name, self.parameters, self.messages, self.warnings = name, {}, [], []

    def declare_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, message_type, topic, depth):
        self.output_topic = topic
        return types.SimpleNamespace(publish=self.messages.append)

    def create_subscription(self, *args):
        return types.SimpleNamespace()

    def get_logger(self):
        return types.SimpleNamespace(info=lambda _: None, warning=lambda message, **_: self.warnings.append(message))


class ArrayMessage:
    def __init__(self):
        self.layout = types.SimpleNamespace(dim=[])
        self.data = []


def load_node_module():
    modules = {
        "rclpy": types.SimpleNamespace(),
        "rclpy.node": types.SimpleNamespace(Node=FakeNode),
        "rclpy.qos": types.SimpleNamespace(qos_profile_sensor_data=object()),
        "sensor_msgs.msg": types.SimpleNamespace(PointCloud2=object),
        "sensor_msgs_py": types.SimpleNamespace(point_cloud2=types.SimpleNamespace(
            read_points=lambda message, **_: message.points)),
        "std_msgs.msg": types.SimpleNamespace(Float32MultiArray=ArrayMessage,
                                             MultiArrayDimension=lambda **kwargs: types.SimpleNamespace(**kwargs)),
    }
    path = ROOT / "ros2/dksh_l1_features/dksh_l1_features/node.py"
    spec = importlib.util.spec_from_file_location("dksh_l1_features.test_node_adapter", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class LidarFeatureNodeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_node_module()

    def cloud(self, points, frame="unilidar_lidar"):
        return types.SimpleNamespace(header=types.SimpleNamespace(frame_id=frame), points=points)

    def test_l2_callback_publishes_negative_slice_with_correct_topic_and_quantization(self):
        node = self.module.L2PolicyFeaturesNode()
        node._on_cloud(self.cloud([(math.cos(math.radians(3)), 0, -math.sin(math.radians(3)))]))
        self.assertEqual(node.output_topic, "/policy/l2_features")
        self.assertEqual(len(node.messages[0].data), 32)
        self.assertEqual(node.messages[0].data[16], 1.0)
        self.assertAlmostEqual(node.messages[0].data[0], 0.999 / 30.0)
        self.assertEqual(node.messages[0].layout.dim[0].label, "l2_policy_features")

    def test_wrong_coordinate_frame_is_rejected_before_publication(self):
        node = self.module.L2PolicyFeaturesNode()
        node._on_cloud(self.cloud([(1, 0, 0)], frame="base_footprint"))
        self.assertEqual(node.messages, [])
        self.assertEqual(len(node.warnings), 1)

    def test_legacy_l1_callback_keeps_its_original_contract(self):
        node = self.module.L1PolicyFeaturesNode()
        node._on_cloud(self.cloud([(1, 0, -0.01)]))
        self.assertEqual(node.output_topic, "/policy/l1_features")
        self.assertEqual(node.messages[0].data, [1.0] * 16 + [0.0] * 16)


if __name__ == "__main__":
    unittest.main()
