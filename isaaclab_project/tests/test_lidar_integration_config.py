"""Cross-runtime checks that keep the L2 simulation and ROS path aligned."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class LidarIntegrationConfigTests(unittest.TestCase):
    def test_hardware_launch_is_valid_python_and_uses_l2_topics(self):
        path = ROOT / "ros2/launch/l2_hardware.launch.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for value in (
            'default_value="/unilidar/cloud"',
            'default_value="/unilidar/imu"',
            'default_value="/scan"',
            '"range_min": 0.05',
            '"range_max": 30.0',
            '"scan_time": 1.0 / 5.55',
            'DeclareLaunchArgument("start_navigation", default_value="false")',
            'condition=IfCondition(LaunchConfiguration("start_navigation"))',
            'package="dksh_l1_features"',
            '"output_topic": "/policy/l2_features"',
            'DeclareLaunchArgument("initialize_type", default_value="2")',
            'DeclareLaunchArgument("work_mode", default_value="1")',
            'DeclareLaunchArgument("cloud_scan_num", default_value="39")',
            'remappings=[("/tf", "/unilidar/sdk_tf")]',
        ):
            with self.subTest(value=value):
                self.assertIn(value, source)

    def test_unity_scene_uses_the_same_l2_measurement_limits(self):
        scene = (ROOT / "DKSH SW Project/Assets/Scenes/LidarScene.unity").read_text(encoding="utf-8")
        expected = {
            "horizontalResolution": "360",
            "verticalResolution": "32",
            "horizontalFovDegrees": "360",
            "verticalFovDegrees": "96",
            "verticalMinAngleDegrees": "-6",
            "maxDistance": "30",
            "minDistance": "0.05",
            "measurementAccuracy": "0.02",
            "measurementResolution": "0.0045",
            "scanFrequencyHz": "5.55",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertRegex(scene, rf"(?m)^\s+{re.escape(name)}: {re.escape(value)}$")

    def test_official_l2_driver_source_is_commit_pinned(self):
        dockerfile = (ROOT / "ros2/Dockerfile").read_text(encoding="utf-8")
        compose = (ROOT / "ros2/compose.yaml").read_text(encoding="utf-8")
        commit = "0e3c51f512e6b8ff60b8c32f160b412cb48445c2"
        self.assertIn(f"ARG UNITREE_LIDAR_COMMIT={commit}", dockerfile)
        self.assertIn(f"UNITREE_LIDAR_COMMIT: {commit}", compose)
        self.assertIn("https://github.com/unitreerobotics/unilidar_sdk2.git", dockerfile)

    def test_hardware_compose_keeps_navigation_opt_in(self):
        for file in ("compose.lidar.yaml", "compose.lidar.serial.yaml"):
            compose = (ROOT / "ros2" / file).read_text(encoding="utf-8")
            self.assertIn('start_navigation:=${L2_START_NAVIGATION:-false}', compose)
            self.assertIn('/l2_policy_features', compose)

    def test_udp_and_serial_select_matching_sdk2_modes(self):
        udp = (ROOT / "ros2/compose.lidar.yaml").read_text(encoding="utf-8")
        serial = (ROOT / "ros2/compose.lidar.serial.yaml").read_text(encoding="utf-8")
        self.assertIn("initialize_type:=2", udp)
        self.assertIn("work_mode:=1", udp)
        self.assertIn("6201:6201/udp", udp)
        self.assertNotIn("devices:", udp)
        self.assertIn("initialize_type:=1", serial)
        self.assertIn("work_mode:=9", serial)
        self.assertIn("baudrate:=4000000", serial)

    def test_policy_feature_encoder_is_built_into_ros_image(self):
        dockerfile = (ROOT / "ros2/Dockerfile").read_text(encoding="utf-8")
        package = ROOT / "ros2/dksh_l1_features/package.xml"
        setup = ROOT / "ros2/dksh_l1_features/setup.py"
        self.assertTrue(package.is_file() and setup.is_file())
        self.assertIn("COPY dksh_l1_features", dockerfile)
        self.assertIn("unitree_lidar_ros2 dksh_l1_features", dockerfile)
        self.assertIn("l2_policy_features = dksh_l1_features.node:main_l2", setup.read_text(encoding="utf-8"))

    def test_new_navigation_runs_use_l2_log_paths(self):
        wrapper = (ROOT / "run_isaaclab.ps1").read_text(encoding="utf-8")
        runner = (ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/agents/rsl_rl_ppo_cfg.py").read_text(encoding="utf-8")
        self.assertIn("navigation_l2v1", wrapper)
        self.assertIn("navigation_l2v1", runner)
        self.assertNotIn("navigation_l1v2", wrapper + runner)

    def test_training_requires_completion_marker_and_new_checkpoint(self):
        trainer = (ROOT / "isaaclab_project/scripts/train.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "run_isaaclab.ps1").read_text(encoding="utf-8")
        marker = "DKSH_ISAACLAB_TRAIN_COMPLETE"
        self.assertIn(f'print("{marker}", flush=True)', trainer)
        self.assertIn(marker, wrapper)
        self.assertIn("-not $trainingCompleted -or -not $createdCheckpoint", wrapper)


if __name__ == "__main__":
    unittest.main()
