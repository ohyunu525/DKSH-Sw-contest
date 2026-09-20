"""Cross-runtime checks that keep the L1 RM simulation and ROS path aligned."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class LidarIntegrationConfigTests(unittest.TestCase):
    def test_hardware_launch_is_valid_python_and_uses_l1_topics(self):
        path = ROOT / "ros2/launch/l1_hardware.launch.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for value in (
            'default_value="/unilidar/cloud"',
            'default_value="/unilidar/imu"',
            'default_value="/scan"',
            '"range_min": 0.05',
            '"range_max": 30.0',
            '"scan_time": 1.0 / 11.0',
            'DeclareLaunchArgument("start_navigation", default_value="false")',
            'condition=IfCondition(LaunchConfiguration("start_navigation"))',
        ):
            with self.subTest(value=value):
                self.assertIn(value, source)

    def test_unity_scene_uses_the_same_l1_measurement_limits(self):
        scene = (ROOT / "DKSH SW Project/Assets/Scenes/LidarScene.unity").read_text(encoding="utf-8")
        expected = {
            "horizontalResolution": "180",
            "verticalResolution": "11",
            "horizontalFovDegrees": "360",
            "verticalFovDegrees": "90",
            "maxDistance": "30",
            "minDistance": "0.05",
            "measurementAccuracy": "0.02",
            "measurementResolution": "0.008",
            "scanFrequencyHz": "11",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertRegex(scene, rf"(?m)^\s+{re.escape(name)}: {re.escape(value)}$")

    def test_official_l1_driver_source_is_commit_pinned(self):
        dockerfile = (ROOT / "ros2/Dockerfile").read_text(encoding="utf-8")
        compose = (ROOT / "ros2/compose.yaml").read_text(encoding="utf-8")
        commit = "1bd7d95d8ab7ce7a22058d2bb07e39fd62612aa6"
        self.assertIn(f"ARG UNITREE_LIDAR_COMMIT={commit}", dockerfile)
        self.assertIn(f"UNITREE_LIDAR_COMMIT: {commit}", compose)

    def test_hardware_compose_keeps_navigation_opt_in(self):
        compose = (ROOT / "ros2/compose.lidar.yaml").read_text(encoding="utf-8")
        self.assertIn('start_navigation:=${L1_START_NAVIGATION:-false}', compose)

    def test_training_requires_completion_marker_and_new_checkpoint(self):
        trainer = (ROOT / "isaaclab_project/scripts/train.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "run_isaaclab.ps1").read_text(encoding="utf-8")
        marker = "DKSH_ISAACLAB_TRAIN_COMPLETE"
        self.assertIn(f'print("{marker}", flush=True)', trainer)
        self.assertIn(marker, wrapper)
        self.assertIn("-not $trainingCompleted -or -not $createdCheckpoint", wrapper)


if __name__ == "__main__":
    unittest.main()
