"""Reject gait residuals and wrong robots even when policy tensors look valid."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

SPEC = importlib.util.spec_from_file_location("joint_angle_checkpoint", Path(__file__).resolve().parents[1] / "scripts/joint_angle_checkpoint.py")
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)
TASK = "Isaac-DKSH-MG90S-CAD6-JointAngles-v1"


class CheckpointTests(unittest.TestCase):
    def create_checkpoint(self, root, environment="stairs", task=TASK, semantics="absolute_joint_angles_v1"):
        observations, actions = CHECKPOINT.expected_dimensions(task, environment)
        path = Path(root) / "model_100.pt"
        torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(4, observations), "actor.6.weight": torch.zeros(actions, 4)}, "iter": 100}, path)
        (path.parent / "run_metadata.json").write_text(json.dumps({
            "task": task, "observations": observations, "actions": actions, "action_semantics": semantics,
        }), encoding="utf-8")
        return path

    def test_nonflat_tasks_can_transfer_between_difficulties_and_courses(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            CHECKPOINT.validate_checkpoint(path, TASK, "rough")
            CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_flat_and_nonflat_observations_are_not_interchangeable(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat")
            with self.assertRaises(ValueError):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_wrong_robot_servo_or_residual_semantics_is_rejected(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            for task in (TASK.replace("CAD6", "CAD8"), TASK.replace("MG90S", "Spider")):
                with self.subTest(task=task), self.assertRaises(ValueError):
                    CHECKPOINT.validate_checkpoint(path, task, "stairs")
            path = self.create_checkpoint(root, semantics="wave_gait_residual")
            with self.assertRaises(ValueError):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_missing_metadata_and_mislabeled_network_dimensions_are_rejected(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(4, 68), "actor.6.weight": torch.zeros(18, 4)}}, path)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")
            (Path(root) / "run_metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "metadata"):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_reordered_joints_or_changed_angle_ranges_are_rejected(self):
        names = ["hip", "femur", "tibia"]
        limits = torch.tensor([[-1.4, 1.4], [-1.1, 1.4], [-0.4, 1.4]])
        metadata = {"joint_names": names, "angle_limits_rad": limits.tolist()}
        CHECKPOINT.validate_control_contract(metadata, names, limits)
        with self.assertRaises(ValueError):
            CHECKPOINT.validate_control_contract(metadata, list(reversed(names)), limits)
        with self.assertRaises(ValueError):
            CHECKPOINT.validate_control_contract(metadata, names, limits + 0.01)


if __name__ == "__main__":
    unittest.main()
