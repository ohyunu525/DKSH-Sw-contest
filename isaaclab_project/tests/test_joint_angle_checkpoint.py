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
            "actuator_contract": CHECKPOINT.expected_actuator_contract(task),
            "cad_asset_sha256": CHECKPOINT.expected_cad_asset_sha256(task),
            "terrain_observation_contract": CHECKPOINT.TERRAIN_OBSERVATION_CONTRACT,
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

    def test_current_cad6_rejects_checkpoint_from_the_old_geometry(self):
        task = "Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v1"
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, task=task)
            CHECKPOINT.validate_checkpoint(path, task, "stairs")
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["cad_asset_sha256"] = "old_cad_geometry"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CAD asset"):
                CHECKPOINT.validate_checkpoint(path, task, "stairs")

    def test_same_size_old_terrain_features_cannot_be_reused(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("terrain_observation_contract")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terrain observation contract"):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")


if __name__ == "__main__":
    unittest.main()
