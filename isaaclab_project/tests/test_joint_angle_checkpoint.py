"""Reject gait residuals and wrong robots even when policy tensors look valid."""

import importlib.util
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

SPEC = importlib.util.spec_from_file_location("joint_angle_checkpoint", Path(__file__).resolve().parents[1] / "scripts/joint_angle_checkpoint.py")
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)
TASK = "Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2"


class CheckpointTests(unittest.TestCase):
    def test_original_v7_com_checkpoint_and_distal_v7_are_distinct(self):
        old = {**CHECKPOINT.BEHAVIOR_PROFILES["revised_v7_com"], "behavior_profile": "revised_v7"}
        new = {**CHECKPOINT.BEHAVIOR_PROFILES["revised_v7"], "behavior_profile": "revised_v7"}
        self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(old), "revised_v7_com")
        self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(new), "revised_v7")
        old["behavior_profile"] = "revised_v6"
        with self.assertRaises(ValueError):
            CHECKPOINT.behavior_profile_from_metadata(old)

    def test_tracked_model_350_preserves_legacy_behavior_and_requires_explicit_training_transfer(self):
        checkpoint = (Path(__file__).resolve().parents[2]
                      / "logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2/2026-10-23_00-54-23_673348/model_350.pt")
        self.assertTrue(checkpoint.is_file(), "Tracked model_350 regression artifact is missing")
        metadata = CHECKPOINT.validate_checkpoint(checkpoint, TASK, "flat")
        self.assertEqual((metadata["observations"], metadata["actions"]), (140, 18))
        self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(metadata), "legacy_v2")
        revised = CHECKPOINT.BEHAVIOR_PROFILES["revised_v3"]
        compatibility = {"expected_reward_contract": revised["reward_contract"],
                         "expected_contact_contract": revised["contact_contract"]}
        with self.assertRaisesRegex(ValueError, "allow_behavior_transfer"):
            CHECKPOINT.validate_checkpoint(checkpoint, TASK, "flat", **compatibility)
        CHECKPOINT.validate_checkpoint(checkpoint, TASK, "flat", allow_behavior_transfer=True, **compatibility)
        CHECKPOINT.validate_control_contract(
            metadata, [f"leg_{index}_{joint}_joint" for index in range(6) for joint in ("hip", "femur", "tibia")],
            torch.tensor([[-math.pi / 2, math.pi / 2]] * 18),
            target_rate_limit=1.5, target_smoothing=0.35, control_frequency_hz=50,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        std = saved["model_state_dict"]["std"]
        self.assertAlmostEqual(float(std.mean()), 0.013206, places=6)
        self.assertGreater(float(std.min()), 0.0084)
        self.assertLess(float(std.max()), 0.0175)
        self.assertAlmostEqual(saved["optimizer_state_dict"]["param_groups"][0]["lr"], 1e-5)

    def create_checkpoint(self, root, environment="stairs", task=TASK, semantics="absolute_joint_angles_v2", profile=None):
        observations, actions = CHECKPOINT.expected_dimensions(task, environment)
        path = Path(root) / "model_100.pt"
        torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(4, observations), "actor.6.weight": torch.zeros(actions, 4), "std": torch.full((actions,), 0.02)}, "iter": 100}, path)
        (path.parent / "run_metadata.json").write_text(json.dumps({
            "task": task, "observations": observations, "actions": actions, "action_semantics": semantics,
            "actuator_contract": CHECKPOINT.expected_actuator_contract(task),
            "cad_asset_sha256": CHECKPOINT.expected_cad_asset_sha256(task),
            **CHECKPOINT.SCHEMA_CONTRACTS,
            **({**CHECKPOINT.BEHAVIOR_PROFILES[profile], "behavior_profile": profile} if profile else {}),
            "target_rate_limit_rad_s": 1.5, "target_smoothing": 0.35,
            "control_frequency_hz": 50, "exploration_std_cap": 0.0211844041781,
        }), encoding="utf-8")
        return path

    def test_nonflat_tasks_can_transfer_between_difficulties_and_courses(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            CHECKPOINT.validate_checkpoint(path, TASK, "rough")
            CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_flat_and_nonflat_share_complete_v2_observation_schema(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat")
            CHECKPOINT.validate_checkpoint(path, TASK, "stairs")
            self.assertEqual(CHECKPOINT.expected_dimensions(TASK, "flat"), (140, 18))
            self.assertEqual(CHECKPOINT.expected_dimensions(TASK.replace("CAD6", "CAD8"), "stairs"), (166, 24))

    def test_wrong_robot_servo_or_residual_semantics_is_rejected(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            for task in (TASK.replace("CAD6", "CAD8"), TASK.replace("MG996R-Hip", "Spider")):
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
        task = "Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2"
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

    def test_v1_tasks_and_semantics_fail_with_clear_retraining_message(self):
        with self.assertRaisesRegex(ValueError, "retrain"):
            CHECKPOINT.require_v2_task(TASK.replace("v2", "v1"))
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, semantics="absolute_joint_angles_v1")
            with self.assertRaisesRegex(ValueError, "retrain"):
                CHECKPOINT.validate_checkpoint(path, TASK, "flat")

    def test_removed_servo_tasks_are_rejected_before_loading_checkpoint_data(self):
        for task in ("Isaac-DKSH-MG90S-CAD6-JointAngles-v2", "Isaac-DKSH-MG90S-JointAngles-v2"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "removed"):
                CHECKPOINT.validate_checkpoint("missing_model.pt", task, "flat")

    def test_every_behavior_contract_is_required_even_on_flat_terrain(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat")
            metadata_path = path.parent / "run_metadata.json"
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in CHECKPOINT.SCHEMA_CONTRACTS:
                altered = original.copy()
                altered.pop(key)
                metadata_path.write_text(json.dumps(altered), encoding="utf-8")
                with self.subTest(contract=key), self.assertRaisesRegex(ValueError, "contract"):
                    CHECKPOINT.validate_checkpoint(path, TASK, "flat")

    def test_exploration_parameter_must_respect_saved_cap(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            state = torch.load(path, weights_only=True)
            state["model_state_dict"]["std"].fill_(0.12)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "exploration std"):
                CHECKPOINT.validate_checkpoint(path, TASK, "stairs")

    def test_resume_requires_matching_applied_control_dynamics(self):
        names = ["hip", "femur", "tibia"]
        limits = torch.tensor([[-1.4, 1.4], [-1.1, 1.4], [-0.4, 1.4]])
        metadata = {"joint_names": names, "angle_limits_rad": limits.tolist(),
                    "target_rate_limit_rad_s": 1.5, "target_smoothing": 0.35,
                    "control_frequency_hz": 50, "exploration_std_cap": 0.02}
        CHECKPOINT.validate_control_contract(metadata, names, limits, target_rate_limit=1.5,
                                             target_smoothing=0.35, control_frequency_hz=50)
        for parameter, value in (("target_rate_limit", 2), ("target_smoothing", 0.5),
                                 ("control_frequency_hz", 100)):
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                CHECKPOINT.validate_control_contract(metadata, names, limits, **{parameter: value})

    def test_std_bounds_are_saved_optimizer_preferences_not_physical_control(self):
        names, limits = ["hip"], torch.tensor([[-1.4, 1.4]])
        metadata = {"joint_names": names, "angle_limits_rad": limits.tolist(), "exploration_std_cap": 0.021}
        CHECKPOINT.validate_control_contract(metadata, names, limits)
        metadata["exploration_std_cap"] = 0.30
        CHECKPOINT.validate_control_contract(metadata, names, limits)

    def test_old_v2_model_retains_its_saved_std_cap_and_known_behavior(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(CHECKPOINT.BEHAVIOR_PROFILES["legacy_v2"])
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = CHECKPOINT.validate_checkpoint(path, TASK, "flat")
            self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(loaded), "legacy_v2")
            revised = CHECKPOINT.BEHAVIOR_PROFILES["revised_v3"]
            with self.assertRaisesRegex(ValueError, "allow_behavior_transfer"):
                CHECKPOINT.validate_checkpoint(path, TASK, "flat", expected_reward_contract=revised["reward_contract"],
                                               expected_contact_contract=revised["contact_contract"])
            CHECKPOINT.validate_checkpoint(path, TASK, "flat", expected_reward_contract=revised["reward_contract"],
                                           expected_contact_contract=revised["contact_contract"], allow_behavior_transfer=True)
            legacy = CHECKPOINT.BEHAVIOR_PROFILES["legacy_v2"]
            CHECKPOINT.validate_checkpoint(path, TASK, "flat", expected_reward_contract=legacy["reward_contract"],
                                           expected_contact_contract=legacy["contact_contract"])

    def test_new_profile_validates_actual_std_against_its_own_minimum_and_maximum(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(exploration_profile=CHECKPOINT.EXPLORATION_PROFILE, exploration_initial_std=0.08,
                            exploration_min_std=0.03, exploration_max_std=0.30, exploration_std_cap=0.30)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            state = torch.load(path, weights_only=True)
            for value, valid in ((0.08, True), (0.3, True), (0.01, False), (0.31, False)):
                state["model_state_dict"]["std"].fill_(value)
                torch.save(state, path)
                with self.subTest(value=value):
                    if valid:
                        CHECKPOINT.validate_checkpoint(path, TASK, "flat")
                    else:
                        with self.assertRaisesRegex(ValueError, "exploration std"):
                            CHECKPOINT.validate_checkpoint(path, TASK, "flat")

    def test_transfer_does_not_accept_unknown_or_mixed_behavior_contracts(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root)
            metadata_path = path.parent / "run_metadata.json"
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            for changed in ({"reward_contract": "unknown"},
                            {"contact_contract": CHECKPOINT.BEHAVIOR_PROFILES["legacy_v2"]["contact_contract"]},
                            {"behavior_profile": "legacy_v2"}):
                metadata = original | changed
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "contract|profile"):
                    CHECKPOINT.validate_checkpoint(path, TASK, "flat", allow_behavior_transfer=True)

    def test_full_foot_slip_preserves_saved_profiles_and_requires_explicit_transfer(self):
        expected_rewards = {
            "revised_v4": "signed_progress_huber_speed_v3",
            "revised_v5": "signed_progress_huber_speed_knee_clearance_v1",
            "revised_v6": "signed_progress_huber_speed_knee_clearance_full_foot_slip_v1",
        }
        current = CHECKPOINT.BEHAVIOR_PROFILES["revised_v6"]
        compatibility = {"expected_reward_contract": current["reward_contract"],
                         "expected_contact_contract": current["contact_contract"]}
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat")
            metadata_path = path.parent / "run_metadata.json"
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            for profile, reward in expected_rewards.items():
                with self.subTest(profile=profile):
                    contracts = CHECKPOINT.BEHAVIOR_PROFILES[profile]
                    self.assertEqual(contracts["reward_contract"], reward)
                    self.assertEqual(contracts["contact_contract"], "cad6_toe_button_hysteresis_v1")
                    metadata = original | contracts | {"behavior_profile": profile}
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    saved = CHECKPOINT.validate_checkpoint(path, TASK, "flat")
                    self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(saved), profile)
                    self.assertEqual((saved["observations"], saved["actions"]), (140, 18))
                    if profile != "revised_v6":
                        with self.assertRaisesRegex(ValueError, f"--behavior_profile={profile}.*--allow_behavior_transfer"):
                            CHECKPOINT.validate_checkpoint(path, TASK, "flat", **compatibility)
                    CHECKPOINT.validate_checkpoint(path, TASK, "flat", allow_behavior_transfer=True,
                                                   **compatibility)

    def test_distal_support_preserves_all_saved_profiles_and_requires_explicit_training_transfer(self):
        current = CHECKPOINT.BEHAVIOR_PROFILES["revised_v7"]
        self.assertEqual(current, {
            "reward_contract": "signed_progress_huber_speed_distal_support_v1",
            "contact_contract": "cad6_distal_support_toe_button_v1",
        })
        compatibility = {"expected_reward_contract": current["reward_contract"],
                         "expected_contact_contract": current["contact_contract"]}
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat")
            metadata_path = path.parent / "run_metadata.json"
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            for profile, contracts in CHECKPOINT.BEHAVIOR_PROFILES.items():
                with self.subTest(profile=profile):
                    metadata = original | contracts | {"behavior_profile": profile}
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    saved = CHECKPOINT.validate_checkpoint(path, TASK, "flat")
                    self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(saved), profile)
                    if profile != "revised_v7":
                        with self.assertRaisesRegex(ValueError, f"--behavior_profile={profile}.*--allow_behavior_transfer"):
                            CHECKPOINT.validate_checkpoint(path, TASK, "flat", **compatibility)
                    else:
                        CHECKPOINT.validate_checkpoint(path, TASK, "flat", **compatibility)
                    CHECKPOINT.validate_checkpoint(path, TASK, "flat", allow_behavior_transfer=True,
                                                   **compatibility)

    def test_strict_support_roundtrips_and_rejects_same_size_older_behavior_without_transfer(self):
        strict = CHECKPOINT.BEHAVIOR_PROFILES["revised_v8"]
        self.assertEqual(strict, {
            "reward_contract": "signed_progress_huber_speed_strict_support_v1",
            "contact_contract": "cad6_strict_distal_all_links_v1",
        })
        compatibility = {"expected_reward_contract": strict["reward_contract"],
                         "expected_contact_contract": strict["contact_contract"]}
        with TemporaryDirectory() as root:
            for profile in CHECKPOINT.BEHAVIOR_PROFILES:
                with self.subTest(profile=profile):
                    path = self.create_checkpoint(root, "flat", profile=profile)
                    saved = CHECKPOINT.validate_checkpoint(path, TASK, "flat")
                    self.assertEqual((saved["observations"], saved["actions"]), (140, 18))
                    self.assertEqual(CHECKPOINT.behavior_profile_from_metadata(saved), profile)
                    if profile == "revised_v8":
                        CHECKPOINT.validate_checkpoint(path, TASK, "flat", **compatibility)
                    else:
                        with self.assertRaisesRegex(ValueError, f"--behavior_profile={profile}.*--allow_behavior_transfer"):
                            CHECKPOINT.validate_checkpoint(path, TASK, "flat", **compatibility)
                        CHECKPOINT.validate_checkpoint(path, TASK, "flat", allow_behavior_transfer=True,
                                                       **compatibility)
                        historical = CHECKPOINT.BEHAVIOR_PROFILES[profile]
                        CHECKPOINT.validate_checkpoint(path, TASK, "flat",
                                                       expected_reward_contract=historical["reward_contract"],
                                                       expected_contact_contract=historical["contact_contract"])

    def test_strict_checkpoint_discovery_skips_newer_historical_models(self):
        strict = CHECKPOINT.BEHAVIOR_PROFILES["revised_v8"]
        compatibility = {"expected_reward_contract": strict["reward_contract"],
                         "expected_contact_contract": strict["contact_contract"]}
        with TemporaryDirectory() as root:
            strict_dir, old_dir = Path(root) / "strict", Path(root) / "historical"
            strict_dir.mkdir()
            old_dir.mkdir()
            strict_path = self.create_checkpoint(strict_dir, "flat", profile="revised_v8")
            old_path = self.create_checkpoint(old_dir, "flat", profile="revised_v7")
            os.utime(strict_path, (100, 100))
            os.utime(old_path, (200, 200))
            resolved = CHECKPOINT.resolve_checkpoint("", root, TASK, "flat", **compatibility)
            self.assertEqual(resolved, strict_path.resolve())
            with self.assertRaisesRegex(ValueError, "explicit historical comparison"):
                CHECKPOINT.resolve_checkpoint(str(old_path), root, TASK, "flat", **compatibility)

    def test_transfer_cannot_relabel_historical_contacts_as_strict_v8(self):
        with TemporaryDirectory() as root:
            path = self.create_checkpoint(root, "flat", profile="revised_v7")
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["behavior_profile"] = "revised_v8"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile disagrees"):
                CHECKPOINT.validate_checkpoint(path, TASK, "flat", allow_behavior_transfer=True)


if __name__ == "__main__":
    unittest.main()
