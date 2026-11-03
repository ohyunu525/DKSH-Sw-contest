"""Upgraded CAD6 regressions using independent CAD FK and tensor environment fixtures."""

import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import test_joint_angle_env as FIXTURE
import test_joint_angle_foot_geometry as GEOMETRY_TEST


FOLDER = FIXTURE.FOLDER
CONTROL = FIXTURE.load_helper("Upgraded_cad6_control")
FOOT = FIXTURE.load_helper("joint_angle_foot_geometry")
COLLISION = FIXTURE.load_helper("joint_angle_non_distal_geometry")


def Upgraded_kinematics():
    return CONTROL.UpgradedCad6Kinematics(
        FOOT.ONSHAPE_TOE_BUTTON_OFFSETS, COLLISION.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS,
        FOOT.ONSHAPE_PROXIMAL_FOOT_OFFSETS, dtype=torch.float64,
    )


def Upgraded_environment():
    path = FOLDER / "Upgraded_cad6_env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import)
                 and not (isinstance(node, ast.ImportFrom) and node.module != "__future__")
                 and not (isinstance(node, ast.ClassDef) and node.name.endswith("Cfg"))]
    namespace = {
        "torch": torch, "DirectRLEnv": FIXTURE.Parent, "JointAngleNavigationEnv": FIXTURE.Environment,
        "normalize_angles": FIXTURE.CONTROL.normalize_angles,
        "UpgradedCad6Kinematics": CONTROL.UpgradedCad6Kinematics,
        "UpgradedFootTrajectory": CONTROL.UpgradedFootTrajectory,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    base = FIXTURE.StrictContactEnvironmentTests().strict_env(behavior_profile="revised_v12")
    settings = vars(base.cfg).copy()
    settings.update(
        history_frames=4, gait_period=2.0, gait_duty=0.65, gait_lift=0.02, foothold_offset_limit=0.015,
        ik_clearance=0.008, target_speed=0.03, imu_angular_noise=0.01, imu_gravity_noise=0.005,
        toe_button_offsets=FOOT.ONSHAPE_TOE_BUTTON_OFFSETS,
        non_distal_collision_offsets=COLLISION.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS,
        proximal_foot_offsets=FOOT.ONSHAPE_PROXIMAL_FOOT_OFFSETS,
        robot=SimpleNamespace(copy=lambda: deepcopy(base.cfg.robot)),
    )
    env = namespace["UpgradedCad6Env"](SimpleNamespace(**settings))
    env._reset_idx(torch.tensor([0, 1]))
    env._out_of_bounds = torch.zeros(2, dtype=torch.bool)
    return env


class UpgradedKinematicsTests(unittest.TestCase):
    def test_fk_matches_independent_urdf_toe_coordinates(self):
        kinematics = Upgraded_kinematics()
        GEOMETRY_TEST.FootGeometryTests.setUpClass()
        independent = GEOMETRY_TEST.FootGeometryTests()
        q = torch.tensor([[[0.1, -0.2, -0.4]] * 6], dtype=torch.float64)
        actual = kinematics.forward(q)[0][0].numpy()
        for leg in range(6):
            angles = {f"leg_{leg}_{part}_joint": float(q[0, leg, index])
                      for index, part in enumerate(("hip", "femur", "tibia"))}
            frame = independent.world_frame(f"leg_{leg}_tibia", angles)
            expected = GEOMETRY_TEST.transform(frame, FOOT.ONSHAPE_TOE_BUTTON_OFFSETS[leg])
            np.testing.assert_allclose(actual[leg], expected, atol=1e-8)

    def test_ik_tracks_constant_height_stance_and_survives_servo_filter(self):
        kinematics = Upgraded_kinematics()
        q = torch.tensor([0.0, -0.3, -0.3], dtype=torch.float64).repeat(3, 6, 1)
        home = kinematics.forward(q)[0]
        home[..., 2] = home[..., 2].amin(-1, keepdim=True)
        trajectory = CONTROL.UpgradedFootTrajectory(home)
        command = q.new_tensor([[0.03, 0, 0], [0, -0.03, 0], [0, 0, 0.12]])
        filtered = q.clone()
        previous_stance = trajectory.phase() < trajectory.duty
        previous_z = home[..., 2].clone()
        for step in range(250):
            target = trajectory.advance(command, q.new_zeros((3, 18)), 0.02)
            q, accepted, error = kinematics.inverse(target, q)
            self.assertTrue(accepted.all(), f"IK rejected a nominal target at step {step}: {error}")
            actual = kinematics.forward(q)[0]
            torch.testing.assert_close(actual, target, atol=0.0001, rtol=0)
            stance = trajectory.phase() < trajectory.duty
            continuous_stance = stance & previous_stance
            torch.testing.assert_close(actual[..., 2][continuous_stance], previous_z[continuous_stance], atol=1e-6, rtol=0)
            previous_z = actual[..., 2].clone()
            previous_stance = stance
            limits = kinematics.limits.expand(3, -1, -1, -1)
            filtered = FIXTURE.CONTROL.bounded_angle_targets(
                FIXTURE.CONTROL.normalize_angles(q, limits), filtered, limits, dt=0.02, rate_limit=1.5, smoothing=0.35,
            )
            if step > 100:
                # Exclude the immediate post-touchdown filter transient.
                midstance = (trajectory.phase() > 0.15) & (trajectory.phase() < 0.55)
                z_error = (kinematics.forward(filtered)[0][..., 2] - home[..., 2]).abs()
                if midstance.any():
                    self.assertLess(float(z_error[midstance].max()), 0.002)

    def test_invalid_or_unreachable_ik_preserves_last_command(self):
        kinematics = Upgraded_kinematics()
        seed = torch.tensor([0.0, -0.3, -0.3], dtype=torch.float64).repeat(1, 6, 1)
        target = kinematics.forward(seed)[0]
        target[0, 0, 0] = float("nan")
        target[0, 1, 0] += 3.0
        angles, valid, _ = kinematics.inverse(target, seed)
        self.assertFalse(valid[0, 0])
        self.assertFalse(valid[0, 1])
        torch.testing.assert_close(angles[0, :2], seed[0, :2])
        self.assertTrue(torch.isfinite(angles).all())

    def test_actions_are_latched_and_partial_reset_is_local(self):
        home = torch.tensor([[[0.2, 0.1, -0.2]] * 6] * 2)
        gait = CONTROL.UpgradedFootTrajectory(home)
        gait.elapsed[:] = 1.28
        gait.advance(torch.tensor([[0.03, 0, 0]] * 2), torch.ones(2, 18), 0.03)
        first_end = gait.end.clone()
        with torch.inference_mode():
            for _ in range(5):
                gait.advance(torch.tensor([[0.03, 0, 0]] * 2), -torch.ones(2, 18), 0.02)
        torch.testing.assert_close(gait.end[:, ::2], first_end[:, ::2])
        second = gait.features()[1].clone()
        gait.reset(torch.tensor([0]))
        torch.testing.assert_close(gait.position[0], home[0])
        torch.testing.assert_close(gait.features()[1], second)


class UpgradedEnvironmentTests(unittest.TestCase):
    def test_actor_excludes_privileged_state_and_history_is_idempotent(self):
        env = Upgraded_environment()
        first = env._get_observations()
        self.assertEqual(first["policy"].shape, (2, 744))
        self.assertEqual(first["critic"].shape, (2, 884))
        self.assertEqual(env.cfg.state_space, 884)
        torch.testing.assert_close(env._get_observations()["policy"], first["policy"])
        env._robot.data.joint_pos += 0.05
        env._robot.data.joint_vel += 0.5
        env._robot.data.root_lin_vel_b += 0.1
        second = env._get_observations()
        torch.testing.assert_close(first["policy"], second["policy"])
        self.assertFalse(torch.equal(first["critic"], second["critic"]))
        env.common_step_counter += 1
        third = env._get_observations()
        torch.testing.assert_close(third["policy"][:, :-env.frame_size], first["policy"][:, env.frame_size:])

    def test_inner_control_and_partial_reset(self):
        env = Upgraded_environment()
        env._get_observations()
        previous = env.joint_angle_targets_rad.clone()
        env._pre_physics_step(torch.zeros(2, 18))
        with torch.inference_mode():
            for _ in range(5):
                env._advance_angle_targets(0.02)
        self.assertLessEqual(float((env.joint_angle_targets_rad - previous).abs().max()), 0.15 + 1e-6)
        env.common_step_counter += 1
        before = env._get_observations()["policy"].clone()
        env._reset_idx(torch.tensor([0]))
        after = env._get_observations()["policy"]
        torch.testing.assert_close(before[1], after[1])
        self.assertEqual(float(env._trajectory.elapsed[0]), 0)
        self.assertGreater(float(env._trajectory.elapsed[1]), 0)
        with self.assertRaises(ValueError):
            env._pre_physics_step(torch.full((2, 18), float("nan")))


class UpgradedRunnerTests(unittest.TestCase):
    def test_checkpoint_write_is_atomic_and_uses_upgraded_prefix(self):
        script = Path(__file__).resolve().parents[1] / "scripts/Upgraded_cad6_runner.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        definition = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "UpgradedRunner")

        class StorageRunner:
            fail = False

            def save(self, path, infos=None):
                if self.fail:
                    Path(path).write_text("interrupted write")
                    raise OSError("simulated disk failure")
                torch.save({"infos": infos}, path)

        contract = {"task": "Upgraded-CAD6-PMTG-v0"}
        namespace = {"OnPolicyRunner": StorageRunner, "Path": Path, "NamedTemporaryFile": tempfile.NamedTemporaryFile,
                     "contract": contract}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(script), "exec"), namespace)
        runner = namespace["UpgradedRunner"]()
        with tempfile.TemporaryDirectory(prefix="Upgraded_atomic_") as directory:
            requested = Path(directory) / "model_1.pt"
            runner.save(str(requested))
            saved = Path(directory) / "Upgraded_model_1.pt"
            original = saved.read_bytes()
            self.assertEqual(torch.load(saved, weights_only=True)["infos"]["Upgraded_contract"], contract)
            runner.fail = True
            with self.assertRaises(OSError):
                runner.save(str(requested))
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [saved])

    def test_baseline_gate_and_checkpoint_contract(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("Upgraded_runner_test", scripts / "Upgraded_cad6_runner.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
        finally:
            sys.path.pop(0)
        contract = {"task": "Upgraded-CAD6-PMTG-v0", "actor_observations": 744}
        with tempfile.TemporaryDirectory(prefix="Upgraded_test_") as directory:
            report_path = Path(directory) / "Upgraded_baseline.json"
            report = {"Upgraded_contract": contract, "mode": "baseline", "environment": "flat",
                      "locomotion_pass": False, "Upgraded_ik_rejection_fraction": 0}
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                runner.Upgraded_validate_baseline(report_path, contract)
            report["locomotion_pass"] = True
            report_path.write_text(json.dumps(report))
            runner.Upgraded_validate_baseline(report_path, contract)
            checkpoint = Path(directory) / "Upgraded_model_0.pt"
            torch.save({"infos": {"Upgraded_contract": contract}}, checkpoint)
            runner.Upgraded_validate_checkpoint(checkpoint, contract)
            with self.assertRaises(ValueError):
                runner.Upgraded_validate_checkpoint(checkpoint, {**contract, "actor_observations": 140})


if __name__ == "__main__":
    unittest.main()
