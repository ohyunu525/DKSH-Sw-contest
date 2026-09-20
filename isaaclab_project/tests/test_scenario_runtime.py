"""Runtime state-machine unit tests with real torch and Isaac Lab unit doubles.

These tests do not launch PhysX and cannot verify collision, friction, USD
spawning, sensor registration, or actual platform motion. They exercise tensor
state transitions and the buffers submitted to the simulator's asset APIs.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def _load_runtime_with_unit_doubles():
    """Keep substitute simulator modules scoped to these isolated imports."""
    folder = (
        Path(__file__).resolve().parents[1]
        / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"
    )
    package_name = "_dksh_runtime_unit_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    module_names = (
        "isaaclab", "isaaclab.sim", "isaaclab.assets", "isaaclab.sensors",
        "isaaclab.utils", "isaaclab.utils.math", "pxr", "pxr.UsdPhysics",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    modules[package_name] = package
    modules["isaaclab"].sim = modules["isaaclab.sim"]
    modules["pxr"].UsdPhysics = modules["pxr.UsdPhysics"]
    modules["isaaclab.assets"].RigidObject = object
    modules["isaaclab.assets"].RigidObjectCfg = object
    modules["isaaclab.sensors"].ContactSensor = object
    modules["isaaclab.sensors"].ContactSensorCfg = object

    def inverse_rotation(quat, vector):
        # Quaternion convention shared by the robot state is wxyz.
        xyz = quat[:, 1:]
        uv = 2.0 * torch.cross(xyz, vector, dim=-1)
        return vector - quat[:, :1] * uv + torch.cross(xyz, uv, dim=-1)

    def forward_rotation(quat, vector):
        xyz = quat[:, 1:]
        uv = 2.0 * torch.cross(xyz, vector, dim=-1)
        return vector + quat[:, :1] * uv + torch.cross(xyz, uv, dim=-1)

    modules["isaaclab.utils.math"].quat_rotate = forward_rotation
    modules["isaaclab.utils.math"].quat_rotate_inverse = inverse_rotation
    with patch.dict(sys.modules, modules):
        loaded = {}
        for short_name in ("scenario_layout", "scenario_sensing", "scenario_runtime"):
            name = f"{package_name}.{short_name}"
            spec = importlib.util.spec_from_file_location(name, folder / f"{short_name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[short_name] = module
    return loaded["scenario_runtime"], loaded["scenario_layout"]


if torch is not None:
    RUNTIME, LAYOUT = _load_runtime_with_unit_doubles()


class FakeRigidObject:
    def __init__(self, origins, position):
        count = len(origins)
        default = torch.zeros((count, 13))
        default[:, :3] = torch.tensor(position)
        default[:, 3] = 1.0
        current = default.clone()
        current[:, :3] += origins
        self.data = types.SimpleNamespace(
            default_root_state=default,
            root_state_w=current,
            root_pos_w=current[:, :3],
            root_lin_vel_w=current[:, 7:10],
        )
        self._ALL_INDICES = torch.arange(count, dtype=torch.int32)
        self.writes = []
        self.velocity_writes = []
        self.resets = []

    def reset(self, env_ids):
        self.resets.append(env_ids.clone())

    def write_root_state_to_sim(self, state, env_ids=None):
        ids = self._ALL_INDICES if env_ids is None else env_ids
        self.data.root_state_w[ids] = state
        self.writes.append((state.clone(), ids.clone()))

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        ids = self._ALL_INDICES if env_ids is None else env_ids
        self.data.root_state_w[ids, 7:] = velocity
        self.velocity_writes.append((velocity.clone(), ids.clone()))


@unittest.skipIf(torch is None, "PyTorch is required; use the Isaac Lab Python environment")
class ScenarioRuntimeTests(unittest.TestCase):
    def make_runtime(self, preset="mixed", difficulty=0.5, count=4):
        torch.manual_seed(71)
        origins = torch.zeros((count, 3))
        origins[:, 0] = torch.arange(count) * 10.0
        origins[:, 2] = torch.arange(count) * 0.25
        layout = LAYOUT.build_layout(preset, difficulty, robot_width=0.69, robot_height=0.28)
        robot_pos = origins + torch.tensor(layout.spawn)
        robot_pos[:, 2] += 0.2
        robot_quat = torch.zeros((count, 4))
        robot_quat[:, 0] = 1.0
        env = types.SimpleNamespace(
            num_envs=count,
            device="cpu",
            physics_dt=0.02,
            cfg=types.SimpleNamespace(
                debris_impact_threshold=2.0,
                lidar_min_range_m=0.05,
                lidar_max_range_m=30.0,
                lidar_horizontal_scan_frequency_hz=11.0,
                lidar_vertical_fov_deg=90.0,
                lidar_vertical_projection_bins=3,
                lidar_azimuth_samples_per_bin=5,
                lidar_measurement_accuracy_m=0.02,
                lidar_measurement_resolution_m=0.008,
                lidar_observation_bins=16,
                lidar_noise_enabled=False,
                lidar_mount_position_b=(0.0, 0.0, 0.030),
            ),
            scene=types.SimpleNamespace(env_origins=origins),
            _robot=types.SimpleNamespace(data=types.SimpleNamespace(
                root_pos_w=robot_pos, root_quat_w=robot_quat,
            )),
        )
        # Asset construction/registration belongs to simulator integration
        # coverage. Only replace those external resources in these unit tests.
        runtime = RUNTIME.ScenarioRuntime.__new__(RUNTIME.ScenarioRuntime)
        runtime.env = env
        runtime.layout = layout
        runtime.difficulty = difficulty
        runtime.radius = 0.045 + 0.025 * difficulty
        runtime.platform = FakeRigidObject(origins, (0.0, 0.0, 0.05)) if layout.vibration_active else None
        runtime.debris = [
            FakeRigidObject(origins, runtime._parking_position(slot))
            for slot in range(runtime.DEBRIS_COUNT)
        ] if layout.debris_active else []
        runtime.contacts = [
            types.SimpleNamespace(data=types.SimpleNamespace(force_matrix_w=torch.zeros((count, 1, 2, 3))))
            for _ in runtime.debris
        ]
        runtime.initialize()
        runtime.reset(torch.arange(count))
        return runtime

    def test_subset_reset_preserves_other_environment_state_and_bodies(self):
        runtime = self.make_runtime()
        runtime.time[:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        runtime.drop_count[:] = torch.tensor([4, 5, 6, 7])
        runtime.impact[:] = True
        runtime._contact_active[:] = True
        changed, unchanged = torch.tensor([1, 3]), torch.tensor([0, 2])
        buffers = (
            "time", "phase", "amplitude", "frequency", "next_drop", "drop_count",
            "impact", "_contact_active", "floor_height", "floor_velocity",
        )
        before = {name: getattr(runtime, name).clone() for name in buffers}
        bodies = [runtime.platform, *runtime.debris]
        body_before = [body.data.root_state_w.clone() for body in bodies]

        runtime.reset(changed)

        for name, expected in before.items():
            with self.subTest(buffer=name):
                torch.testing.assert_close(getattr(runtime, name)[unchanged], expected[unchanged])
        for body, expected in zip(bodies, body_before):
            torch.testing.assert_close(body.data.root_state_w[unchanged], expected[unchanged])
            torch.testing.assert_close(body.writes[-1][1], changed)
        torch.testing.assert_close(runtime.time[changed], torch.zeros(2))
        torch.testing.assert_close(runtime.drop_count[changed], torch.zeros(2, dtype=torch.long))
        self.assertFalse(runtime.impact[changed].any())
        self.assertFalse(runtime._contact_active[changed].any())
        for slot, body in enumerate(runtime.debris):
            local_position = body.data.root_pos_w[changed] - runtime.env.scene.env_origins[changed]
            expected = torch.tensor(runtime._parking_position(slot)).expand(2, -1)
            torch.testing.assert_close(local_position, expected)
            torch.testing.assert_close(body.data.root_state_w[changed, 7:], torch.zeros((2, 6)))
        self.assertEqual(len(runtime.platform.velocity_writes), 0)
        torch.testing.assert_close(
            runtime.platform.data.root_state_w[changed, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
        )
        torch.testing.assert_close(
            runtime.floor_height[changed],
            runtime.platform.data.root_pos_w[changed, 2] - runtime.env.scene.env_origins[changed, 2] + 0.03,
        )

    def test_vibration_velocity_advances_to_analytic_position_without_pose_writes(self):
        runtime = self.make_runtime("vibrating")
        runtime.time[:] = torch.tensor([0.0, 0.1, 0.2, 0.3])
        runtime.phase[:] = torch.tensor([0.0, 0.5, 1.0, 1.5])
        runtime.amplitude[:] = torch.tensor([0.01, 0.02, 0.015, 0.005])
        runtime.frequency[:] = torch.tensor([1.0, 2.0, 1.5, 0.5])
        omega = 2.0 * math.pi * runtime.frequency
        angle = omega * runtime.time + runtime.phase
        displacement = runtime.amplitude * angle.sin()
        position_before = runtime.platform.data.root_pos_w.clone()
        pose_write_count = len(runtime.platform.writes)

        runtime._move_platform()

        velocity, indices = runtime.platform.velocity_writes[-1]
        next_position = position_before + velocity[:, :3] * runtime.env.physics_dt
        local = next_position - runtime.env.scene.env_origins
        torch.testing.assert_close(local[:, 0], displacement * 0.5, atol=2e-6, rtol=1e-4)
        torch.testing.assert_close(local[:, 2], 0.05 + displacement)
        torch.testing.assert_close(local[:, 1], torch.zeros(4))
        torch.testing.assert_close(runtime.floor_velocity, velocity[:, :3])
        torch.testing.assert_close(velocity[:, 3:], torch.zeros((4, 3)))
        torch.testing.assert_close(indices, runtime.platform._ALL_INDICES)
        self.assertEqual(velocity.shape, (4, 6))
        self.assertEqual(len(runtime.platform.writes), pose_write_count)
        torch.testing.assert_close(runtime.platform.data.root_pos_w, position_before)

    def test_timed_debris_release_recycles_only_due_slots_and_environments(self):
        runtime = self.make_runtime("falling_debris")
        runtime.next_drop.fill_(torch.inf)
        runtime.next_drop[1, 0] = 0.01
        runtime.next_drop[2, 2] = 0.03
        previous = [body.data.root_state_w.clone() for body in runtime.debris]
        identities = [id(body) for body in runtime.debris]

        runtime.physics_step()

        torch.testing.assert_close(runtime.drop_count, torch.tensor([0, 1, 0, 0]))
        torch.testing.assert_close(runtime.debris[0].data.root_state_w[[0, 2, 3]], previous[0][[0, 2, 3]])
        for slot in (1, 2, 3):
            torch.testing.assert_close(runtime.debris[slot].data.root_state_w, previous[slot])
        self.assert_spawn_is_safe(runtime, runtime.debris[0], torch.tensor([1]))
        period = runtime.DEBRIS_COUNT * (1.7 - runtime.difficulty)
        self.assertGreaterEqual(runtime.next_drop[1, 0].item(), runtime.time[1].item() + period - 1e-6)
        self.assertLessEqual(runtime.next_drop[1, 0].item(), runtime.time[1].item() + period + 0.4)

        runtime.physics_step()
        torch.testing.assert_close(runtime.drop_count, torch.tensor([0, 1, 1, 0]))
        self.assert_spawn_is_safe(runtime, runtime.debris[2], torch.tensor([2]))
        # Repeated due events must reuse the same bounded object pool.
        for _ in range(20):
            runtime.next_drop[:, 0] = runtime.time
            runtime.physics_step()
            self.assert_spawn_is_safe(runtime, runtime.debris[0], torch.arange(4))
        self.assertEqual([id(body) for body in runtime.debris], identities)
        self.assertEqual(len(runtime.debris), runtime.DEBRIS_COUNT)
        self.assertTrue(torch.isfinite(runtime.debris[0].data.root_state_w).all())

    def assert_spawn_is_safe(self, runtime, body, env_ids):
        local = body.data.root_pos_w[env_ids] - runtime.env.scene.env_origins[env_ids]
        self.assertTrue((local[:, 0].abs() <= 0.85001).all())
        self.assertTrue((local[:, 1].abs() <= runtime.layout.corridor_width * 0.325 + 1e-6).all())
        expected_height = 1.2 + runtime.difficulty + runtime.layout.spawn[2]
        torch.testing.assert_close(local[:, 2], torch.full((len(env_ids),), expected_height))
        self.assertTrue((local[:, 0] > runtime.layout.spawn[0] + 0.6).all())
        self.assertTrue((local[:, 0] < runtime.layout.goal[0] - 0.6).all())
        torch.testing.assert_close(body.data.root_state_w[env_ids, 7:], torch.zeros((len(env_ids), 6)))

    def test_contact_events_are_edges_and_remain_latched_for_control_step(self):
        runtime = self.make_runtime("falling_debris")
        runtime.begin_step()
        runtime.contacts[0].data.force_matrix_w[1, 0, 0, 0] = 3.0
        runtime.collect_impacts()
        torch.testing.assert_close(runtime.impact, torch.tensor([False, True, False, False]))
        runtime.begin_step()
        runtime.collect_impacts()
        self.assertFalse(runtime.impact.any())
        runtime.contacts[0].data.force_matrix_w.zero_()
        runtime.collect_impacts()
        runtime.contacts[1].data.force_matrix_w[1, 0, 1, 2] = -4.0
        runtime.collect_impacts()
        self.assertTrue(runtime.impact[1])
        runtime.contacts[1].data.force_matrix_w.zero_()
        runtime.collect_impacts()
        self.assertTrue(runtime.impact[1])

    def test_ground_height_uses_rough_supports_excludes_walls_roofs_and_off_platform(self):
        rough = self.make_runtime("rough")
        width = rough.layout.corridor_width
        positions = rough.env.scene.env_origins + torch.tensor([
            [-0.8, 0.0, 0.4], [0.0, 0.0, 0.4], [0.8, 0.0, 0.4], [0.0, width / 2 + 0.08, 0.4],
        ])
        step_tops = [box.position[2] + box.size[2] / 2 for box in rough.layout.boxes if box.kind == "rough"]
        torch.testing.assert_close(
            rough.ground_height(positions), rough.env.scene.env_origins[:, 2] + torch.tensor([*step_tops, 0.0]),
        )

        narrow = self.make_runtime("narrow")
        positions = narrow.env.scene.env_origins.clone()
        positions[:, 2] += 0.2  # Three points under the ceiling and one within a wall.
        positions[1, 1] += narrow.layout.corridor_width / 2 + 0.08
        torch.testing.assert_close(narrow.ground_height(positions), narrow.env.scene.env_origins[:, 2])

        vibrating = self.make_runtime("vibrating")
        positions = vibrating.env.scene.env_origins + torch.tensor([
            [0.0, 0.0, 0.3], [2.9, 0.0, 0.3], [-2.9, 0.0, 0.3],
            [0.0, vibrating.layout.corridor_width / 2 + 0.1, 0.3],
        ])
        expected = vibrating.env.scene.env_origins[:, 2].clone()
        expected[0] += vibrating.floor_height[0]
        torch.testing.assert_close(vibrating.ground_height(positions), expected)

    def test_every_preset_returns_48_finite_bounded_observations(self):
        for preset in LAYOUT.ENVIRONMENT_PRESETS:
            for difficulty in (0.0, 0.5, 1.0):
                with self.subTest(preset=preset, difficulty=difficulty):
                    runtime = self.make_runtime(preset, difficulty)
                    runtime.physics_step()
                    observation = runtime.observations()
                    self.assertEqual(observation.shape, (4, 48))
                    self.assertTrue(torch.isfinite(observation).all())
                    self.assertTrue((observation.abs() <= 1.0).all())
                    self.assertTrue((observation[:, :16] >= 0.0).all())
                    self.assertTrue(((observation[:, 16:32] == 0.0) | (observation[:, 16:32] == 1.0)).all())
                    torch.testing.assert_close(observation[:, 16:32], (observation[:, :16] < 1.0).float())
                    if not runtime.layout.vibration_active:
                        torch.testing.assert_close(observation[:, 41:44], torch.zeros((4, 3)))
                    if not runtime.layout.debris_active:
                        torch.testing.assert_close(observation[:, 44:], torch.zeros((4, 4)))
                    if preset == "flat":
                        torch.testing.assert_close(observation[:, :16], torch.ones((4, 16)))
                        torch.testing.assert_close(observation[:, 16:32], torch.zeros((4, 16)))

    def test_lidar_scan_is_held_until_next_l1_rm_horizontal_frame(self):
        runtime = self.make_runtime("narrow", count=1)
        first = runtime.observations()[:, :16].clone()
        runtime.env._robot.data.root_pos_w[:, 1] += 0.2
        runtime.time[:] = 0.05
        held = runtime.observations()[:, :16]
        torch.testing.assert_close(held, first)
        runtime.time[:] = 0.10
        updated = runtime.observations()[:, :16]
        self.assertFalse(torch.equal(updated, first))
        self.assertAlmostEqual(runtime._lidar_next_update.item(), 2.0 / 11.0, places=6)

        runtime.env._robot.data.root_pos_w[:, 1] += 0.2
        runtime.time[:] = 0.18
        torch.testing.assert_close(runtime.observations()[:, :16], updated)
        runtime.time[:] = 0.19
        self.assertFalse(torch.equal(runtime.observations()[:, :16], updated))
        self.assertAlmostEqual(runtime._lidar_next_update.item(), 3.0 / 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
