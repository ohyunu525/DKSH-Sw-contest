"""Course state/physics writer tests with tensor assets, without running PhysX."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


FOLDER = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"


class Config(NS):
    InitialStateCfg = NS


class FakeRigidObject:
    def __init__(self, cfg, origins):
        self.cfg = cfg
        state = torch.zeros((len(origins), 13))
        state[:, :3] = torch.tensor(cfg.init_state.pos)
        state[:, 3] = 1
        current = state.clone()
        current[:, :3] += origins
        self.data = NS(default_root_state=state, root_state_w=current,
                       root_pos_w=current[:, :3], root_lin_vel_w=current[:, 7:10])
        self.writes, self.resets = [], []

    def reset(self, ids):
        self.resets.append(ids.clone())

    def write_root_state_to_sim(self, state, ids):
        self.data.root_state_w[ids] = state
        self.writes.append((state.clone(), ids.clone()))


def load_runtime():
    package_name = "_dksh_curriculum_runtime_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(FOLDER)]
    names = ("isaaclab", "isaaclab.sim", "isaaclab.assets", "isaaclab.sensors",
             "isaaclab.utils", "isaaclab.utils.math", "pxr", "pxr.UsdPhysics")
    modules = {name: types.ModuleType(name) for name in names}
    modules[package_name] = package
    modules["isaaclab"].sim = modules["isaaclab.sim"]
    modules["pxr"].UsdPhysics = modules["pxr.UsdPhysics"]
    modules["isaaclab.assets"].RigidObject = FakeRigidObject
    modules["isaaclab.assets"].RigidObjectCfg = Config
    modules["isaaclab.sensors"].ContactSensor = object
    modules["isaaclab.sensors"].ContactSensorCfg = object
    sim = modules["isaaclab.sim"]
    for name in ("CuboidCfg", "RigidBodyPropertiesCfg", "CollisionPropertiesCfg",
                 "MassPropertiesCfg", "PreviewSurfaceCfg"):
        setattr(sim, name, Config)

    def rotate(q, v, inverse=False):
        xyz = -q[:, 1:] if inverse else q[:, 1:]
        uv = 2 * torch.cross(xyz, v, dim=-1)
        return v + q[:, :1] * uv + torch.cross(xyz, uv, dim=-1)

    modules["isaaclab.utils.math"].quat_rotate = rotate
    modules["isaaclab.utils.math"].quat_rotate_inverse = lambda q, v: rotate(q, v, True)
    with patch.dict(sys.modules, modules):
        loaded = {}
        for short in ("scenario_layout", "scenario_sensing", "scenario_runtime", "terrain_curriculum", "terrain_curriculum_runtime"):
            name = f"{package_name}.{short}"
            spec = importlib.util.spec_from_file_location(name, FOLDER / f"{short}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[short] = module
    return loaded["terrain_curriculum_runtime"], loaded["terrain_curriculum"]


if torch is not None:
    RUNTIME, SAMPLER = load_runtime()


@unittest.skipIf(torch is None, "PyTorch is required")
class TerrainCurriculumRuntimeTests(unittest.TestCase):
    def runtime(self, count=4, split="train", enabled=True, seed=42, difficulty=0.0):
        origins = torch.zeros((count, 3))
        origins[:, 0] = torch.arange(count) * 7
        origins[:, 2] = torch.arange(count) * 0.25
        quat = torch.zeros((count, 4)); quat[:, 0] = 1
        cfg = NS(
            seed=42 if seed is None else seed, course_seed=seed, course_split=split, environment_difficulty=difficulty,
            robot_width=0.56, robot_height=0.24, curriculum_enabled=enabled,
            curriculum_flat_fraction=0.25, curriculum_window=4, curriculum_min_episodes=4,
            curriculum_promote_threshold=0.75, curriculum_demote_threshold=0.25,
            curriculum_step=0.1, sim=NS(physics_material="unchanged physical material"),
            lidar_observation_bins=16, lidar_max_range_m=30.0, lidar_min_range_m=0.05,
            lidar_horizontal_scan_frequency_hz=5.55, lidar_vertical_fov_deg=96.0,
            lidar_vertical_min_angle_deg=-6.0, lidar_vertical_projection_bins=5,
            lidar_azimuth_samples_per_bin=5, lidar_noise_enabled=False,
            lidar_measurement_accuracy_m=0.02, lidar_measurement_resolution_m=0.0045,
            lidar_mount_position_b=(0.0, 0.0, 0.03),
        )
        env = NS(cfg=cfg, num_envs=count, device="cpu", physics_dt=0.005,
                 scene=NS(env_origins=origins, rigid_objects={}),
                 _robot=NS(data=NS(root_pos_w=origins + torch.tensor([-1.6, 0.0, 0.1375]), root_quat_w=quat)))
        with patch.object(RUNTIME, "RigidObject", side_effect=lambda body_cfg: FakeRigidObject(body_cfg, origins)):
            result = RUNTIME.CurriculumScenarioRuntime(env)
        result.initialize()
        result.reset(torch.arange(count))
        return result

    def test_fixed_physical_pool_is_kinematic_and_has_both_flat_and_obstacles(self):
        runtime = self.runtime()
        self.assertEqual(runtime.course_families, ["flat", "rough", "stairs", "narrow"])
        self.assertEqual(len(runtime.boxes), len(SAMPLER.POOL_BOX_SPECS))
        for body, spec in zip(runtime.boxes, SAMPLER.POOL_BOX_SPECS):
            self.assertEqual(body.cfg.spawn.size, spec.size)
            self.assertTrue(body.cfg.spawn.rigid_props.kinematic_enabled)
            self.assertTrue(body.cfg.spawn.rigid_props.disable_gravity)
            self.assertEqual(body.cfg.spawn.physics_material, "unchanged physical material")
        self.assertFalse(runtime._active_boxes[0, runtime._terrain_indices].any())
        self.assertTrue(runtime._active_boxes[0, :2].all())
        self.assertTrue(runtime._active_boxes[1:].any())

    def test_reset_pose_writers_and_sensing_share_exact_geometry(self):
        runtime = self.runtime()
        origins = runtime.env.scene.env_origins
        for slot, body in enumerate(runtime.boxes):
            torch.testing.assert_close(body.data.root_pos_w, runtime._world_box_centers()[:, slot])
            torch.testing.assert_close(body.data.root_state_w[:, 7:], torch.zeros((4, 6)))
        # Query every obstacle's actual top-center. The ground query excludes
        # walls/ceilings, but rough/stair tops exactly match collider positions.
        for row in (1, 2):
            for slot in runtime._terrain_indices:
                if not runtime._active_boxes[row, slot]:
                    continue
                point = runtime._world_box_centers()[:, slot:slot + 1].clone()
                expected = point[row, 0, 2] + runtime._halves[slot, 2]
                heights = runtime.surface_heights(point)
                self.assertAlmostEqual(float(heights[row, 0]), float(expected), places=6)
        # A cloned floor at z=.25 remains at .25 when its obstacle is parked.
        flat = runtime.surface_heights(origins[:, None, :] + torch.tensor([[[-1.6, 0.0, 0.0]]]))
        torch.testing.assert_close(flat[:, 0], origins[:, 2])

    def test_partial_reset_preserves_all_other_geometry_counters_and_sensor_state(self):
        runtime = self.runtime()
        runtime.time[:] = torch.arange(4) + 1
        runtime._lidar_ranges[:] = 0.25
        runtime._lidar_next_update[:] = 3.0
        keep, changed = [0, 2], torch.tensor([1, 3])
        tensors = {name: getattr(runtime, name).clone() for name in (
            "_centers", "_active_boxes", "spawn_positions", "goal_positions", "corridor_widths",
            "difficulties", "time", "_lidar_ranges", "_lidar_next_update",
        )}
        before_ids = runtime.course_ids.copy()
        before_counters = runtime.episode_counters.copy()
        poses = [body.data.root_state_w.clone() for body in runtime.boxes]
        runtime.reset(changed)
        for name, previous in tensors.items():
            torch.testing.assert_close(getattr(runtime, name)[keep], previous[keep])
        for row in keep:
            self.assertEqual(runtime.course_ids[row], before_ids[row])
            self.assertEqual(runtime.episode_counters[row], before_counters[row])
        for slot, body in enumerate(runtime.boxes):
            torch.testing.assert_close(body.data.root_state_w[keep], poses[slot][keep])
            torch.testing.assert_close(body.writes[-1][1], changed)
        self.assertNotEqual(runtime.course_ids[1], before_ids[1])
        self.assertNotEqual(runtime.course_ids[3], before_ids[3])
        torch.testing.assert_close(runtime.time[changed], torch.zeros(2))

    def test_course_sequence_is_independent_of_reset_batch_order_and_torch_rng(self):
        first, second = self.runtime(), self.runtime()
        torch.manual_seed(912)
        first.reset(torch.tensor([1, 3]))
        torch.rand(100)
        second.reset(torch.tensor([3, 1]))
        self.assertEqual(first.episode_metadata(range(4)), second.episode_metadata(range(4)))
        torch.testing.assert_close(first._centers, second._centers)

    def test_single_environment_replays_flat_and_each_obstacle_family(self):
        runtime = self.runtime(count=1)
        families = []
        for _ in range(16):
            families += runtime.course_families
            runtime.reset([0])
        self.assertEqual(families.count("flat"), 4)
        self.assertEqual(set(families), set(SAMPLER.FAMILIES))

    def test_completed_obstacle_successes_promote_only_the_matching_level(self):
        runtime = self.runtime(count=20)
        runtime.record_outcomes([0, 1, 2, 3, 4], [True] * 5)
        runtime.record_outcomes([5, 8, 11, 14], [True] * 4)
        self.assertAlmostEqual(runtime.controller.difficulty_for("rough"), 0.1)
        self.assertEqual(runtime.controller.difficulty_for("stairs"), 0.0)
        self.assertEqual(runtime.controller.difficulty_for("narrow"), 0.0)
        # The existing rough course was generated at the old level; additional
        # delayed completions cannot count toward another promotion.
        for _ in range(8):
            runtime.record_outcomes([17], [True])
        self.assertAlmostEqual(runtime.controller.difficulty_for("rough"), 0.1)
        runtime.reset([5])
        self.assertEqual(runtime.course_families[0], "flat")

    def test_missing_course_seed_uses_and_records_the_environment_seed(self):
        runtime = self.runtime(seed=None)
        self.assertEqual(runtime.env.cfg.course_seed, 42)
        self.assertEqual(runtime.course_ids, self.runtime(seed=42).course_ids)

    def test_holdout_freezes_even_when_a_direct_config_requests_adaptation(self):
        runtime = self.runtime(split="holdout", enabled=True)
        self.assertTrue(runtime.controller.frozen)
        before = runtime.controller.state_dict()
        runtime.record_outcomes([1, 2, 3], [True] * 3)
        self.assertEqual(runtime.controller.state_dict(), before)

    def test_holdout_namespace_is_disjoint_reproducible_and_never_adapts(self):
        train = self.runtime(difficulty=0.3)
        first = self.runtime(split="holdout", enabled=False, difficulty=0.3)
        second = self.runtime(split="holdout", enabled=False, difficulty=0.3)
        self.assertEqual(first.course_ids, second.course_ids)
        self.assertFalse(set(train.course_ids) & set(first.course_ids))
        before = first.controller.state_dict()
        for _ in range(10):
            first.record_outcomes([1, 2, 3], [True, True, True])
        self.assertEqual(first.controller.state_dict(), before)

    def test_parked_colliders_are_invisible_to_lidar_and_observations_keep_schema(self):
        runtime = self.runtime()
        centers = runtime._world_box_centers()
        # Force the entire batch flat, including formerly active obstacles, to
        # exercise the active-mask path independently from the floor height.
        runtime._active_boxes.zero_()
        pos = runtime.env._robot.data.root_pos_w
        quat = runtime.env._robot.data.root_quat_w
        scan = runtime._lidar_observations(pos, quat, centers)
        torch.testing.assert_close(scan[:, :16], torch.ones((4, 16)))
        torch.testing.assert_close(scan[:, 16:], torch.zeros((4, 16)))
        observation = runtime.observations()
        self.assertEqual(observation.shape, (4, 48))
        self.assertTrue(torch.isfinite(observation).all())
        torch.testing.assert_close(observation[:, 41:], torch.zeros((4, 7)))

    def test_state_roundtrip_preserves_current_course_and_next_episode_sequence(self):
        source = self.runtime()
        source.reset([1, 3])
        saved = copy.deepcopy(source.state_dict())
        target = self.runtime()
        target.load_state_dict(saved)
        self.assertEqual(source.state_dict(), target.state_dict())
        torch.testing.assert_close(source._centers, target._centers)
        source.reset([0, 1, 2, 3])
        target.reset([0, 1, 2, 3])
        self.assertEqual(source.episode_metadata(range(4)), target.episode_metadata(range(4)))

    def test_saved_state_cannot_change_split_topology_or_course_identity(self):
        runtime = self.runtime()
        initial = copy.deepcopy(runtime.state_dict())
        for field, bad in (("split", "holdout"), ("num_envs", 8), ("episode_counters", [0, -1, 2, 3])):
            with self.subTest(field=field):
                invalid = copy.deepcopy(initial); invalid[field] = bad
                with self.assertRaises(ValueError):
                    runtime.load_state_dict(invalid)
                self.assertEqual(runtime.state_dict(), initial)
        invalid = copy.deepcopy(initial)
        invalid["current_courses"][1]["course_id"] = "forged"
        with self.assertRaisesRegex(ValueError, "identity"):
            runtime.load_state_dict(invalid)
        self.assertEqual(runtime.state_dict(), initial)


if __name__ == "__main__":
    unittest.main()
