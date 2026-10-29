"""Course randomization and curriculum regressions without Isaac Sim."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest


FOLDER = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"
PACKAGE = "_curriculum_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(FOLDER)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.terrain_curriculum", FOLDER / "terrain_curriculum.py")
MODULE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


class RandomizedCourseTests(unittest.TestCase):
    def sample(self, **values):
        defaults = dict(seed=42, split="train", env_id=0, episode=0, family="stairs",
                        difficulty=0.5, robot_width=0.56, robot_height=0.24)
        return MODULE.sample_course(**(defaults | values))

    def test_reproducible_per_environment_without_shared_rng_state(self):
        original = self.sample()
        self.sample(env_id=2, episode=912)
        self.sample(env_id=0, episode=1)
        self.assertEqual(original, self.sample())
        self.assertNotEqual(original.course_id, self.sample(episode=1).course_id)
        self.assertNotEqual(original.boxes, self.sample(episode=1).boxes)
        self.assertNotEqual(original.course_id, self.sample(env_id=1).course_id)

    def test_holdout_seed_domains_and_ids_are_disjoint_even_with_equal_base_seed(self):
        for index in range(100):
            train = self.sample(seed=index, episode=index)
            holdout = self.sample(seed=index, episode=index, split="holdout")
            self.assertLess(train.course_seed, 1 << 62)
            self.assertGreaterEqual(holdout.course_seed, 1 << 62)
            self.assertLess(holdout.course_seed, 1 << 63)
            self.assertNotEqual(train.course_seed, holdout.course_seed)
            self.assertNotEqual(train.course_id, holdout.course_id)
            self.assertNotEqual(train.boxes, holdout.boxes)

    def test_every_course_uses_identical_fixed_collider_pool_and_parked_unused_slots(self):
        expected = tuple((box.name, box.kind, box.size) for box in MODULE.POOL_BOX_SPECS)
        for family in MODULE.FAMILIES:
            for difficulty in (0.0, 0.5, 1.0):
                course = self.sample(family=family, difficulty=difficulty)
                self.assertEqual(tuple((box.name, box.kind, box.size) for box in course.boxes), expected)
                active = [box for box in course.boxes if box.position[2] + box.size[2] / 2 > -1]
                kinds = {box.kind for box in active}
                self.assertEqual(kinds, {"wall"} | ({"stairs"} if family == "stairs" else
                                                  {"rough"} if family == "rough" else
                                                  {"ceiling"} if family == "narrow" else set()))
                for box in course.boxes:
                    self.assertTrue(all(math.isfinite(v) for v in (*box.position, *box.size)))
                    self.assertTrue(all(v > 0 for v in box.size))

    def test_many_randomized_courses_have_safe_spawn_goal_and_supported_stairs(self):
        for width, height in ((0.56, 0.24), (0.69, 0.28), (1.2, 0.5)):
            for seed in range(40):
                for family in MODULE.FAMILIES:
                    for difficulty in (0.0, 1.0):
                        course = self.sample(seed=seed, family=family, difficulty=difficulty,
                                             robot_width=width, robot_height=height)
                        self.assertGreaterEqual(course.corridor_width, 1.08 * width - 1e-10)
                        self.assertLessEqual(course.corridor_width, 1.6)
                        for point in (course.spawn, course.goal):
                            self.assertLess(abs(point[1]) + width / 2, course.corridor_width / 2)
                            low = (point[0] - width / 2, point[1] - width / 2, point[2] + 1e-9)
                            high = (point[0] + width / 2, point[1] + width / 2, point[2] + height)
                            for box in course.boxes:
                                intersects = all(low[axis] < box.position[axis] + box.size[axis] / 2
                                                 and high[axis] > box.position[axis] - box.size[axis] / 2
                                                 for axis in (0, 1, 2))
                                self.assertFalse(intersects, (family, seed, width, box.name, point))
                        if family == "stairs":
                            treads = [box for box in course.boxes if box.kind == "stairs"]
                            for left, right in zip(treads, treads[1:]):
                                self.assertAlmostEqual(left.position[0] + left.size[0] / 2,
                                                       right.position[0] - right.size[0] / 2)
                                self.assertLess(left.position[2] + left.size[2] / 2,
                                                right.position[2] + right.size[2] / 2)
                            self.assertAlmostEqual(course.goal[2], treads[-1].position[2] + treads[-1].size[2] / 2)

    def test_randomization_changes_obstacle_geometry_not_only_identifier(self):
        for family, kind in (("rough", "rough"), ("stairs", "stairs"), ("narrow", "ceiling")):
            geometries = set()
            for episode in range(20):
                course = self.sample(family=family, episode=episode)
                geometries.add(tuple(box.position for box in course.boxes if box.kind == kind))
            self.assertEqual(len(geometries), 20)

    def test_difficulty_controls_physical_obstacle_height_and_narrowness(self):
        for family, kind in (("rough", "rough"), ("stairs", "stairs")):
            easy = self.sample(family=family, difficulty=0)
            hard = self.sample(family=family, difficulty=1)
            easy_heights = [box.position[2] + box.size[2] / 2 for box in easy.boxes if box.kind == kind]
            hard_heights = [box.position[2] + box.size[2] / 2 for box in hard.boxes if box.kind == kind]
            self.assertTrue(all(a < b for a, b in zip(easy_heights, hard_heights)))
        self.assertGreater(self.sample(family="narrow", difficulty=0).corridor_width,
                           self.sample(family="narrow", difficulty=1).corridor_width)

    def test_invalid_parameters_fail_without_starting_simulator(self):
        for argument in ("seed", "env_id", "episode"):
            for value in (-1, True, 1.5, "2", None):
                with self.assertRaises(ValueError):
                    self.sample(**{argument: value})
        for argument in ("difficulty", "robot_width", "robot_height"):
            for value in (math.nan, math.inf, "0.5", None, True, -0.1):
                with self.assertRaises(ValueError):
                    self.sample(**{argument: value})
        for values in ({"difficulty": 1.1}, {"robot_width": 1.21}, {"robot_height": 0.51},
                       {"family": "mixed"}, {"split": "validation"}):
            with self.assertRaises(ValueError):
                self.sample(**values)


class CurriculumControllerTests(unittest.TestCase):
    def test_family_success_window_promotes_and_clears_only_its_history(self):
        controller = MODULE.CurriculumController(initial_difficulty=0.2)
        for _ in range(19):
            self.assertFalse(controller.record("rough", True, difficulty=0.2))
        controller.record("stairs", False)
        self.assertTrue(controller.record("rough", True, difficulty=0.2))
        self.assertEqual(controller.levels, {"rough": 0.3, "stairs": 0.2, "narrow": 0.2})
        self.assertEqual(list(controller.history["rough"]), [])
        self.assertEqual(list(controller.history["stairs"]), [False])
        self.assertEqual(controller.completed["rough"], 20)
        self.assertEqual(controller.promotions["rough"], 1)

    def test_failure_rate_demotes_and_thresholds_are_inclusive(self):
        controller = MODULE.CurriculumController(initial_difficulty=0.5)
        for success in [True] * 16 + [False] * 4:
            controller.record("stairs", success)
        self.assertEqual(controller.difficulty_for("stairs"), 0.6)
        for success in [True] * 6 + [False] * 14:
            controller.record("stairs", success)
        self.assertEqual(controller.difficulty_for("stairs"), 0.5)
        self.assertEqual(controller.demotions["stairs"], 1)

    def test_flat_incomplete_frozen_and_stale_results_cannot_promote_obstacles(self):
        controller = MODULE.CurriculumController(initial_difficulty=0.2)
        before = controller.state_dict()
        for _ in range(30):
            self.assertFalse(controller.record("flat", True))
            self.assertFalse(controller.record("rough", True, completed=False))
            self.assertFalse(controller.record("rough", True, difficulty=0.1))
        self.assertEqual(before, controller.state_dict())
        controller.frozen = True
        for _ in range(30):
            controller.record("rough", True)
        self.assertEqual(before, controller.state_dict())

    def test_intermediate_success_rate_retains_rolling_window(self):
        controller = MODULE.CurriculumController(window=4, min_episodes=4)
        for success in (True, False, True, False):
            controller.record("rough", success)
        self.assertEqual(controller.levels["rough"], 0)
        self.assertEqual(list(controller.history["rough"]), [True, False, True, False])
        for success in (True, True, True, True):
            controller.record("rough", success)
        self.assertEqual(controller.levels["rough"], 0.1)

    def test_difficulty_is_bounded_and_does_not_invent_level_changes(self):
        low = MODULE.CurriculumController(initial_difficulty=0, window=2, min_episodes=2)
        high = MODULE.CurriculumController(initial_difficulty=1, window=2, min_episodes=2)
        for _ in range(25):
            self.assertFalse(low.record("rough", False))
            self.assertFalse(high.record("rough", True))
        self.assertEqual(low.levels["rough"], 0)
        self.assertEqual(high.levels["rough"], 1)
        self.assertEqual(low.demotions["rough"], 0)
        self.assertEqual(high.promotions["rough"], 0)

    def test_checkpoint_roundtrip_preserves_partial_windows_and_evaluation_freeze(self):
        controller = MODULE.CurriculumController(initial_difficulty=0.2, window=4, min_episodes=4)
        for success in (True, True, False):
            controller.record("stairs", success)
        state = controller.state_dict()
        restored = MODULE.CurriculumController(initial_difficulty=0.2, window=4, min_episodes=4)
        restored.load_state_dict(deepcopy(state))
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.record("stairs", True), controller.record("stairs", True))
        self.assertEqual(restored.state_dict(), controller.state_dict())
        frozen = MODULE.CurriculumController(initial_difficulty=0.2, window=4, min_episodes=4, frozen=True)
        frozen.load_state_dict(state)
        self.assertTrue(frozen.frozen)
        self.assertFalse(frozen.record("stairs", True))
        self.assertEqual(frozen.state_dict(), state)

    def test_invalid_saved_state_is_rejected_before_mutating_live_state(self):
        controller = MODULE.CurriculumController()
        before = controller.state_dict()
        for key, value in (("contract", "other"), ("levels", {"rough": 2}),
                           ("history", {"rough": [1], "stairs": [], "narrow": []}),
                           ("completed", {"rough": -1, "stairs": 0, "narrow": 0})):
            state = deepcopy(before)
            state[key] = value
            with self.assertRaises(ValueError):
                controller.load_state_dict(state)
            self.assertEqual(controller.state_dict(), before)

    def test_statistics_expose_level_rates_and_sample_counts(self):
        controller = MODULE.CurriculumController()
        controller.record("rough", True)
        controller.record("rough", False)
        stats = controller.statistics()
        self.assertEqual(stats["success_rate/rough"], 0.5)
        self.assertEqual(stats["window_episodes/rough"], 2)
        self.assertEqual(stats["completed/rough"], 2)

    def test_bad_configuration_and_outcomes_are_rejected(self):
        for values in ({"window": 0}, {"min_episodes": 21}, {"step": 0}, {"step": math.nan},
                       {"initial_difficulty": 2}, {"demote_threshold": 0.9}, {"frozen": "true"}):
            with self.assertRaises(ValueError):
                MODULE.CurriculumController(**values)
        controller = MODULE.CurriculumController()
        for values in (("unknown", True), ("rough", 1), ("rough", None)):
            with self.assertRaises(ValueError):
                controller.record(*values)


if __name__ == "__main__":
    unittest.main()
