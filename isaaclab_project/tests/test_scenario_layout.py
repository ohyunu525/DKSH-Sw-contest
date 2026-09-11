"""Course geometry checks that run without Isaac Sim or a GPU."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


LAYOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dksh_isaaclab"
    / "dksh_isaaclab"
    / "tasks"
    / "direct"
    / "spider_navigation"
    / "scenario_layout.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_scenario_layout", LAYOUT_PATH)
assert SPEC is not None and SPEC.loader is not None
LAYOUT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAYOUT
SPEC.loader.exec_module(LAYOUT)


class ScenarioLayoutTests(unittest.TestCase):
    def test_all_presets_have_finite_geometry_within_environment_allocation(self) -> None:
        for preset in LAYOUT.ENVIRONMENT_PRESETS:
            for difficulty in (0.0, 0.5, 1.0):
                with self.subTest(preset=preset, difficulty=difficulty):
                    layout = LAYOUT.build_layout(preset, difficulty)
                    self.assertEqual(len({box.name for box in layout.boxes}), len(layout.boxes))
                    for box in layout.boxes:
                        self.assertTrue(all(math.isfinite(value) for value in (*box.position, *box.size)))
                        self.assertTrue(all(value > 0.0 for value in box.size))
                        for axis in (0, 1):
                            self.assertLessEqual(abs(box.position[axis]) + box.size[axis] / 2.0, 3.0)
                        self.assertGreaterEqual(box.position[2] - box.size[2] / 2.0, -1e-10)

    def test_spawn_and_goal_standing_envelopes_do_not_intersect_static_solids(self) -> None:
        for width, height in ((0.62, 0.28), (0.69, 0.28), (0.46, 0.24), (1.2, 2.0)):
            for preset in LAYOUT.ENVIRONMENT_PRESETS:
                for difficulty in (0.0, 1.0):
                    layout = LAYOUT.build_layout(preset, difficulty, width, height)
                    for point in (layout.spawn, layout.goal):
                        with self.subTest(width=width, preset=preset, difficulty=difficulty, point=point):
                            robot_min = (point[0] - width / 2.0, point[1] - width / 2.0, point[2])
                            robot_max = (point[0] + width / 2.0, point[1] + width / 2.0, point[2] + height)
                            for box in layout.boxes:
                                overlaps = all(
                                    robot_min[axis] < box.position[axis] + box.size[axis] / 2.0
                                    and robot_max[axis] > box.position[axis] - box.size[axis] / 2.0
                                    for axis in (0, 1, 2)
                                )
                                self.assertFalse(overlaps, box.name)

    def test_difficulty_and_footprint_validation(self) -> None:
        for preset in ("unknown", "", "Flat", None):
            with self.subTest(preset=preset), self.assertRaises(ValueError):
                LAYOUT.validate_environment(preset, 0.5)
        for difficulty in (-0.01, 1.01, math.nan, math.inf, -math.inf, "0.5", None, True):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                LAYOUT.validate_environment("flat", difficulty)
        for value in (-1.0, 0.0, math.nan, math.inf, "0.5", None, True, 6.0):
            for argument in ("robot_width", "robot_height"):
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    LAYOUT.build_layout("narrow", 0.5, **{argument: value})

    def test_narrow_course_has_clearance_but_no_side_route(self) -> None:
        for width in (0.45, 0.62, 0.69):
            easy = LAYOUT.build_layout("narrow", 0.0, robot_width=width)
            hard = LAYOUT.build_layout("narrow", 1.0, robot_width=width)
            self.assertGreater(easy.corridor_width, hard.corridor_width)
            self.assertGreater(hard.corridor_width, width)
            self.assertLess(hard.corridor_width, width * 1.1)
            for layout in (easy, hard):
                for wall in (box for box in layout.boxes if box.kind == "wall"):
                    self.assertAlmostEqual(abs(wall.position[1]) - wall.size[1] / 2.0, layout.corridor_width / 2.0)
                    self.assertLess(wall.position[0] - wall.size[0] / 2.0, layout.spawn[0] - width / 2.0)
                    self.assertGreater(wall.position[0] + wall.size[0] / 2.0, layout.goal[0] + width / 2.0)
                ceiling = next(box for box in layout.boxes if box.kind == "ceiling")
                self.assertGreater(ceiling.position[2] - ceiling.size[2] / 2.0, 0.28)
                self.assertLess(layout.spawn[0], ceiling.position[0] - ceiling.size[0] / 2.0)
                self.assertGreater(layout.goal[0], ceiling.position[0] + ceiling.size[0] / 2.0)

    def test_rough_steps_are_between_endpoints_and_cover_every_lateral_route(self) -> None:
        easy = LAYOUT.build_layout("rough", 0.0)
        hard = LAYOUT.build_layout("rough", 1.0)
        easy_steps = [box for box in easy.boxes if box.kind == "rough"]
        hard_steps = [box for box in hard.boxes if box.kind == "rough"]
        self.assertEqual(len(hard_steps), 3)
        for easy_step, hard_step in zip(easy_steps, hard_steps):
            self.assertGreater(hard_step.size[2], easy_step.size[2])
            self.assertLess(hard_step.size[2], 0.07)
            self.assertEqual(hard_step.size[1], hard.corridor_width)
            self.assertGreater(hard_step.position[0] - hard_step.size[0] / 2.0, hard.spawn[0])
            self.assertLess(hard_step.position[0] + hard_step.size[0] / 2.0, hard.goal[0])

    def test_dynamic_presets_have_matching_surface_heights_and_open_hazard_zone(self) -> None:
        expected_flags = {
            "flat": (False, False),
            "narrow": (False, False),
            "vibrating": (True, False),
            "falling_debris": (False, True),
            "rough": (False, False),
            "mixed": (True, True),
        }
        for preset, flags in expected_flags.items():
            with self.subTest(preset=preset):
                layout = LAYOUT.build_layout(preset, 0.5)
                self.assertEqual((layout.vibration_active, layout.debris_active), flags)
                expected_surface = 0.08 if flags[0] else 0.0
                self.assertEqual(layout.spawn[2], expected_surface)
                self.assertEqual(layout.goal[2], expected_surface)
                if flags[1]:
                    self.assertFalse(any(box.kind == "ceiling" for box in layout.boxes))
                if flags[0]:
                    for box in layout.boxes:
                        self.assertAlmostEqual(box.position[2] - box.size[2] / 2.0, expected_surface)

    def test_flat_is_empty_and_layouts_are_deterministic(self) -> None:
        self.assertEqual(LAYOUT.build_layout("flat", 0.5).boxes, ())
        for preset in LAYOUT.ENVIRONMENT_PRESETS:
            self.assertEqual(LAYOUT.build_layout(preset, 0.5), LAYOUT.build_layout(preset, 0.5))


if __name__ == "__main__":
    unittest.main()
