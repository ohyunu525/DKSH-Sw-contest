"""CPU regressions for sole support, shaft impacts and link-force history."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


MODULE = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_contact.py"
SPEC = importlib.util.spec_from_file_location("joint_angle_contact", MODULE)
CONTACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTACT)


class SoleContactStateTests(unittest.TestCase):
    def setUp(self):
        self.forces = torch.zeros((2, 4, 5, 3), dtype=torch.float64)
        self.soles = torch.zeros((2, 2, 3), dtype=torch.float64)
        self.surfaces = torch.zeros((2, 2), dtype=torch.float64)
        self.options = dict(
            sensor_feet=[3, 1], sensor_body=[0, 2], sole_positions_w=self.soles,
            surface_heights=self.surfaces, force_threshold=1.0, height_tolerance=0.03125,
        )

    def state(self):
        return CONTACT.sole_contact_state(self.forces, **self.options)

    def test_upward_force_and_surface_height_are_both_required(self):
        self.forces[0, -1, 3, 2] = 2
        feet, body = self.state()
        torch.testing.assert_close(feet, torch.tensor([[True, False], [False, False]]))
        torch.testing.assert_close(body, torch.tensor([False, False]))

    def test_horizontal_wall_force_is_not_foot_support_even_near_floor(self):
        self.forces[0, -1, 3, 0] = 100
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        torch.testing.assert_close(body, torch.tensor([True, False]))

    def test_downward_only_force_is_an_undesired_contact(self):
        self.forces[1, -1, 1, 2] = -10
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_upward_shaft_force_with_floating_or_deeply_penetrating_sole_is_bad(self):
        self.forces[:, -1, 3, 2] = 10
        self.soles[0, 0, 2] = 0.08
        self.soles[1, 0, 2] = -0.08
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        self.assertTrue(bool(body.all()))

    def test_support_is_retained_from_force_history_in_public_leg_order(self):
        self.forces[0, 0, 1, 2] = 2
        self.forces[1, 1, 3, 2] = 3
        feet, body = self.state()
        torch.testing.assert_close(feet, torch.tensor([[False, True], [True, False]]))
        self.assertFalse(bool(body.any()))

    def test_each_sole_uses_its_own_terrain_height_and_inclusive_tolerance(self):
        self.surfaces[:] = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
        self.soles[..., 2] = self.surfaces + torch.tensor([[0.03125, -0.03125], [0.0, 0.03126]])
        self.forces[:, -1, [3, 1], 2] = 2
        feet, body = self.state()
        torch.testing.assert_close(feet, torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_nonfoot_body_contact_uses_force_magnitude_and_history(self):
        self.forces[0, 0, 2, :2] = torch.tensor([0.8, 0.8])
        self.forces[1, -1, 0, 2] = -2
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        self.assertTrue(bool(body.all()))

    def test_valid_foot_and_separate_body_collision_can_coexist(self):
        self.forces[0, -1, 3, 2] = 3
        self.forces[0, -1, 0, 0] = 2
        feet, body = self.state()
        self.assertTrue(bool(feet[0, 0]))
        self.assertTrue(bool(body[0]))

    def test_unlisted_bodies_are_not_misclassified_and_empty_body_list_is_supported(self):
        self.options["sensor_body"] = []
        self.forces[:, -1, 4, 2] = 100
        self.forces[0, -1, 3, 2] = 2
        feet, body = self.state()
        self.assertTrue(bool(feet[0, 0]))
        self.assertFalse(bool(body.any()))

    def test_contact_force_threshold_is_strict(self):
        self.forces[0, -1, 3, 2] = 1
        self.forces[1, -1, 3, 2] = 1.001
        feet, body = self.state()
        torch.testing.assert_close(feet[:, 0], torch.tensor([False, True]))
        self.assertFalse(bool(body.any()))

    def test_zero_height_tolerance_allows_exact_surface_support(self):
        self.options["height_tolerance"] = 0
        self.forces[0, -1, 3, 2] = 2
        self.forces[1, -1, 3, 2] = 2
        self.soles[1, 0, 2] = 1e-6
        feet, body = self.state()
        torch.testing.assert_close(feet[:, 0], torch.tensor([True, False]))
        torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_mismatched_tensor_shapes_fail_explicitly(self):
        for force_shape in ((2, 5, 3), (2, 0, 5, 3), (2, 4, 5, 2)):
            with self.subTest(force_shape=force_shape), self.assertRaises(ValueError):
                CONTACT.sole_contact_state(torch.zeros(force_shape), **self.options)
        for key, shape in (("sole_positions_w", (1, 2, 3)), ("sole_positions_w", (2, 3, 3)), ("surface_heights", (2, 2, 1))):
            options = {**self.options, key: torch.zeros(shape)}
            with self.subTest(key=key, shape=shape), self.assertRaises(ValueError):
                CONTACT.sole_contact_state(self.forces, **options)

    def test_bad_indices_and_overlapping_classifications_are_rejected(self):
        for key, indices in (("sensor_feet", []), ("sensor_feet", [3, 3]), ("sensor_feet", [-1, 1]), ("sensor_feet", [5, 1]), ("sensor_feet", [True, 1]), ("sensor_body", [0.5]), ("sensor_body", [0, 0]), ("sensor_body", [3])):
            with self.subTest(key=key, indices=indices), self.assertRaises(ValueError):
                CONTACT.sole_contact_state(self.forces, **{**self.options, key: indices})

    def test_scalar_parameters_must_be_finite_and_in_range(self):
        for key, value in (("force_threshold", 0), ("force_threshold", -1), ("force_threshold", math.inf), ("force_threshold", math.nan), ("height_tolerance", -1), ("height_tolerance", math.inf), ("height_tolerance", math.nan)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CONTACT.sole_contact_state(self.forces, **{**self.options, key: value})

    def test_integer_inputs_or_mismatched_devices_fail_explicitly(self):
        with self.assertRaises(ValueError):
            CONTACT.sole_contact_state(self.forces.long(), **self.options)
        for key in ("sole_positions_w", "surface_heights"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONTACT.sole_contact_state(self.forces, **{**self.options, key: self.options[key].to("meta")})


class FootContactStateTests(unittest.TestCase):
    def setUp(self):
        self.forces = torch.zeros((2, 4, 5, 3), dtype=torch.float64)
        self.points = torch.zeros((2, 2, 3, 3), dtype=torch.float64)
        self.heights = torch.zeros((2, 2, 3), dtype=torch.float64)
        self.options = dict(
            sensor_feet=[3, 1], sensor_body=[0, 2], foot_positions_w=self.points,
            surface_heights=self.heights, force_threshold=1.0, height_tolerance=0.012,
        )

    def state(self):
        return CONTACT.foot_contact_state(self.forces, **self.options)

    def test_heel_support_is_valid_when_legacy_toe_is_above_tolerance(self):
        # Witness 0 is the historical toe; witness 1 is the actual grounded heel.
        self.points[..., 0, 2] = 0.01275
        self.points[..., 1, 2] = 0
        self.points[..., 2, 2] = 0.03
        self.forces[:, -1, [3, 1], 2] = 2
        feet, body = self.state()
        self.assertTrue(bool(feet.all()))
        self.assertFalse(bool(body.any()))
        legacy_feet, legacy_body = CONTACT.sole_contact_state(
            self.forces, sensor_feet=[3, 1], sensor_body=[0, 2],
            sole_positions_w=self.points[..., 0, :], surface_heights=self.heights[..., 0],
            force_threshold=1.0, height_tolerance=0.012,
        )
        self.assertFalse(bool(legacy_feet.any()))
        self.assertTrue(bool(legacy_body.all()))

    def test_closest_witness_preserves_xyz_height_and_indices_for_slip(self):
        self.points[:] = torch.tensor([
            [[[10, 11, 0.25], [20, 21, 0.501], [30, 31, 0.78]],
             [[40, 41, 0.01], [50, 51, 0.01], [60, 61, -0.0001]]],
            [[[70, 71, 2.00], [80, 81, 2.02], [90, 91, 2.03]],
             [[100, 101, 2.21], [110, 111, 2.19], [120, 121, 2.25]]],
        ], dtype=torch.float64)
        self.heights[:] = torch.tensor([
            [[0.2, 0.5, 0.8], [0, 0, 0]],
            [[2, 2, 2], [2.2, 2.19, 2.2]],
        ], dtype=torch.float64)
        positions, heights, indices = CONTACT.closest_foot_support(self.points, self.heights)
        expected_indices = torch.tensor([[1, 2], [0, 1]])
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(positions, torch.tensor([
            [[20, 21, 0.501], [60, 61, -0.0001]],
            [[70, 71, 2], [110, 111, 2.19]],
        ], dtype=torch.float64))
        torch.testing.assert_close(heights, torch.tensor([[0.5, 0], [2, 2.19]], dtype=torch.float64))
        witness_velocity = torch.arange(36, dtype=torch.float64).reshape(2, 2, 3, 3)
        selected_velocity = witness_velocity.gather(2, indices[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        torch.testing.assert_close(selected_velocity, torch.stack((
            torch.stack((witness_velocity[0, 0, 1], witness_velocity[0, 1, 2])),
            torch.stack((witness_velocity[1, 0, 0], witness_velocity[1, 1, 1])),
        )))

    def test_each_witness_uses_its_own_stair_height_and_clone_origin(self):
        self.heights[:] = torch.tensor([
            [[0, 0.006, 0.018], [0.006, 0.018, 0.03]],
            [[1.75, 1.756, 1.768], [1.756, 1.768, 1.78]],
        ], dtype=torch.float64)
        self.points[..., 2] = self.heights + torch.tensor([
            [[0.03, 0, 0.02], [0.04, 0.02, 0]],
            [[0.02, 0.03, 0], [0.013, 0.02, 0.04]],
        ], dtype=torch.float64)
        self.forces[:, 0, [3, 1], 2] = 2
        feet, body = self.state()
        torch.testing.assert_close(feet, torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_grounded_witness_does_not_turn_horizontal_or_downward_force_into_support(self):
        self.forces[0, -1, 3, 0] = 100
        self.forces[1, -1, 1, 2] = -10
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        self.assertTrue(bool(body.all()))

    def test_near_ground_without_force_is_not_support(self):
        self.points[..., 2] = 0.001
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        self.assertFalse(bool(body.any()))

    def test_force_on_shaft_with_all_foot_witnesses_airborne_or_deep_is_bad(self):
        self.points[0, ..., 2] = 0.02
        self.points[1, ..., 2] = -0.02
        self.forces[:, -1, [3, 1], 2] = 2
        feet, body = self.state()
        self.assertFalse(bool(feet.any()))
        self.assertTrue(bool(body.all()))

    def test_support_does_not_hide_separate_nonfoot_body_contact(self):
        self.forces[:, -1, 3, 2] = 3
        self.forces[0, 0, 0, 0] = 3
        feet, body = self.state()
        torch.testing.assert_close(feet, torch.tensor([[True, False], [True, False]]))
        torch.testing.assert_close(body, torch.tensor([True, False]))

    def test_single_witness_matches_legacy_sole_exactly(self):
        points = self.points[:, :, :1]
        heights = self.heights[:, :, :1]
        self.forces[0, 1, 3, 2] = 2
        self.forces[1, -1, 1, 0] = 2
        expected = CONTACT.sole_contact_state(
            self.forces, sensor_feet=[3, 1], sensor_body=[0, 2],
            sole_positions_w=points.squeeze(2), surface_heights=heights.squeeze(2),
            force_threshold=1.0, height_tolerance=0.012,
        )
        actual = CONTACT.foot_contact_state(self.forces, **{
            **self.options, "foot_positions_w": points, "surface_heights": heights,
        })
        for observed, legacy in zip(actual, expected):
            torch.testing.assert_close(observed, legacy)

    def test_foot_witness_shapes_devices_and_dtypes_fail_before_selection(self):
        for shape in ((2, 2, 3), (2, 2, 0, 3), (2, 0, 3, 3), (2, 2, 3, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                CONTACT.closest_foot_support(torch.zeros(shape), self.heights)
        with self.assertRaises(ValueError):
            CONTACT.closest_foot_support(self.points, self.heights[..., :1])
        with self.assertRaises(ValueError):
            CONTACT.closest_foot_support(self.points, self.heights.to("meta"))
        for points, heights in ((self.points.long(), self.heights), (self.points, self.heights.long())):
            with self.subTest(points=points.dtype, heights=heights.dtype), self.assertRaises(ValueError):
                CONTACT.closest_foot_support(points, heights)

    def test_force_validation_and_foot_count_matching_are_preserved(self):
        for force in (self.forces.long(), torch.zeros((2, 0, 5, 3)), self.forces.to("meta")):
            with self.subTest(force_shape=force.shape, force_dtype=force.dtype, device=force.device), self.assertRaises(ValueError):
                CONTACT.foot_contact_state(force, **self.options)
        for key, value in (("sensor_feet", [3]), ("sensor_body", [3]), ("force_threshold", 0), ("height_tolerance", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONTACT.foot_contact_state(self.forces, **{**self.options, key: value})


if __name__ == "__main__":
    unittest.main()
