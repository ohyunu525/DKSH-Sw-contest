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


class DistalFootContactStateTests(unittest.TestCase):
    def setUp(self):
        self.forces = torch.zeros((2, 4, 5, 3), dtype=torch.float64)
        self.distal = torch.zeros((2, 2, 3, 3), dtype=torch.float64)
        self.distal_heights = torch.zeros((2, 2, 3), dtype=torch.float64)
        self.proximal = torch.zeros((2, 2, 2, 3), dtype=torch.float64)
        self.proximal[..., 2] = 0.06
        self.proximal_heights = torch.zeros((2, 2, 2), dtype=torch.float64)
        self.options = dict(
            sensor_feet=[3, 1], sensor_body=[0, 2],
            distal_foot_positions_w=self.distal, distal_surface_heights=self.distal_heights,
            proximal_foot_positions_w=self.proximal, proximal_surface_heights=self.proximal_heights,
            force_threshold=1.0, height_tolerance=0.004, proximal_height_tolerance=0.004,
        )

    def state(self):
        return CONTACT.distal_foot_contact_state(self.forces, **self.options)

    def test_distal_load_with_knee_raised_is_valid_in_public_leg_order(self):
        self.forces[0, 0, 1, 2] = 2
        self.forces[1, -1, 3, 2] = 3
        support, body, wrong_load = self.state()
        torch.testing.assert_close(support, torch.tensor([[False, True], [True, False]]))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(wrong_load.any()))

    def test_knee_only_load_is_rejected_even_though_full_hull_would_support_it(self):
        self.distal[..., 2] = 0.08
        self.proximal[..., 2] = 0
        self.forces[:, -1, [3, 1], 2] = 5
        support, body, wrong_load = self.state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(wrong_load.all()))
        historical_support, historical_body = CONTACT.foot_contact_state(
            self.forces, sensor_feet=[3, 1], sensor_body=[0, 2],
            foot_positions_w=torch.cat((self.distal, self.proximal), dim=2),
            surface_heights=torch.cat((self.distal_heights, self.proximal_heights), dim=2),
            force_threshold=1.0, height_tolerance=0.004,
        )
        self.assertTrue(bool(historical_support.all()))
        self.assertFalse(bool(historical_body.any()))

    def test_flat_shaft_load_is_rejected_when_distal_and_knee_both_reach_floor(self):
        self.forces[:, -1, [3, 1], 2] = 5
        self.proximal[..., 0, 2] = 0
        support, body, wrong_load = self.state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(wrong_load.all()))

    def test_actual_middle_shaft_on_stair_is_rejected_with_knee_and_tip_on_other_surfaces(self):
        geometry_spec = importlib.util.spec_from_file_location(
            "actual_foot_geometry", MODULE.with_name("joint_angle_foot_geometry.py"),
        )
        geometry = importlib.util.module_from_spec(geometry_spec)
        geometry_spec.loader.exec_module(geometry)
        # Pitch the real CAD tibia down towards its tip. Its distal patch rests
        # on z=0, while the knee and middle shaft clear this lower tread.
        angle = 0.8
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
        distal = torch.tensor(geometry.ONSHAPE_DISTAL_FOOT_SUPPORT_OFFSETS[0], dtype=torch.float64) @ rotation.T
        proximal = torch.tensor(geometry.ONSHAPE_PROXIMAL_FOOT_OFFSETS[0], dtype=torch.float64) @ rotation.T
        floor_shift = -distal[:, 2].min()
        distal[:, 2] += floor_shift
        proximal[:, 2] += floor_shift
        forces = torch.zeros((1, 1, 1, 3), dtype=torch.float64)
        forces[..., 2] = 5
        surfaces = torch.zeros((1, 1, len(proximal)), dtype=torch.float64)
        options = dict(
            sensor_feet=[0], sensor_body=[],
            distal_foot_positions_w=distal[None, None],
            distal_surface_heights=torch.zeros((1, 1, len(distal)), dtype=torch.float64),
            proximal_foot_positions_w=proximal[None, None], proximal_surface_heights=surfaces,
            force_threshold=1.0, height_tolerance=0.004, proximal_height_tolerance=0.008,
        )
        support, body, wrong_load = CONTACT.distal_foot_contact_state(forces, **options)
        self.assertTrue(bool(support.all()))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(wrong_load.any()))
        middle_indices = [i for i, point in enumerate(geometry._PROXIMAL_FOOT_HULL_POINTS_M)
                          if 0.050 < point[1] < 0.125]
        self.assertTrue(middle_indices)
        # A narrow raised stair nosing now reaches a real middle-shaft witness.
        # Knee-side witnesses still see the lower tread; the distal tip remains
        # on its own lower surface. A first-50-mm-only check misses this load.
        middle = min(middle_indices, key=lambda i: float(proximal[i, 2]))
        self.assertGreater(float(proximal[middle, 2]), 0.008)
        surfaces[0, 0, middle] = proximal[middle, 2]
        support, body, wrong_load = CONTACT.distal_foot_contact_state(forces, **options)
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(wrong_load.all()))
        knee_indices = [i for i, point in enumerate(geometry._PROXIMAL_FOOT_HULL_POINTS_M)
                        if point[1] <= 0.050]
        historical, _, _ = CONTACT.distal_foot_contact_state(forces, **{
            **options, "proximal_foot_positions_w": proximal[None, None, knee_indices],
            "proximal_surface_heights": surfaces[:, :, knee_indices],
        })
        self.assertTrue(bool(historical.all()))

    def test_deep_knee_penetration_cannot_be_hidden_by_distal_support(self):
        self.forces[:, -1, [3, 1], 2] = 5
        self.proximal[..., 1, 2] = -0.03
        support, body, wrong_load = self.state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        self.assertTrue(bool(wrong_load.all()))

    def test_unloaded_near_ground_swing_leg_does_not_create_bad_contact(self):
        self.proximal[..., 2] = 0
        support, body, wrong_load = self.state()
        self.assertFalse(bool(support.any()))
        self.assertFalse(bool(body.any()))
        self.assertFalse(bool(wrong_load.any()))

    def test_wall_and_downward_loads_are_bad_even_with_distal_patch_at_floor(self):
        self.forces[0, 1, 3, 0] = 5
        self.forces[1, 0, 1, 2] = -5
        support, body, wrong_load = self.state()
        self.assertFalse(bool(support.any()))
        self.assertTrue(bool(body.all()))
        torch.testing.assert_close(wrong_load, torch.tensor([[True, False], [False, True]]))

    def test_each_patch_checks_its_own_terrain_on_stairs_and_clone_origins(self):
        self.distal_heights[:] = torch.tensor([[[0, .1, .2], [0, .1, .2]],
                                             [[1.5, 1.6, 1.7], [1.5, 1.6, 1.7]]])
        self.distal[..., 2] = self.distal_heights
        self.proximal_heights[:] = torch.tensor([[[.1, .2], [.3, .4]],
                                               [[1.6, 1.7], [1.8, 1.9]]])
        self.proximal[..., 2] = self.proximal_heights + .05
        self.proximal[1, 1, 1, 2] = self.proximal_heights[1, 1, 1] + .003
        self.forces[:, -1, [3, 1], 2] = 5
        support, body, wrong_load = self.state()
        torch.testing.assert_close(support, torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(body, torch.tensor([False, True]))
        torch.testing.assert_close(wrong_load, torch.tensor([[False, False], [False, True]]))

    def test_distal_support_does_not_mask_separate_nonfoot_collision(self):
        self.forces[:, -1, [3, 1], 2] = 5
        self.forces[0, 0, 2, 0] = 5
        support, body, wrong_load = self.state()
        self.assertTrue(bool(support.all()))
        torch.testing.assert_close(body, torch.tensor([True, False]))
        self.assertFalse(bool(wrong_load.any()))

    def test_proximal_tolerance_is_inclusive_and_does_not_relax_distal_tolerance(self):
        self.forces[:, -1, [3, 1], 2] = 5
        self.proximal[0, 0, :, 2] = 0.004
        self.proximal[0, 1, :, 2] = 0.00401
        self.distal[1, 0, :, 2] = 0.00401
        support, _, wrong_load = self.state()
        torch.testing.assert_close(support, torch.tensor([[False, True], [False, True]]))
        torch.testing.assert_close(wrong_load, ~support)

    def test_proximal_geometry_and_tolerance_validation(self):
        invalid = (
            ("proximal_foot_positions_w", torch.zeros((1, 2, 2, 3))),
            ("proximal_foot_positions_w", torch.zeros((2, 2, 0, 3))),
            ("proximal_foot_positions_w", self.proximal.long()),
            ("proximal_surface_heights", self.proximal_heights.long()),
            ("proximal_foot_positions_w", self.proximal.to("meta")),
            ("proximal_height_tolerance", -1),
            ("proximal_height_tolerance", math.inf),
            ("proximal_height_tolerance", math.nan),
        )
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONTACT.distal_foot_contact_state(self.forces, **{**self.options, key: value})


class StrictLegContactStateTests(unittest.TestCase):
    def setUp(self):
        # Sensor order deliberately differs from public leg order:
        # base, leg1 femur, leg0 tibia, leg0 hip, leg1 tibia, leg0 femur, leg1 hip.
        self.forces = torch.zeros((2, 4, 7, 3), dtype=torch.float64)
        self.supported = torch.zeros((2, 2), dtype=torch.bool)
        self.options = dict(
            sensor_feet=[2, 4], sensor_leg_nonfeet=[[3, 5], [6, 1]],
            sensor_body=[0, 1, 3, 5, 6], distal_supported=self.supported,
            force_threshold=1.0,
        )

    def state(self):
        return CONTACT.strict_leg_contact_state(self.forces, **self.options)

    def test_femur_only_grounding_counts_without_any_tibia_force(self):
        self.forces[0, 0, 5, 2] = 5
        self.forces[1, -1, 1, 2] = 5
        loaded, invalid, body = self.state()
        expected = torch.tensor([[True, False], [False, True]])
        torch.testing.assert_close(loaded, expected)
        torch.testing.assert_close(invalid, expected)
        self.assertTrue(bool(body.all()))

    def test_hip_only_grounding_counts_in_public_leg_order(self):
        self.forces[0, 2, 6, 2] = 5
        self.forces[1, 1, 3, 2] = 5
        loaded, invalid, body = self.state()
        expected = torch.tensor([[False, True], [True, False]])
        torch.testing.assert_close(loaded, expected)
        torch.testing.assert_close(invalid, expected)
        self.assertTrue(bool(body.all()))

    def test_supported_tip_does_not_mask_same_legs_femur_contact(self):
        self.supported[:] = True
        self.forces[:, -1, [2, 4], 2] = 5
        self.forces[0, 0, 5, 0] = 2
        self.forces[1, 1, 1, 2] = 2
        loaded, invalid, body = self.state()
        self.assertTrue(bool(loaded.all()))
        torch.testing.assert_close(invalid, torch.tensor([[True, False], [False, True]]))
        self.assertTrue(bool(body.all()))

    def test_valid_distal_support_is_loaded_without_invalid_contact(self):
        self.supported[0, 1] = True
        self.supported[1, 0] = True
        self.forces[0, 0, 4, 2] = 3
        self.forces[1, -1, 2, 2] = 3
        loaded, invalid, body = self.state()
        torch.testing.assert_close(loaded, self.supported)
        self.assertFalse(bool(invalid.any()))
        self.assertFalse(bool(body.any()))

    def test_unloaded_swing_does_not_become_invalid_from_missing_support(self):
        loaded, invalid, body = self.state()
        self.assertFalse(bool(loaded.any()))
        self.assertFalse(bool(invalid.any()))
        self.assertFalse(bool(body.any()))

    def test_base_only_collision_reports_body_without_inventing_a_loaded_leg(self):
        self.forces[0, 0, 0, 2] = 10
        loaded, invalid, body = self.state()
        self.assertFalse(bool(loaded.any()))
        self.assertFalse(bool(invalid.any()))
        torch.testing.assert_close(body, torch.tensor([True, False]))

    def test_wall_and_downward_tibia_impacts_are_invalid_loads(self):
        self.forces[0, 0, 2, 0] = 5
        self.forces[1, -1, 4, 2] = -5
        loaded, invalid, body = self.state()
        expected = torch.tensor([[True, False], [False, True]])
        torch.testing.assert_close(loaded, expected)
        torch.testing.assert_close(invalid, expected)
        self.assertFalse(bool(body.any()))  # Tibia impacts are in invalid, not nonfoot.

    def test_mapped_femur_reports_body_even_if_body_list_only_contains_base(self):
        self.options["sensor_body"] = [0]
        self.forces[0, 1, 5, 2] = 5
        _, invalid, body = self.state()
        self.assertTrue(bool(invalid[0, 0]))
        torch.testing.assert_close(body, torch.tensor([True, False]))

    def test_empty_nonfoot_groups_and_body_list_are_supported(self):
        self.options.update(sensor_leg_nonfeet=[[], []], sensor_body=[])
        self.forces[0, 0, 2, 2] = 5
        loaded, invalid, body = self.state()
        torch.testing.assert_close(loaded, torch.tensor([[True, False], [False, False]]))
        torch.testing.assert_close(invalid, loaded)
        self.assertFalse(bool(body.any()))

    def test_threshold_uses_magnitude_strictly_and_retains_history(self):
        self.forces[0, 0, 5, 2] = 1
        self.forces[1, 0, 1, :2] = torch.tensor([0.8, 0.8])
        loaded, invalid, body = self.state()
        expected = torch.tensor([[False, False], [False, True]])
        torch.testing.assert_close(loaded, expected)
        torch.testing.assert_close(invalid, expected)
        torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_nonfinite_femur_and_hip_forces_fail_closed_only_in_their_own_environment(self):
        for index, leg, value, axis in ((5, 0, math.nan, 0), (6, 1, math.inf, 1),
                                        (1, 1, -math.inf, 2)):
            with self.subTest(index=index, value=value):
                self.forces.zero_()
                self.forces[0, 1, index, axis] = value
                loaded, invalid, body = self.state()
                expected = torch.zeros((2, 2), dtype=torch.bool)
                expected[0, leg] = True
                torch.testing.assert_close(loaded, expected)
                torch.testing.assert_close(invalid, expected)
                torch.testing.assert_close(body, torch.tensor([True, False]))

    def test_nonfinite_tibia_force_invalidates_even_a_supported_distal_tip(self):
        self.supported[:] = True
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.forces.zero_()
                self.forces[:, -1, [2, 4], 2] = 5
                self.forces[0, 0, 4, 0] = value
                loaded, invalid, body = self.state()
                self.assertTrue(bool(loaded.all()))
                torch.testing.assert_close(invalid, torch.tensor([[False, True], [False, False]]))
                self.assertFalse(bool(body.any()))  # Invalid tibia is reported in the per-leg mask.

    def test_nonfinite_base_force_reports_body_without_loading_unrelated_legs(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.forces.zero_()
                self.forces[1, 0, 0, 2] = value
                loaded, invalid, body = self.state()
                self.assertFalse(bool(loaded.any()))
                self.assertFalse(bool(invalid.any()))
                torch.testing.assert_close(body, torch.tensor([False, True]))

    def test_invalid_mapping_is_rejected(self):
        cases = (
            ("sensor_feet", []), ("sensor_feet", [2, 2]),
            ("sensor_feet", [-1, 4]), ("sensor_feet", [2, 7]),
            ("sensor_feet", [True, 4]),
            ("sensor_leg_nonfeet", [[3, 5]]),
            ("sensor_leg_nonfeet", [[3, 5], [5, 1]]),
            ("sensor_leg_nonfeet", [[3, 3], [6, 1]]),
            ("sensor_leg_nonfeet", [[2, 5], [6, 1]]),
            ("sensor_leg_nonfeet", [[3, 7], [6, 1]]),
            ("sensor_leg_nonfeet", [[3, 0.5], [6, 1]]),
            ("sensor_leg_nonfeet", [3, [6, 1]]),
            ("sensor_leg_nonfeet", None),
            ("sensor_body", [2]), ("sensor_body", [0, 0]),
            ("sensor_body", [7]), ("sensor_body", [False]),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CONTACT.strict_leg_contact_state(self.forces, **{**self.options, key: value})

    def test_force_and_support_metadata_are_validated(self):
        for forces in (torch.zeros((2, 7, 3)), torch.zeros((2, 0, 7, 3)),
                       torch.zeros((2, 4, 7, 2)), self.forces.long()):
            with self.subTest(shape=forces.shape), self.assertRaises(ValueError):
                CONTACT.strict_leg_contact_state(forces, **self.options)
        for support in (torch.zeros((2, 3), dtype=torch.bool), self.supported.float(),
                        self.supported.to("meta")):
            with self.subTest(shape=support.shape, dtype=support.dtype), self.assertRaises(ValueError):
                CONTACT.strict_leg_contact_state(self.forces, **{**self.options, "distal_supported": support})
        for threshold in (0, -1, math.inf, math.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                CONTACT.strict_leg_contact_state(self.forces, **{**self.options, "force_threshold": threshold})


if __name__ == "__main__":
    unittest.main()
