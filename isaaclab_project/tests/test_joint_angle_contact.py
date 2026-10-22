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


if __name__ == "__main__":
    unittest.main()
