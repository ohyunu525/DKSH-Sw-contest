"""Exercise slip measurement at a classifier-valid pose of the actual CAD foot.

The tensor environment runs the production substep capture and reward methods.
It verifies a geometric measurement defect; it does not simulate PhysX contact
points or claim that this instantaneous pose is a dynamically stable gait.
"""

import math
import unittest

import torch

import test_joint_angle_env as environment
import test_joint_angle_foot_geometry as foot_geometry


class DistalSlipMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        foot_geometry.FootGeometryTests.setUpClass()
        helper = foot_geometry.FootGeometryTests()
        geometry = foot_geometry.GEOMETRY
        frame = helper.world_frame(
            "leg_0_tibia", {"leg_0_femur_joint": -0.03, "leg_0_tibia_joint": -0.03},
        )
        proximal = [foot_geometry.transform(frame, point)
                    for point in geometry.ONSHAPE_PROXIMAL_FOOT_OFFSETS[0]]
        shift = 0.0011 - min(point[2] for point in proximal)

        def points_in_fixture_frame(points):
            return tuple((x, y, z + shift)
                         for x, y, z in (foot_geometry.transform(frame, point) for point in points))

        cls.full = points_in_fixture_frame(geometry.ONSHAPE_FOOT_SUPPORT_OFFSETS[0])
        cls.distal = points_in_fixture_frame(geometry.ONSHAPE_DISTAL_FOOT_SUPPORT_OFFSETS[0])
        cls.proximal = points_in_fixture_frame(geometry.ONSHAPE_PROXIMAL_FOOT_OFFSETS[0])
        cls.pivot = min(cls.distal, key=lambda point: abs(point[2]))

    def capture_sample(self, profile, *, angular_velocity=(0.0, 0.0, 0.0),
                       translation=(0.0, 0.0, 0.0)):
        # Repeat one real CAD foot pose across six tensor bodies to isolate
        # contact-point selection from unrelated assembly and actuator motion.
        collision = {"base": ((0.0, 0.0, 0.5),)}
        collision.update({f"leg_{leg}_{part}": ((0.0, 0.0, 0.5),)
                          for leg in range(6) for part in ("hip", "femur", "tibia")})
        env = environment.make_env(
            support_offsets=(self.full,) * 6, distal_offsets=(self.distal,) * 6,
            proximal_offsets=(self.proximal,) * 6, toe_button_offsets=(self.pivot,) * 6,
            collision_offsets=collision, behavior_profile=profile,
        )
        omega, pivot, translation = map(torch.tensor, (angular_velocity, self.pivot, translation))
        link_velocity = translation - torch.linalg.cross(omega, pivot)
        for body in env._feet:
            env._robot.data.body_link_ang_vel_w[:, body] = omega
            env._robot.data.body_link_lin_vel_w[:, body] = link_velocity
        for name, sensor in zip(env._self_contact_sources, env._self_contact_sensors):
            if name.endswith("_tibia"):
                sensor.data.net_forces_w[..., 2] = 2.0
        env._pre_physics_step(env._actions.clone())
        for _ in range(env.cfg.decimation):
            env._apply_action()
        terminated, _ = env._get_dones()
        self.assertFalse(terminated.any())
        self.assertTrue(env._strict_contact_state()[0].all())
        self.assertEqual(env._stance_slip_samples, env.cfg.decimation)
        env._get_rewards()
        return env

    def test_actual_cad_pose_has_valid_distal_support_but_nearer_proximal_witness(self):
        distal_gap = min(point[2] for point in self.distal)
        proximal_gap = min(point[2] for point in self.proximal)
        self.assertAlmostEqual(proximal_gap, 0.0011, places=10)
        self.assertAlmostEqual(distal_gap, 0.0028523263819478784, places=10)
        self.assertGreater(proximal_gap, 0.001)
        self.assertLess(distal_gap, 0.004)
        nearest_full = min(self.full, key=lambda point: abs(point[2]))
        self.assertIn(nearest_full, self.proximal)
        self.assertAlmostEqual(math.dist(nearest_full, self.pivot), 0.10341952054754572, places=9)

    def test_production_substeps_keep_fixed_distal_point_free_of_false_rolling_slip(self):
        old = self.capture_sample("revised_v11", angular_velocity=(0.0, 0.0, 1.0))
        fixed = self.capture_sample("revised_v12", angular_velocity=(0.0, 0.0, 1.0))
        old_rms = old.step_metrics["foot_slip_m2_s2"].sqrt()
        torch.testing.assert_close(old_rms, torch.full((2,), 0.10340467389115136))
        torch.testing.assert_close(fixed.step_metrics["foot_slip_m2_s2"], torch.zeros(2), atol=1e-12, rtol=0)
        torch.testing.assert_close(fixed._episode_sums["foot_slip"], torch.zeros(2), atol=1e-8, rtol=0)
        self.assertTrue((old._episode_sums["foot_slip"] < -0.06).all())

    def test_production_substeps_still_measure_real_translation_of_supported_distal_point(self):
        env = self.capture_sample("revised_v12", translation=(0.02, 0.0, 0.0))
        torch.testing.assert_close(env.step_metrics["foot_slip_m2_s2"], torch.full((2,), 0.02 ** 2))
        expected = -env.cfg.stance_slip_cost_coefficient * 0.02 * env.step_dt
        torch.testing.assert_close(env._episode_sums["foot_slip"], torch.full((2,), expected))


if __name__ == "__main__":
    unittest.main()
