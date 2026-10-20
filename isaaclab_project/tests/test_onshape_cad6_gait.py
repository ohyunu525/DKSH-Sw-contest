"""Fast checks for conservative current-Onshape joint-space references."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dksh_isaaclab"
    / "dksh_isaaclab"
    / "tasks"
    / "direct"
    / "spider_navigation"
    / "onshape_cad6_gait.py"
)
SPEC = importlib.util.spec_from_file_location("dksh_onshape_cad6_gait", MODULE)
assert SPEC is not None and SPEC.loader is not None
GAIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GAIT)


class OnshapeCad6GaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        urdf = Path(__file__).resolve().parents[1] / 'assets/spiderbot_variants/onshape_20260928/robot.training.urdf'
        cls.cad_joints = {joint.get('name'): joint for joint in ET.parse(urdf).getroot().findall('joint')}

    @staticmethod
    def _rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
        return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)

    def _cad_foot_position(self, leg, q):
        """Independent FK using CAD origins and axes, not the gait's gains."""
        transform = np.eye(4)
        for index, kind in enumerate(('hip', 'femur', 'tibia', 'foot')):
            suffix = 'fixed' if kind == 'foot' else 'joint'
            joint = self.cad_joints[f'leg_{leg}_{kind}_{suffix}']
            origin = joint.find('origin')
            roll, pitch, yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            fixed = np.eye(4)
            fixed[:3, :3] = (self._rotation((0, 0, 1), yaw)
                             @ self._rotation((0, 1, 0), pitch)
                             @ self._rotation((1, 0, 0), roll))
            fixed[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform = transform @ fixed
            if kind != 'foot':
                motion = np.eye(4)
                motion[:3, :3] = self._rotation(np.fromstring(joint.find('axis').get('xyz'), sep=' '), q[index])
                transform = transform @ motion
        return transform[:3, 3]

    def _assert_stance_direction(self, offsets, duty, commands, relative_error_limit=None):
        period = 2.4
        phase_delta = 0.0001
        for command in commands:
            for leg, offset in enumerate(offsets):
                with self.subTest(command=command, leg=leg, duty=duty):
                    # Both samples are in this foot's planted stance interval.
                    phase = (0.30 - offset / 6) % 1
                    q = GAIT.joint_positions_for_twist(
                        torch.tensor([phase, phase + phase_delta], dtype=torch.float64),
                        torch.tensor([command] * 2, dtype=torch.float64),
                        period, 0.004, 0.012, offsets=offsets, duty=duty,
                    ).numpy()[:, leg]
                    position = self._cad_foot_position(leg, q[0])
                    velocity = (self._cad_foot_position(leg, q[1]) - position) / (phase_delta * period)
                    # A stationary world foot travels opposite to body yaw in
                    # body coordinates: dp/dt = -omega cross p.
                    required = -np.array((*command[:2], 0.)) - np.cross((0., 0., command[2]), position)
                    self.assertGreater(float(velocity @ required), 0.0,
                                       'CAD stance foot moves opposite to the no-slip body-twist requirement')
                    if relative_error_limit is not None:
                        self.assertLess(float(np.linalg.norm(velocity - required) / np.linalg.norm(required)),
                                        relative_error_limit,
                                        'Small-command Cartesian travel is incorrectly scaled against CAD')

    def test_wave_stance_matches_cad_no_slip_yaw_direction(self) -> None:
        self._assert_stance_direction(GAIT.WAVE_OFFSETS, 5 / 6, ((0., 0., -0.04), (0., 0., 0.04)))

    def test_tripod_stance_matches_cad_no_slip_yaw_direction(self) -> None:
        self._assert_stance_direction(GAIT.TRIPOD_OFFSETS, 0.55, ((0., 0., -0.04), (0., 0., 0.04)))

    def test_wave_translation_stance_matches_cad_no_slip_direction(self) -> None:
        self._assert_stance_direction(GAIT.WAVE_OFFSETS, 5 / 6,
                                     ((0.003, 0., 0.), (-0.003, 0., 0.), (0., 0.003, 0.), (0., -0.003, 0.)))

    def test_tripod_translation_stance_matches_cad_no_slip_direction(self) -> None:
        self._assert_stance_direction(GAIT.TRIPOD_OFFSETS, 0.55,
                                     ((0.003, 0., 0.), (-0.003, 0., 0.), (0., 0.003, 0.), (0., -0.003, 0.)))

    def test_small_unsaturated_commands_match_cad_velocity_scale(self) -> None:
        commands = ((0.003, 0., 0.), (-0.003, 0., 0.), (0., 0.003, 0.), (0., -0.003, 0.),
                    (0., 0., 0.004), (0., 0., -0.004))
        self._assert_stance_direction(GAIT.WAVE_OFFSETS, 5 / 6, commands, relative_error_limit=0.20)

    def test_stationary_reference_is_the_saved_zero_pose(self) -> None:
        phase = torch.linspace(0, 1, 41)
        q = GAIT.wave_joint_positions(phase, torch.zeros_like(phase), 2.4, 0.0, 0.012)
        torch.testing.assert_close(q, torch.zeros_like(q))

    def test_references_remain_well_within_joint_limits(self) -> None:
        phase = torch.linspace(0, 1, 2401)
        commands = torch.tensor([[0.012, 0.006, 0.08]]).expand(len(phase), -1)
        q = GAIT.joint_positions_for_twist(phase, commands, 2.4, 0.005, 0.012)
        self.assertEqual(q.shape, (len(phase), 6, 3))
        self.assertTrue(torch.isfinite(q).all())
        self.assertLess(float(q.abs().max()) + 0.035, math.pi / 2 * 0.30)

    def test_swing_lifts_only_femur_and_tibia_from_zero(self) -> None:
        phase = torch.tensor([0.90])
        q = GAIT.wave_joint_positions(phase, torch.tensor([0.0]), 2.4, 0.004, 0.012)
        self.assertTrue((q[:, :, 1] >= 0).all())
        self.assertTrue((q[:, :, 2] >= 0).all())
        self.assertGreater(float(q[:, :, 1].max()), 0.0)
        # Check actual CAD foot-point height, rather than assuming that a
        # joint's positive/negative sign always corresponds to vertical lift.
        for leg in range(6):
            if q[0, leg, 1] > 0.01:
                self.assertGreater(self._cad_foot_position(leg, q[0, leg].numpy())[2],
                                   self._cad_foot_position(leg, np.zeros(3))[2])


if __name__ == "__main__":
    unittest.main()
