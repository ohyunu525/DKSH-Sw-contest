"""Original ZIP equations versus the SI tensor port, without hardware imports."""

import ast
import hashlib
import math
from pathlib import Path
import unittest
from zipfile import ZipFile

import torch

import test_joint_angle_env as FIXTURE


def Upgraded_original_equations():
    """Compile only the reviewed mathematical functions; never import the ZIP."""
    path = Path(__file__).resolve().parents[2] / 'HexapodRobot2.zip'
    with ZipFile(path) as archive:
        tree = ast.parse(archive.read('hexapod_joy_walk.py').decode('utf-8-sig'))
    constants = {'DUTY', 'SWING_FLAT', 'SWING_UP_END', 'SWING_DOWN_START', 'LAND_APPROACH'}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in constants]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'swing_profile']
    gait = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Gait')
    methods = [node for node in gait.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_target', '_build', '_budget', 'land_above'}]
    shell = ast.parse('class OriginalEquations:\n    pass\n').body[0]
    shell.body = methods
    nodes.append(shell)
    namespace = {'math': math, 'NUM_LEGS': 6}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path) + ':reviewed_equations', 'exec'), namespace)
    return namespace


class UpgradedOriginalGaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.port = FIXTURE.load_helper('Upgraded_original_gait')
        cls.original = Upgraded_original_equations()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_swing_profile_matches_original_at_boundaries_and_between_them(self):
        samples = torch.cat((torch.linspace(0, 1, 151, dtype=torch.float64),
                             torch.tensor([0.08, 0.4, 0.5, 0.92], dtype=torch.float64)))
        actual = self.port.Upgraded_swing_profile(samples)
        expected = torch.tensor([self.original['swing_profile'](float(u)) for u in samples],
                                dtype=torch.float64)
        torch.testing.assert_close(actual[0], expected[:, 0], atol=1e-12, rtol=0)
        torch.testing.assert_close(actual[1], expected[:, 1], atol=1e-12, rtol=0)

    def port_and_original(self, count=5):
        theta = torch.arange(6, dtype=torch.float64) * math.pi / 3
        home = torch.stack((0.23 * theta.cos(), 0.20 * theta.sin(),
                            torch.full_like(theta, -0.192)), -1)[None].repeat(count, 1, 1)
        port = self.port.UpgradedOriginalGait(home, land_approach=self.original['LAND_APPROACH'] / 1000)
        port.position[..., 0] -= 0.014
        port.start[..., 0] += 0.006
        port.position[..., 2] += torch.linspace(-0.003, 0.003, 6)
        port.ground.copy_(port.position[..., 2])
        port.ref_off[:] = 0.001
        port.in_swing[:, 0::2] = True
        originals = []
        for i in range(count):
            original = self.original['OriginalEquations']()
            original.home = (port.home[i, :, :2] * 1000).tolist()
            original.feet = (port.position[i] * 1000).tolist()
            original.start = (port.start[i, :, :2] * 1000).tolist()
            original.ground = (port.ground[i] * 1000).tolist()
            original.base_ground = (port.base_ground[i] * 1000).tolist()
            original.offset = port.offset[0].tolist()
            original.in_swing = port.in_swing[i].tolist()
            original.lift = port.lift * 1000
            original.stride = port.stride * 1000
            original.ref_off = float(port.ref_off[i]) * 1000
            original.land_bias = [0.0] * 6
            original.probe_on = True
            originals.append(original)
        return port, originals

    def test_build_and_stride_budget_match_original_in_si_units(self):
        port, originals = self.port_and_original()
        phase = torch.tensor([0.001, 0.25, 0.45, 0.75, 0.999], dtype=torch.float64)
        dt = torch.tensor([0.02, 0.01, 0.03, 0.02, 0.005], dtype=torch.float64)
        v = torch.tensor([[0.1, 0.04], [0.01, -0.02], [0.12, 0], [-0.03, 0.01], [0, 0]], dtype=torch.float64)
        w = torch.tensor([0.2, -0.1, 0, 0.12, -0.2], dtype=torch.float64)
        target_v, target_w = v * 0.7, w * 0.8
        feet, swing, starts = port._build(phase, dt, v, w, target_v, target_w, lift_scale=0.7)
        budget = port._budget(phase, dt, v, w)
        for i, original in enumerate(originals):
            original.v = (v[i] * 1000).tolist()
            original.w = math.degrees(float(w[i]))
            expected = original._build(float(phase[i]), float(dt[i]), original.v, original.w,
                                       port.period * port.duty, 0.7,
                                       (target_v[i] * 1000).tolist(), math.degrees(float(target_w[i])))
            torch.testing.assert_close(feet[i], torch.tensor(expected[0], dtype=feet.dtype) / 1000,
                                       atol=1e-12, rtol=0)
            self.assertEqual(swing[i].tolist(), expected[1])
            torch.testing.assert_close(starts[i, :, :2], torch.tensor(expected[2], dtype=feet.dtype) / 1000,
                                       atol=1e-12, rtol=0)
            expected_budget = original._budget(float(phase[i]), float(dt[i]), port.period * port.duty)
            self.assertEqual(float(budget[i]), expected_budget)
        self.assertTrue((budget < 1).any(), 'Exercise a real stride-budget reduction')

    def test_proposal_is_transactional_and_partial_reset_is_isolated(self):
        port, _ = self.port_and_original(count=2)
        before = {name: getattr(port, name).clone() for name in port.state_names}
        command = port.home.new_tensor([[0.02, 0, 0.1], [0, -0.02, -0.1]])
        proposal = port.proposal(0.02, command)
        for name, value in before.items():
            torch.testing.assert_close(getattr(port, name), value)
        port.commit(proposal, torch.tensor([True, False]))
        self.assertGreater(float(port.phase[0]), 0)
        for name, value in before.items():
            torch.testing.assert_close(getattr(port, name)[1], value[1])
        self.assertEqual(port.features().shape, (2, port.feature_count))
        first = {name: getattr(port, name)[0].clone() for name in port.state_names}
        with torch.inference_mode():
            port.reset(torch.tensor([1]))
        for name, value in first.items():
            torch.testing.assert_close(getattr(port, name)[0], value)
        torch.testing.assert_close(port.position[1], port.home[1])

    def test_port_provenance_matches_the_supplied_source(self):
        archive_path = Path(__file__).resolve().parents[2] / 'HexapodRobot2.zip'
        with ZipFile(archive_path) as archive:
            source = archive.read('hexapod_joy_walk.py')
        self.assertEqual(hashlib.sha256(source).hexdigest(), self.port.UPGRADED_GAIT_SOURCE_SHA256)


if __name__ == '__main__':
    unittest.main()
