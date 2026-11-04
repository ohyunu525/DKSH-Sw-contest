"""Imitation alignment, rejection, episode splits and CAD reference regressions."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_joint_angle_env as FIXTURE
import Upgraded_test_cad6 as CAD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import Upgraded_imitation_learning as LEARNING
import Upgraded_imitation_runner as RUNNER

REFERENCE = FIXTURE.load_helper('Upgraded_hexapod_reference')


def Upgraded_home(count=2):
    k = CAD.Upgraded_kinematics()
    q = torch.tensor([0.0, -0.3, -0.3], dtype=torch.float64).repeat(count, 6, 1)
    home = k.forward(q)[0]
    home[..., 2] = home[..., 2].amin(-1, keepdim=True)
    return k, q, home


def Upgraded_fixture():
    path = FIXTURE.FOLDER / 'Upgraded_imitation_env.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UpgradedImitationEnv')
    namespace = dict(torch=torch, DirectRLEnv=FIXTURE.Parent, JointAngleNavigationEnv=FIXTURE.Environment,
                     normalize_angles=FIXTURE.CONTROL.normalize_angles,
                     UpgradedCad6Kinematics=CAD.CONTROL.UpgradedCad6Kinematics,
                     UpgradedCrawlReference=REFERENCE.UpgradedCrawlReference)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    base = FIXTURE.StrictContactEnvironmentTests().strict_env(behavior_profile='revised_v12')
    settings = vars(base.cfg).copy()
    settings.update(history_frames=4, decimation=4, reference_settings={}, ik_clearance=0.008,
                    target_speed=0.003, imu_angular_noise=0.01, imu_gravity_noise=0.005,
                    toe_button_offsets=CAD.FOOT.ONSHAPE_TOE_BUTTON_OFFSETS,
                    non_distal_collision_offsets=CAD.COLLISION.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS,
                    proximal_foot_offsets=CAD.FOOT.ONSHAPE_PROXIMAL_FOOT_OFFSETS,
                    robot=SimpleNamespace(copy=lambda: deepcopy(base.cfg.robot)))
    original = FIXTURE.Robot

    def robot(count, legs):
        result = original(count, legs)
        for i, name in enumerate(result.joint_names):
            result.data.default_joint_pos[:, i] = 0 if '_hip_' in name else -0.3
        result.data.joint_pos.copy_(result.data.default_joint_pos)
        return result

    with patch.object(FIXTURE, 'Robot', robot):
        env = namespace['UpgradedImitationEnv'](SimpleNamespace(**settings))
    env._reset_idx(torch.tensor([0, 1]))
    env._out_of_bounds = torch.zeros(2, dtype=torch.bool)
    env._contact_state = lambda: (torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    return env


class UpgradedImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_archive_inspection_never_imports_hardware(self):
        report = RUNNER.Upgraded_inspect_archive(RUNNER.UPGRADED_ROOT / 'HexapodRobot.zip')
        self.assertEqual(set(report['missing_local_modules']), {'hexapod_body.py', 'hexapod_joy_walk.py'})
        self.assertFalse(report['original_code_executed'])
        self.assertEqual(len(report['members']), 5)
        self.assertNotIn('hexapod_sdk', sys.modules)

    def test_single_foot_sequence_and_cad_ik(self):
        k, q, home = Upgraded_home()
        reference = REFERENCE.UpgradedCrawlReference(home)
        command = q.new_tensor([[0.003, 0, 0], [0, 0.003, 0]])
        gravity = q.new_tensor([[0, 0, -1]]).expand(2, -1)
        visited, legs = set(), set()
        for step in range(1050):
            contacts = reference.position[..., 2] <= home[..., 2] + 0.0001
            position = reference.advance(contacts, command, gravity, 0.02)
            self.assertFalse(reference.failed.any(), f'{step}: {reference.fault}')
            self.assertTrue(((position[..., 2] > home[..., 2] + 0.001).sum(-1) <= 1).all())
            visited.add(int(reference.stage[0]))
            legs.add(int(reference.active_leg()[0]))
            if step % 10 == 0:
                q, valid, error = k.inverse(position, q)
                self.assertTrue(valid.all(), f'{step}: {error}')
        self.assertEqual(visited, {0, 1, 2, 3, 4})
        self.assertEqual(legs, set(range(6)))

    def test_stuck_contact_and_missing_support_cannot_advance(self):
        _, _, home = Upgraded_home()
        reference = REFERENCE.UpgradedCrawlReference(home)
        contact = torch.ones(2, 6, dtype=torch.bool)
        command = home.new_tensor([[0.003, 0, 0]]).expand(2, -1)
        gravity = home.new_tensor([[0, 0, -1]]).expand(2, -1)
        for _ in range(50):
            reference.advance(contact, command, gravity, 0.02)
        self.assertTrue((reference.fault == 2).all())
        reference.reset(torch.tensor([0]))
        self.assertEqual(reference.stage[0], reference.WAIT)
        self.assertEqual(reference.stage[1], reference.FAILED)
        for _ in range(310):
            with torch.inference_mode():
                reference.advance(torch.zeros_like(contact), command, gravity, 0.02)
        self.assertEqual(reference.fault[0], 1)
        reference.reset(torch.tensor([0]))  # normal reset after inference-mode updates

    def test_body_shift_requires_the_recently_landed_foot_too(self):
        _, _, home = Upgraded_home()
        reference = REFERENCE.UpgradedCrawlReference(home)
        reference.stage[:] = reference.SHIFT
        contacts = torch.ones(2, 6, dtype=torch.bool)
        contacts[:, 0] = False
        for _ in range(6):
            reference.advance(contacts, home.new_tensor([[.003, 0, 0]]).expand(2, -1),
                              home.new_tensor([[0, 0, -1]]).expand(2, -1), .02)
        self.assertTrue((reference.fault == 3).all())

    def test_observation_timing_privilege_boundary_and_reset(self):
        env = Upgraded_fixture()
        first = env._get_observations()
        self.assertEqual(first['policy'].shape, (2, 456))
        self.assertEqual(first['critic'].shape, (2, 596))
        label = env.reference_actions.clone()
        clock = env._reference.elapsed.clone()
        env._robot.data.joint_pos += 0.01
        env._robot.data.root_lin_vel_b += 0.1
        repeat = env._get_observations()
        torch.testing.assert_close(first['policy'], repeat['policy'])
        torch.testing.assert_close(label, env.reference_actions)
        torch.testing.assert_close(clock, env._reference.elapsed)
        self.assertFalse(torch.equal(first['critic'], repeat['critic']))
        env._pre_physics_step(label)
        old = env.joint_angle_targets_rad.clone()
        env._advance_angle_targets(0.02)
        self.assertLessEqual(float((old - env.joint_angle_targets_rad).abs().max()), 0.030001)
        env.common_step_counter += 1
        with torch.inference_mode():
            second = env._get_observations()
        torch.testing.assert_close(first['policy'], repeat['policy'])  # retained rollout tensors must not alias history
        env._reset_idx(torch.tensor([0]))
        third = env._get_observations()
        torch.testing.assert_close(second['policy'][1], third['policy'][1])
        with self.assertRaises(ValueError):
            env._pre_physics_step(torch.full((2, 18), float('nan')))

    def metrics(self, bad=False):
        result = {name: torch.zeros(2) for name in ('fallen', 'out_of_bounds', 'strict_contact_failure',
                  'body_contact', 'non_distal_support_fraction', 'non_distal_collision_fraction', 'Upgraded_reference_fault')}
        result.update(progress_m=torch.full((2,), 0.01), Upgraded_reference_valid=torch.ones(2))
        if bad:
            result['strict_contact_failure'][1] = 1
        return result

    def test_collection_rejects_entire_failed_episode_and_preserves_alignment(self):
        with tempfile.TemporaryDirectory(prefix='Upgraded_data_test_') as folder:
            contract = {'actor_observations': 4}
            collector = LEARNING.UpgradedDemonstrations(Path(folder) / 'Upgraded_data', contract, 2,
                                                        minimum_duration=0.02, minimum_progress=0.02)
            obs = torch.arange(8).reshape(2, 4).float()
            for step in range(3):
                collector.update(obs + step, torch.full((2, 18), step / 10), torch.tensor([step == 2] * 2),
                                 self.metrics(bad=step == 1), 0.02)
            # A second good episode supports episode-level validation.
            for step in range(3):
                collector.update(obs + 10 + step, torch.full((2, 18), -step / 10), torch.tensor([step == 2] * 2),
                                 self.metrics(), 0.02)
            collector.update(obs, torch.zeros(2, 18), torch.zeros(2, dtype=torch.bool), self.metrics(), 0.02)
            manifest = collector.finish()
            self.assertEqual(manifest['accepted_episodes'], 3)
            self.assertEqual(manifest['rejected_reasons']['strict_contact_failure'], 1)
            self.assertEqual(manifest['discarded_incomplete_episodes'], 2)
            _, episodes = LEARNING.Upgraded_load_demonstrations(collector.directory)
            torch.testing.assert_close(episodes[0][0], torch.stack([obs[0] + i for i in range(3)]))
            torch.testing.assert_close(episodes[0][1][:, 0], torch.tensor([0., .1, .2]))
            output = Path(folder) / 'Upgraded_bc.pt'
            report = LEARNING.Upgraded_bc(collector.directory, output, epochs=2, batch_size=4)
            self.assertFalse(set(report['train_episodes']) & set(report['validation_episodes']))
            actor, _ = LEARNING.Upgraded_load_bc(output, contract)
            self.assertTrue(torch.isfinite(actor(obs)).all())
            with self.assertRaises(ValueError):
                LEARNING.Upgraded_load_bc(output, {'actor_observations': 5})
            with self.assertRaises(FileExistsError):
                LEARNING.Upgraded_bc(collector.directory, output, epochs=1)
            shard = collector.directory / manifest['episodes'][0]['file']
            shard.write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                LEARNING.Upgraded_load_demonstrations(collector.directory)

    def test_empty_dataset_and_outside_shard_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='Upgraded_data_test_') as folder:
            data = Path(folder) / 'Upgraded_data'
            collector = LEARNING.UpgradedDemonstrations(data, {'actor_observations': 4}, 2)
            manifest = collector.finish()
            with self.assertRaisesRegex(ValueError, 'two accepted'):
                LEARNING.Upgraded_load_demonstrations(data)
            manifest['episodes'] = [dict(file='../Upgraded_outside.npz', samples=1)] * 2
            LEARNING.Upgraded_json(data / 'Upgraded_manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Invalid or duplicate'):
                LEARNING.Upgraded_load_demonstrations(data)

    def test_atomic_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='Upgraded_atomic_test_') as folder:
            p = Path(folder) / 'Upgraded_file.pt'
            p.write_bytes(b'previous')
            def fail(temporary):
                temporary.write_bytes(b'partial')
                raise OSError('disk failure')
            with self.assertRaises(OSError):
                LEARNING.Upgraded_atomic(p, fail)
            self.assertEqual(p.read_bytes(), b'previous')
            self.assertEqual(list(p.parent.iterdir()), [p])


if __name__ == '__main__':
    unittest.main()
