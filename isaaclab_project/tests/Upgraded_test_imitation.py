"""Imitation alignment, rejection, episode splits and CAD reference regressions."""

import ast
from copy import deepcopy
import json
import math
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
ORIGINAL = FIXTURE.load_helper('Upgraded_original_gait')


def Upgraded_load_gait_adapter():
    path = FIXTURE.FOLDER / 'Upgraded_gait_reference.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = dict(torch=torch, math=math, F=torch.nn.functional,
                     UpgradedOriginalGait=ORIGINAL.UpgradedOriginalGait,
                     bounded_angle_targets=FIXTURE.CONTROL.bounded_angle_targets,
                     normalize_angles=FIXTURE.CONTROL.normalize_angles)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['UpgradedGaitReference']


GAIT_REFERENCE = Upgraded_load_gait_adapter()


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
                     UpgradedGaitReference=GAIT_REFERENCE)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    base = FIXTURE.StrictContactEnvironmentTests().strict_env(behavior_profile='revised_v12')
    settings = vars(base.cfg).copy()
    settings.update(history_frames=4, decimation=4, reference_settings={}, ik_clearance=0.008,
                    target_speed=0.003, imu_angular_noise=0.01, imu_gravity_noise=0.005,
                    toe_button_offsets=CAD.FOOT.ONSHAPE_TOE_BUTTON_OFFSETS,
                    non_distal_collision_offsets=CAD.COLLISION.ONSHAPE_NON_DISTAL_COLLISION_OFFSETS,
                    proximal_foot_offsets=CAD.FOOT.ONSHAPE_PROXIMAL_FOOT_OFFSETS,
                    robot=SimpleNamespace(copy=lambda: deepcopy(base.cfg.robot)))
    configuration = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                         and n.name == 'UpgradedImitationEnvCfg')
    for assignment in configuration.body:
        if not isinstance(assignment, ast.Assign):
            continue
        try:
            value = assignment.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict':
                parsed = {item.arg: ast.literal_eval(item.value) for item in value.keywords}
            else:
                parsed = ast.literal_eval(value)
            settings[assignment.targets[0].id] = parsed
        except (ValueError, TypeError, AttributeError):
            pass
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

    def test_supplement_resolves_the_original_gait_sources(self):
        report = RUNNER.Upgraded_inspect_archive(RUNNER.UPGRADED_ROOT / 'HexapodRobot.zip',
                                                RUNNER.UPGRADED_ROOT / 'HexapodRobot2.zip')
        self.assertEqual(report['missing_local_modules'], [])
        self.assertEqual(len(report['members']), 7)
        self.assertEqual(len(report['archives']), 2)
        self.assertEqual(len({entry['sha256'] for entry in report['archives']}), 2)
        self.assertIn('Gait', report['members']['hexapod_joy_walk.py']['classes'])
        self.assertTrue(report['reviewed_gait_source_matches'])
        self.assertEqual(report['reviewed_gait_port']['source_sha256'],
                         ORIGINAL.UPGRADED_GAIT_SOURCE_SHA256)
        self.assertFalse(report['original_code_executed'])
        self.assertNotIn('hexapod_joy_walk', sys.modules)
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

    def original_reference(self, count=3):
        kinematics, neutral, home = Upgraded_home(count)
        neutral, valid, _ = kinematics.inverse(home, neutral)
        self.assertTrue(valid.all())
        limits = kinematics.limits.expand(count, -1, -1, -1)
        reference = GAIT_REFERENCE(home, kinematics, neutral, limits)
        return reference, kinematics, neutral, limits

    def test_original_tripod_tracks_cad_with_the_actual_target_filter(self):
        reference, kinematics, applied, limits = self.original_reference()
        command = applied.new_tensor([[0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.10]])
        gravity = applied.new_tensor([[0, 0, -1]]).expand(3, -1)
        saw_probe = torch.zeros(3, dtype=torch.bool)
        saw_swing = torch.zeros(3, 6, dtype=torch.bool)
        saw_slowdown = False
        for step in range(400):
            contacts = kinematics.forward(applied)[0][..., 2] <= reference.home[..., 2] + 0.0003
            previous = reference.angles.clone()
            previous_phase = reference.core.phase.clone()
            was_probing = (reference.stage == reference.PROBE) | (reference.stage == reference.RECENTER)
            q = reference.advance(contacts, command, gravity, 0.02, applied)
            self.assertFalse(reference.failed.any(), f'step={step}, faults={reference.fault.tolist()}')
            self.assertLessEqual(float((q - previous).abs().max()), 0.030001)
            torch.testing.assert_close(reference.core.phase[was_probing], previous_phase[was_probing])
            old_applied = applied
            applied = FIXTURE.CONTROL.bounded_angle_targets(
                FIXTURE.CONTROL.normalize_angles(q, limits), applied, limits,
                dt=0.02, rate_limit=1.5, smoothing=0.35)
            self.assertLessEqual(float((applied - old_applied).abs().max()), 0.030001)
            self.assertTrue(torch.isfinite(applied).all())
            self.assertTrue((reference.core.in_swing.sum(-1) <= 3).all())
            saw_probe |= reference.stage == reference.PROBE
            saw_swing |= reference.core.in_swing
            saw_slowdown |= bool((reference.scale < 1).any())
        self.assertTrue(saw_probe.all())
        self.assertTrue(saw_swing.all())
        self.assertTrue(saw_slowdown)

    def test_original_reference_faults_instead_of_ignoring_contact_feedback(self):
        reference, _, applied, _ = self.original_reference(count=2)
        command = applied.new_tensor([[0.015, 0, 0]]).expand(2, -1)
        gravity = applied.new_tensor([[0, 0, -1]]).expand(2, -1)
        contacts = torch.ones(2, 6, dtype=torch.bool)
        for _ in range(180):
            q = reference.advance(contacts, command, gravity, 0.02, applied)
            applied = FIXTURE.CONTROL.bounded_angle_targets(
                FIXTURE.CONTROL.normalize_angles(q, reference.limits), applied, reference.limits,
                dt=0.02, rate_limit=1.5, smoothing=0.35)
        self.assertTrue((reference.fault == 2).all(), reference.fault.tolist())
        reference.reset(torch.tensor([0]))
        held = reference.features()[1].clone()
        for _ in range(310):
            reference.advance(torch.zeros_like(contacts), command, gravity, 0.02, applied,
                              mask=torch.tensor([True, False]))
        self.assertEqual(int(reference.fault[0]), 1)
        torch.testing.assert_close(reference.features()[1], held)

    def test_original_probe_lowers_one_foot_and_keeps_phase_frozen(self):
        reference, _, applied, _ = self.original_reference(count=2)
        reference.stage[:] = reference.PROBE
        reference.pending[:, 0::2] = True
        reference.released[:, 0::2] = True
        contacts = torch.ones(2, 6, dtype=torch.bool)
        contacts[:, 0::2] = False
        before = reference.position.clone()
        phase = reference.core.phase.clone()
        command = applied.new_tensor([[0.015, 0, 0]]).expand(2, -1)
        gravity = applied.new_tensor([[0, 0, -1]]).expand(2, -1)
        reference.advance(contacts, command, gravity, 0.02, applied)
        moved = (reference.position[..., 2] - before[..., 2]).abs() > 1e-10
        self.assertTrue((moved.sum(-1) == 1).all())
        self.assertTrue(moved[:, 0].all())
        torch.testing.assert_close(reference.core.phase, phase)
        reference.core.position[..., 2] = reference.probe_origin - reference.settings['probe_depth'] - 0.001
        reference.advance(contacts, command, gravity, 0.02, applied)
        self.assertTrue((reference.fault == 5).all())

    def test_original_recenter_removes_repeated_height_and_tilt_offsets(self):
        reference, kinematics, applied, limits = self.original_reference(count=2)
        contacts = torch.ones(2, 6, dtype=torch.bool)
        command = applied.new_zeros((2, 3))
        gravity = applied.new_tensor([[0, 0, -1]]).expand(2, -1)
        untouched = reference.features()[1].clone()
        for _ in range(3):
            # Model the commanded plane after compliant stance/landing probes.
            reference.position[0, :, 2] += 0.003 + 0.004 * reference.home[0, :, 0]
            reference.core.ground[0] = reference.position[0, :, 2]
            q, valid, _ = kinematics.inverse(reference.position, reference.angles, limits)
            self.assertTrue(valid.all())
            reference.angles[0] = q[0]
            applied[0] = q[0]
            reference.stage[0] = reference.RECENTER
            reference.elapsed[0] = 0
            phase = reference.core.phase.clone()
            for step in range(100):
                previous = reference.angles.clone()
                q = reference.advance(contacts, command, gravity, 0.02, applied,
                                      mask=torch.tensor([True, False]))
                self.assertFalse(reference.failed.any())
                self.assertLessEqual(float((q - previous).abs().max()), 0.030001)
                applied = FIXTURE.CONTROL.bounded_angle_targets(
                    FIXTURE.CONTROL.normalize_angles(q, limits), applied, limits,
                    dt=0.02, rate_limit=1.5, smoothing=0.35)
                torch.testing.assert_close(reference.core.phase, phase)
                if reference.stage[0] == reference.WALK:
                    break
            self.assertLess(step, 99)
            self.assertLessEqual(float(reference._plane_shift()[0].abs().max()),
                                 reference.settings['recenter_deadband'] + 1e-9)
            torch.testing.assert_close(reference.core.ground[0], reference.position[0, :, 2])
            torch.testing.assert_close(reference.features()[1], untouched)

    def test_observation_timing_privilege_boundary_and_reset(self):
        env = Upgraded_fixture()
        first = env._get_observations()
        self.assertEqual(first['policy'].shape, (2, env.frame_size * env.cfg.history_frames))
        self.assertEqual(first['critic'].shape, (2, env.cfg.observation_space + 140))
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

    def test_bc_actor_transfer_preserves_predictions_and_rejects_activation_drift(self):
        source = LEARNING.Upgraded_actor(11)
        target = LEARNING.Upgraded_actor(11)
        observations = torch.randn(7, 11)
        with torch.no_grad():
            source[-1].weight.fill_(0.02)
        error = LEARNING.Upgraded_transfer_bc_actor(source, target, observations)
        self.assertLessEqual(error, 1e-6)
        torch.testing.assert_close(source(observations), target(observations))
        target[1] = torch.nn.Tanh()
        with self.assertRaises((ValueError, RuntimeError)):
            LEARNING.Upgraded_transfer_bc_actor(source, target, observations)


if __name__ == '__main__':
    unittest.main()
