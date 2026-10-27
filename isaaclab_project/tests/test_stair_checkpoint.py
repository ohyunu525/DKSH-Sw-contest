"""Prevent resuming a flat/privileged policy or a mismatched sensor contract."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/mg90s_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('stair_checkpoint_guard', PATH)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
TASK = 'Isaac-DKSH-MG90S-CAD6-Stairs-Sensors-Direct-v1'


class StairCheckpointTests(unittest.TestCase):
    def test_scenario_geometry_and_strategy_must_match(self):
        record = {'stair_geometry': {'tread_m': 0.15, 'width_m': 1.2}, 'training_strategy': 'staged_v1'}
        GUARD.validate_stair_scenario(record, tread_m=0.15, width_m=1.2, strategy='staged_v1')
        for tread, width, strategy in ((0.25, 1.2, 'staged_v1'),
                                       (0.15, 0.15, 'staged_v1'), (0.15, 1.2, 'legacy_v1')):
            with self.assertRaisesRegex(ValueError, 'scenario'):
                GUARD.validate_stair_scenario(record, tread_m=tread, width_m=width, strategy=strategy)
        GUARD.validate_stair_scenario({}, tread_m=0.25, width_m=1.2, strategy='legacy_v1')

    def test_sensor_contract_and_task_required_even_when_shapes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'iter': 3, 'model_state_dict': {'actor.0.weight': torch.zeros(16, 244),
                                                       'actor.2.weight': torch.zeros(18, 16)}}, path)
            metadata = path.parent / 'run_metadata.json'
            with self.assertRaises(ValueError):
                GUARD.validate_checkpoint(path, 244, 18, TASK)
            for record in ({'task': 'Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0'},
                           {'task': TASK, 'actuator_contract': 'mg996r_4v8_rc920dmg_5v_v1',
                            'observation_contract': 'simulation_only_privileged_state_v1'}):
                metadata.write_text(json.dumps(record), encoding='utf-8')
                with self.assertRaises(ValueError):
                    GUARD.validate_checkpoint(path, 244, 18, TASK)
            metadata.write_text(json.dumps({'task': TASK,
                                           'actuator_contract': 'mg996r_4v8_rc920dmg_5v_v1',
                                           'observation_contract': 'l2_bno086_buttons_absolute_targets_v2',
                                           'action_semantics': 'absolute_joint_angles_v1'}), encoding='utf-8')
            self.assertEqual(GUARD.validate_checkpoint(path, 244, 18, TASK), 3)
            for semantics in (None, 'residual_joint_angles_v1'):
                metadata.write_text(json.dumps({'task': TASK,
                                               'actuator_contract': 'mg996r_4v8_rc920dmg_5v_v1',
                                               'observation_contract': 'l2_bno086_buttons_absolute_targets_v2',
                                               'action_semantics': semantics}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'action semantics'):
                    GUARD.validate_checkpoint(path, 244, 18, TASK)
            for old_contract in (None, 'mg90s_4v8_rc920dmg_5v_v1'):
                metadata.write_text(json.dumps({'task': TASK, 'actuator_contract': old_contract,
                                               'observation_contract': 'assumed_l2_bno086_command_history_v1'}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'actuator contract'):
                    GUARD.validate_checkpoint(path, 244, 18, TASK)

    def test_flat_policy_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'iter': 3, 'model_state_dict': {'actor.0.weight': torch.zeros(16, 68),
                                                       'actor.2.weight': torch.zeros(18, 16)}}, path)
            with self.assertRaisesRegex(ValueError, 'Incompatible checkpoint'):
                GUARD.validate_checkpoint(path, 244, 18, TASK)

    def test_previous_240_dimension_residual_policy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'iter': 3, 'model_state_dict': {'actor.0.weight': torch.zeros(16, 240),
                                                       'actor.2.weight': torch.zeros(18, 16)}}, path)
            with self.assertRaisesRegex(ValueError, 'Incompatible checkpoint'):
                GUARD.validate_checkpoint(path, 244, 18, TASK)


if __name__ == '__main__':
    unittest.main()
