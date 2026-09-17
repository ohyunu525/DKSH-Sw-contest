"""Fail before simulation if a policy belongs to a different robot/task."""
import argparse
from pathlib import Path
import torch


def validate_checkpoint(path, observations=68, actions=18, task='Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0'):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = data['model_state_dict']
    layers = [(k, v) for k, v in state.items() if k.startswith('actor.') and k.endswith('weight')]
    actual = (layers[0][1].shape[1], layers[-1][1].shape[0])
    if actual != (observations, actions):
        raise ValueError(f'Incompatible checkpoint: observations/actions={actual}; expected {(observations, actions)}. Do not reuse an eight-leg policy on CAD6.')
    if observations == 68:
        metadata = Path(path).parent / 'run_metadata.json'
        import json
        if not metadata.is_file() or json.loads(metadata.read_text(encoding='utf-8')).get('task') != task:
            raise ValueError(f'CAD6 checkpoint must include matching run_metadata.json for {task}.')
    return int(data['iter'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--observations', type=int, default=68)
    parser.add_argument('--actions', type=int, default=18)
    parser.add_argument('--task', default='Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0')
    args = parser.parse_args()
    print('CHECKPOINT_COMPATIBLE iteration=' + str(validate_checkpoint(args.checkpoint, args.observations, args.actions, args.task)))
