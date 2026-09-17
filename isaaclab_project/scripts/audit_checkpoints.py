"""Inspect local policy dimensions without starting Isaac Sim."""
from pathlib import Path
import collections
import json
import torch


def main():
    root = Path(__file__).resolve().parents[2]
    records = []
    for folder in (root / 'logs', root / 'isaaclab_project'):
        for path in sorted(folder.rglob('*.pt')):
            record = {'path': str(path.relative_to(root)).replace('\\', '/')}
            try:
                if path.name == 'policy.pt':
                    state = torch.jit.load(str(path), map_location='cpu').state_dict()
                    prefix = 'actor.' if any(k.startswith('actor.') for k in state) else 'actor'
                else:
                    data = torch.load(path, map_location='cpu', weights_only=True)
                    state = data['model_state_dict']
                    record['iteration'] = data.get('iter')
                    prefix = 'actor.'
                layers = [(k, v) for k, v in state.items() if k.startswith(prefix) and k.endswith('weight')]
                if not layers:
                    # Exported Isaac policies name the sequential module "actor".
                    layers = [(k, v) for k, v in state.items() if v.ndim == 2 and k.endswith('weight')]
                record['observations'] = layers[0][1].shape[1]
                record['actions'] = layers[-1][1].shape[0]
                record['six_leg_candidate'] = record['actions'] == 18
            except Exception as exc:
                record['error'] = str(exc)
            records.append(record)
    output = root / 'logs' / 'checkpoint_audit.json'
    output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    counts = collections.Counter((r.get('observations'), r.get('actions')) for r in records)
    print(json.dumps({'files': len(records), 'dimensions': {str(k): v for k, v in counts.items()},
                      'six_leg_candidates': [r for r in records if r.get('six_leg_candidate')],
                      'errors': [r for r in records if 'error' in r], 'report': str(output)}, indent=2))


if __name__ == '__main__':
    main()
