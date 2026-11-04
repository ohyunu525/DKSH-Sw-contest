"""CPU-compatible demonstration storage and behavior cloning; no simulator imports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn


UPGRADED_HIDDEN = (128, 128, 64)
UPGRADED_SCHEMA = 'Upgraded_demonstrations_v1'


def Upgraded_path(path):
    path = Path(path).expanduser().resolve()
    if not path.name.startswith('Upgraded'):
        raise ValueError('Generated file/directory names must start with Upgraded')
    return path


def Upgraded_atomic(path, writer):
    path = Upgraded_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix='Upgraded_pending_', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def Upgraded_json(path, data):
    Upgraded_atomic(path, lambda p: p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'))


def Upgraded_actor(observations):
    layers, width = [], observations
    for hidden in UPGRADED_HIDDEN:
        layers += [nn.Linear(width, hidden), nn.ELU()]
        width = hidden
    return nn.Sequential(*layers, nn.Linear(width, 18))


def Upgraded_transfer_bc_actor(source_actor, target_actor, observations):
    """Transfer only the deterministic mean; verify BC and PPO action parity.

    Strict tensor shapes alone cannot detect a different activation function.
    Probe both the current observations and bounded synthetic observations before
    allowing PPO updates. The critic and exploration distribution stay separate.
    """
    if observations.ndim != 2 or not torch.isfinite(observations).all():
        raise ValueError('BC transfer requires finite batched actor observations')
    target_actor.load_state_dict(source_actor.state_dict(), strict=True)
    width = observations.shape[1]
    probe = torch.cat((observations.detach()[:32], torch.zeros_like(observations[:1]),
                       torch.linspace(-1, 1, width, device=observations.device,
                                      dtype=observations.dtype)[None]), dim=0)
    with torch.inference_mode():
        expected, actual = source_actor(probe), target_actor(probe)
        if (expected.shape != (len(probe), 18) or actual.shape != expected.shape
                or not torch.isfinite(expected).all() or not torch.isfinite(actual).all()
                or not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)):
            raise ValueError('BC to PPO transfer changed actor outputs; verify architecture and observation scaling')
        return float((expected - actual).abs().max())


class UpgradedDemonstrations:
    """Only complete, progressing, strict-contact-clean episodes become labels.

    Observations are BEFORE the action; validation is split by entire episodes.
    Terminal metrics must be sampled before the simulator's automatic reset.
    """

    def __init__(self, directory, contract, num_envs, *, minimum_progress=0.02,
                 minimum_duration=10.0, maximum_episode_steps=4000):
        self.directory = Upgraded_path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.contract, self.num_envs = contract, num_envs
        self.minimum_progress, self.minimum_duration = minimum_progress, minimum_duration
        self.maximum_episode_steps = maximum_episode_steps
        self.pending = [[] for _ in range(num_envs)]
        self.progress = np.zeros(num_envs)
        self.duration = np.zeros(num_envs)
        self.bad = [set() for _ in range(num_envs)]
        self.records = []
        self.rejections = Counter()
        self.episode_number = np.zeros(num_envs, dtype=int)

    def update(self, observations, actions, dones, metrics, dt):
        obs = torch.as_tensor(observations).detach().cpu().numpy().copy()
        act = torch.as_tensor(actions).detach().cpu().numpy().copy()
        done = torch.as_tensor(dones).detach().cpu().numpy().astype(bool)
        if (obs.shape != (self.num_envs, self.contract['actor_observations'])
                or act.shape != (self.num_envs, 18) or done.shape != (self.num_envs,)):
            raise ValueError('Invalid demonstration batch dimensions')
        if not np.isfinite(obs).all() or not np.isfinite(act).all() or np.abs(act).max() > 1.00001:
            raise ValueError('Invalid observation/action in demonstration batch')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid demonstration interval')
        required = ('progress_m', 'fallen', 'out_of_bounds', 'strict_contact_failure',
                    'body_contact', 'non_distal_support_fraction', 'non_distal_collision_fraction',
                    'Upgraded_reference_valid', 'Upgraded_reference_fault')
        values = {}
        for name in required:
            v = torch.as_tensor(metrics[name]).detach().cpu().numpy()
            if v.shape != (self.num_envs,) or not np.isfinite(v).all():
                raise ValueError(f'Invalid pre-reset metric {name}')
            values[name] = v
        for i in range(self.num_envs):
            self.progress[i] += float(values['progress_m'][i])
            self.duration[i] += dt
            for name in required[1:7]:
                if values[name][i] > 0:
                    self.bad[i].add(name)
            if values['Upgraded_reference_valid'][i] != 1:
                self.bad[i].add('reference_invalid')
            if values['Upgraded_reference_fault'][i] != 0:
                self.bad[i].add(f"reference_fault_{int(values['Upgraded_reference_fault'][i])}")
            if len(self.pending[i]) >= self.maximum_episode_steps:
                self.bad[i].add('episode_buffer_limit')
            elif not self.bad[i]:
                self.pending[i].append((obs[i], act[i]))
            if done[i]:
                if self.progress[i] < self.minimum_progress:
                    self.bad[i].add('insufficient_progress')
                if self.duration[i] < self.minimum_duration:
                    self.bad[i].add('too_short')
                if not self.bad[i] and self.pending[i]:
                    name = f'Upgraded_episode_{len(self.records):06d}.npz'
                    path = self.directory / name
                    xs, ys = zip(*self.pending[i])
                    with path.open('wb') as stream:
                        np.savez_compressed(stream, observations=np.asarray(xs, dtype=np.float32),
                                            actions=np.asarray(ys, dtype=np.float32))
                    self.records.append(dict(file=name, samples=len(xs), environment=i,
                                             episode=int(self.episode_number[i]), duration_s=float(self.duration[i]),
                                             progress_m=float(self.progress[i]), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                else:
                    self.rejections.update(self.bad[i])
                self.episode_number[i] += 1
                self.pending[i].clear()
                self.bad[i].clear()
                self.progress[i] = self.duration[i] = 0

    def finish(self):
        manifest = dict(schema=UPGRADED_SCHEMA, Upgraded_contract=self.contract, episodes=self.records,
                        accepted_episodes=len(self.records), rejected_reasons=dict(self.rejections),
                        discarded_incomplete_episodes=int((self.duration > 0).sum()),
                        acceptance=dict(minimum_progress_m=self.minimum_progress, minimum_duration_s=self.minimum_duration,
                                        strict_contact_failures=0, reference_faults=0))
        Upgraded_json(self.directory / 'Upgraded_manifest.json', manifest)
        return manifest


def Upgraded_load_demonstrations(directory, maximum_samples=200000):
    directory = Path(directory).resolve()
    manifest_path = directory / 'Upgraded_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != UPGRADED_SCHEMA:
        raise ValueError('Unsupported imitation dataset schema')
    episodes = manifest['episodes']
    if len(episodes) < 2:
        raise ValueError('BC needs at least two accepted complete episodes for a separate validation episode')
    if sum(e['samples'] for e in episodes) > maximum_samples:
        raise ValueError('Dataset exceeds --maximum_samples; use a smaller collection or explicitly increase the limit')
    loaded = []
    seen = set()
    width = manifest['Upgraded_contract']['actor_observations']
    for record in episodes:
        path = (directory / record['file']).resolve()
        if path.parent != directory or not path.name.startswith('Upgraded') or path in seen:
            raise ValueError('Invalid or duplicate demonstration shard')
        seen.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Demonstration checksum mismatch: {path.name}')
        with np.load(path, allow_pickle=False) as shard:
            x, y = shard['observations'], shard['actions']
            if (x.shape != (record['samples'], width) or y.shape != (len(x), 18)
                    or not np.isfinite(x).all() or not np.isfinite(y).all() or np.abs(y).max() > 1.00001):
                raise ValueError(f'Invalid demonstration arrays: {path.name}')
            loaded.append((torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()))
    return manifest, loaded


def Upgraded_bc(directory, output, *, epochs=100, batch_size=512, seed=42, device='cpu',
                learning_rate=3e-4, maximum_samples=200000):
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid BC optimizer settings')
    output = Upgraded_path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing checkpoint: {output}')
    manifest, episodes = Upgraded_load_demonstrations(directory, maximum_samples)
    torch.manual_seed(seed)
    # Keep adjacent frames of an episode in the same split to prevent leakage.
    order = torch.randperm(len(episodes)).tolist()
    nval = max(1, round(0.2 * len(order)))
    validation_ids, train_ids = order[:nval], order[nval:]
    x, y = (torch.cat([episodes[i][j] for i in train_ids]) for j in range(2))
    vx, vy = (torch.cat([episodes[i][j] for i in validation_ids]) for j in range(2))
    actor = Upgraded_actor(x.shape[1]).to(device)
    # Start near average demonstrated stance instead of arbitrary joint extremes.
    with torch.no_grad():
        actor[-1].weight.mul_(0.01)
        actor[-1].bias.copy_(y.mean(0).to(device))
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    best, history = math.inf, []
    for epoch in range(epochs):
        actor.train()
        total = 0.0
        for ids in torch.randperm(len(x)).split(batch_size):
            loss = (actor(x[ids].to(device)) - y[ids].to(device)).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite BC loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(ids)
        actor.eval()
        with torch.inference_mode():
            error = sum((actor(a.to(device)) - b.to(device)).square().sum().item()
                        for a, b in zip(vx.split(batch_size), vy.split(batch_size)))
        validation = error / vy.numel()
        history.append(dict(epoch=epoch + 1, train_mse=total / len(x), validation_mse=validation))
        if validation < best:
            best = validation
            state = dict(kind='Upgraded_BC_v1', actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                         hidden_dims=list(UPGRADED_HIDDEN), Upgraded_contract=manifest['Upgraded_contract'],
                         epoch=epoch + 1, validation_mse=validation, seed=seed,
                         train_episodes=train_ids, validation_episodes=validation_ids,
                         dataset_manifest_sha256=hashlib.sha256((Path(directory) / 'Upgraded_manifest.json').read_bytes()).hexdigest())
            Upgraded_atomic(output, lambda p: torch.save(state, p))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'Upgraded BC epoch={epoch + 1} train_mse={total / len(x):.6g} validation_mse={validation:.6g}', flush=True)
    report = dict(best_validation_mse=best, train_samples=len(x), validation_samples=len(vx),
                  train_episodes=train_ids, validation_episodes=validation_ids, history=history,
                  physics_evaluated=False, checkpoint=str(output))
    Upgraded_json(output.with_suffix('.json'), report)
    return report


def Upgraded_load_bc(path, contract, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('kind') != 'Upgraded_BC_v1' or state.get('Upgraded_contract') != contract
            or state.get('hidden_dims') != list(UPGRADED_HIDDEN)):
        raise ValueError('BC checkpoint action/observation/CAD/controller contract mismatch')
    actor = Upgraded_actor(contract['actor_observations']).to(device)
    actor.load_state_dict(state['actor'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.parameters()):
        raise ValueError('BC checkpoint has non-finite weights')
    return actor.eval(), state
