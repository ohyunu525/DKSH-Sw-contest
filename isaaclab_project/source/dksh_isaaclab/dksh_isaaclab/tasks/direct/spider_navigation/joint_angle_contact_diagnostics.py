"""Bounded pre-reset contact evidence; no Isaac Sim imports or policy changes.

Filtered contact matrices have no history in Isaac Lab 2.1. Call ``capture``
after EACH physics substep, before reset, rather than sampling only the last
matrix at the policy step. Forces identify a source/target rigid-body pair;
they do not identify a contact point or a particular collider on a rigid body.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import torch


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _json_value(value, path, nonfinite_fields):
    """Preserve invalid measurement locations while producing strict JSON."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, float):
        if not math.isfinite(value):
            nonfinite_fields.append(path)
            return None
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{path}[{index}]", nonfinite_fields)
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Contact diagnostic object keys must be strings")
        return {key: _json_value(item, f"{path}.{key}", nonfinite_fields)
                for key, item in value.items()}
    raise TypeError(f"Unsupported contact diagnostic value at {path}: {type(value).__name__}")


class ContactDiagnosticWriter:
    """Append a bounded number of complete JSONL snapshots, flushing each batch.

    ``select`` avoids copying thousands of failed environments to the host.
    Pass only its selected records to ``record_batch``. Records excluded by
    either operation increment ``dropped_records``. Limits concern evidence
    volume only and must never change rewards, reset flags or policy actions.
    """

    def __init__(self, path, max_records=64, max_per_step=4, context=None):
        self.path = Path(path).expanduser().resolve()
        self.max_records = _positive_integer(max_records, "max_records")
        self.max_per_step = _positive_integer(max_per_step, "max_per_step")
        self.context = {} if context is None else dict(context)
        self.dropped_records = 0
        self.written_records = 0
        self._needs_newline = False
        # Resume/append without replacing earlier failure evidence. Do not scan
        # an unexpectedly large existing log once its cap is already reached.
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        try:
                            previous = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError("Existing contact diagnostic log has an incomplete JSON record; use a new file") from error
                        if not isinstance(previous, dict):
                            raise ValueError("Existing contact diagnostic records must be JSON objects")
                        self.written_records += 1
                        self._needs_newline = not line.endswith("\n")
                        if self.written_records >= self.max_records:
                            break

    @property
    def remaining_records(self):
        return max(0, self.max_records - self.written_records)

    def select(self, environment_ids):
        if isinstance(environment_ids, torch.Tensor):
            if environment_ids.ndim != 1:
                raise ValueError("environment_ids must be a one-dimensional list of integer indices")
            environment_ids = environment_ids.detach().cpu().tolist()
        values = list(environment_ids)
        if any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in values):
            raise ValueError("environment_ids must contain nonnegative integer indices")
        if len(set(values)) != len(values):
            raise ValueError("environment_ids must contain unique indices")
        values = sorted(map(int, values))
        selected = values[:min(self.max_per_step, self.remaining_records)]
        self.dropped_records += len(values) - len(selected)
        return selected

    def record_batch(self, records):
        records = list(records)
        count = min(len(records), self.max_per_step, self.remaining_records)
        self.dropped_records += len(records) - count
        if count == 0:
            return 0
        lines = []
        for record in records[:count]:
            if not isinstance(record, dict):
                raise TypeError("Each contact diagnostic record must be an object")
            nonfinite_fields = []
            payload = _json_value(record, "record", nonfinite_fields)
            payload["schema_version"] = 1
            payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            payload["context"] = _json_value(self.context, "context", nonfinite_fields)
            if nonfinite_fields:
                payload["nonfinite_fields"] = nonfinite_fields
            lines.append(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        # Validate/serialize the full selected batch before touching the file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            if self._needs_newline:
                stream.write("\n")
            stream.write("\n".join(lines) + "\n")
            stream.flush()
        self.written_records += count
        self._needs_newline = False
        return count


class ContactPairHistory:
    """Retain explicitly mapped filtered normal forces for one policy step.

    Sources must each correspond to ONE rigid body per environment. Target
    paths must match the actual filter columns in precisely the given order;
    ``capture`` rejects a different column count rather than guessing labels.
    This buffer is diagnostic only, separate from the support classifier.
    """

    def __init__(self, num_envs, source_names, target_paths, history_length, device="cpu", unsupported_targets=()):
        self.num_envs = _positive_integer(num_envs, "num_envs")
        self.history_length = _positive_integer(history_length, "history_length")
        self.source_names = tuple(source_names)
        self.target_paths = tuple(target_paths)
        self.unsupported_targets = []
        for target in unsupported_targets:
            if (not isinstance(target, dict) or not isinstance(target.get("path"), str)
                    or not target["path"].startswith("/") or not isinstance(target.get("reason"), str)
                    or not target["reason"]):
                raise ValueError("Unsupported targets must identify a path and reason")
            if target["path"] in self.target_paths:
                raise ValueError("An unsupported target cannot also be an active filter")
            self.unsupported_targets.append(dict(target))
        for name, values in (("source_names", self.source_names), ("target_paths", self.target_paths)):
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain nonempty names")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique names")
        # An environment wildcard denotes one known target in each clone.
        # A body wildcard would merge targets, making an exact pair label false.
        for path in self.target_paths:
            expanded = self._environment_path(path, 0)
            if not expanded.startswith("/") or any(character in expanded for character in ("*", "?", "[", "]")):
                raise ValueError("target_paths must identify one target per environment, without body wildcards")
        self.device = torch.device(device)
        self._forces = None
        self._captured = set()

    @staticmethod
    def _environment_path(path, environment_id):
        return (path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{environment_id}")
                .replace("/env_.*/", f"/env_{environment_id}/")
                .replace("/env_*/", f"/env_{environment_id}/"))

    def begin_step(self):
        self._captured.clear()

    def capture(self, matrices, physics_substep):
        if (isinstance(physics_substep, bool) or not isinstance(physics_substep, Integral)
                or not 0 <= physics_substep < self.history_length):
            raise ValueError("physics_substep must be a zero-based index inside history_length")
        physics_substep = int(physics_substep)
        if physics_substep in self._captured:
            raise ValueError("The same physics substep cannot be captured twice")
        expected = (self.num_envs, len(self.source_names), len(self.target_paths), 3)
        if isinstance(matrices, torch.Tensor):
            forces = matrices
        else:
            matrices = tuple(matrices)
            single_shape = (self.num_envs, 1, len(self.target_paths), 3)
            if len(matrices) != len(self.source_names):
                raise ValueError("One filtered contact matrix is required per source body")
            if any(not isinstance(matrix, torch.Tensor) or matrix.shape != single_shape for matrix in matrices):
                raise ValueError("Each filtered contact matrix must have shape (N,1,T,3) matching target paths")
            forces = torch.cat(matrices, dim=1)
        if forces.shape != expected:
            raise ValueError("Filtered contact matrices must have shape (N,S,T,3) matching source and target mappings")
        if not forces.is_floating_point() or forces.device != self.device:
            raise ValueError("Filtered contact matrices must be floating point on the configured device")
        if self._forces is None:
            self._forces = torch.empty((self.history_length, *expected), dtype=forces.dtype, device=self.device)
        elif forces.dtype != self._forces.dtype:
            raise ValueError("Filtered contact matrix dtype must stay constant")
        self._forces[physics_substep].copy_(forces.detach())
        self._captured.add(physics_substep)

    def snapshot(self, environment_id, force_threshold=0.0):
        if (isinstance(environment_id, bool) or not isinstance(environment_id, Integral)
                or not 0 <= environment_id < self.num_envs):
            raise ValueError("environment_id must identify a configured environment")
        if not math.isfinite(force_threshold) or force_threshold < 0:
            raise ValueError("force_threshold must be finite and nonnegative")
        environment_id = int(environment_id)
        substeps = sorted(self._captured)
        result = {
            "status": "captured" if substeps else "not_captured",
            "complete": len(substeps) == self.history_length,
            "physics_substeps_captured": substeps,
            "expected_physics_substeps": self.history_length,
            "source_names": list(self.source_names),
            "filter_target_paths": [self._environment_path(path, environment_id) for path in self.target_paths],
            "unsupported_filter_targets": [
                {**target, "path": self._environment_path(target["path"], environment_id)}
                for target in self.unsupported_targets
            ],
            # Absence of a known limitation is not a successful physical
            # calibration of every filter. Unknown coverage stays explicit.
            "filter_coverage_complete": False if self.unsupported_targets else None,
            "filter_matching_verified": False,
            # Correct dimensions and configured labels do not establish that
            # PhysX actually matched every filter. Empty matrices alone cannot
            # establish that a body had no contact with an unverified target.
            "empty_pairs_prove_no_contact": False,
            "force_semantics": "normal_contact_force_on_source_in_world_frame",
            "contact_point_available": False,
            "pairs": [],
        }
        if not substeps:
            return result
        forces = self._forces[substeps, environment_id].detach().cpu()
        finite = torch.isfinite(forces).all(-1)
        magnitudes = torch.linalg.vector_norm(forces, dim=-1)
        active = ((magnitudes > force_threshold) | ~finite).any(0)
        for source_id, target_id in active.nonzero(as_tuple=False).tolist():
            samples = forces[:, source_id, target_id]
            valid_samples = finite[:, source_id, target_id]
            valid_magnitudes = magnitudes[:, source_id, target_id][valid_samples]
            source = self.source_names[source_id]
            source_path = source if source.startswith("/") else f"/World/envs/env_.*/Robot/{source}"
            result["pairs"].append({
                "source_name": source,
                "source_path": self._environment_path(source_path, environment_id),
                "target_path": self._environment_path(self.target_paths[target_id], environment_id),
                "peak_force_norm_n": float(valid_magnitudes.max()) if len(valid_magnitudes) else None,
                "has_nonfinite": not bool(valid_samples.all()),
                "samples": [{"physics_substep": substep,
                             "force_w_n": [float(component) if math.isfinite(float(component)) else None
                                           for component in vector]}
                            for substep, vector in zip(substeps, samples)],
            })
        return result
