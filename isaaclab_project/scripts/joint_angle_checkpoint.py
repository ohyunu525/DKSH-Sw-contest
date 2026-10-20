"""Checkpoint compatibility for absolute-angle policies; no simulator imports."""

import json
from pathlib import Path

import torch

ACTION_SEMANTICS = "absolute_joint_angles_v1"


def expected_actuator_contract(task):
    if "MG996R-Hip-CAD6" in task:
        return "mg996r_4v8_rc920dmg_5v_v1"
    if "MG90S" in task:
        return "mg90s_4v8_historical_v1"
    return "mg996r_4v8_v1"


def expected_dimensions(task, environment):
    actions = 18 if "CAD6" in task else 24
    return 14 + 3 * actions + actions // 3 + (32 if environment != "flat" else 0), actions


def validate_checkpoint(path, task, environment):
    path = Path(path).expanduser().resolve()
    metadata_path = path.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Absolute-angle checkpoints require their run_metadata.json; v0 residual policies cannot be reused")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = expected_dimensions(task, environment)
    actuator_contract = metadata.get("actuator_contract")
    # Historical all-MG90S v1 runs predate actuator-contract metadata but retain
    # the same MG90S model. New or changed MG996R contracts must be explicit.
    legacy_mg90s_metadata = actuator_contract is None and "MG90S" in task
    if (metadata.get("action_semantics") != ACTION_SEMANTICS or metadata.get("task") != task
            or (metadata.get("observations"), metadata.get("actions")) != expected
            or (actuator_contract != expected_actuator_contract(task) and not legacy_mg90s_metadata)):
        raise ValueError("Checkpoint robot, servo, terrain observations or action semantics do not match the selected task")
    state = torch.load(path, map_location="cpu", weights_only=True)
    layers = sorted(
        ((key, value) for key, value in state["model_state_dict"].items()
         if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda item: int(item[0].split(".")[1]),
    )
    actual = (layers[0][1].shape[1], layers[-1][1].shape[0]) if layers else None
    if actual != expected:
        raise ValueError(f"Checkpoint network dimensions {actual} do not match {expected}")
    return metadata


def resolve_checkpoint(checkpoint, log_root, task, environment):
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        validate_checkpoint(path, task, environment)
        return path
    for path in sorted(Path(log_root).glob("**/model_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            validate_checkpoint(path, task, environment)
            return path
        except (ValueError, KeyError):
            continue
    raise FileNotFoundError(f"No compatible absolute-angle checkpoint in {log_root}. Train a new v1 policy first")


def validate_control_contract(metadata, joint_names, angle_limits):
    """A network's tensor sizes alone cannot prove that action meanings match."""
    saved_limits = torch.tensor(metadata.get("angle_limits_rad", []), dtype=angle_limits.dtype, device=angle_limits.device)
    if (metadata.get("joint_names") != joint_names or saved_limits.shape != angle_limits.shape
            or not torch.allclose(saved_limits, angle_limits, atol=1e-6, rtol=0)):
        raise ValueError("Checkpoint joint order or absolute angle ranges differ from the environment")
