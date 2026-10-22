"""Checkpoint compatibility for absolute-angle policies; no simulator imports."""

import hashlib
import json
import math
from pathlib import Path

import torch

ACTION_SEMANTICS = "absolute_joint_angles_v2"
TERRAIN_OBSERVATION_CONTRACT = "terrain_full48_all_presets_v2"
OBSERVATION_CONTRACT = "joint_angles_targets_target_velocity_terrain48_v2"
REWARD_CONTRACT = "signed_progress_sustained_speed_v2"
CONTACT_CONTRACT = "support_surface_vertical_sole_v2"
CONTROL_CONTRACT = "absolute_targets_rate_limited_observed_velocity_v2"
SCHEMA_CONTRACTS = {
    "terrain_observation_contract": TERRAIN_OBSERVATION_CONTRACT,
    "observation_contract": OBSERVATION_CONTRACT,
    "reward_contract": REWARD_CONTRACT,
    "contact_contract": CONTACT_CONTRACT,
    "control_contract": CONTROL_CONTRACT,
}


def require_v2_task(task):
    if task.endswith("-JointAngles-v1"):
        raise ValueError("JointAngles-v1 checkpoints and tasks are incompatible with v2; retrain a new v2 policy")
    if not task.endswith("-JointAngles-v2"):
        raise ValueError("This runner requires a JointAngles-v2 absolute-angle task; use the legacy runner for gait v0")


def expected_cad_asset_sha256(task):
    if "CAD6" not in task or "MG90S" in task:
        return None
    path = Path(__file__).resolve().parents[1] / "assets/spiderbot_variants/onshape_20260928/robot.training.urdf"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_actuator_contract(task):
    if "MG996R-Hip-CAD6" in task:
        return "mg996r_4v8_rc920dmg_5v_v1"
    if "MG90S" in task:
        return "mg90s_4v8_historical_v1"
    return "mg996r_4v8_v1"


def expected_dimensions(task, environment):
    actions = 18 if "CAD6" in task else 24
    # Every preset observes target velocity and the same complete terrain state.
    return 14 + 4 * actions + actions // 3 + 48, actions


def validate_checkpoint(path, task, environment):
    require_v2_task(task)
    path = Path(path).expanduser().resolve()
    metadata_path = path.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Absolute-angle v2 checkpoints require their run_metadata.json; retrain v1/v0 policies")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = expected_dimensions(task, environment)
    if metadata.get("action_semantics") != ACTION_SEMANTICS:
        raise ValueError("Checkpoint uses v1/legacy action semantics; retrain a new absolute-angle v2 policy")
    if (metadata.get("task") != task
            or (metadata.get("observations"), metadata.get("actions")) != expected
            or metadata.get("actuator_contract") != expected_actuator_contract(task)):
        raise ValueError("Checkpoint robot, servo, terrain observations or action semantics do not match the selected task")
    for name, expected_contract in SCHEMA_CONTRACTS.items():
        if metadata.get(name) != expected_contract:
            raise ValueError(f"Checkpoint {name.replace('_', ' ')} differs from v2; retrain the policy")
    cad_hash = expected_cad_asset_sha256(task)
    if cad_hash is not None and metadata.get("cad_asset_sha256") != cad_hash:
        raise ValueError("Checkpoint CAD asset differs from the current Onshape model; train a new policy")
    for name in ("target_rate_limit_rad_s", "control_frequency_hz", "exploration_std_cap"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Checkpoint is missing a valid v2 {name}")
    smoothing = metadata.get("target_smoothing")
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not 0 < smoothing <= 1:
        raise ValueError("Checkpoint is missing a valid v2 target_smoothing")
    state = torch.load(path, map_location="cpu", weights_only=True)
    layers = sorted(
        ((key, value) for key, value in state["model_state_dict"].items()
         if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda item: int(item[0].split(".")[1]),
    )
    actual = (layers[0][1].shape[1], layers[-1][1].shape[0]) if layers else None
    if actual != expected:
        raise ValueError(f"Checkpoint network dimensions {actual} do not match {expected}")
    model = state["model_state_dict"]
    std = model.get("std")
    if std is None and "log_std" in model:
        std = model["log_std"].exp()
    if (std is None or std.numel() not in (1, expected[1])
            or not torch.isfinite(std).all() or not (std > 0).all()
            or bool((std > metadata["exploration_std_cap"] + 1e-7).any())):
        raise ValueError("Checkpoint exploration std is missing, invalid or exceeds its v2 saturation cap")
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
    raise FileNotFoundError(f"No compatible absolute-angle checkpoint in {log_root}. Train a new v2 policy first")


def validate_control_contract(metadata, joint_names, angle_limits, *, target_rate_limit=None,
                              target_smoothing=None, control_frequency_hz=None, exploration_std_cap=None):
    """A network's tensor sizes alone cannot prove that action meanings match."""
    saved_limits = torch.tensor(metadata.get("angle_limits_rad", []), dtype=angle_limits.dtype, device=angle_limits.device)
    if (metadata.get("joint_names") != joint_names or saved_limits.shape != angle_limits.shape
            or not torch.allclose(saved_limits, angle_limits, atol=1e-6, rtol=0)):
        raise ValueError("Checkpoint joint order or absolute angle ranges differ from the environment")
    for name, expected in (
        ("target_rate_limit_rad_s", target_rate_limit), ("target_smoothing", target_smoothing),
        ("control_frequency_hz", control_frequency_hz), ("exploration_std_cap", exploration_std_cap),
    ):
        if expected is not None:
            saved = metadata.get(name)
            if not isinstance(saved, (int, float)) or not math.isclose(saved, expected, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Checkpoint {name} differs from the environment; retrain with the matching v2 control contract")
