"""Build a CAD6 MG90S observation and optionally run one ONNX inference.

This program only prints and logs data. It never sends motor commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


OBSERVATIONS = 68
ACTIONS = 18
ACTION_SCALE_RAD = 0.035
ACTION_SMOOTHING = 0.2
# CAD6 link estimate plus the simulated 0.230 kg L1 RM payload.
EXPECTED_SIM_MASS_KG = 2.54362
MASS_TOLERANCE_KG = 0.02


@dataclass(frozen=True)
class PolicyProfile:
    task: str
    min_speed_mps: float
    max_speed_mps: float


# Keep these values aligned with MG90SCad6EnvCfg and MG90SCad6SprintEnvCfg.
PROFILES = {
    "walk": PolicyProfile("Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0", 0.003, 0.006),
    "sprint": PolicyProfile("Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0", 0.012, 0.050),
}


def _profile(name: str) -> PolicyProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"알 수 없는 정책 프로필: {name}") from error


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: 숫자가 필요합니다")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다")
    return number


def _vector(state: dict, key: str, size: int) -> list[float]:
    value = state.get(key)
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{key}: 길이 {size}의 배열이 필요합니다")
    return [_number(item, f"{key}[{index}]") for index, item in enumerate(value)]


def sample_state(profile: str = "walk") -> dict:
    """A stationary, upright virtual sensor reading for format checks."""
    spec = _profile(profile)
    return {
        "root_lin_vel_b": [0.0, 0.0, 0.0],
        "root_ang_vel_b": [0.0, 0.0, 0.0],
        "projected_gravity_b": [0.0, 0.0, -1.0],
        "command": [(spec.min_speed_mps + spec.max_speed_mps) / 2.0, 0.0, 0.0],
        "joint_pos": [0.0, 0.60, -0.20] * 6,
        "default_joint_pos": [0.0, 0.60, -0.20] * 6,
        "joint_vel": [0.0] * ACTIONS,
        "filtered_actions": [0.0] * ACTIONS,
        "phase": 0.0,
    }


def build_observation(state: dict, profile: str = "walk") -> list[float]:
    """Match MG90SWalkEnv._get_observations, including its field order."""
    spec = _profile(profile)
    if not isinstance(state, dict):
        raise ValueError("상태 입력은 JSON 객체여야 합니다")
    lin_vel = _vector(state, "root_lin_vel_b", 3)
    ang_vel = _vector(state, "root_ang_vel_b", 3)
    gravity = _vector(state, "projected_gravity_b", 3)
    command = _vector(state, "command", 3)
    if command[1] != 0.0 or command[2] != 0.0:
        raise ValueError("현재 보행 학습의 command는 [전진속도, 0, 0] 형식입니다")
    if not spec.min_speed_mps <= command[0] <= spec.max_speed_mps:
        raise ValueError(
            f"{profile} 학습 속도 범위는 {spec.min_speed_mps}~{spec.max_speed_mps} m/s입니다"
        )
    joint_pos = _vector(state, "joint_pos", ACTIONS)
    default_joint_pos = _vector(state, "default_joint_pos", ACTIONS)
    joint_vel = _vector(state, "joint_vel", ACTIONS)
    filtered_actions = _vector(state, "filtered_actions", ACTIONS)
    if any(abs(action) > 1.0 for action in filtered_actions):
        raise ValueError("filtered_actions는 학습 환경에서 -1~1 범위입니다")
    phase = _number(state.get("phase"), "phase")
    if not 0.0 <= phase < 1.0:
        raise ValueError("phase: 0 이상 1 미만이어야 합니다")

    observation = (
        lin_vel
        + ang_vel
        + gravity
        + command
        + [position - default for position, default in zip(joint_pos, default_joint_pos)]
        + [velocity * 0.1 for velocity in joint_vel]
        + filtered_actions
        + [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)]
    )
    assert len(observation) == OBSERVATIONS
    return observation


def process_action(raw_action: list[float], previous_filtered: list[float]) -> dict:
    """Match _pre_physics_step's action clamp, smoothing, and residual scale."""
    if len(raw_action) != ACTIONS or len(previous_filtered) != ACTIONS:
        raise ValueError("AI 행동과 이전 평활화 행동은 각각 18개여야 합니다")
    clipped = [max(-1.0, min(1.0, _number(value, f"action_raw[{i}]")))
               for i, value in enumerate(raw_action)]
    previous = [_number(value, f"filtered_actions[{i}]") for i, value in enumerate(previous_filtered)]
    if any(abs(value) > 1.0 for value in previous):
        raise ValueError("filtered_actions는 -1~1 범위여야 합니다")
    filtered = [old + ACTION_SMOOTHING * (new - old)
                for old, new in zip(previous, clipped)]
    return {
        "action_clipped": clipped,
        "filtered_actions_next": filtered,
        "joint_residual_rad": [ACTION_SCALE_RAD * value for value in filtered],
    }


def _metadata_path(model: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (model.parent / "run_metadata.json", model.parent.parent / "run_metadata.json"):
        if candidate.is_file():
            return candidate
    raise ValueError("run_metadata.json이 없습니다. --metadata로 지정하세요")


def _validate_metadata(path: Path, profile: str = "walk") -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata.json은 JSON 객체여야 합니다")
    expected = {"task": _profile(profile).task, "observations": OBSERVATIONS,
                "actions": ACTIONS, "legs": 6, "status": "completed"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"모델 메타데이터 불일치: {key}={metadata.get(key)!r}, 예상값={value!r}")
    mass = _number(metadata.get("mass_kg"), "mass_kg")
    if abs(mass - EXPECTED_SIM_MASS_KG) > MASS_TOLERANCE_KG:
        raise ValueError(
            f"모델 학습 질량 {mass} kg이 현재 L1 RM 포함 설정 {EXPECTED_SIM_MASS_KG} kg과 다릅니다"
        )


def infer_once(model: Path, observation: list[float], metadata: Path | None = None,
               profile: str = "walk") -> list[float]:
    """Load one feedforward ONNX policy and return its 18 raw actions."""
    if model.suffix.lower() != ".onnx" or not model.is_file():
        raise ValueError(f"ONNX 모델 파일이 필요합니다: {model}")
    _validate_metadata(_metadata_path(model, metadata), profile)
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("추론에는 numpy와 onnxruntime이 필요합니다: pip install -r requirements-core.txt") from error

    try:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise RuntimeError(f"ONNX 모델을 열 수 없습니다: {error}") from error
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("입력 1개와 출력 1개인 feedforward 정책만 지원합니다")
    if inputs[0].shape[-2:] != [1, OBSERVATIONS] or outputs[0].shape[-2:] != [1, ACTIONS]:
        raise ValueError(f"모델 차원 불일치: 입력={inputs[0].shape}, 출력={outputs[0].shape}; 예상=[1,68]→[1,18]")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("float32 입력·출력 모델만 지원합니다")
    values = np.asarray([observation], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("관측값을 float32로 변환할 수 없습니다")
    try:
        action = session.run([outputs[0].name], {inputs[0].name: values})[0]
    except Exception as error:
        raise RuntimeError(f"ONNX 추론 실패: {error}") from error
    if action.shape != (1, ACTIONS) or not np.isfinite(action).all():
        raise ValueError("모델 출력의 차원 또는 숫자가 올바르지 않습니다")
    return action[0].astype(float).tolist()


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="walk", help="학습 작업: walk 또는 sprint")
    parser.add_argument("--state", type=Path, help="상태 JSON 파일. 생략하면 가상 상태 사용")
    parser.add_argument("--model", type=Path, help="선택한 6족 MG90S Walk/Sprint ONNX 정책")
    parser.add_argument("--metadata", type=Path, help="학습 폴더의 run_metadata.json")
    parser.add_argument("--log", type=Path, help="결과를 추가할 JSONL 파일")
    args = parser.parse_args(argv)
    if args.metadata and not args.model:
        parser.error("--metadata는 --model과 함께 사용하세요")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": _profile(args.profile).task,
        "profile": args.profile,
        "source": str(args.state) if args.state else "virtual_sample",
        "model": str(args.model) if args.model else None,
    }
    try:
        state = json.loads(args.state.read_text(encoding="utf-8")) if args.state else sample_state(args.profile)
        record["state"] = state
        record["observation"] = build_observation(state, args.profile)
        if args.model:
            record["action_raw"] = infer_once(args.model, record["observation"], args.metadata, args.profile)
            record.update(process_action(record["action_raw"], state["filtered_actions"]))
            record["status"] = "inferred"
        else:
            record["action_raw"] = None
            record["status"] = "observation_only"
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        record["status"] = "error"
        record["error"] = str(error)
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    if args.log:
        try:
            _append_log(args.log, record)
        except OSError as error:
            print(f"로그 저장 실패: {error}", file=sys.stderr)
            return 1
    return 0 if record["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
