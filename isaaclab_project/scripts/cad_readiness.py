"""Validate that the current Onshape design is ready to replace the legacy CAD asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "spiderbot_variants"
    / "cad_source.json"
)


class CadNotReadyError(RuntimeError):
    """Raised when release evidence is incomplete and no override was requested."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("cad_source.json schema_version must be 1")
    for key in ("source", "observed_structure", "target_hardware", "release_evidence"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"cad_source.json is missing object: {key}")
    return data


def collect_blockers(data: dict[str, Any]) -> list[str]:
    source = data["source"]
    evidence = data["release_evidence"]
    step = evidence["step_export"]
    joints = evidence["joint_spec"]
    mass = evidence["mass_properties"]
    servo = evidence["servo_layout"]
    blockers: list[str] = []

    if source.get("lifecycle") != "released_for_simulation":
        blockers.append("Onshape lifecycle is not released_for_simulation")
    if not step.get("path") or not step.get("sha256") or not step.get("parts_preserved"):
        blockers.append("part-preserving STEP export and SHA-256 are missing")
    if not joints.get("path") or not joints.get("axes_and_centers_complete"):
        blockers.append("joint axes and centers are incomplete")
    if not joints.get("zero_pose_complete"):
        blockers.append("joint zero pose is incomplete")
    if not joints.get("limits_complete"):
        blockers.append("joint limits are incomplete")
    if not mass.get("path") or not mass.get("complete") or not mass.get("motor_inclusion_labeled"):
        blockers.append("link mass/COM data with motor inclusion labels are incomplete")
    if not servo.get("verified_in_cad"):
        blockers.append("MG90S/RC920DMG layout is not verified in CAD")
    if not servo.get("performance_source") or not servo.get("operating_point_verified"):
        blockers.append("servo operating-point specs lack a versioned source or bench verification")
    return blockers


def readiness_report(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    data = load_manifest(path)
    blockers = collect_blockers(data)
    return {
        "ready": not blockers,
        "manifest": str(path.resolve()),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_url": data["source"].get("url"),
        "lifecycle": data["source"].get("lifecycle"),
        "blockers": blockers,
    }


def assert_cad_ready(
    path: Path = DEFAULT_MANIFEST, *, allow_provisional: bool = False
) -> dict[str, Any]:
    report = readiness_report(path)
    report["provisional_override"] = bool(allow_provisional and not report["ready"])
    if not report["ready"] and not allow_provisional:
        details = "; ".join(report["blockers"])
        raise CadNotReadyError(
            "Current CAD is not released for training: "
            f"{details}. Use --allow-provisional only for an explicitly provisional experiment."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        report = readiness_report(args.manifest)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        if args.as_json:
            print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"CAD_NOT_READY {exc}")
        return 2
    report["provisional_override"] = bool(args.allow_provisional and not report["ready"])
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        marker = (
            "CAD_READY"
            if report["ready"]
            else "CAD_PROVISIONAL_OVERRIDE"
            if args.allow_provisional
            else "CAD_NOT_READY"
        )
        print(f"{marker} {report['manifest_sha256']}")
    return 0 if report["ready"] or args.allow_provisional else 2


if __name__ == "__main__":
    raise SystemExit(main())
