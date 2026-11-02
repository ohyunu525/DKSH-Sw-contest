"""Read a training run's metadata and TensorBoard scalars without Isaac Sim.

The report describes logged training snapshots, not a policy evaluation. It
never estimates total episodes by adding averaged TensorBoard counters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs/rsl_rl/dksh_mg996r_hip_cad6_jointangles_v2"


def latest_run(log_root):
    candidates = [path.parent for path in Path(log_root).glob("*/run_metadata.json")]
    if not candidates:
        raise FileNotFoundError(f"No run_metadata.json below {log_root}")
    # Run directories have sortable creation timestamps. File modification time
    # can reflect a copied snapshot and does not identify the newest training.
    return max(candidates, key=lambda path: path.name)


def read_scalar_events(run_directory):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required; run this with the training Python environment") from error
    accumulator = EventAccumulator(str(run_directory), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags().get("scalars", [])}


def summarize_run(run_directory, metadata, scalars, *, window=100, expected_profile=None):
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    warnings = []
    metrics = {}
    selected_steps = {}
    last_wall_time = None
    for tag, events in sorted(scalars.items()):
        # Resume files can repeat an iteration; the newest written sample wins.
        by_step = {}
        for event in events:
            if not math.isfinite(event.value):
                warnings.append({"code": "nonfinite_scalar", "tag": tag, "step": int(event.step)})
                continue
            previous = by_step.get(int(event.step))
            if previous is None or event.wall_time >= previous.wall_time:
                by_step[int(event.step)] = event
        selected = sorted(by_step.values(), key=lambda event: event.step)[-window:]
        if not selected:
            continue
        values = [float(event.value) for event in selected]
        metrics[tag] = {
            "samples": len(selected), "first_step": int(selected[0].step),
            "last_step": int(selected[-1].step), "last": values[-1],
            "mean": sum(values) / len(values), "min": min(values), "max": max(values),
        }
        selected_steps[tag] = tuple(int(event.step) for event in selected)
        newest_time = max(event.wall_time for event in selected)
        last_wall_time = newest_time if last_wall_time is None else max(last_wall_time, newest_time)
    profile = metadata.get("behavior_profile")
    if expected_profile is not None and profile != expected_profile:
        warnings.append({"code": "profile_mismatch", "expected": expected_profile, "actual": profile})
    if not metrics:
        warnings.append({"code": "no_scalar_events"})
    target = metadata.get("target_speed_mps")
    speed_tag = next((tag for tag in ("Metrics/step_progress_speed_mps", "Metrics/forward_speed_mps")
                      if tag in metrics), None)
    speed_fraction = None
    if speed_tag is not None and isinstance(target, (int, float)) and math.isfinite(target) and target > 0:
        speed = metrics[speed_tag]
        speed_fraction = speed["mean"] / target
        if speed["samples"] >= min(window, 20) and speed["last_step"] >= 100 and abs(speed_fraction) < 0.05:
            warnings.append({"code": "near_stationary", "speed_tag": speed_tag,
                             "note": "Low logged training speed; this does not identify its cause."})
    def mean(tag):
        return metrics[tag]["mean"] if tag in metrics else None
    locomotion_tags = ("Metrics/progress_reward_per_s", "Metrics/speed_reward_per_s")
    slip_tag = "Metrics/stance_slip_reward_per_s"
    # Compare only same-unit metrics from matching iteration windows. Episode
    # reward summaries have a different aggregation and cannot be substituted.
    ratio = None
    if all(tag in metrics for tag in (*locomotion_tags, slip_tag)):
        aligned = all(selected_steps[tag] == selected_steps[slip_tag] for tag in locomotion_tags)
        if aligned:
            gain = sum(mean(tag) for tag in locomotion_tags)
            if gain > 1e-9:
                ratio = max(0.0, -mean(slip_tag)) / gain
                if ratio >= 1:
                    warnings.append({"code": "slip_cost_exceeds_locomotion_reward",
                                     "note": "Inspect stance geometry and contact-frame slip samples."})
        else:
            warnings.append({"code": "unaligned_reward_windows"})
    # RSL-RL also emits Train/*/time tags with elapsed SECONDS as the event
    # step. Mixing those with iteration-axis tags fabricates training progress.
    iteration_metrics = {tag: value for tag, value in metrics.items()
                         if not tag.endswith("/time")}
    latest_iteration = max((value["last_step"] for value in iteration_metrics.values()), default=None)
    return {
        "schema": "dksh_joint_angle_training_diagnosis_v1",
        "run_directory": str(Path(run_directory).resolve()),
        "scope": "training_scalar_snapshots_only_not_policy_evaluation",
        "behavior_profile": profile, "metadata_status": metadata.get("status"),
        "reward_contract": metadata.get("reward_contract"),
        "contact_contract": metadata.get("contact_contract"),
        "stance_slip_measurement_contract": metadata.get("stance_slip_measurement_contract"),
        "control_frequency_hz": metadata.get("control_frequency_hz"),
        "target_speed_mps": target, "episode_length_s": metadata.get("episode_length_s"),
        "completed_episodes_at_metadata_write": metadata.get("completed_episodes"),
        "latest_logged_step": latest_iteration,
        "latest_logged_training_iteration": latest_iteration,
        "last_scalar_write_utc": (datetime.fromtimestamp(last_wall_time, timezone.utc).isoformat()
                                  if last_wall_time is not None else None),
        "window_samples_per_tag": window, "speed_tag": speed_tag,
        "logged_speed_fraction_of_target": speed_fraction,
        "slip_cost_to_locomotion_reward_ratio": ratio,
        "metrics": metrics, "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Exact run directory; defaults to newest local timestamp directory")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--window", type=int, default=100, help="Last N logged samples per tag")
    parser.add_argument("--expected-profile", help="Warn if the selected run used another profile")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    try:
        run = args.run.resolve() if args.run is not None else latest_run(args.log_root)
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        report = summarize_run(run, metadata, read_scalar_events(run), window=args.window,
                               expected_profile=args.expected_profile)
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Training diagnosis failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
