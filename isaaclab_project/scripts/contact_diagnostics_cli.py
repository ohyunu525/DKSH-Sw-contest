"""Configure bounded pre-reset contact logs without importing Isaac Sim."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import uuid


def _positive_integer(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_contact_diagnostics_args(parser):
    parser.add_argument("--contact_diagnostics", action="store_true",
                        help="Record filtered contact pairs at each physics substep (additional sensors).")
    parser.add_argument("--contact_diagnostics_output", default="",
                        help="JSONL snapshot path; enables basic snapshots even without pair sensors.")
    parser.add_argument("--contact_diagnostics_max_records", type=_positive_integer, default=64)
    parser.add_argument("--contact_diagnostics_per_step", type=_positive_integer, default=4)


def validate_contact_diagnostics_args(args):
    explicit = bool(args.contact_diagnostics or args.contact_diagnostics_output)
    if explicit and not args.task.endswith("-JointAngles-v2"):
        raise ValueError("Contact diagnostics require a JointAngles-v2 direct-angle task")
    if args.contact_diagnostics_output and not args.contact_diagnostics_output.strip():
        raise ValueError("Contact diagnostics output must name a JSONL file")
    return args


def apply_contact_diagnostics_cfg(cfg, args, *, project_root, run_dir=None, mode=None):
    """Training always records basic failures; expensive pair sensors are opt-in."""
    validate_contact_diagnostics_args(args)
    mode = mode or getattr(args, "mode", "check")
    enabled = mode == "train" or args.contact_diagnostics or bool(args.contact_diagnostics_output)
    cfg.contact_diagnostics_pairs = bool(args.contact_diagnostics)
    cfg.contact_diagnostics_max_records = args.contact_diagnostics_max_records
    cfg.contact_diagnostics_per_step = args.contact_diagnostics_per_step
    if not enabled:
        cfg.contact_diagnostics_path = None
        return None
    if args.contact_diagnostics_output:
        path = Path(args.contact_diagnostics_output).expanduser().resolve()
    elif run_dir is not None:
        path = Path(run_dir).resolve() / "contact_failures.jsonl"
    else:
        identity = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + uuid.uuid4().hex[:8]
        path = Path(project_root).resolve() / "logs/contact_diagnostics" / identity / "contact_failures.jsonl"
    if path.exists() and path.is_dir():
        raise ValueError("Contact diagnostics output must name a file, not a directory")
    cfg.contact_diagnostics_path = str(path)
    return path
