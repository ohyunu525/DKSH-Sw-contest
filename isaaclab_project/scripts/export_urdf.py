"""Export the deterministic DKSH spiderbot URDF without starting Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


parser = argparse.ArgumentParser(description="Generate the DKSH spiderbot URDF source asset.")
parser.add_argument("--output", required=True, help="Destination .urdf path")
args = parser.parse_args()

generator_path = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dksh_isaaclab"
    / "dksh_isaaclab"
    / "assets"
    / "urdf_generator.py"
)
spec = importlib.util.spec_from_file_location("dksh_urdf_generator", generator_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load URDF generator: {generator_path}")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

output_path = Path(args.output).expanduser().resolve()
if output_path.suffix.lower() != ".urdf":
    raise ValueError(f"Output must use the .urdf extension: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(generator.build_spiderbot_urdf(), encoding="utf-8")
print(f"DKSH_URDF_READY path={output_path}", flush=True)
