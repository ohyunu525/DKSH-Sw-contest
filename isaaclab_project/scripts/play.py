"""Register DKSH tasks and delegate to Isaac Lab's RSL-RL player."""

import runpy
import sys
from pathlib import Path

import dksh_isaaclab  # noqa: F401


workspace_root = Path(__file__).resolve().parents[2]
runner_dir = workspace_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(runner_dir))
runpy.run_path(str(runner_dir / "play.py"), run_name="__main__")
