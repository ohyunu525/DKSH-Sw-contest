"""Register DKSH tasks and delegate to Isaac Lab's RSL-RL trainer."""

import runpy
import sys
from pathlib import Path

import dksh_isaaclab  # noqa: F401
from environment_cli import configure_runner


workspace_root = Path(__file__).resolve().parents[2]
sys.argv = [sys.argv[0]] + configure_runner(sys.argv[1:], training=True)
runner_dir = workspace_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(runner_dir))
runner = runpy.run_path(str(runner_dir / "train.py"), run_name="dksh_isaaclab_rsl_rl_train")
try:
    runner["main"]()
    # Emit before SimulationApp.close(): Kit may terminate with status 1 during
    # an otherwise clean Isaac Sim 4.5 headless shutdown on Windows.
    print("DKSH_ISAACLAB_TRAIN_COMPLETE", flush=True)
finally:
    runner["simulation_app"].close()
