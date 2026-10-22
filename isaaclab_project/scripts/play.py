"""Register DKSH tasks and delegate to Isaac Lab's RSL-RL player."""

import runpy
import sys
import argparse
from pathlib import Path

import dksh_isaaclab  # noqa: F401
from environment_cli import configure_runner


selector = argparse.ArgumentParser(add_help=False)
selector.add_argument("--task", default="Isaac-DKSH-Spider-JointAngles-v2")
selected, _ = selector.parse_known_args()
if selected.task.endswith("-JointAngles-v1"):
    selector.error("JointAngles-v1 is retired because the control/reward/observation contract changed; retrain a v2 policy")
if selected.task.endswith("-JointAngles-v2"):
    sys.argv = [sys.argv[0], "--mode=play", f"--task={selected.task}"] + sys.argv[1:]
    runpy.run_path(str(Path(__file__).with_name("joint_angle_runner.py")), run_name="__main__")
else:
    workspace_root = Path(__file__).resolve().parents[2]
    sys.argv = [sys.argv[0]] + configure_runner(sys.argv[1:], training=False)
    runner_dir = workspace_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"
    sys.path.insert(0, str(runner_dir))
    runpy.run_path(str(runner_dir / "play.py"), run_name="__main__")
