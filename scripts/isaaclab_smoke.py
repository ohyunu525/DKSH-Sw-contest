"""Run a finite, headless Isaac Lab smoke test.

Run this through IsaacLab\\isaaclab.bat so it uses Isaac Sim's bundled
Python interpreter and environment.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a finite Isaac Lab smoke test.")
parser.add_argument("--steps", type=int, default=64, help="Number of physics steps to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    """Create an empty scene and advance physics for a bounded number of steps."""
    sim = SimulationContext(SimulationCfg(dt=0.01, device=args_cli.device))
    sim.reset()

    executed_steps = 0
    while simulation_app.is_running() and executed_steps < args_cli.steps:
        sim.step()
        executed_steps += 1

    if executed_steps != args_cli.steps:
        raise RuntimeError(f"Simulation stopped after {executed_steps} of {args_cli.steps} requested steps.")

    print(f"ISAAC_LAB_SMOKE_TEST_PASS: steps={executed_steps}, device={args_cli.device}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
