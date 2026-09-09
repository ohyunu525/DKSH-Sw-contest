"""DKSH spiderbot environments for Isaac Lab."""

# Importing this package only registers Gym environments. Isaac Sim modules are
# deliberately imported lazily after AppLauncher starts the simulator.
from .tasks import *  # noqa: F401, F403
