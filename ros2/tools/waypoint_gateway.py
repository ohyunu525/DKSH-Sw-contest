#!/usr/bin/env python3
"""Run the gateway from a checkout without pip, colcon or a LiDAR."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dksh_remote_navigation"))
from dksh_remote_navigation.server import main

if __name__ == "__main__":
    main()
