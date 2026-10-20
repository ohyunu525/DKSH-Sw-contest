"""Encode one obstacle-only Unitree L1 RM point-cloud frame for the actor.

The canonical encoder is shared with the ROS2 ``l1_policy_features`` node so
offline checks and hardware frames cannot silently diverge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PACKAGE_SOURCE = Path(__file__).resolve().parents[2] / "ros2" / "dksh_l1_features"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from dksh_l1_features.encoder import (  # noqa: E402,F401
    L1_MAX_RANGE_M,
    L1_MIN_RANGE_M,
    L1_RESOLUTION_M,
    L1_SECTORS,
    L1_VERTICAL_FOV_DEG,
    encode_l1_obstacles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode an obstacle-only L1 RM frame as 32 policy features")
    parser.add_argument("points_json", help="JSON file containing one L1-frame [x, y, z] point list")
    args = parser.parse_args()
    with open(args.points_json, encoding="utf-8") as source:
        points = json.load(source)
    print(json.dumps(encode_l1_obstacles(points)))


if __name__ == "__main__":
    main()
