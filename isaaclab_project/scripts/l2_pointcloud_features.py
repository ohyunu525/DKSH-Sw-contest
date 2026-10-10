"""Encode one obstacle-only Unitree L2 frame using the shared ROS2 encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PACKAGE_SOURCE = Path(__file__).resolve().parents[2] / "ros2" / "dksh_l1_features"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from dksh_l1_features.encoder import (  # noqa: E402,F401
    L2_MAX_RANGE_M, L2_MIN_RANGE_M, L2_RESOLUTION_M, L2_SECTORS,
    L2_VERTICAL_FOV_DEG, L2_VERTICAL_MIN_DEG, encode_l2_obstacles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode an obstacle-only L2 NEGA frame as 32 policy features")
    parser.add_argument("points_json", help="JSON file containing one L2-frame [x, y, z] point list")
    args = parser.parse_args()
    with open(args.points_json, encoding="utf-8") as source:
        points = json.load(source)
    print(json.dumps(encode_l2_obstacles(points)))


if __name__ == "__main__":
    main()
