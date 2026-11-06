# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import RLforHexapod.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    for task_id in ("Hexapod-Walk", "Hexapod-Stair", "Hexapod-Stair-Student-A"):
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].startswith("RLforHexapod.tasks.")
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith("RLforHexapod.tasks.")
