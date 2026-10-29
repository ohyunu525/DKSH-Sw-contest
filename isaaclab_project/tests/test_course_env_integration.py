"""Exercise the actual base reset with per-environment course tensors.

Only the external robot and scene are doubled; no PhysX simulation is claimed.
"""

import ast
from collections.abc import Sequence
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


FOLDER = Path(__file__).resolve().parents[1] / "source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"


class DirectParent:
    def _reset_idx(self, ids):
        self.episode_length_buf[ids] = 0


def quat_from_euler_xyz(roll, pitch, yaw):
    result = torch.zeros((len(yaw), 4))
    result[:, 0], result[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    return result


source = ast.parse((FOLDER / "spider_navigation_env.py").read_text(encoding="utf-8"))
original = next(node for node in source.body if isinstance(node, ast.ClassDef))
reset = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
definition = ast.ClassDef(name="BaseEnvironment", bases=[ast.Name(id="DirectRLEnv", ctx=ast.Load())],
                          keywords=[], body=[reset], decorator_list=[])
namespace = dict(torch=torch, math=math, Sequence=Sequence, DirectRLEnv=DirectParent,
                 quat_from_euler_xyz=quat_from_euler_xyz)
exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
             "actual_base_reset", "exec"), namespace)
BaseEnvironment = namespace["BaseEnvironment"]


class CourseResetIntegrationTests(unittest.TestCase):
    def test_subset_reset_uses_new_course_spawn_goal_and_height(self):
        env = BaseEnvironment()
        env.num_envs, env.device = 2, "cpu"
        env.cfg = NS(goal_radius=0.08, goal_min_distance=0.5, goal_max_distance=1.5, joint_reset_noise=0)
        env.max_episode_length_s = 240
        env.episode_length_buf = torch.tensor([15, 20])
        env.reset_terminated = torch.tensor([False, True])
        env.reset_time_outs = torch.zeros(2, dtype=torch.bool)
        env._episode_sums = {"progress": torch.zeros(2)}
        env._goal_data = lambda: (torch.ones(2), None, None)
        env._fallen = lambda: torch.zeros(2, dtype=torch.bool)
        env._debris_hit = lambda: torch.zeros(2, dtype=torch.bool)
        env.extras = {}
        env._actions = torch.ones((2, 18))
        env._previous_actions = env._actions.clone()
        env.scene = NS(env_origins=torch.tensor([[0., 0., 0.25], [10., 0., 0.5]]))
        env._target_positions_w = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        env._previous_goal_distance = torch.ones(2)
        default_root = torch.zeros((2, 13))
        default_root[:, 2], default_root[:, 3] = 0.1375, 1.0
        writes = {}
        env._robot = NS(
            _ALL_INDICES=torch.arange(2), reset=lambda ids: None,
            data=NS(default_joint_pos=torch.zeros((2, 18)), default_joint_vel=torch.zeros((2, 18)),
                    default_root_state=default_root),
            write_root_pose_to_sim=lambda pose, ids: writes.update(pose=pose.clone(), ids=ids.clone()),
            write_root_velocity_to_sim=lambda velocity, ids: writes.update(velocity=velocity.clone()),
            write_joint_state_to_sim=lambda *args: None,
        )
        # The fallback representative layout is deliberately wrong. Each reset
        # must read the requested environment's freshly generated course instead.
        scenario = NS(layout=NS(spawn=(99, 99, 99), goal=(99, 99, 99)),
                      spawn_positions=torch.zeros((2, 3)), goal_positions=torch.zeros((2, 3)),
                      floor_height=torch.zeros(2), floor_velocity=torch.zeros((2, 3)))

        def replace_course(ids):
            scenario.spawn_positions[ids] = torch.tensor([-1.6, 0.05, 0.0])
            scenario.goal_positions[ids] = torch.tensor([1.6, -0.03, 0.08])

        scenario.reset = replace_course
        env._scenario = scenario
        untouched_target = env._target_positions_w[0].clone()
        env._reset_idx(torch.tensor([1]))
        torch.testing.assert_close(writes["pose"][0, :3], torch.tensor([8.4, 0.05, 0.6375]))
        torch.testing.assert_close(env._target_positions_w[1], torch.tensor([11.6, -0.03, 0.595]))
        torch.testing.assert_close(env._target_positions_w[0], untouched_target)
        torch.testing.assert_close(env._previous_goal_distance[1], torch.tensor((3.2 ** 2 + 0.08 ** 2) ** 0.5))
        torch.testing.assert_close(env._actions[0], torch.ones(18))
        self.assertEqual(env.episode_length_buf.tolist(), [15, 0])


if __name__ == "__main__":
    unittest.main()
