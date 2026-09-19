"""Direct Isaac Lab environment for DKSH spiderbot goal navigation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse

from .spider_navigation_env_cfg import SpiderNavigationEnvCfg
from .scenario_layout import validate_environment
from .scenario_runtime import ScenarioRuntime


class SpiderNavigationEnv(DirectRLEnv):
    """Train a 24-DoF spiderbot to walk toward randomized planar goals."""

    cfg: SpiderNavigationEnvCfg

    def __init__(self, cfg: SpiderNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        validate_environment(cfg.environment_preset, cfg.environment_difficulty)
        scenario_observations = cfg.lidar_observation_bins + 16
        cfg.observation_space = 12 + 3 * cfg.action_space + (
            scenario_observations if cfg.environment_preset != "flat" else 0
        )
        self._scenario = None
        self._viewer_visuals = None
        super().__init__(cfg, render_mode, **kwargs)
        if self._scenario is not None:
            self._scenario.initialize()
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros((self.num_envs, action_dim), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._target_positions_w = self.scene.env_origins.clone()
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in (
                "progress",
                "velocity_to_goal",
                "heading",
                "upright",
                "action_rate",
                "torque",
                "vertical_velocity",
                "stillness",
                "goal",
                "failure",
                "debris_impact",
            )
        }
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                color=(0.16, 0.18, 0.20),
                physics_material=self.cfg.sim.physics_material,
            ),
        )
        if self.cfg.environment_preset != "flat":
            self._scenario = ScenarioRuntime(self)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._scenario is not None:
            self._scenario.begin_step()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._processed_actions = self._robot.data.default_joint_pos + self.cfg.action_scale * self._actions

    def _apply_action(self) -> None:
        if self._scenario is not None:
            self._scenario.physics_step()
        self._robot.set_joint_position_target(self._processed_actions)

    def _goal_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset_w = self._target_positions_w - self._robot.data.root_pos_w
        offset_w[:, 2] = 0.0
        distance = torch.linalg.vector_norm(offset_w[:, :2], dim=1)
        direction_w = offset_w / distance.clamp_min(1.0e-6).unsqueeze(1)
        direction_b = quat_rotate_inverse(self._robot.data.root_quat_w, direction_w)
        return distance, direction_w, direction_b

    def _sync_cad_visual_for_gui(self) -> None:
        """Show a visual-only CAD copy at the GPU articulation's body poses.

        Isaac Sim on this host keeps the referenced CAD USD hierarchy at its
        source pose while Direct GPU PhysX moves the articulation.  Editing that
        physics hierarchy is forbidden by PhysX, so the viewport receives a
        separate, mesh-only copy instead.  It is created only for an interactive
        GUI and has no collision or physics schemas.
        """
        if not self.sim.has_gui():
            return
        if self._viewer_visuals is None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            display_root = "/World/envs/env_0/RobotDisplay"
            stage.DefinePrim(display_root, "Xform")
            asset_path = (
                Path(__file__).resolve().parents[6]
                / "assets"
                / "spiderbot_variants"
                / "spiderbot_6leg"
                / "visuals.usdc"
            )
            mesh_for_body = {
                "base": "base",
                "hip": "hip",
                "femur": "femur",
                "tibia": "tibia",
            }
            self._viewer_visuals = []
            for index, body_name in enumerate(self._robot.body_names):
                mesh_name = next(kind for kind in mesh_for_body if body_name == "base" or body_name.endswith(kind))
                prim = stage.DefinePrim(f"{display_root}/{body_name}", "Mesh")
                prim.GetReferences().AddReference(str(asset_path), f"/{mesh_for_body[mesh_name]}")
                xformable = UsdGeom.Xformable(prim)
                self._viewer_visuals.append((index, xformable.MakeMatrixXform()))

        from pxr import Gf

        body_positions = self._robot.data.body_pos_w[0].detach().cpu().tolist()
        body_quaternions = self._robot.data.body_quat_w[0].detach().cpu().tolist()
        for index, transform_op in self._viewer_visuals:
            position = body_positions[index]
            quaternion = body_quaternions[index]
            transform = Gf.Matrix4d(1.0)
            transform.SetRotate(Gf.Quatd(quaternion[0], Gf.Vec3d(quaternion[1], quaternion[2], quaternion[3])))
            transform.SetTranslateOnly(Gf.Vec3d(*position))
            transform_op.Set(transform)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._sync_cad_visual_for_gui()
        distance, _, direction_b = self._goal_data()
        observation = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                direction_b[:, :2],
                (distance / self.cfg.goal_max_distance).unsqueeze(1),
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel * self.cfg.joint_velocity_scale,
                self._actions,
            ),
            dim=-1,
        )
        if self._scenario is not None:
            observation = torch.cat((observation, self._scenario.observations()), dim=-1)
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        distance, direction_w, direction_b = self._goal_data()
        progress = (self._previous_goal_distance - distance).clamp(-0.25, 0.25)
        velocity_to_goal = torch.sum(self._robot.data.root_lin_vel_w[:, :2] * direction_w[:, :2], dim=1)
        velocity_to_goal = velocity_to_goal.clamp(-0.50, 0.75)
        heading = direction_b[:, 0].clamp(-1.0, 1.0)
        upright = (-self._robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        torque_cost = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        vertical_velocity = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        reached_goal = distance < self.cfg.goal_radius
        fallen = self._fallen()
        reward_terms = {
            "progress": self.cfg.progress_reward_scale * progress,
            "velocity_to_goal": self.cfg.velocity_to_goal_reward_scale * velocity_to_goal * self.step_dt,
            "heading": self.cfg.heading_reward_scale * heading * self.step_dt,
            "upright": self.cfg.upright_reward_scale * upright * self.step_dt,
            "action_rate": self.cfg.action_rate_penalty_scale * action_rate * self.step_dt,
            "torque": self.cfg.torque_penalty_scale * torque_cost * self.step_dt,
            "vertical_velocity": self.cfg.vertical_velocity_penalty_scale * vertical_velocity * self.step_dt,
            "stillness": self.cfg.stillness_penalty_scale * (velocity_to_goal < 0.03).float() * self.step_dt,
            "goal": self.cfg.goal_reward * reached_goal.float(),
            "failure": self.cfg.failure_penalty * fallen.float(),
            "debris_impact": self.cfg.debris_impact_penalty * self._debris_hit().float(),
        }
        reward = torch.sum(torch.stack(tuple(reward_terms.values())), dim=0)
        for name, value in reward_terms.items():
            self._episode_sums[name] += value
        self._previous_goal_distance = distance
        self._previous_actions.copy_(self._actions)
        return reward

    def _fallen(self) -> torch.Tensor:
        ground = (self._scenario.ground_height(self._robot.data.root_pos_w)
                  if self._scenario is not None else self.scene.env_origins[:, 2])
        too_low = self._robot.data.root_pos_w[:, 2] - ground < self.cfg.minimum_base_height
        too_tilted = self._robot.data.projected_gravity_b[:, 2] > -0.35
        return too_low | too_tilted

    def _debris_hit(self) -> torch.Tensor:
        if self._scenario is not None:
            return self._scenario.impact
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._scenario is not None:
            self._scenario.collect_impacts()
        distance, _, _ = self._goal_data()
        reached_goal = distance < self.cfg.goal_radius
        local_position = self._robot.data.root_pos_w - self.scene.env_origins
        out_of_bounds = torch.linalg.vector_norm(local_position[:, :2], dim=1) > self.cfg.max_distance_from_origin
        if self._scenario is not None:
            out_of_bounds = (local_position[:, 0].abs() > 2.5) | (
                local_position[:, 1].abs() > self._scenario.layout.corridor_width / 2
            )
        terminated = reached_goal | self._fallen() | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        distance, _, _ = self._goal_data()
        success = (distance[env_ids] < self.cfg.goal_radius) & self.reset_terminated[env_ids]
        fallen = self._fallen()[env_ids] & self.reset_terminated[env_ids]
        log = {
            f"Episode_Reward/{name}": torch.mean(values[env_ids]).item() / self.max_episode_length_s
            for name, values in self._episode_sums.items()
        }
        log.update(
            {
                "Episode_Termination/goal": torch.count_nonzero(success).item(),
                "Episode_Termination/fallen": torch.count_nonzero(fallen).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
                "Metrics/final_goal_distance": torch.mean(distance[env_ids]).item(),
                "Metrics/debris_impact": torch.count_nonzero(self._debris_hit()[env_ids]).item(),
            }
        )
        self.extras["log"] = log
        for values in self._episode_sums.values():
            values[env_ids] = 0.0

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs and self._scenario is None:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=self.max_episode_length)

        count = len(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if self._scenario is not None:
            self._scenario.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += 0.03 * (2.0 * torch.rand_like(joint_pos) - 1.0)
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        yaw = 2.0 * math.pi * torch.rand(count, device=self.device) - math.pi
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0

        goal_angle = 2.0 * math.pi * torch.rand(count, device=self.device)
        goal_distance = self.cfg.goal_min_distance + (
            self.cfg.goal_max_distance - self.cfg.goal_min_distance
        ) * torch.rand(count, device=self.device)
        self._target_positions_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + goal_distance * torch.cos(goal_angle)
        self._target_positions_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + goal_distance * torch.sin(goal_angle)
        self._target_positions_w[env_ids, 2] = 0.05
        self._previous_goal_distance[env_ids] = goal_distance

        if self._scenario is not None:
            layout = self._scenario.layout
            # Face through the course. Large random yaw would place spread legs
            # through a narrow wall before the policy can take its first action.
            yaw = (torch.rand(count, device=self.device) - 0.5) * 0.10
            root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
            root_state[:, 0] += layout.spawn[0]
            root_state[:, 1] += layout.spawn[1]
            root_state[:, 2] += self._scenario.floor_height[env_ids]
            root_state[:, 7:10] = self._scenario.floor_velocity[env_ids]
            self._target_positions_w[env_ids] = self.scene.env_origins[env_ids] + torch.tensor(
                layout.goal, device=self.device
            )
            self._target_positions_w[env_ids, 2] += 0.015
            self._previous_goal_distance[env_ids] = torch.linalg.vector_norm(
                self._target_positions_w[env_ids, :2] - root_state[:, :2], dim=1
            )

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/DKSH/goal",
                    markers={
                        "goal": sim_utils.CylinderCfg(
                            radius=self.cfg.goal_radius,
                            height=0.012,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.10, 0.90, 0.25), opacity=0.45
                            ),
                        )
                    },
                )
                self._goal_visualizer = VisualizationMarkers(marker_cfg)
            self._goal_visualizer.set_visibility(True)
        elif hasattr(self, "_goal_visualizer"):
            self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        self._goal_visualizer.visualize(translations=self._target_positions_w)
