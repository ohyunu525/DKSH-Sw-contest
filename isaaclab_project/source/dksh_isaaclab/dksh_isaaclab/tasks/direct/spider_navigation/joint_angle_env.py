"""Learn locomotion by selecting absolute motor angles, including on stairs."""

from __future__ import annotations

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_rotate

from .spider_navigation_env import SpiderNavigationEnv
from .joint_angle_control import (
    action_to_angles, bounded_angle_targets, joint_names, motion_costs,
    normalize_angles, observation_size, point_velocity, signed_progress, speed_tracking_cost,
    uncapped_signed_progress, huber_speed_tracking_cost,
)
from .joint_angle_contact import (
    sole_contact_state, foot_contact_state, closest_foot_support, toe_button_contact_state,
)


class JointAngleNavigationEnv(SpiderNavigationEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.robot = cfg.robot.copy()
        cfg.robot.spawn.activate_contact_sensors = True
        cfg.robot.spawn.articulation_props.enabled_self_collisions = cfg.enable_self_collisions
        super().__init__(cfg, render_mode, **kwargs)
        # Parent retains its v0 observation contract. Set the v2 space before
        # DirectRLEnv configures Gym spaces, via _configure_gym_env_spaces below.
        self._joint_ids = [self._robot.joint_names.index(name) for name in joint_names(cfg.action_space // 3)]
        limits = (self._robot.data.joint_pos_limits if cfg.use_asset_joint_limits
                  else self._robot.data.soft_joint_pos_limits)
        asset_limits = limits[:, self._joint_ids].clone()
        working = torch.tensor(
            [cfg.joint_angle_ranges[part] for _ in range(cfg.action_space // 3)
             for part in ("hip", "femur", "tibia")], device=self.device,
        )
        self._angle_limits = torch.stack((
            torch.maximum(asset_limits[..., 0], working[:, 0]),
            torch.minimum(asset_limits[..., 1], working[:, 1]),
        ), dim=-1)
        if not torch.all(self._angle_limits[..., 0] < self._angle_limits[..., 1]):
            raise ValueError("Working angle ranges do not overlap the asset's joint limits")
        # Constrain actual physics too: contacts must not bend knees through the
        # opposite IK branch while commanded targets remain inside their limits.
        physical_limits = self._robot.data.joint_pos_limits.clone()
        physical_limits[:, self._joint_ids] = self._angle_limits
        self._robot.write_joint_position_limit_to_sim(physical_limits)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self._target_velocity = torch.zeros_like(self._actions)
        self._previous_target_velocity = torch.zeros_like(self._actions)
        self._goal_hold_time = torch.zeros(self.num_envs, device=self.device)
        self._rate_saturation = torch.zeros_like(self._goal_hold_time)
        self._previous_root_xy = self._robot.data.root_pos_w[:, :2].clone()
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._feet = [self._robot.body_names.index(f"leg_{leg}_tibia") for leg in range(cfg.action_space // 3)]
        # In this URDF each tibia link frame is located at the femur/tibia hinge.
        self._knees = [self._robot.body_names.index(f"leg_{leg}_tibia") for leg in range(cfg.action_space // 3)]
        self._approach_steps = torch.zeros(self.num_envs, device=self.device)
        self._approach_low_knee_steps = torch.zeros(self.num_envs, device=self.device)
        self._sensor_feet = [self._contacts.body_names.index(self._robot.body_names[i]) for i in self._feet]
        self._sensor_body = [i for i, name in enumerate(self._contacts.body_names) if not name.endswith("_tibia")]
        self._sole_offset = torch.tensor(cfg.sole_offset, device=self.device).expand(self.num_envs, len(self._feet), 3)
        support_offsets = cfg.foot_support_offsets
        if support_offsets is None:
            self._foot_support_offsets = self._sole_offset.unsqueeze(2)
        else:
            self._foot_support_offsets = torch.tensor(support_offsets, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        button_offsets = cfg.toe_button_offsets
        self._toe_button_offsets = (None if button_offsets is None else
                                    torch.tensor(button_offsets, device=self.device).unsqueeze(0)
                                    .expand(self.num_envs, -1, -1))
        if self._toe_button_offsets is not None and self._toe_button_offsets.shape != (self.num_envs, len(self._feet), 3):
            raise ValueError("toe_button_offsets must provide one 3-D CAD point per foot")
        self._toe_button_pressed = torch.zeros(
            (self.num_envs, len(self._feet)), dtype=torch.bool, device=self.device
        )
        self._effort_limits = self._robot.data.joint_effort_limits[:, self._joint_ids]
        self._servo_velocity_limits = torch.as_tensor(
            cfg.servo_velocity_limit, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if self._servo_velocity_limits.shape[-1] not in (1, cfg.action_space):
            raise ValueError("servo_velocity_limit must be scalar or provide one value per action")
        self.completed = {name: 0 for name in ("episodes", "goal", "fallen", "time_out", "out_of_bounds")}
        self._episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "progress", "speed", "upright", "joint_limit", "joint_velocity", "torque", "power",
            "target_acceleration", "body_contact", "foot_slip", "angular_velocity", "stillness",
            "goal", "failure", "debris_impact", "knee_clearance",
        )}
        self.step_metrics = {}

    @property
    def joint_angle_targets_rad(self):
        """Applied absolute targets in public leg-major order, ready for 50 Hz commands."""
        return self._processed_actions[:, self._joint_ids]

    def _configure_gym_env_spaces(self):
        self.cfg.observation_space = observation_size(self.cfg.action_space, self.cfg.environment_preset)
        self.cfg.state_space = 0
        super()._configure_gym_env_spaces()

    def _setup_scene(self):
        super()._setup_scene()
        # All link colliders have contact reporting enabled by the spawn config.
        self._contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0,
            history_length=self.cfg.decimation,
        ))
        self.scene.sensors["robot_contacts"] = self._contacts

    def _pre_physics_step(self, actions):
        if self._scenario is not None:
            self._scenario.begin_step()
        self.extras["log"] = {}
        self._previous_actions.copy_(self._actions)
        previous = self._processed_actions[:, self._joint_ids]
        self._actions.copy_(torch.where(torch.isfinite(actions), actions, normalize_angles(previous, self._angle_limits)).clamp(-1, 1))
        requested_step = self.cfg.target_smoothing * (action_to_angles(self._actions, self._angle_limits) - previous)
        self._rate_saturation = (requested_step.abs() > self.cfg.target_rate_limit * self.step_dt).float().mean(-1)
        targets = bounded_angle_targets(
            self._actions, previous, self._angle_limits, dt=self.step_dt,
            rate_limit=self.cfg.target_rate_limit, smoothing=self.cfg.target_smoothing,
        )
        self._previous_target_velocity.copy_(self._target_velocity)
        self._target_velocity.copy_((targets - previous) / self.step_dt)
        self._processed_actions[:, self._joint_ids] = targets

    def _contact_state(self):
        if self.cfg.behavior_profile in ("revised_v4", "revised_v5", "revised_v6", "revised_v7"):
            if self._toe_button_offsets is None or self._foot_support_offsets is None:
                raise ValueError("Toe-button profiles require CAD toe-button and foot-support geometry")
            points, surfaces = self._foot_support_geometry()
            button_points = self._toe_button_world_positions()
            button_surfaces = self._surface_heights(button_points)
            pressed, body = toe_button_contact_state(
                self._contacts.data.net_forces_w_history,
                sensor_feet=self._sensor_feet, sensor_body=self._sensor_body,
                foot_positions_w=points, foot_surface_heights=surfaces,
                button_positions_w=button_points, button_surface_heights=button_surfaces,
                previous_pressed=self._toe_button_pressed,
                force_threshold=self.cfg.contact_threshold,
                press_force_threshold=self.cfg.toe_button_press_force_threshold,
                release_force_threshold=self.cfg.toe_button_release_force_threshold,
                foot_height_tolerance=self.cfg.foot_contact_height_tolerance,
                button_height_tolerance=self.cfg.toe_button_height_tolerance,
            )
            self._toe_button_pressed.copy_(pressed)
            return self._toe_button_pressed, body
        if self.cfg.behavior_profile != "legacy_v2" and self.cfg.foot_support_offsets is not None:
            points, surfaces = self._foot_support_geometry()
            return foot_contact_state(
                self._contacts.data.net_forces_w_history,
                sensor_feet=self._sensor_feet, sensor_body=self._sensor_body,
                foot_positions_w=points, surface_heights=surfaces,
                force_threshold=self.cfg.contact_threshold,
                height_tolerance=self.cfg.foot_contact_height_tolerance,
            )
        offset = self._sole_world_offset()
        soles = self._robot.data.body_link_pos_w[:, self._feet] + offset
        surface = (self.scene.env_origins[:, 2:3].expand(-1, len(self._feet))
                   if self._scenario is None else self._scenario.surface_heights(soles))
        return sole_contact_state(
            self._contacts.data.net_forces_w_history,
            sensor_feet=self._sensor_feet, sensor_body=self._sensor_body,
            sole_positions_w=soles, surface_heights=surface,
            force_threshold=self.cfg.contact_threshold,
            height_tolerance=self.cfg.sole_contact_height_tolerance,
        )

    def _foot_support_geometry(self):
        offsets = self._foot_support_offsets
        rotations = self._robot.data.body_link_quat_w[:, self._feet, None, :].expand(-1, -1, offsets.shape[2], -1)
        rotated = quat_rotate(rotations.reshape(-1, 4), offsets.reshape(-1, 3)).reshape_as(offsets)
        points = self._robot.data.body_link_pos_w[:, self._feet, None, :] + rotated
        surfaces = (self.scene.env_origins[:, None, 2:3].expand(points.shape[:-1])
                    if self._scenario is None else self._scenario.surface_heights(points.flatten(1, 2)).reshape(points.shape[:-1]))
        return points, surfaces

    def _toe_button_world_positions(self):
        rotations = self._robot.data.body_link_quat_w[:, self._feet]
        offsets = self._toe_button_offsets
        rotated = quat_rotate(rotations.reshape(-1, 4), offsets.reshape(-1, 3)).reshape_as(offsets)
        return self._robot.data.body_link_pos_w[:, self._feet] + rotated

    def _surface_heights(self, points_w):
        if self._scenario is None:
            return self.scene.env_origins[:, 2:3].expand(points_w.shape[:-1])
        return self._scenario.surface_heights(points_w)

    def _sole_world_offset(self):
        return quat_rotate(
            self._robot.data.body_link_quat_w[:, self._feet].reshape(-1, 4),
            self._sole_offset.reshape(-1, 3),
        ).reshape_as(self._sole_offset)

    def _ground_height(self):
        if self._scenario is None:
            return self.scene.env_origins[:, 2]
        return self._scenario.ground_height(self._robot.data.root_pos_w)

    def _get_observations(self):
        # Keep the CAD viewport synchronization already used by the v0 player.
        if "spiderbot_variants" in self.cfg.robot.spawn.usd_path:
            self._sync_cad_visual_for_gui()
        data = self._robot.data
        distance, _, direction_b = self._goal_data()
        feet, _ = self._contact_state()
        clearance = data.root_pos_w[:, 2] - self._ground_height()
        goal_surface = (self._target_positions_w[:, 2] - 0.015 if self._scenario is not None
                        else self.scene.env_origins[:, 2])
        observation = torch.cat((
            data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
            direction_b[:, :2], (distance / self.cfg.goal_max_distance).unsqueeze(-1),
            normalize_angles(data.joint_pos[:, self._joint_ids], self._angle_limits),
            data.joint_vel[:, self._joint_ids] / self._servo_velocity_limits,
            normalize_angles(self._processed_actions[:, self._joint_ids], self._angle_limits),
            self._target_velocity / self.cfg.target_rate_limit,
            feet.float(), (clearance / self.cfg.robot_height).unsqueeze(-1),
            ((goal_surface - data.root_pos_w[:, 2]) / self.cfg.robot_height).unsqueeze(-1),
        ), dim=-1)
        if self._scenario is not None:
            terrain = self._scenario.observations()
        else:
            # Same schema on flat and stairs, permitting explicit flat-to-terrain
            # transfer. Height samples are simulated terrain information.
            terrain = torch.zeros((self.num_envs, 48), device=self.device)
            terrain[:, :16] = 1.0
            terrain[:, 32:41] = ((self.scene.env_origins[:, 2:3] - data.root_pos_w[:, 2:3]) / 0.5).clamp(-1, 1)
        observation = torch.cat((observation, terrain), dim=-1)
        return {"policy": observation}

    def _get_dones(self):
        if self._scenario is not None:
            self._scenario.collect_impacts()
        distance, _, _ = self._goal_data()
        feet, body_contact = self._contact_state()
        goal_surface = (self._target_positions_w[:, 2] - 0.015 if self._scenario is not None
                        else self.scene.env_origins[:, 2])
        stable_goal = (
            (distance < self.cfg.goal_radius)
            & (self._robot.data.root_pos_w[:, 2] >= goal_surface + self.cfg.minimum_base_height)
            & (self._robot.data.projected_gravity_b[:, 2] < -0.80)
            & (feet.sum(-1) >= self.cfg.minimum_support_feet) & ~body_contact & ~self._fallen()
        )
        self._goal_hold_time = torch.where(stable_goal, self._goal_hold_time + self.step_dt, 0.0)
        self._reached_goal = self._goal_hold_time >= self.cfg.goal_hold_seconds
        local = self._robot.data.root_pos_w - self.scene.env_origins
        if self._scenario is None:
            self._out_of_bounds = torch.linalg.vector_norm(local[:, :2], dim=-1) > self.cfg.max_distance_from_origin
        else:
            self._out_of_bounds = (local[:, 0].abs() > 2.5) | (local[:, 1].abs() > self._scenario.layout.corridor_width / 2)
        return self._reached_goal | self._fallen() | self._out_of_bounds, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        data = self._robot.data
        distance, direction_w, _ = self._goal_data()
        net_progress = self._previous_goal_distance - distance
        legacy = self.cfg.behavior_profile == "legacy_v2"
        progress = (signed_progress(self._previous_goal_distance, distance, self.cfg.target_speed * self.step_dt)
                    if legacy else uncapped_signed_progress(self._previous_goal_distance, distance))
        instantaneous_forward = (data.root_lin_vel_w[:, :2] * direction_w[:, :2]).sum(-1)
        # A transition's actual distance reduction cannot be faked by velocity
        # sampled at one phase of a zero-displacement oscillation.
        forward = instantaneous_forward if legacy else net_progress / self.step_dt
        upright = (-data.projected_gravity_b[:, 2]).clamp(0, 1)
        costs = motion_costs(
            data.joint_pos[:, self._joint_ids], data.joint_vel[:, self._joint_ids],
            data.applied_torque[:, self._joint_ids], self._angle_limits,
            self._effort_limits, self._servo_velocity_limits,
        )
        feet, body_contact = self._contact_state()
        slip_feet = feet
        if not legacy and self.cfg.foot_support_offsets is not None:
            points, surfaces = self._foot_support_geometry()
            support, support_height, _ = closest_foot_support(points, surfaces)
            offset = support - data.body_link_pos_w[:, self._feet]
            if self.cfg.behavior_profile in ("revised_v6", "revised_v7"):
                # A raised toe button does not remove a grounded heel's slip.
                # Use the closest hull witness already selected for velocity,
                # while observations and goal gates retain the button bits.
                slip_feet, _ = sole_contact_state(
                    self._contacts.data.net_forces_w_history,
                    sensor_feet=self._sensor_feet, sensor_body=self._sensor_body,
                    sole_positions_w=support, surface_heights=support_height,
                    force_threshold=self.cfg.contact_threshold,
                    height_tolerance=self.cfg.foot_contact_height_tolerance,
                )
        else:
            offset = self._sole_world_offset()
            support = data.body_link_pos_w[:, self._feet] + offset
            support_height = (self.scene.env_origins[:, 2:3] if self._scenario is None
                              else self._scenario.surface_heights(support))
        if self.cfg.behavior_profile == "revised_v7":
            offset = support - data.body_com_pos_w[:, self._feet]
        velocity = point_velocity(data.body_link_lin_vel_w[:, self._feet], data.body_link_ang_vel_w[:, self._feet], offset)
        if self._scenario is not None:
            velocity = velocity - self._scenario.floor_velocity[:, None, :]
        slip = (velocity[..., :2].square().sum(-1) * slip_feet).sum(-1) / slip_feet.sum(-1).clamp_min(1)
        acceleration = ((self._target_velocity - self._previous_target_velocity) / self.cfg.target_rate_limit).square().mean(-1)
        failed = self._fallen() | self._out_of_bounds
        knee_below_body_fraction = torch.zeros_like(self._goal_hold_time)
        approach_low_knee_fraction = torch.zeros_like(self._goal_hold_time)
        knee_clearance_rate = torch.zeros_like(self._goal_hold_time)
        if self.cfg.behavior_profile in ("revised_v5", "revised_v6", "revised_v7"):
            knee_z = data.body_link_pos_w[:, self._knees, 2]
            body_bottom_offset = torch.zeros_like(data.root_pos_w)
            body_bottom_offset[:, 2] = self.cfg.body_bottom_offset_z_m
            body_bottom_z = (data.root_pos_w + quat_rotate(data.root_quat_w, body_bottom_offset))[:, 2]
            knee_below_body_fraction = (knee_z < body_bottom_z.unsqueeze(-1)).float().mean(-1)
            approach_active = ~(self._reached_goal | failed)
            majority_of_knees_below = knee_below_body_fraction >= 0.5
            self._approach_steps += approach_active.float()
            self._approach_low_knee_steps += (approach_active & majority_of_knees_below).float()
            approach_low_knee_fraction = (
                self._approach_low_knee_steps / self._approach_steps.clamp_min(1.0)
            )
            threshold = self.cfg.knee_below_body_approach_fraction
            knee_violation = ((approach_low_knee_fraction - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
            knee_clearance_rate = self.cfg.knee_below_body_penalty_scale * knee_violation
        rates = {
            "speed": self.cfg.speed_cost_scale * (speed_tracking_cost if legacy else huber_speed_tracking_cost)(
                forward, self.cfg.target_speed, self.cfg.speed_tracking_sigma
            ),
            "upright": self.cfg.upright_cost_scale * (1 - upright).square(),
            **{name: getattr(self.cfg, f"{name}_cost_scale") * value for name, value in costs.items()},
            "target_acceleration": self.cfg.target_acceleration_cost_scale * acceleration,
            "body_contact": self.cfg.body_contact_cost_scale * body_contact.float(),
            "foot_slip": self.cfg.foot_slip_cost_scale * slip,
            "angular_velocity": self.cfg.angular_velocity_cost_scale * data.root_ang_vel_b[:, :2].square().sum(-1),
            "stillness": self.cfg.stillness_penalty_scale * torch.relu(
                0.1 * self.cfg.target_speed - forward
            ) / self.cfg.speed_tracking_sigma,
            "knee_clearance": knee_clearance_rate,
        }
        terms = {name: rate * self.step_dt for name, rate in rates.items()}
        terms.update(
            progress=self.cfg.progress_reward_scale * progress,
            goal=self.cfg.goal_reward * self._reached_goal.float(),
            failure=self.cfg.failure_penalty * failed.float(),
            debris_impact=self.cfg.debris_impact_penalty * self._debris_hit().float(),
        )
        for name, value in terms.items():
            self._episode_sums[name] += value
        self._previous_goal_distance.copy_(distance)
        travel = torch.linalg.vector_norm(data.root_pos_w[:, :2] - self._previous_root_xy, dim=-1)
        self._previous_root_xy.copy_(data.root_pos_w[:, :2])
        # Capture before auto-reset, so evaluation includes terminal falls and effort.
        self.step_metrics = {
            "forward_speed_mps": instantaneous_forward, "step_progress_speed_mps": net_progress / self.step_dt,
            "mechanical_power_w": costs["power"],
            "progress_m": net_progress, "reward_progress_m": progress, "body_contact": body_contact.float(),
            "joint_limit_cost": costs["joint_limit"], "foot_slip_m2_s2": slip,
            "goal": self._reached_goal.float(), "fallen": self._fallen().float(),
            "out_of_bounds": (self._out_of_bounds & ~self._fallen()).float(),
            "target_rate_rad_s": self._target_velocity.abs().amax(-1),
            "travel_distance_m": travel,
            "target_rate_saturation": self._rate_saturation.clone(),
            "torque_saturation": (data.applied_torque[:, self._joint_ids].abs() >= 0.95 * self._effort_limits).float().mean(-1),
            "tracking_error_rad": (data.joint_pos[:, self._joint_ids] - self.joint_angle_targets_rad).abs().mean(-1),
            "feet_contact": feet.float().sum(-1),
            "target_excursion_rad": (self.joint_angle_targets_rad - data.default_joint_pos[:, self._joint_ids]).abs().mean(-1),
            "joint_excursion_rad": (data.joint_pos[:, self._joint_ids] - data.default_joint_pos[:, self._joint_ids]).abs().mean(-1),
            "joint_velocity_rms_rad_s": data.joint_vel[:, self._joint_ids].square().mean(-1).sqrt(),
            "mean_foot_clearance_m": (support[..., 2] - support_height).mean(-1),
            "knees_below_body_fraction": knee_below_body_fraction,
            "approach_low_knee_time_fraction": approach_low_knee_fraction,
        }
        self.extras["log"].update({f"Metrics/{name}": value.mean() for name, value in self.step_metrics.items()})
        return torch.stack(tuple(terms.values())).sum(0)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        valid = env_ids[self.episode_length_buf[env_ids] > 0]
        counts = {name: 0 for name in self.completed}
        counts["episodes"] = len(valid)
        if len(valid):
            counts["goal"] = int(self._reached_goal[valid].sum())
            counts["fallen"] = int(self._fallen()[valid].sum())
            counts["out_of_bounds"] = int((self._out_of_bounds[valid] & ~self._fallen()[valid]).sum())
            counts["time_out"] = int((self.reset_time_outs[valid] & ~self.reset_terminated[valid]).sum())
        for name, count in counts.items():
            self.completed[name] += count
        super()._reset_idx(env_ids)
        if hasattr(self, "_toe_button_pressed"):
            self._toe_button_pressed[env_ids] = False
        if self.cfg.evaluation_mode and self._scenario is None:
            # Fixed one-metre route and heading for deterministic sustained-walk
            # evaluation; stochastic training goals retain the parent reset.
            state = self._robot.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            state[:, 3:7] = state.new_tensor((1.0, 0.0, 0.0, 0.0))
            state[:, 7:] = 0.0
            self._robot.write_root_state_to_sim(state, env_ids)
            self._target_positions_w[env_ids] = state[:, :3]
            self._target_positions_w[env_ids, 0] += self.cfg.evaluation_goal_distance
            self._target_positions_w[env_ids, 2] = self.scene.env_origins[env_ids, 2] + 0.015
            self._previous_goal_distance[env_ids] = self.cfg.evaluation_goal_distance
        for name in ("goal", "fallen", "time_out", "out_of_bounds"):
            self.extras["log"][f"Episode_Termination/{name}"] = counts[name]
        # No random episode age and no precomputed gait phase.
        self.episode_length_buf[env_ids] = 0
        positions = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
        limits = self._angle_limits[env_ids]
        positions = positions.clamp(limits[..., 0], limits[..., 1])
        full_positions = self._robot.data.joint_pos[env_ids].clone()
        full_positions[:, self._joint_ids] = positions
        self._robot.write_joint_state_to_sim(full_positions, torch.zeros_like(full_positions), env_ids=env_ids)
        self._processed_actions[env_ids] = full_positions
        self._robot.set_joint_position_target(full_positions, env_ids=env_ids)
        self._actions[env_ids] = normalize_angles(positions, limits)
        self._previous_actions[env_ids] = self._actions[env_ids]
        self._target_velocity[env_ids] = 0
        self._previous_target_velocity[env_ids] = 0
        self._previous_root_xy[env_ids] = self._robot.data.root_pos_w[env_ids, :2]
        self._rate_saturation[env_ids] = 0
        self._goal_hold_time[env_ids] = 0
        self._approach_steps[env_ids] = 0
        self._approach_low_knee_steps[env_ids] = 0
        self._reached_goal[env_ids] = False
