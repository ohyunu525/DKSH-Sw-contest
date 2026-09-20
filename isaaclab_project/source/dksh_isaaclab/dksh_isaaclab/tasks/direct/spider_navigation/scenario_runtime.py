"""Physical obstacle courses and per-environment hazard state for Isaac Lab 2.1."""

from __future__ import annotations

import math

import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

from .scenario_layout import build_layout
from .scenario_sensing import apply_lidar_measurement_model, sample_box_heights, spatial_obstacle_ranges


class ScenarioRuntime:
    """Create before scene cloning; initialize buffers after simulation starts."""

    DEBRIS_COUNT = 4

    def __init__(self, env):
        self.env = env
        cfg = env.cfg
        self.layout = build_layout(
            cfg.environment_preset, cfg.environment_difficulty, cfg.robot_width, cfg.robot_height
        )
        self.difficulty = cfg.environment_difficulty
        self.platform = None
        self.debris = []
        self.contacts = []
        self.radius = 0.045 + 0.025 * self.difficulty
        for box in self.layout.boxes:
            color = (0.38, 0.42, 0.48) if box.kind != "rough" else (0.42, 0.32, 0.22)
            spawn_cfg = sim_utils.CuboidCfg(
                size=box.size,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=cfg.sim.physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            spawn_cfg.func(f"/World/envs/env_0/{box.name}", spawn_cfg, translation=box.position)

        if self.layout.vibration_active:
            self.platform = RigidObject(RigidObjectCfg(
                prim_path="/World/envs/env_.*/VibratingFloor",
                spawn=sim_utils.CuboidCfg(
                    size=(5.6, self.layout.corridor_width, 0.06),
                    # A massive motor-driven body uses the velocity API available
                    # in Isaac Sim 4.5 and transfers real contact/friction forces.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False, disable_gravity=True,
                        max_depenetration_velocity=0.5, solver_position_iteration_count=8,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                    physics_material=cfg.sim.physics_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.40, 0.65)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
            ))
            env.scene.rigid_objects["vibrating_floor"] = self.platform

        if self.layout.debris_active:
            # Each filter must name one link per cloned environment. A single
            # Robot/.* pattern ambiguously groups every link in the PhysX view.
            robot_links = sim_utils.get_all_matching_child_prims(
                "/World/envs/env_0/Robot", lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            robot_filters = [str(prim.GetPath()).replace("/env_0/", "/env_.*/") for prim in robot_links]
            if not robot_filters:
                raise RuntimeError("The robot has no rigid bodies for debris contact filtering.")
            # A bounded reusable pool prevents unbounded scene growth in long PPO runs.
            for slot in range(self.DEBRIS_COUNT):
                path = f"/World/envs/env_.*/Debris_{slot}"
                debris = RigidObject(RigidObjectCfg(
                    prim_path=path,
                    spawn=sim_utils.SphereCfg(
                        radius=self.radius,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False, max_depenetration_velocity=1.0,
                            max_linear_velocity=10.0, max_angular_velocity=100.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.15 + 0.35 * self.difficulty),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=0.8, dynamic_friction=0.6, restitution=0.05,
                        ),
                        activate_contact_sensors=True,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.24, 0.08)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=self._parking_position(slot)),
                ))
                env.scene.rigid_objects[f"debris_{slot}"] = debris
                self.debris.append(debris)
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=path,
                    update_period=0.0,
                    history_length=1,
                    # One sensor body per environment; filter all robot links only.
                    filter_prim_paths_expr=robot_filters,
                ))
                env.scene.sensors[f"debris_contact_{slot}"] = sensor
                self.contacts.append(sensor)

    def _parking_position(self, slot):
        return (2.1 + slot * 0.18, self.layout.corridor_width * 0.5 + 0.45, self.radius + 0.02)

    def initialize(self):
        env = self.env
        self.time = torch.zeros(env.num_envs, device=env.device)
        self.phase = torch.zeros_like(self.time)
        self.amplitude = torch.zeros_like(self.time)
        self.frequency = torch.zeros_like(self.time)
        self.next_drop = torch.zeros((env.num_envs, self.DEBRIS_COUNT), device=env.device)
        self.drop_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.impact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._contact_active = torch.zeros_like(self.impact)
        self.floor_height = torch.full_like(self.time, self.layout.spawn[2])
        self.floor_velocity = torch.zeros((env.num_envs, 3), device=env.device)
        self._lidar_ranges = torch.ones(
            (env.num_envs, env.cfg.lidar_observation_bins), device=env.device
        )
        self._lidar_valid = torch.zeros_like(self._lidar_ranges)
        self._lidar_next_update = torch.zeros(env.num_envs, device=env.device)
        self._centers = torch.tensor([b.position for b in self.layout.boxes], device=env.device).reshape(-1, 3)
        self._halves = torch.tensor([b.size for b in self.layout.boxes], device=env.device).reshape(-1, 3) / 2
        # Roof and walls are never interpreted as supporting terrain.
        self._terrain_indices = [i for i, b in enumerate(self.layout.boxes) if b.kind == "rough"]
        self._height_offsets = torch.tensor(
            [(x, y) for x in (-0.3, 0.0, 0.3) for y in (-0.3, 0.0, 0.3)], device=env.device
        )

    def reset(self, env_ids):
        count = len(env_ids)
        self.time[env_ids] = 0.0
        self.phase[env_ids] = torch.rand(count, device=self.env.device) * (2.0 * math.pi)
        self.amplitude[env_ids] = (0.004 + 0.014 * self.difficulty) * (
            0.8 + 0.2 * torch.rand(count, device=self.env.device)
        )
        self.frequency[env_ids] = (0.8 + 2.2 * self.difficulty) * (
            0.85 + 0.3 * torch.rand(count, device=self.env.device)
        )
        self.impact[env_ids] = False
        self._contact_active[env_ids] = False
        self._lidar_ranges[env_ids] = 1.0
        self._lidar_valid[env_ids] = 0.0
        self._lidar_next_update[env_ids] = 0.0
        self.drop_count[env_ids] = 0
        interval = 1.7 - self.difficulty
        for slot, debris in enumerate(self.debris):
            debris.reset(env_ids)
            state = debris.data.default_root_state[env_ids].clone()
            state[:, :3] = torch.tensor(self._parking_position(slot), device=self.env.device)
            state[:, :3] += self.env.scene.env_origins[env_ids]
            state[:, 7:] = 0.0
            debris.write_root_state_to_sim(state, env_ids)
            self.next_drop[env_ids, slot] = (
                0.8 + slot * interval + torch.rand(count, device=self.env.device) * 0.5
            )
        if self.platform is not None:
            self.platform.reset(env_ids)
            self._move_platform(env_ids, reset=True)

    def begin_step(self):
        self.impact.zero_()

    def _move_platform(self, env_ids=None, reset=False):
        ids = slice(None) if env_ids is None else env_ids
        omega = 2.0 * math.pi * self.frequency[ids]
        angle = omega * self.time[ids] + self.phase[ids]
        amplitude = self.amplitude[ids]
        state = self.platform.data.default_root_state[ids].clone()
        state[:, :3] += self.env.scene.env_origins[ids]
        state[:, 0] += 0.5 * amplitude * torch.sin(angle)
        state[:, 2] += amplitude * torch.sin(angle)
        state[:, 7] = 0.5 * amplitude * omega * torch.cos(angle)
        state[:, 9] = amplitude * omega * torch.cos(angle)
        self.floor_height[ids] = state[:, 2] - self.env.scene.env_origins[ids, 2] + 0.03
        self.floor_velocity[ids] = state[:, 7:10]
        if reset:
            self.platform.write_root_state_to_sim(state, env_ids)
            return
        # Drive toward the next sinusoid sample using velocity. PhysX integrates
        # the motion and resolves contacts; poses are written only at reset.
        # Use the public velocity writer supplied by this repository's Isaac Lab.
        current = self.platform.data.root_state_w[ids]
        velocity = torch.zeros_like(state[:, 7:])
        velocity[:, :3] = ((state[:, :3] - current[:, :3]) / self.env.physics_dt).clamp(-1.0, 1.0)
        # Restore the floor's level orientation after contact torques. Its high
        # inertia and small per-step correction model a powered shaker table.
        sign = torch.where(current[:, 3:4] < 0, -1.0, 1.0)
        velocity[:, 3:] = (-2.0 * sign * current[:, 4:7] / self.env.physics_dt).clamp(-5.0, 5.0)
        self.platform.write_root_velocity_to_sim(velocity, env_ids)
        self.floor_velocity[ids] = velocity[:, :3]

    def physics_step(self):
        # Read the preceding physics substep before preparing the next one.
        self.collect_impacts()
        self.time += self.env.physics_dt
        if self.platform is not None:
            self._move_platform()
        for slot, debris in enumerate(self.debris):
            ids = (self.time >= self.next_drop[:, slot]).nonzero(as_tuple=False).flatten()
            if ids.numel() == 0:
                continue
            count = len(ids)
            state = debris.data.default_root_state[ids].clone()
            state[:, :3] = self.env.scene.env_origins[ids]
            # Safe spawn and goal aprons lie outside the drop region.
            state[:, 0] += 1.7 * torch.rand(count, device=self.env.device) - 0.85
            state[:, 1] += (torch.rand(count, device=self.env.device) - 0.5) * self.layout.corridor_width * 0.65
            state[:, 2] += 1.2 + self.difficulty + self.layout.spawn[2]
            state[:, 7:] = 0.0
            debris.write_root_state_to_sim(state, ids)
            self.next_drop[ids, slot] = self.time[ids] + self.DEBRIS_COUNT * (
                1.7 - self.difficulty
            ) + 0.4 * torch.rand(count, device=self.env.device)
            self.drop_count[ids] += 1

    def collect_impacts(self):
        if self.platform is not None:
            self.floor_height.copy_(
                self.platform.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2] + 0.03
            )
            self.floor_velocity.copy_(self.platform.data.root_lin_vel_w)
        if not self.contacts:
            return
        contacting = torch.zeros_like(self.impact)
        for sensor in self.contacts:
            forces = sensor.data.force_matrix_w
            contacting |= torch.linalg.vector_norm(forces, dim=-1).flatten(1).amax(dim=1) > (
                self.env.cfg.debris_impact_threshold
            )
        self.impact |= contacting & ~self._contact_active
        self._contact_active.copy_(contacting)

    def ground_height(self, positions_w):
        origins = self.env.scene.env_origins
        local = positions_w - origins
        on_floor = (local[:, 0].abs() < 2.8) & (local[:, 1].abs() < self.layout.corridor_width / 2)
        floor = origins[:, 2] + torch.where(on_floor, self.floor_height, 0.0)
        centers = self._centers.unsqueeze(0) + origins.unsqueeze(1)
        return sample_box_heights(
            positions_w[:, None, :2], centers[:, self._terrain_indices], self._halves[self._terrain_indices],
            default_height=floor,
        ).squeeze(1)

    def _lidar_observations(self, positions_w, quaternion_w, centers_w):
        """Update and hold a 3-D-projected L1 RM scan at its physical 11 Hz rate."""
        cfg = self.env.cfg
        due = self.time >= self._lidar_next_update
        due_ids = due.nonzero(as_tuple=True)[0]
        if due_ids.numel():
            sphere_centers = None
            if self.debris:
                sphere_centers = torch.stack([body.data.root_pos_w[due_ids] for body in self.debris], dim=1)
            ranges = spatial_obstacle_ranges(
                positions_w[due_ids], quaternion_w[due_ids], centers_w[due_ids], self._halves,
                max_range=cfg.lidar_max_range_m,
                horizontal_count=cfg.lidar_observation_bins,
                azimuth_samples_per_bin=cfg.lidar_azimuth_samples_per_bin,
                vertical_fov_degrees=cfg.lidar_vertical_fov_deg,
                vertical_count=cfg.lidar_vertical_projection_bins,
                sphere_centers_w=sphere_centers,
                sphere_radii=self.radius if sphere_centers is not None else None,
            )
            noise = torch.zeros_like(ranges)
            if cfg.lidar_noise_enabled:
                noise.uniform_(-1.0, 1.0)
            measured = apply_lidar_measurement_model(
                ranges,
                max_range=cfg.lidar_max_range_m,
                min_range=cfg.lidar_min_range_m,
                accuracy=cfg.lidar_measurement_accuracy_m,
                resolution=cfg.lidar_measurement_resolution_m,
                noise=noise,
                validate_tensors=False,
            )
            self._lidar_ranges[due_ids] = measured
            self._lidar_valid[due_ids] = (measured < 1.0).to(measured.dtype)
        period = 1.0 / cfg.lidar_horizontal_scan_frequency_hz
        # Advance from the ideal sensor clock instead of the policy clock.  At
        # 50 Hz control, scheduling from ``time + period`` would turn 11 Hz into
        # a drifting 10 Hz stream because scans can only be consumed on a step.
        elapsed_periods = torch.floor(
            (self.time - self._lidar_next_update).clamp_min(0.0) / period
        ) + 1.0
        self._lidar_next_update.copy_(torch.where(
            due,
            self._lidar_next_update + elapsed_periods * period,
            self._lidar_next_update,
        ))
        return torch.cat((self._lidar_ranges, self._lidar_valid), dim=1)

    def observations(self):
        """Return 32 L1 actor features followed by 16 privileged critic features."""
        env = self.env
        pos = env._robot.data.root_pos_w
        quat = env._robot.data.root_quat_w
        yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                          1 - 2 * (quat[:, 2].square() + quat[:, 3].square()))
        centers = self._centers.unsqueeze(0) + env.scene.env_origins.unsqueeze(1)
        mount_b = pos.new_tensor(env.cfg.lidar_mount_position_b).expand(env.num_envs, -1)
        lidar_pos = pos + quat_rotate(quat, mount_b)
        ranges = self._lidar_observations(lidar_pos, quat, centers)
        offsets = self._height_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
        cos, sin = yaw.cos().unsqueeze(1), yaw.sin().unsqueeze(1)
        points = torch.stack((cos * offsets[:, :, 0] - sin * offsets[:, :, 1],
                              sin * offsets[:, :, 0] + cos * offsets[:, :, 1]), dim=-1) + pos[:, None, :2]
        heights = sample_box_heights(
            points, centers[:, self._terrain_indices], self._halves[self._terrain_indices],
            default_height=self.floor_height + env.scene.env_origins[:, 2],
        )
        heights = ((heights - pos[:, 2:3]) / 0.5).clamp(-1, 1)
        floor_velocity_b = quat_rotate_inverse(quat, self.floor_velocity).clamp(-1, 1)
        hazard = torch.zeros((env.num_envs, 4), device=env.device)
        if self.debris:
            positions = torch.stack([d.data.root_pos_w for d in self.debris], dim=1)
            velocities = torch.stack([d.data.root_lin_vel_w for d in self.debris], dim=1)
            offset = positions - pos[:, None, :]
            distance = torch.linalg.vector_norm(offset, dim=-1)
            index = distance.argmin(dim=1)
            rows = torch.arange(env.num_envs, device=env.device)
            nearest = quat_rotate_inverse(quat, offset[rows, index])
            hazard[:, :3] = (nearest / 3.0).clamp(-1, 1)
            hazard[:, 3] = (velocities[rows, index, 2] / 8.0).clamp(-1, 1)
        return torch.cat((ranges, heights, floor_velocity_b, hazard), dim=1)
