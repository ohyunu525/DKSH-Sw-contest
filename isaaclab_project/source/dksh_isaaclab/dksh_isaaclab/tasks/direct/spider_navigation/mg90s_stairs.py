"""CAD6 absolute-angle stair learning with L2, BNO086 and six capped buttons."""
import torch
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from .mg90s_cad6 import MG90SCad6Env, MG90SCad6EnvCfg, MG90SCad6RunnerCfg
from . import stair_sensing as sensing
from .button_contact import ButtonBank, add_button_caps
from .joint_angle_control import ACTION_SEMANTICS, bounded_angle_targets, normalize_angles


@configclass
class MG90SCad6StairsEnvCfg(MG90SCad6EnvCfg):
    observation_space = sensing.OBSERVATIONS
    observation_contract = sensing.OBSERVATION_CONTRACT
    action_semantics = ACTION_SEMANTICS
    episode_length_s = 80.0
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=20.0, replicate_physics=True)
    action_scale = None  # No residual path; the actor selects an absolute angle.
    target_smoothing = 0.15
    button_count = 6
    button_cap_radius_m = 0.005
    button_cap_mass_kg = 0.001
    button_press_force_n = 0.5
    button_release_force_n = 0.25
    button_debounce_s = 0.040
    stair_tread_m = sensing.TREAD_M
    stair_width_m = sensing.WIDTH_M
    stair_heights_m = sensing.HEIGHTS_M
    training_strategy = 'legacy_v1'
    foot_contact_sensor = ContactSensorCfg(
        prim_path='/World/envs/env_.*/Robot/leg_[0-5]_button_cap',
        update_period=0.0, history_length=1, track_air_time=False,
    )
    command_speed_min = 0.02
    command_speed_max = 0.03
    stair_max_level = len(sensing.HEIGHTS_M) - 1
    stair_initial_level = 0
    stair_curriculum = True
    lidar_update_period_s = 1 / 5.55
    lidar_range_noise_std_m = 0.01  # Assumption, not a published noise distribution.
    lidar_dropout = 0.05
    imu_gyro_noise_std_rps = 0.01
    imu_gravity_noise_std = 0.01
    sensor_assumptions = {
        'lidar': 'L2 inverted mounting, decimated nearest-surface analytic rays',
        'local_processing_range_m': 1.5,
        'lidar_self_occlusion': False,
        'imu': 'BNO086 fused gravity/angular velocity, 50 Hz consumer, 20 ms delay',
        'encoder_feedback': False,
        'contact_feedback_to_policy': True,
        'contact': '6 binary cap-only switches in leg order; hysteresis and 40 ms debounce',
        'button_geometry': 'unmeasured 5 mm radius, 1 g rigid cap centred at CAD foot point; no button travel model',
        'hardware_deployable': False,
    }


@configclass
class MG90SCad6StairsRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_cad6_stairs_mg996r_buttons_absolute_v2'
    run_name = 'provisional_button_caps'

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.algorithm.entropy_coef = 0.005


class MG90SCad6StairsEnv(MG90SCad6Env):
    cfg: MG90SCad6StairsEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        if not 0 <= cfg.stair_initial_level <= cfg.stair_max_level < len(cfg.stair_heights_m):
            raise ValueError('Invalid stair curriculum range')
        super().__init__(cfg, render_mode, **kwargs)
        # OYW action convention, mapped explicitly rather than relying on USD joint order.
        self._angle_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids].clone()
        self._sensor_button_ids = [self._foot_contact.body_names.index(f'leg_{i}_button_cap') for i in range(6)]
        self._buttons = ButtonBank((self.num_envs, 6), self.device, cfg.button_press_force_n,
                                   cfg.button_release_force_n, cfg.button_debounce_s)
        self._button_values = torch.zeros((self.num_envs, 6), device=self.device)
        self.button_press_samples = torch.zeros(6, dtype=torch.long, device=self.device)
        self._levels = torch.full((self.num_envs,), cfg.stair_initial_level, device=self.device, dtype=torch.long)
        self._streaks = torch.zeros_like(self._levels)
        self._episode_origins = self.scene.env_origins.clone()
        self._success = torch.zeros_like(self._fallen)
        self._success_hold = torch.zeros_like(self._age)
        self._lidar_age = torch.full_like(self._age, cfg.lidar_update_period_s)
        self._lidar_features = torch.zeros((self.num_envs, 192), device=self.device)
        self._directions = sensing.lidar_directions(self.device)
        self._imu = torch.zeros((self.num_envs, 6), device=self.device)
        self._imu_pending = self._imu.clone()
        self._previous_x = torch.zeros_like(self._age)
        self._progress = torch.zeros_like(self._age)
        self.stair_stats = {'attempts': [0] * len(cfg.stair_heights_m), 'successes': [0] * len(cfg.stair_heights_m)}

    @property
    def joint_angle_targets_rad(self):
        """Applied absolute command, leg-major, radians relative to assembled CAD zero."""
        return self._targets[:, self._joint_ids]

    def _setup_foot_hardware(self, stage, robot_path):
        add_button_caps(stage, robot_path, self.cfg.button_cap_radius_m, self.cfg.button_cap_mass_kg)

    def _setup_task_terrain(self):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        vertices, faces = [], []
        triangles = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                     (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                     (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
        lower, upper = sensing.stair_boxes(torch.tensor(self.cfg.stair_heights_m), self.cfg.stair_tread_m,
                                          self.cfg.stair_width_m)
        for origin in self.scene.env_origins.cpu().tolist():
            for level in range(len(self.cfg.stair_heights_m)):
                offset = (origin[0], origin[1] + level * sensing.PATCH_SPACING_M, origin[2])
                for lo, hi in zip(lower[level].tolist(), upper[level].tolist()):
                    start = len(vertices)
                    for x, y, z in ((lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
                                    (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
                                    (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
                                    (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])):
                        vertices.append(Gf.Vec3f(x + offset[0], y + offset[1], z + offset[2]))
                    faces.extend(start + index for face in triangles for index in face)
        mesh = UsdGeom.Mesh.Define(stage, '/World/Stairs')
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        material = UsdShade.Material(stage.GetPrimAtPath('/World/ground/physicsMaterial'))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')

    def _task_terrain_paths(self):
        return ['/World/Stairs']

    def _pre_physics_step(self, actions):
        self._previous_x.copy_(self._robot.data.root_pos_w[:, 0] - self._episode_origins[:, 0])
        self.extras['log'] = {}
        self._previous_actions.copy_(self._actions)
        previous = self.joint_angle_targets_rad
        safe = torch.where(torch.isfinite(actions), actions, normalize_angles(previous, self._angle_limits))
        self._actions.copy_(safe.clamp(-1, 1))
        targets = bounded_angle_targets(self._actions, previous, self._angle_limits,
                                       dt=self.step_dt, rate_limit=self.cfg.target_rate_limit,
                                       smoothing=self.cfg.target_smoothing)
        self._targets[:, self._joint_ids] = targets
        # Keep parent reset buffers, but never generate or add a wave reference.
        self._filtered_actions.copy_(normalize_angles(targets, self._angle_limits))
        self._age += self.step_dt
        self._lidar_age += self.step_dt

    def _sample_imu(self):
        # Sample-once-per-control-tick, one tick delay, not resampled per observation call.
        self._imu.copy_(self._imu_pending)
        gyro = self._robot.data.root_ang_vel_b + torch.randn_like(self._imu[:, :3]) * self.cfg.imu_gyro_noise_std_rps
        gravity = self._robot.data.projected_gravity_b + torch.randn_like(self._imu[:, 3:]) * self.cfg.imu_gravity_noise_std
        self._imu_pending[:, :3] = gyro
        self._imu_pending[:, 3:] = torch.nn.functional.normalize(gravity, dim=-1)

    def _update_lidar(self):
        ids = (self._lidar_age >= self.cfg.lidar_update_period_s).nonzero().flatten()
        if not len(ids):
            return
        q = self._robot.data.root_quat_w[ids]
        root = self._robot.data.root_pos_w[ids] - self._episode_origins[ids]
        mount = torch.tensor(self.cfg.lidar_mount_position_b, device=self.device).expand(len(ids), -1)
        origin = root + quat_apply(q, mount)
        q_rays = q[:, None].expand(-1, len(self._directions), -1)
        rays = quat_apply(q_rays, self._directions.expand(len(ids), -1, -1))
        heights = torch.tensor(self.cfg.stair_heights_m, device=self.device)[self._levels[ids]]
        lower, upper = sensing.stair_boxes(heights, self.cfg.stair_tread_m, self.cfg.stair_width_m)
        ranges, valid = sensing.raycast_stairs(origin, rays, lower, upper)
        valid &= torch.rand_like(ranges) >= self.cfg.lidar_dropout
        ranges = (ranges + torch.randn_like(ranges) * self.cfg.lidar_range_noise_std_m).clamp(0.05, 1.5)
        points = origin[:, None] + ranges[..., None] * rays
        points_b = quat_apply(quat_conjugate(q_rays), points - root[:, None])
        self._lidar_features[ids] = sensing.observed_height_grid(points_b, valid)
        self._lidar_age[ids] = 0

    def _get_observations(self):
        self._update_lidar()
        obs = torch.cat((self._imu, self._commands, self._actions,
                         normalize_angles(self._targets[:, self._joint_ids], self._angle_limits),
                         self._button_values, self._lidar_features,
                         self._lidar_age[:, None].clamp(0, 1)), -1)
        if obs.shape[1] != sensing.OBSERVATIONS:
            raise RuntimeError('Stair sensor observation contract mismatch')
        return {'policy': obs}

    def _get_dones(self):
        self._sample_imu()
        forces = self._foot_contact.data.net_forces_w[:, self._sensor_button_ids].norm(dim=-1)
        self._button_values.copy_(self._buttons.update(forces, self.step_dt))
        self.button_press_samples += self._button_values.sum(0).long()
        pos = self._robot.data.root_pos_w - self._episode_origins
        heights = torch.tensor(self.cfg.stair_heights_m, device=self.device)[self._levels]
        ground = sensing.terrain_height(pos[:, 0], heights, self.cfg.stair_tread_m,
                                        pos[:, 1], self.cfg.stair_width_m)
        gravity = self._robot.data.projected_gravity_b
        self._fallen = (pos[:, 2] - ground < 0.045) | (gravity[:, 2] > -0.55)
        self._escaped = (pos[:, 1].abs() > 0.45) | (pos[:, 0] < -0.35) | (pos[:, 0] > 1.75)
        # COM must be well onto the top platform, elevated and held for 1 s.
        finish_x = sensing.FIRST_RISER_M + sensing.STEP_COUNT * self.cfg.stair_tread_m + 0.25
        on_top = (pos[:, 0] > finish_x) & (pos[:, 2] > sensing.STEP_COUNT * heights + 0.075)
        on_top &= pos[:, 1].abs() <= self.cfg.stair_width_m / 2
        on_top &= ~self._fallen & ~self._escaped & (self._button_values.sum(-1) >= 3)
        self._success_hold = torch.where(on_top, self._success_hold + self.step_dt, 0)
        self._success = self._success_hold >= 1.0
        self._progress = pos[:, 0] - self._previous_x
        return self._fallen | self._escaped | self._success, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        upright = (-self._robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
        pos = self._robot.data.root_pos_w - self._episode_origins
        torque_ratio = self._robot.data.applied_torque.abs() / self._robot.data.joint_effort_limits
        active = (self._age > self.cfg.settling_seconds + 1).float()
        if self.cfg.training_strategy == 'staged_v1':
            reward = sensing.progress_and_speed_reward(self._progress, self._commands[:, 0], self.step_dt) * active
        else:
            reward = 30 * self._progress.clamp(-0.01, 0.01) * active
        reward -= self.step_dt * (0.2 + 2 * (1 - upright).square() + 3 * pos[:, 1].square()
                                 + 0.1 * self._robot.data.root_ang_vel_b.square().sum(-1)
                                 + 0.01 * (self._actions - self._previous_actions).square().sum(-1)
                                 + 0.03 * (torque_ratio > 0.7).float().mean(-1))
        reward += 20 * self._success.float() - 5 * (self._fallen | self._escaped).float()
        self.extras['log'].update({'Stairs/progress_m': self._progress.mean(),
                                  'Stairs/level': self._levels.float().mean(),
                                  'Stairs/success': self._success.float().mean(),
                                  'Stairs/torque_near_cap': (torque_ratio > 0.7).float().mean(),
                                  'Buttons/pressed_count': self._button_values.sum(-1).mean(),
                                  'Actions/absolute_target_max_rad': self.joint_angle_targets_rad.abs().max()})
        # Capture terminal metrics before DirectRLEnv auto-resets an episode.
        self.step_metrics = {'max_effort_ratio': torque_ratio.max().item(),
                             'max_progress_x_m': pos[:, 0].max().item(),
                             'min_body_clearance_m': (pos[:, 2] - sensing.terrain_height(
                                 pos[:, 0], torch.tensor(self.cfg.stair_heights_m, device=self.device)[self._levels],
                                 self.cfg.stair_tread_m, pos[:, 1], self.cfg.stair_width_m)).min().item()}
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        valid = env_ids[self._age[env_ids] > 0]
        for level in range(len(self.cfg.stair_heights_m)):
            selected = valid[self._levels[valid] == level]
            self.stair_stats['attempts'][level] += len(selected)
            self.stair_stats['successes'][level] += int(self._success[selected].sum())
        if self.cfg.stair_curriculum:
            levels, streaks = sensing.next_levels(self._levels[valid], self._success[valid],
                                                  self._fallen[valid] | self._escaped[valid],
                                                  self._streaks[valid], self.cfg.stair_max_level)
            self._levels[valid], self._streaks[valid] = levels, streaks
        super()._reset_idx(env_ids)
        self._buttons.reset(env_ids)
        self._button_values[env_ids] = 0
        normalized = normalize_angles(self._targets[env_ids][:, self._joint_ids], self._angle_limits[env_ids])
        self._actions[env_ids] = normalized
        self._previous_actions[env_ids] = normalized
        self._filtered_actions[env_ids] = normalized
        self._robot.set_joint_position_target(self._targets[env_ids], env_ids=env_ids)
        self._episode_origins[env_ids] = self.scene.env_origins[env_ids]
        self._episode_origins[env_ids, 1] += self._levels[env_ids] * sensing.PATCH_SPACING_M
        state = self._robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self._episode_origins[env_ids]
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._success[env_ids] = False
        self._success_hold[env_ids] = 0
        self._lidar_age[env_ids] = self.cfg.lidar_update_period_s
        self._lidar_features[env_ids] = 0
        self._imu[env_ids] = 0
        self._imu[env_ids, 5] = -1
        self._imu_pending[env_ids] = self._imu[env_ids]
        self._previous_x[env_ids] = 0
        self._progress[env_ids] = 0
