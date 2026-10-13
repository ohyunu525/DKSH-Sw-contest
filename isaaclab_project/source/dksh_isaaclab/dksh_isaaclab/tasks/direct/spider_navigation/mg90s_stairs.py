"""Provisional CAD6 stair learning with assumed L2 + BNO086 sensor inputs."""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from .mg90s_cad6 import MG90SCad6Env, MG90SCad6EnvCfg, MG90SCad6RunnerCfg
from . import stair_sensing as sensing


@configclass
class MG90SCad6StairsEnvCfg(MG90SCad6EnvCfg):
    observation_space = sensing.OBSERVATIONS
    observation_contract = 'assumed_l2_bno086_command_history_v1'
    episode_length_s = 80.0
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=20.0, replicate_physics=True)
    action_scale = 0.8  # Allow learned body lift/placement, not only tiny flat residuals.
    action_smoothing = 0.15
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
        'contact_feedback_to_policy': False,
        'hardware_deployable': False,
    }


@configclass
class MG90SCad6StairsRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_cad6_stairs_l2_bno086_v1'
    run_name = 'provisional_sensors'

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.algorithm.entropy_coef = 0.005


class MG90SCad6StairsEnv(MG90SCad6Env):
    cfg: MG90SCad6StairsEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        if not 0 <= cfg.stair_initial_level <= cfg.stair_max_level < len(sensing.HEIGHTS_M):
            raise ValueError('Invalid stair curriculum range')
        super().__init__(cfg, render_mode, **kwargs)
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
        self.stair_stats = {'attempts': [0] * len(sensing.HEIGHTS_M), 'successes': [0] * len(sensing.HEIGHTS_M)}

    def _setup_task_terrain(self):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        vertices, faces = [], []
        triangles = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                     (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                     (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
        lower, upper = sensing.stair_boxes(torch.tensor(sensing.HEIGHTS_M))
        for origin in self.scene.env_origins.cpu().tolist():
            for level in range(len(sensing.HEIGHTS_M)):
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
        super()._pre_physics_step(actions)
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
        heights = torch.tensor(sensing.HEIGHTS_M, device=self.device)[self._levels[ids]]
        lower, upper = sensing.stair_boxes(heights)
        ranges, valid = sensing.raycast_stairs(origin, rays, lower, upper)
        valid &= torch.rand_like(ranges) >= self.cfg.lidar_dropout
        ranges = (ranges + torch.randn_like(ranges) * self.cfg.lidar_range_noise_std_m).clamp(0.05, 1.5)
        points = origin[:, None] + ranges[..., None] * rays
        points_b = quat_apply(quat_conjugate(q_rays), points - root[:, None])
        self._lidar_features[ids] = sensing.observed_height_grid(points_b, valid)
        self._lidar_age[ids] = 0

    def _get_observations(self):
        self._update_lidar()
        phase = 2 * math.pi * self._phase()
        obs = torch.cat((self._imu, self._commands, self._filtered_actions,
                         self._targets[:, self._joint_ids] / (math.pi / 2),
                         phase.sin()[:, None], phase.cos()[:, None], self._lidar_features,
                         self._lidar_age[:, None].clamp(0, 1)), -1)
        if obs.shape[1] != sensing.OBSERVATIONS:
            raise RuntimeError('Stair sensor observation contract mismatch')
        return {'policy': obs}

    def _get_dones(self):
        self._sample_imu()
        pos = self._robot.data.root_pos_w - self._episode_origins
        heights = torch.tensor(sensing.HEIGHTS_M, device=self.device)[self._levels]
        ground = sensing.terrain_height(pos[:, 0], heights)
        gravity = self._robot.data.projected_gravity_b
        self._fallen = (pos[:, 2] - ground < 0.045) | (gravity[:, 2] > -0.55)
        self._escaped = (pos[:, 1].abs() > 0.45) | (pos[:, 0] < -0.35) | (pos[:, 0] > 1.75)
        # COM must be well onto the top platform, elevated and held for 1 s.
        finish_x = sensing.FIRST_RISER_M + sensing.STEP_COUNT * sensing.TREAD_M + 0.25
        on_top = (pos[:, 0] > finish_x) & (pos[:, 2] > sensing.STEP_COUNT * heights + 0.075)
        on_top &= ~self._fallen & ~self._escaped
        self._success_hold = torch.where(on_top, self._success_hold + self.step_dt, 0)
        self._success = self._success_hold >= 1.0
        self._progress = pos[:, 0] - self._previous_x
        return self._fallen | self._escaped | self._success, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        upright = (-self._robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
        pos = self._robot.data.root_pos_w - self._episode_origins
        torque_ratio = self._robot.data.applied_torque.abs() / self._robot.data.joint_effort_limits
        active = (self._age > self.cfg.settling_seconds + 1).float()
        reward = 30 * self._progress.clamp(-0.01, 0.01) * active
        reward -= self.step_dt * (0.2 + 2 * (1 - upright).square() + 3 * pos[:, 1].square()
                                 + 0.1 * self._robot.data.root_ang_vel_b.square().sum(-1)
                                 + 0.01 * (self._actions - self._previous_actions).square().sum(-1)
                                 + 0.03 * (torque_ratio > 0.7).float().mean(-1))
        reward += 20 * self._success.float() - 5 * (self._fallen | self._escaped).float()
        self.extras['log'].update({'Stairs/progress_m': self._progress.mean(),
                                  'Stairs/level': self._levels.float().mean(),
                                  'Stairs/success': self._success.float().mean(),
                                  'Stairs/torque_near_cap': (torque_ratio > 0.7).float().mean()})
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        valid = env_ids[self._age[env_ids] > 0]
        for level in range(len(sensing.HEIGHTS_M)):
            selected = valid[self._levels[valid] == level]
            self.stair_stats['attempts'][level] += len(selected)
            self.stair_stats['successes'][level] += int(self._success[selected].sum())
        if self.cfg.stair_curriculum:
            levels, streaks = sensing.next_levels(self._levels[valid], self._success[valid],
                                                  self._fallen[valid] | self._escaped[valid],
                                                  self._streaks[valid], self.cfg.stair_max_level)
            self._levels[valid], self._streaks[valid] = levels, streaks
        super()._reset_idx(env_ids)
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
