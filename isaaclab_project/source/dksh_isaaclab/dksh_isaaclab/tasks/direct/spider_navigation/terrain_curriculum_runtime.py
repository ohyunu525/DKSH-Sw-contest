"""Per-environment randomized courses on a fixed PhysX cuboid pool.

Only poses change at reset. Collider dimensions never change during simulation,
so the same buffers describe both physical geometry and terrain/LiDAR queries.
Training reserves flat replay environments; evaluation uses a frozen curriculum
and a separate deterministic course-seed namespace.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

from .scenario_runtime import ScenarioRuntime
from .scenario_sensing import sample_box_heights
from .terrain_curriculum import (
    COURSE_CONTRACT, FAMILIES, OBSTACLE_FAMILIES, POOL_BOX_SPECS,
    CurriculumController, sample_course,
)


class CurriculumScenarioRuntime(ScenarioRuntime):
    """A mixed course batch whose pool is spawned before scene cloning."""

    def __init__(self, env):
        self.env = env
        cfg = env.cfg
        course_seed = cfg.seed if cfg.course_seed is None else cfg.course_seed
        if isinstance(course_seed, bool) or not isinstance(course_seed, Integral) or course_seed < 0:
            raise ValueError("course_seed must be a nonnegative integer")
        cfg.course_seed = int(course_seed)
        self.platform, self.debris, self.contacts = None, [], []
        self.radius = 0.0
        self.difficulty = float(cfg.environment_difficulty)
        self.layout = sample_course(
            seed=int(cfg.course_seed), split=cfg.course_split, env_id=0, episode=0,
            family="flat", difficulty=0.0, robot_width=cfg.robot_width,
            robot_height=cfg.robot_height,
        )
        self.boxes = []
        for box in POOL_BOX_SPECS:
            body = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Curriculum_{box.name}",
                spawn=sim_utils.CuboidCfg(
                    size=box.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=cfg.sim.physics_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.42, 0.32, 0.22) if box.kind == "rough" else (0.38, 0.42, 0.48),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=box.position),
            ))
            env.scene.rigid_objects[f"curriculum_{box.name}"] = body
            self.boxes.append(body)

    def initialize(self):
        env, cfg = self.env, self.env.cfg
        fraction = cfg.curriculum_flat_fraction
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("curriculum_flat_fraction must be finite and between 0 and 1")
        self.flat_count = (min(env.num_envs - 1, max(1, math.ceil(env.num_envs * fraction)))
                           if env.num_envs > 1 else 0)
        self._single_flat_period = max(2, round(1.0 / fraction))
        self.controller = CurriculumController(
            initial_difficulty=cfg.environment_difficulty,
            window=cfg.curriculum_window,
            min_episodes=cfg.curriculum_min_episodes,
            promote_threshold=cfg.curriculum_promote_threshold,
            demote_threshold=cfg.curriculum_demote_threshold,
            step=cfg.curriculum_step,
            frozen=not cfg.curriculum_enabled or cfg.course_split == "holdout",
        )
        self.time = torch.zeros(env.num_envs, device=env.device)
        self.impact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.floor_height = torch.zeros_like(self.time)
        self.floor_velocity = torch.zeros((env.num_envs, 3), device=env.device)
        self._lidar_ranges = torch.ones((env.num_envs, cfg.lidar_observation_bins), device=env.device)
        self._lidar_valid = torch.zeros_like(self._lidar_ranges)
        self._lidar_next_update = torch.zeros_like(self.time)
        self._centers = torch.tensor([b.position for b in POOL_BOX_SPECS], device=env.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
        self._halves = torch.tensor([b.size for b in POOL_BOX_SPECS], device=env.device) / 2
        self._active_boxes = torch.zeros(self._centers.shape[:2], dtype=torch.bool, device=env.device)
        self._terrain_indices = [i for i, box in enumerate(POOL_BOX_SPECS) if box.kind in ("rough", "stairs")]
        self._height_offsets = torch.tensor(
            [(x, y) for x in (-0.3, 0.0, 0.3) for y in (-0.3, 0.0, 0.3)], device=env.device,
        )
        self.spawn_positions = torch.zeros((env.num_envs, 3), device=env.device)
        self.goal_positions = torch.zeros_like(self.spawn_positions)
        self.corridor_widths = torch.ones(env.num_envs, device=env.device)
        self.difficulties = torch.zeros_like(self.time)
        self.course_families = ["flat"] * env.num_envs
        self.course_ids = [None] * env.num_envs
        self.courses = [None] * env.num_envs
        # Host counters only advance for the reset environment. Seed derivation
        # never consumes Torch's global RNG or depends on reset batch ordering.
        self.episode_counters = [0] * env.num_envs
        self._current_episode_indices = [-1] * env.num_envs

    def _ids(self, env_ids):
        if env_ids is None:
            return list(range(self.env.num_envs))
        values = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if (any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.env.num_envs for i in values)
                or len(set(values)) != len(values)):
            raise ValueError("Environment indices must be unique integers within the batch")
        return [int(i) for i in values]

    def _family_for(self, env_id, episode):
        if self.env.num_envs == 1:
            if episode % self._single_flat_period == 0:
                return "flat"
            return OBSTACLE_FAMILIES[(episode - 1 - episode // self._single_flat_period) % len(OBSTACLE_FAMILIES)]
        if env_id < self.flat_count:
            return "flat"
        return OBSTACLE_FAMILIES[(env_id - self.flat_count + episode) % len(OBSTACLE_FAMILIES)]

    def _apply_courses(self, indices, courses):
        if not indices:
            return
        ids = torch.tensor(indices, dtype=torch.long, device=self.env.device)
        for i, course in zip(indices, courses):
            if (len(course.boxes) != len(POOL_BOX_SPECS)
                    or any((a.name, a.size, a.kind) != (b.name, b.size, b.kind)
                           for a, b in zip(course.boxes, POOL_BOX_SPECS))):
                raise ValueError("Sampled course changed the physical collider pool")
            self.courses[i] = course
            self.course_families[i], self.course_ids[i] = course.family, course.course_id
        self._centers[ids] = self._centers.new_tensor([[box.position for box in course.boxes] for course in courses])
        self._active_boxes[ids] = (self._centers[ids, :, 2] + self._halves[:, 2]) > 0
        self.spawn_positions[ids] = self.spawn_positions.new_tensor([course.spawn for course in courses])
        self.goal_positions[ids] = self.goal_positions.new_tensor([course.goal for course in courses])
        self.corridor_widths[ids] = self.corridor_widths.new_tensor([course.corridor_width for course in courses])
        self.difficulties[ids] = self.difficulties.new_tensor([course.difficulty for course in courses])
        for slot, body in enumerate(self.boxes):
            body.reset(ids)
            state = body.data.default_root_state[ids].clone()
            state[:, :3] = self._centers[ids, slot] + self.env.scene.env_origins[ids]
            state[:, 3:7] = state.new_tensor((1.0, 0.0, 0.0, 0.0))
            state[:, 7:] = 0.0
            body.write_root_state_to_sim(state, ids)

    def reset(self, env_ids):
        indices = self._ids(env_ids)
        cfg = self.env.cfg
        courses = []
        for i in indices:
            episode = self.episode_counters[i]
            family = self._family_for(i, episode)
            courses.append(sample_course(
                seed=cfg.course_seed, split=cfg.course_split, env_id=i, episode=episode,
                family=family, difficulty=self.controller.difficulty_for(family),
                robot_width=cfg.robot_width, robot_height=cfg.robot_height,
            ))
            self._current_episode_indices[i] = episode
            self.episode_counters[i] += 1
        self._apply_courses(indices, courses)
        self.time[indices] = 0
        self.impact[indices] = False
        self._lidar_ranges[indices] = 1
        self._lidar_valid[indices] = 0
        self._lidar_next_update[indices] = 0

    def record_outcomes(self, env_ids, successes):
        """Call only for completed episodes, before replacing their course."""
        indices = self._ids(env_ids)
        values = successes.detach().cpu().tolist() if isinstance(successes, torch.Tensor) else list(successes)
        if len(values) != len(indices) or any(not isinstance(value, bool) for value in values):
            raise ValueError("Success flags must be one boolean per completed environment")
        # Sort a simultaneous completion batch so reset-index ordering cannot
        # change a threshold crossing and the corresponding next-course level.
        for i, success in sorted(zip(indices, values)):
            course = self.courses[i]
            if course is None:
                continue
            self.controller.record(course.family, success, difficulty=course.difficulty)

    def episode_metadata(self, env_ids):
        metadata = []
        for i in self._ids(env_ids):
            course = self.courses[i]
            if course is not None:
                metadata.append({
                    "environment_index": i, "family": course.family,
                    "difficulty": course.difficulty, "course_id": course.course_id,
                    "course_seed": course.course_seed, "split": course.split,
                    "episode_index": self._current_episode_indices[i],
                })
        return metadata

    def statistics(self):
        return self.controller.statistics()

    def state_dict(self):
        cfg = self.env.cfg
        return {
            "contract": COURSE_CONTRACT, "seed": cfg.course_seed, "split": cfg.course_split,
            "num_envs": self.env.num_envs, "flat_fraction": cfg.curriculum_flat_fraction,
            "robot_width": cfg.robot_width, "robot_height": cfg.robot_height,
            "controller": self.controller.state_dict(),
            "episode_counters": self.episode_counters.copy(),
            "current_courses": self.episode_metadata(range(self.env.num_envs)),
        }

    def load_state_dict(self, state):
        """Restore course progress; the caller must reset robot episodes next.

        This is episode-boundary continuation, not a PhysX/rollout snapshot.
        Matching training topology and seed prevent silently changing the course
        sequence. Frozen holdout runtimes cannot load a training namespace.
        """
        cfg = self.env.cfg
        expected = {
            "contract": COURSE_CONTRACT, "seed": cfg.course_seed, "split": cfg.course_split,
            "num_envs": self.env.num_envs, "flat_fraction": cfg.curriculum_flat_fraction,
            "robot_width": cfg.robot_width, "robot_height": cfg.robot_height,
        }
        if not isinstance(state, dict) or any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("Saved curriculum course configuration differs from this runtime")
        counters = state.get("episode_counters")
        if (not isinstance(counters, list) or len(counters) != self.env.num_envs
                or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counters)):
            raise ValueError("Saved course episode counters are invalid")
        descriptors = state.get("current_courses")
        if not isinstance(descriptors, list):
            raise ValueError("Saved current course descriptors are invalid")
        indices, courses, episodes = [], [], []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ValueError("Saved current course descriptors are invalid")
            i, episode = descriptor.get("environment_index"), descriptor.get("episode_index")
            if (isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.env.num_envs
                    or i in indices or isinstance(episode, bool) or not isinstance(episode, Integral)
                    or episode < 0 or episode >= counters[i]):
                raise ValueError("Saved current course indices are invalid")
            family = descriptor.get("family")
            if family not in FAMILIES or family != self._family_for(i, episode):
                raise ValueError("Saved course family disagrees with the replay schedule")
            course = sample_course(
                seed=cfg.course_seed, split=cfg.course_split, env_id=i, episode=episode,
                family=family, difficulty=descriptor.get("difficulty"),
                robot_width=cfg.robot_width, robot_height=cfg.robot_height,
            )
            if any(descriptor.get(name) != getattr(course, name) for name in ("course_id", "course_seed", "split")):
                raise ValueError("Saved course identity does not match its generation parameters")
            indices.append(int(i)); courses.append(course); episodes.append(int(episode))
        if len(indices) != sum(value > 0 for value in counters):
            raise ValueError("Saved curriculum is missing an active course")
        self.controller.load_state_dict(state.get("controller"))
        self.episode_counters = [int(value) for value in counters]
        self._apply_courses(indices, courses)
        for i, episode in zip(indices, episodes):
            self._current_episode_indices[i] = episode

    def _world_box_centers(self):
        return self._centers + self.env.scene.env_origins[:, None, :]

    def surface_heights(self, positions_w):
        origins = self.env.scene.env_origins
        return sample_box_heights(
            positions_w[..., :2], self._world_box_centers()[:, self._terrain_indices],
            self._halves[self._terrain_indices], default_height=origins[:, 2:3].expand(positions_w.shape[:2]),
        )

    def _lidar_observations(self, positions_w, quaternion_w, centers_w):
        # Parked pool boxes are physical bodies below the ground plane. They
        # cannot be visible through that opaque floor; exclude them explicitly
        # from synthetic rays, including rays with a downward L2 angle.
        visible = centers_w.clone()
        visible[..., 2] = torch.where(self._active_boxes, visible[..., 2], visible.new_full((), -1e6))
        return super()._lidar_observations(positions_w, quaternion_w, visible)

    def physics_step(self):
        self.time += self.env.physics_dt

    def collect_impacts(self):
        pass  # This course pool contains static obstacles only.
