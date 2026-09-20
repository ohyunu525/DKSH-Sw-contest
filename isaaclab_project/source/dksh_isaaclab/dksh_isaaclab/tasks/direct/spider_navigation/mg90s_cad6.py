"""MG90S locomotion on the actual six-leg CAD asset: 18 joints / 68 observations."""
import torch

from isaaclab.utils import configclass
from dksh_isaaclab.assets.spiderbot import cad_spiderbot_cfg
from .mg90s_env import MG90SWalkEnv
from .mg90s_env_cfg import MG90SWalkEnvCfg, MG90SWalkRunnerCfg, mg90s_robot_cfg
from . import cad6_wave_gait as gait


def robot_cfg():
    robot = cad_spiderbot_cfg(6)
    robot.actuators = mg90s_robot_cfg().actuators
    robot.init_state.pos = (0, 0, -gait.STANCE_Z + 0.016)
    robot.init_state.joint_pos = {'.*_hip_joint': 0., '.*_femur_joint': gait.STANCE_FEMUR,
                                  '.*_tibia_joint': gait.STANCE_TIBIA}
    return robot.replace(prim_path='/World/envs/env_.*/Robot')


@configclass
class MG90SCad6EnvCfg(MG90SWalkEnvCfg):
    robot = robot_cfg()
    action_space = 18
    observation_space = 68
    minimum_height = 0.075
    gait_period = 2.4
    gait_lift = 0.004
    gait_stride_limit = 0.012
    command_speed_min = 0.003
    command_speed_max = 0.006
    tripod_transition_speed = 0.030
    tripod_transition_width = 0.005
    tripod_enabled = False
    mass_assumption = 'CAD6 links 2.31362 kg + L1 RM 0.230 kg; remaining hardware mass unmeasured'


@configclass
class MG90SCad6RunnerCfg(MG90SWalkRunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_walk'
    run_name = 'cad6_4v8'


@configclass
class MG90SCad6SprintEnvCfg(MG90SCad6EnvCfg):
    """A higher-cadence profile kept within the MG90S no-load speed envelope."""
    # Explore a substantially faster command range without increasing the
    # already reach-limited 25 mm stride. At 0.050 m/s the 0.6 s wave cycle
    # produces exactly 25 mm of stance travel (speed * period * 5/6).
    gait_period = 0.6
    gait_lift = 0.005
    gait_stride_limit = 0.025
    command_speed_min = 0.012
    command_speed_max = 0.050
    # Leave a small margin below the modeled 10.47 rad/s no-load speed.
    target_rate_limit = 10.0
    # Low speeds retain wave gait stability. Blend to alternating tripods over
    # 0.025--0.035 m/s to make the high command range physically meaningful.
    tripod_enabled = True
    # Sprint must follow the sampled command instead of maximizing forward
    # velocity regardless of overshoot. Keep the safety penalties unchanged.
    speed_tracking_reward_scale = 6.0
    speed_tracking_sigma = 0.006
    forward_progress_reward_scale = 30.0


@configclass
class MG90SCad6SprintRunnerCfg(MG90SCad6RunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_sprint'
    run_name = 'cad6_4v8_sprint'


class MG90SCad6Env(MG90SWalkEnv):
    cfg: MG90SCad6EnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self._robot.num_joints == 18 and self._robot.num_bodies == 19
        assert len(self._joint_ids) == 18
        self._viewer_visuals = None

    def _gait_joint_positions(self):
        feet = gait.blended_foot_targets(
            self._phase(), self._speed, self.cfg.gait_period, self.cfg.gait_lift,
            self.cfg.gait_stride_limit, self._tripod_weight(),
        )
        return gait.inverse_kinematics(feet).flatten(1)

    def _tripod_weight(self):
        if not self.cfg.tripod_enabled:
            return torch.zeros_like(self._speed)
        width = self.cfg.tripod_transition_width
        if width <= 0:
            return (self._speed >= self.cfg.tripod_transition_speed).float()
        blend = ((self._speed - (self.cfg.tripod_transition_speed - width)) / (2 * width)).clamp(0, 1)
        return blend.square() * (3 - 2 * blend)

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras["log"]["Metrics/tripod_blend"] = self._tripod_weight().mean()
        return reward

    def _get_observations(self):
        if self.sim.has_gui():
            from .spider_navigation_env import SpiderNavigationEnv
            SpiderNavigationEnv._sync_cad_visual_for_gui(self)
        return super()._get_observations()
