"""MG90S locomotion on the actual six-leg CAD asset: 18 joints / 68 observations."""
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
    mass_assumption = 'CAD6 link estimates retained: 2.31362 kg; hardware mass unmeasured'


@configclass
class MG90SCad6RunnerCfg(MG90SWalkRunnerCfg):
    experiment_name = 'dksh_mg90s_cad6_walk'
    run_name = 'cad6_4v8'


@configclass
class MG90SCad6SprintEnvCfg(MG90SCad6EnvCfg):
    """A higher-cadence profile kept within the MG90S no-load speed envelope."""
    gait_period = 1.5
    gait_lift = 0.005
    gait_stride_limit = 0.025
    command_speed_min = 0.012
    command_speed_max = 0.020
    target_rate_limit = 6.0


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
        feet = gait.foot_targets(
            self._phase(), self._speed, self.cfg.gait_period, self.cfg.gait_lift, self.cfg.gait_stride_limit
        )
        return gait.inverse_kinematics(feet).flatten(1)

    def _get_observations(self):
        if self.sim.has_gui():
            from .spider_navigation_env import SpiderNavigationEnv
            SpiderNavigationEnv._sync_cad_visual_for_gui(self)
        return super()._get_observations()
