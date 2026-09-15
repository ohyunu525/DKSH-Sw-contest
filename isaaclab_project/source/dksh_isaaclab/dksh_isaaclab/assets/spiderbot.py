"""Physical configuration for the eight-legged DKSH spiderbot."""

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_SPIDERBOT_USD = _PROJECT_ROOT / "assets" / "spiderbot_usd" / "spiderbot.usd"
_SPIDERBOT_USD = Path(os.environ.get("DKSH_SPIDERBOT_USD", _DEFAULT_SPIDERBOT_USD)).resolve()


SPIDERBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_SPIDERBOT_USD),
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.05,
            angular_damping=0.05,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.19),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_femur_joint": 0.65,
            ".*_tibia_joint": 0.75,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "mg996r_servos": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=0.980665,
            velocity_limit_sim=math.radians(400.0),
            stiffness=8.0,
            damping=0.35,
            armature=0.002,
        )
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Eight-leg, 24-DoF articulation using measured MG996R constraints."""


def cad_spiderbot_cfg(leg_count: int) -> ArticulationCfg:
    """CAD variants have their own assembled zero pose and shorter stance."""
    if leg_count not in (6, 8):
        raise ValueError('CAD spiderbots support 6 or 8 legs')
    cfg = SPIDERBOT_CFG.copy()
    cfg.spawn.usd_path = str(
        _PROJECT_ROOT / 'assets' / 'spiderbot_variants' / f'spiderbot_{leg_count}leg'
        / f'spiderbot_{leg_count}leg.usd'
    )
    cfg.init_state.pos = (0.0, 0.0, 0.12004)
    cfg.init_state.joint_pos = {'.*_joint': 0.0}
    return cfg
