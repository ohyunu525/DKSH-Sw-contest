"""Physical configuration for the eight-legged DKSH MG996R spiderbot."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from .servo_specs import (
    MG996R_NO_LOAD_SPEED_RAD_S,
    MG996R_STALL_TORQUE_NM,
    PROVISIONAL_EFFORT_DERATING,
)


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
        "mg996r_4v8": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            saturation_effort=MG996R_STALL_TORQUE_NM,
            effort_limit=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            effort_limit_sim=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            velocity_limit=MG996R_NO_LOAD_SPEED_RAD_S,
            velocity_limit_sim=MG996R_NO_LOAD_SPEED_RAD_S,
            stiffness=3.0,
            damping=0.06,
            armature=0.0002,
            friction=0.0,
        )
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Eight-leg, 24-DoF articulation using nominal MG996R limits."""


def cad_spiderbot_cfg(leg_count: int) -> ArticulationCfg:
    """CAD variants have their own assembled zero pose and shorter stance."""
    if leg_count not in (6, 8):
        raise ValueError('CAD spiderbots support 6 or 8 legs')
    cfg = SPIDERBOT_CFG.copy()
    if leg_count == 6:
        cfg.spawn.usd_path = str(
            _PROJECT_ROOT / 'assets' / 'spiderbot_variants' / 'onshape_20260928' / 'onshape_20260928.usd'
        )
        # q=0 is the saved Onshape stance. Lowest sole is z=-0.122441 m.
        cfg.init_state.pos = (0.0, 0.0, 0.1275)
    else:
        cfg.spawn.usd_path = str(
            _PROJECT_ROOT / 'assets' / 'spiderbot_variants' / f'spiderbot_{leg_count}leg'
            / f'spiderbot_{leg_count}leg.usd'
        )
        cfg.init_state.pos = (0.0, 0.0, 0.12004)
    cfg.init_state.joint_pos = {'.*_joint': 0.0}
    return cfg
