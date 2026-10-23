"""Actuator configurations for the current mixed-servo CAD6 robot.

MG996R drives the six hip joints at its nominal 4.8 V specification.
RC920DMG drives the twelve femur/tibia joints at its provisional 5 V specification.
The shared 75% torque cap and controller gains remain simulation assumptions.
"""

from isaaclab.actuators import DCMotorCfg

from .servo_specs import (
    MG996R_NO_LOAD_SPEED_RAD_S,
    MG996R_STALL_TORQUE_NM,
    PROVISIONAL_EFFORT_DERATING,
    RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
    RC920DMG_STALL_TORQUE_NM_5V,
)


def mixed_servo_actuators():
    """Return fresh motor configurations without changing robot geometry or pose."""
    return {
        "mg996r_4v8_hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            saturation_effort=MG996R_STALL_TORQUE_NM,
            effort_limit=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            effort_limit_sim=PROVISIONAL_EFFORT_DERATING * MG996R_STALL_TORQUE_NM,
            velocity_limit=MG996R_NO_LOAD_SPEED_RAD_S,
            velocity_limit_sim=MG996R_NO_LOAD_SPEED_RAD_S,
            stiffness=3.0, damping=0.06, armature=0.0002, friction=0.0,
        ),
        "rc920dmg_5v_leg": DCMotorCfg(
            joint_names_expr=[".*_femur_joint", ".*_tibia_joint"],
            saturation_effort=RC920DMG_STALL_TORQUE_NM_5V,
            effort_limit=PROVISIONAL_EFFORT_DERATING * RC920DMG_STALL_TORQUE_NM_5V,
            effort_limit_sim=PROVISIONAL_EFFORT_DERATING * RC920DMG_STALL_TORQUE_NM_5V,
            velocity_limit=RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
            velocity_limit_sim=RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
            stiffness=3.0, damping=0.06, armature=0.0002, friction=0.0,
        ),
    }
