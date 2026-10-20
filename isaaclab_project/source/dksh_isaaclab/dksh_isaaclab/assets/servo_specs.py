"""Servo data shared by actuator models and training metadata.

MG996R values use TowerPro's 4.8 V specification. RC920DMG values are
provisional seller-supplied values at 5 V and have not been bench verified.
The 75% actuator effort limit is a temporary simulation derating, not a
manufacturer continuous-torque rating.
"""

import math


KGF_CM_TO_NM = 0.0980665
PROVISIONAL_EFFORT_DERATING = 0.75
ACTUATOR_CONTRACT = "mg996r_4v8_rc920dmg_5v_v1"

MG996R_SPEC_SOURCE = "https://towerpro.com.tw/product/mg996r/"
MG996R_SPEC_VERIFICATION_STATUS = "manufacturer_nominal_not_bench_verified"
MG996R_REFERENCE_VOLTAGE_V = 4.8
MG996R_STALL_TORQUE_KGF_CM = 9.4
# The specification lists 0.19 s; an older description lists 0.17 s.
MG996R_SECONDS_PER_60_DEGREES = 0.19
MG996R_MASS_KG = 0.055
MG996R_SIZE_M = (0.0407, 0.0197, 0.0429)
MG996R_STALL_TORQUE_NM = MG996R_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM
MG996R_NO_LOAD_SPEED_RAD_S = math.radians(60.0) / MG996R_SECONDS_PER_60_DEGREES

# RC920DMG is modeled at its conservative 5 V seller-supplied specification.
RC920DMG_REFERENCE_VOLTAGE_V = 5.0
RC920DMG_STALL_TORQUE_KGF_CM_5V = 19.0
RC920DMG_SECONDS_PER_60_DEGREES_5V = 0.16
RC920DMG_MASS_KG = 0.060
RC920DMG_SIZE_M = (0.040, 0.0205, 0.0405)
RC920DMG_SPEC_VERIFICATION_STATUS = "provisional_unverified"
RC920DMG_STALL_TORQUE_NM_5V = RC920DMG_STALL_TORQUE_KGF_CM_5V * KGF_CM_TO_NM
RC920DMG_NO_LOAD_SPEED_RAD_S_5V = math.radians(60.0) / RC920DMG_SECONDS_PER_60_DEGREES_5V

# Archived values retained to describe old all-MG90S experiments.
MG90S_REFERENCE_VOLTAGE_V = 4.8
MG90S_STALL_TORQUE_KGF_CM = 1.8
MG90S_SECONDS_PER_60_DEGREES = 0.10
MG90S_MASS_KG = 0.0134
MG90S_STALL_TORQUE_NM = MG90S_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM
MG90S_NO_LOAD_SPEED_RAD_S = math.radians(60.0) / MG90S_SECONDS_PER_60_DEGREES
