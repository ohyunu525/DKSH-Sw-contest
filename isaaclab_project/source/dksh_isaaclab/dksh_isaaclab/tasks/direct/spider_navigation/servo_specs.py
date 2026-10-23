"""Servo specifications used by the spiderbot actuator models.

MG996R uses the TowerPro product Specification section (checked 2026-10-19).
RC920DMG still uses provisional seller-supplied no-load/stall data and has no
manufacturer datasheet or bench curve in this repository.  The 75% effort
limit used by the simulator is a separate provisional derating, not a
manufacturer continuous-torque rating.
"""

import math


KGF_CM_TO_NM = 0.0980665
PROVISIONAL_EFFORT_DERATING = 0.75
ACTUATOR_CONTRACT = 'mg996r_4v8_rc920dmg_5v_v1'

MG996R_SPEC_SOURCE = 'https://towerpro.com.tw/product/mg996r/'
MG996R_SPEC_VERIFICATION_STATUS = 'manufacturer_nominal_not_bench_verified'
MG996R_REFERENCE_VOLTAGE_V = 4.8
MG996R_STALL_TORQUE_KGF_CM = 9.4
# The page's Specification says 0.19 s; an older description table says 0.17 s.
# Use the slower value; neither is a measured loaded joint speed.
MG996R_SECONDS_PER_60_DEGREES = 0.19
MG996R_MASS_KG = 0.055
MG996R_SIZE_M = (0.0407, 0.0197, 0.0429)
MG996R_STALL_TORQUE_NM = MG996R_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM
MG996R_NO_LOAD_SPEED_RAD_S = math.radians(60.0) / MG996R_SECONDS_PER_60_DEGREES

RC920DMG_OPERATING_VOLTAGE_RANGE_V = (4.8, 8.4)
RC920DMG_REFERENCE_VOLTAGE_V = 5.0
RC920DMG_STALL_TORQUE_KGF_CM_5V = 19.0
RC920DMG_STALL_TORQUE_KGF_CM_7V4 = 21.0
RC920DMG_SECONDS_PER_60_DEGREES_5V = 0.16
RC920DMG_SECONDS_PER_60_DEGREES_7V4 = 0.14
RC920DMG_MASS_KG = 0.060
RC920DMG_SIZE_M = (0.040, 0.0205, 0.0405)
RC920DMG_BEARING_COUNT = 2
RC920DMG_CABLE_LENGTH_M = 0.32
RC920DMG_SPEC_VERIFICATION_STATUS = "provisional_unverified"
RC920DMG_STALL_TORQUE_NM_5V = RC920DMG_STALL_TORQUE_KGF_CM_5V * KGF_CM_TO_NM
RC920DMG_STALL_TORQUE_NM_7V4 = RC920DMG_STALL_TORQUE_KGF_CM_7V4 * KGF_CM_TO_NM
RC920DMG_NO_LOAD_SPEED_RAD_S_5V = (
    math.radians(60.0) / RC920DMG_SECONDS_PER_60_DEGREES_5V
)
RC920DMG_NO_LOAD_SPEED_RAD_S_7V4 = (
    math.radians(60.0) / RC920DMG_SECONDS_PER_60_DEGREES_7V4
)
