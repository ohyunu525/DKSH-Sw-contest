"""Pure decision rule for CAD6 body-twist evaluation."""


def velocity_pass(result):
    """Require useful tracking on both axes relative to a stationary robot.

    The zero-velocity baseline is measured on the exact sampled command
    sequence, including stand commands and any gait-envelope projection.
    """
    planar_baseline = result['zero_speed_error_mps']
    yaw_baseline = result['zero_yaw_error_rps']
    return (
        result['falls'] == 0
        and result['out_of_bounds'] == 0
        and result['max_effort_ratio'] <= 1.0001
        and planar_baseline >= 0.004
        and yaw_baseline >= 0.020
        and result['mean_speed_error_mps'] <= min(0.008, 0.8 * planar_baseline)
        and result['mean_yaw_error_rps'] <= min(0.08, 0.8 * yaw_baseline)
    )
