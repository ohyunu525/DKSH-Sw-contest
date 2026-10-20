# MG996R hip + RC920DMG leg servo model

The `Bugfix/MLForLegs-MG996RHip` branch was compared with the current branch and
`Develop/MLForLegs`. `Develop/MLForLegs` still modeled the six body-coxa joints
as MG90S; the correction branch changes those six actuators to MG996R and keeps
RC920DMG on the twelve femur/tibia actuators. The current CAD6 direct-angle task
uses that mixed contract under a new task ID. Old all-MG90S tasks and checkpoints
remain available as historical experiments.

## Actuator assumptions

| Joint group | Model | Reference voltage | Stall torque | No-load speed | Model effort cap |
|---|---|---:|---:|---:|---:|
| 6 hip/body-coxa joints | MG996R | 4.8 V | 9.4 kgf·cm = 0.92183 N·m | 0.19 s/60° = 5.51157 rad/s | 0.69137 N·m |
| 12 femur/tibia joints | RC920DMG | 5.0 V | 19 kgf·cm = 1.86326 N·m | 0.16 s/60° = 6.54498 rad/s | 1.39745 N·m |

The MG996R values are nominal [TowerPro manufacturer specifications](https://towerpro.com.tw/product/mg996r/), not bench results.
The product page lists 0.19 s/60° in the Specification section and 0.17 s in an
older description; this model selects the slower value. RC920DMG data is
provisional seller-supplied data without a manufacturer datasheet in this
repository. The 75% effort caps are temporary simulation deratings, not
continuous-torque ratings. Actual supply voltage, load curve, continuous torque,
heat, backlash, and voltage sag remain unmeasured.

The simulator retains its existing joint PD gains, armature and friction
assumptions. The CAD6 direct-angle controller limits every target to 1.5 rad/s;
the per-joint no-load speed is used to normalize observations and velocity
costs. Actual joint range/PWM calibration is still required before deployment.

Known nominal servo mass is 55 g per MG996R and 60 g per RC920DMG, totaling
1.05 kg across six legs. The existing CAD link mass/inertia is retained because
the CAD mass allocation does not say whether servo mass is already included.
Thus the simulation is not a measured full-robot mass model.

## Current entry point

```powershell
.\run_mg90s.ps1 -Mode check -NumEnvs 4
.\run_mg90s.ps1 -Mode train -NumEnvs 4 -MaxIterations 400
```

These direct-angle invocations select
`Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v1`. The task ID and actuator contract
prevent resuming old MG90S or other-motor checkpoints. The old script name is
kept for compatibility; `run_mg90s.ps1 -ControlMode legacy` remains a historical
all-MG90S gait experiment.
