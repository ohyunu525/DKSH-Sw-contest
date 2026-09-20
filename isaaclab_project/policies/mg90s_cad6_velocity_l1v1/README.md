# MG90S CAD6 velocity policy with L1 RM payload

Selected low-level locomotion policy for the hierarchical deployment path:

`Unitree L1 RM -> ROS2 SLAM/Nav2 -> /cmd_vel -> CAD6 velocity policy`

The policy was trained from scratch with
`Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0` on 2026-09-20.

- 32 environments, seed 42, 2,000 PPO iterations, 3,072,000 transitions.
- 68 observations and 18 residual joint-angle actions.
- Commands: body-frame `vx`, `vy`, and `wz`; LiDAR is consumed by the upper ROS2 layer.
- CAD6 mass with L1 RM payload: 2.54362035 kg.
- Training episodes: 3,072 timeouts, 0 falls, 0 out-of-bounds terminations.
- 60-second evaluation: 12 episodes, 0 falls, 0 out-of-bounds terminations.
- Evaluation planar velocity error: 0.00598743 m/s.
- Evaluation yaw-rate error: 0.03981887 rad/s.
- Evaluation maximum absolute torque: 0.13238977 N·m.

Artifacts:

- `model_1999.pt`: resumable RSL-RL checkpoint; SHA-256
  `c660a23ce256bc79ea5739684ee24dd4fca9dbeee5c16ea7e9b73e83d695b03b`.
- `exported/policy.onnx`: Raspberry Pi inference model; SHA-256
  `04d9de56c93713b8f66d02284c9d586bb1f2d7af81d89221ecf7928be8ded84a`.
- `exported/policy.pt`: TorchScript inference model; SHA-256
  `1fb014979aafedb15c09c60cd12a8de425b7c7add63c8a67ff2ea32d07537092`.

This is a simulation policy. Hardware deployment still requires measured joint ordering,
state estimation, servo calibration, command timeout handling, and an emergency stop.
