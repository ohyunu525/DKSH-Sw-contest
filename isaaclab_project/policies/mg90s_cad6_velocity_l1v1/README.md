# MG90S CAD6 velocity policy with L1 RM payload

Selected low-level locomotion policy for the hierarchical deployment path:

`Unitree L1 RM -> ROS2 SLAM/Nav2 -> /cmd_vel -> CAD6 velocity policy`

The policy was trained from scratch with
`Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0` on 2026-09-22.

- 32 environments, seed 42, 2,000 PPO iterations, 3,072,000 transitions.
- 68 observations and 18 residual joint-angle actions.
- Commands: body-frame `vx`, `vy`, and `wz`; LiDAR is consumed by the upper ROS2 layer.
- Mixed actuators: MG90S body-coxa joints and RC920DMG coxa-femur/femur-tibia joints.
- CAD6 mass with L1 RM payload: 2.54362035 kg.
- Training episodes: 3,072 timeouts, 0 falls, 0 out-of-bounds terminations.
- 60-second evaluation: 12 episodes, 0 falls, 0 out-of-bounds terminations.
- Evaluation planar velocity error: 0.00600232 m/s.
- Evaluation yaw-rate error: 0.03714342 rad/s.
- Evaluation maximum absolute torque: 0.67748183 N·m; maximum per-joint effort ratio: 1.0.

Artifacts:

- `model_1999.pt`: resumable RSL-RL checkpoint; SHA-256
  `d6604a43fe5957c8d2d479aa9fe38a4c5d2070b449d1b5544fc36aebb961d184`.
- `exported/policy.onnx`: Raspberry Pi inference model; SHA-256
  `c67c37090a58a4e5c591c3c821e5e6473d108c2f1587890ee7fc3b19e7752282`.
- `exported/policy.pt`: TorchScript inference model; SHA-256
  `2deb0edc3b0c52aea07e4d0439640bf98a3bffd3b01d4a8586d4c3ebc1a6735c`.

This is a simulation policy. Hardware deployment still requires measured joint ordering,
state estimation, servo calibration, command timeout handling, and an emergency stop.
