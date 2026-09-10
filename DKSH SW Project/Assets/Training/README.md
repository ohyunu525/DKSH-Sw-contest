# Spider Navigation ML-Agents

This training setup intentionally separates navigation policy from locomotion physics.
`SpiderNavigationAgent` outputs local strafe, yaw, and forward commands through
`ISpiderLocomotionController`. The current `SpiderProxyLocomotion` uses a
`CharacterController`; replace it with an articulated 24-servo controller later without
changing observations, rewards, or the PPO behavior name.

## Provided hardware values encoded in the profile

- 8 legs assumed for the spider platform
- 3 x 55 g MG996R servos, 3 x 2.4 g bearings, and 130.07 g printed parts per leg
- 302.27 g known mass per leg; 2.41816 kg for all eight legs, excluding the body/electronics
- Hip 86.17 mm, femur 100 mm, tibia 120 mm
- 6 V, 10 kg-cm stall torque, 180 degree range, 50 Hz commands
- Stair curriculum riser is constrained to 140-160 mm for the proxy phase

## Training

From the Unity project directory, with the matching ML-Agents Python package installed:

```powershell
mlagents-learn Assets/Training/Configs/spider_navigation_ppo.yaml --run-id=spider-nav-v1
```

Open `Assets/Scenes/SpiderNavigationTraining.unity`, press Play, and let the Python
trainer connect. The curriculum advances Flat -> Stairs -> Hills -> Buildings.

## Sim-to-real boundary

The glTF preserves the assembled appearance but has no joint hierarchy, joint axes,
inertias, or collision shapes. Before motor-level transfer, add an articulated prefab,
measure body/electronics mass and each link center of mass, then randomize friction,
servo delay, torque, battery voltage, contact compliance, and sensor noise.
