# MG90S 8-leg 4.8 V selected policy

> **Retired archive.** The matching MG90S task and launcher have been removed.
> These binaries and results are retained as historical evidence, with no active inference or training path.
> See the [archive note](../README.md) and [current CAD6 training guide](../../docs/joint_angle_training.md).

> Superseded physical configuration: this policy predates the 2026-09-22
> RC920DMG replacement at the coxa-femur and femur-tibia joints. Retrain and
> revalidate it before use with the mixed-servo model.

**Historical eight-leg experiment; not compatible with the user's six-leg robot.**
The current CAD6 launcher rejects this checkpoint; no MG90S task remains registered.

`model_1999.pt` was trained with `Isaac-DKSH-MG90S-Walk-Direct-v0` on 2026-09-15.

- 32 environments, seed 42, 2,000 PPO iterations, 3,072,000 transitions.
- 86 observations and 24 residual joint-angle actions.
- 4.8 V MG90S model: 0.13239 N·m provisional effort cap and 10.47198 rad/s no-load speed.
- Primitive eight-leg USD mass/inertia is retained at 2.91816 kg because measured robot mass is not yet available.
- SHA-256: `EF1B58921216314A1DDB6353A6DCA3110C80202876632C53E7DF11FE9F998316`.

The original task configuration is available only in Git history. This simulation result is
not a validation of the current robot.
