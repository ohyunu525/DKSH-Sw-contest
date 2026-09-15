# MG90S 8-leg 4.8 V selected policy

`model_1999.pt` was trained with `Isaac-DKSH-MG90S-Walk-Direct-v0` on 2026-09-15.

- 32 environments, seed 42, 2,000 PPO iterations, 3,072,000 transitions.
- 86 observations and 24 residual joint-angle actions.
- 4.8 V MG90S model: 0.13239 N·m provisional effort cap and 10.47198 rad/s no-load speed.
- Primitive eight-leg USD mass/inertia is retained at 2.91816 kg because measured robot mass is not yet available.
- SHA-256: `EF1B58921216314A1DDB6353A6DCA3110C80202876632C53E7DF11FE9F998316`.

Use only with the matching MG90S task and configuration in this repository. It is a
simulation policy, not evidence of safe real-hardware operation.
