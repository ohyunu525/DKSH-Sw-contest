# DKSH-Sw-contest

## Isaac Lab

This workspace is configured to run Isaac Lab 2.1.0 against the locally installed
Isaac Sim 4.5.0. The matching versions matter: Isaac Sim 4.5 uses Python 3.10.
The `IsaacLab\\_isaac_sim` junction points to the existing Isaac Sim directory, so
it does not duplicate the simulator.

Run these commands from PowerShell at the repository root:

```powershell
# Needed once after cloning. The default path is the detected local Isaac Sim 4.5 install.
git submodule update --init --recursive
.\setup_isaaclab.ps1

# Finite headless startup and physics test (recommended first command).
.\run_isaaclab.ps1

# Open an empty Isaac Sim viewport. Stop with Ctrl+Break in the terminal.
.\run_isaaclab.ps1 -Mode viewer

# Start a modest Cartpole RSL-RL training run for this RTX 3070 (8 GB VRAM).
.\run_isaaclab.ps1 -Mode cartpole -NumEnvs 32
```

If Isaac Sim is installed elsewhere, pass its directory explicitly:

```powershell
.\setup_isaaclab.ps1 -IsaacSimPath "D:\IsaacSim\isaac-sim-standalone-4.5.0-windows-x86_64"
```

The original Unity project remains in `DKSH SW Project`. Its scenes and FBX assets
are not automatically Isaac Lab assets; porting the spiderbot for training requires
a URDF or USD articulation with joint limits, drives, and collision geometry.
