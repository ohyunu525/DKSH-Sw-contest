"""Register the DKSH spiderbot navigation task."""

import gymnasium as gym

# v2 includes filter velocity and terrain heights and uses signed progress.
# Separate IDs prevent reuse of incompatible v1 policies or v0 gait residuals.
for robot, prefix, config in (
    ("baseline", "Spider", "JointAngleNavigationEnvCfg"),
    ("cad8", "Spider-CAD8", "Cad8JointAngleEnvCfg"),
    ("cad6", "Spider-CAD6", "Cad6JointAngleEnvCfg"),
):
    gym.register(
        id=f"Isaac-DKSH-{prefix}-JointAngles-v2",
        entry_point=f"{__name__}.joint_angle_env:JointAngleNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_angle_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.joint_angle_env_cfg:JointAnglePPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2",
    entry_point=f"{__name__}.joint_angle_env:JointAngleNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_angle_env_cfg:MG996RHipCad6JointAngleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.joint_angle_env_cfg:JointAnglePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DKSH-Spider-Navigation-Direct-v0",
    entry_point=f"{__name__}.spider_navigation_env:SpiderNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spider_navigation_env_cfg:SpiderNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SpiderNavigationPPORunnerCfg",
    },
)

for leg_count in (8, 6):
    gym.register(
        id=f'Isaac-DKSH-Spider-CAD{leg_count}-Navigation-Direct-v0',
        entry_point=f'{__name__}.spider_navigation_env:SpiderNavigationEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.spider_navigation_env_cfg:SpiderCad{leg_count}NavigationEnvCfg',
            'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:SpiderCad{leg_count}PPORunnerCfg',
        },
    )
