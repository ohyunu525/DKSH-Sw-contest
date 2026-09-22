"""Register the DKSH spiderbot navigation task."""

import gymnasium as gym

gym.register(
    id="Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v1",
    entry_point=f"{__name__}.mg90s_cad6:MG90SCad6VelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6VelocityV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6VelocityV1RunnerCfg",
    },
)

gym.register(
    id="Isaac-DKSH-MG90S-CAD6-Walk-Direct-v0",
    entry_point=f"{__name__}.mg90s_cad6:MG90SCad6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6RunnerCfg",
    },
)

gym.register(
    id="Isaac-DKSH-MG90S-CAD6-Sprint-Direct-v0",
    entry_point=f"{__name__}.mg90s_cad6:MG90SCad6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6SprintEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6SprintRunnerCfg",
    },
)

gym.register(
    id="Isaac-DKSH-MG90S-CAD6-Velocity-Direct-v0",
    entry_point=f"{__name__}.mg90s_cad6:MG90SCad6VelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6VelocityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mg90s_cad6:MG90SCad6VelocityRunnerCfg",
    },
)

gym.register(
    id="Isaac-DKSH-MG90S-Walk-Direct-v0",
    entry_point=f"{__name__}.mg90s_env:MG90SWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mg90s_env_cfg:MG90SWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mg90s_env_cfg:MG90SWalkRunnerCfg",
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
