"""Register the DKSH spiderbot navigation task."""

import gymnasium as gym


gym.register(
    id="Isaac-DKSH-Spider-Navigation-Direct-v0",
    entry_point=f"{__name__}.spider_navigation_env:SpiderNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spider_navigation_env_cfg:SpiderNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SpiderNavigationPPORunnerCfg",
    },
)
