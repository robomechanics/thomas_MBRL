# source/thomas_MBRL/thomas_MBRL/envs/__init__.py

import gymnasium as gym

gym.register(
    id="Thomas-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # IMPORTANT: this must be a STRING entry point, same style as IsaacLab
        "env_cfg_entry_point": "thomas_MBRL.envs.go2_velocity_env_cfg:ThomasGo2VelocityEnvCfg",
    },
)