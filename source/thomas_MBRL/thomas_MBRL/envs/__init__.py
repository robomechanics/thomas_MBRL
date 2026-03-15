# source/thomas_MBRL/thomas_MBRL/envs/__init__.py

import gymnasium as gym

gym.register(
    id="Thomas-Go2-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thomas_MBRL.envs.go2_velocity_env_cfg:ThomasGo2VelocityEnvCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents:skrl_flat_ppo_cfg.yaml",
    },
)