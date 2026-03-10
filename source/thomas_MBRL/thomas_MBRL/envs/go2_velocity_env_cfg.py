# source/thomas_MBRL/thomas_MBRL/envs/go2_velocity_env_cfg.py

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


class ThomasGo2VelocityEnvCfg(UnitreeGo2FlatEnvCfg):
    """Thomas custom Go2 velocity task config."""

    def __post_init__(self):
        super().__post_init__()

        # number of parallel envs (was num_envs=... in gym.make before)
        self.scene.num_envs = 65

        # device (was device="cuda" in gym.make before)
        self.sim.device = "cuda"