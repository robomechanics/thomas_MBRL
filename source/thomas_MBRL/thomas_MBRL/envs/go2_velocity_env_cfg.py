from isaaclab_tasks.locomotion.velocity.velocity_env_cfg import (
    UnitreeGo2VelocityFlatEnvCfg,
)

class MyGo2VelocityEnvCfg(UnitreeGo2VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Keep identical for now
        self.scene.num_envs = 1024

        # Example for later experiments:
        # self.rewards.lin_vel_xy = 1.5
