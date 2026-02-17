from isaaclab_tasks.locomotion.velocity.velocity_env import VelocityEnv
from .go2_velocity_env_cfg import MyGo2VelocityEnvCfg


class MyGo2VelocityEnv(VelocityEnv):
    cfg_cls = MyGo2VelocityEnvCfg
