import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export built-in Isaac Lab Go2 skrl policy to TorchScript.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained skrl checkpoint.")
parser.add_argument("--output", type=str, required=True, help="Output TorchScript .pt path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))

    policy = runner.agent.models["policy"]
    policy.eval()

    obs, _ = env.reset()

    if isinstance(obs, dict):
        if "policy" in obs:
            example_obs = obs["policy"]
        elif "obs" in obs:
            example_obs = obs["obs"]
        else:
            example_obs = next(v for v in obs.values() if isinstance(v, torch.Tensor))
    else:
        example_obs = obs

    example_obs = example_obs.to(env.device)

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy_model):
            super().__init__()
            self.policy_model = policy_model

        def forward(self, obs_tensor):
            net = self.policy_model.net_container(obs_tensor)
            action_mean = self.policy_model.policy_layer(net)
            return action_mean

    wrapped_policy = PolicyWrapper(policy).to(env.device)
    traced = torch.jit.trace(wrapped_policy, example_obs)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    traced.save(args_cli.output)

    print(f"Saved TorchScript policy to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()