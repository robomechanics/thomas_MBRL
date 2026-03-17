"""
test_exported_policy.py

Run the exported TorchScript Go2 walking policy directly in Isaac Lab with a
fixed base-velocity command, without MPPI.

This is used to verify that the exported policy itself can walk forward cleanly.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "thomas_MBRL")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True, help="Path to exported TorchScript policy.")
    parser.add_argument("--vx", type=float, default=0.4, help="Fixed forward velocity command.")
    parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral velocity command.")
    parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw-rate command.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps to run.")
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N steps.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )
    from thomas_MBRL.controllers.torch_policy_wrapper import TorchVelocityPolicy

    device = "cuda"

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = device

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    obs, _ = env.reset()

    policy = TorchVelocityPolicy(args.policy_path, device)

    robot = env.unwrapped.scene["robot"]

    cmd = torch.tensor([[args.vx, args.vy, args.wz]], device=device, dtype=torch.float32)

    print("Starting exported-policy test...")
    print(f"Fixed command: vx={args.vx}, vy={args.vy}, wz={args.wz}")

    for step in range(args.steps):
        # Force the same command every step.
        cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")
        cmd_buf[0, 0] = cmd[0, 0]
        cmd_buf[0, 1] = cmd[0, 1]
        cmd_buf[0, 2] = cmd[0, 2]

        # Recompute obs if available, otherwise use last obs.
        try:
            fresh_obs = env.unwrapped.observation_manager.compute()
        except Exception:
            fresh_obs = obs

        # Run exported TorchScript policy directly.
        actions = policy(fresh_obs)

        obs, reward, terminated, truncated, info = env.step(actions)

        if step % args.print_every == 0 or step == args.steps - 1:
            base_vel = robot.data.root_state_w[0, 7:10]
            base_pos = robot.data.root_state_w[0, 0:3]
            proj_grav = robot.data.projected_gravity_b[0]
            vel_err = base_vel - cmd[0]

            print(f"\nStep {step}")
            print("terminated:", bool(terminated[0].item()))
            print("truncated:", bool(truncated[0].item()))
            print("commanded vel:", cmd[0].detach().cpu().numpy())
            print("actual vel:", base_vel.detach().cpu().numpy())
            print("vel error:", vel_err.detach().cpu().numpy())
            print("pos:", base_pos.detach().cpu().numpy())
            print("projected gravity:", proj_grav.detach().cpu().numpy())

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            print(f"\nEpisode ended at step {step}.")
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()