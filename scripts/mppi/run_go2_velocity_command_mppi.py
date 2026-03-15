"""
run_go2_velocity_command_mppi.py

Command-space MPPI wrapper for Isaac Lab Go2 velocity control.

This script does not optimize low-level robot actions directly. Instead, it
optimizes a residual on the commanded base velocity. A lower-level locomotion
policy then converts that command into joint actions.

The intended architecture is:

    MPPI over command residuals
        ->
    trained velocity policy
        ->
    Isaac Lab env action space
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "thomas_MBRL")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)


def _get_fresh_obs(env: Any, last_obs: Any) -> Any:
    """
    Try to recompute observations after command or state writes.

    Isaac Lab environments commonly expose an observation manager. If that path
    is unavailable in the current setup, this function falls back to the most
    recent observation returned by env.step or env.reset.

    Args:
        env: Gymnasium Isaac Lab environment.
        last_obs: Previously returned observation object.

    Returns:
        Current observation object suitable for the policy.
    """
    try:
        return env.unwrapped.observation_manager.compute()
    except Exception:
        return last_obs


def _set_base_velocity_command(env: Any, env_ids: torch.Tensor, commands: torch.Tensor) -> None:
    """
    Write base velocity commands into the command buffer for selected envs.

    Args:
        env: Gymnasium Isaac Lab environment.
        env_ids: Tensor of environment indices.
        commands: Tensor with shape (len(env_ids), 3).
    """
    cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")
    cmd_buf[env_ids, 0] = commands[:, 0]
    cmd_buf[env_ids, 1] = commands[:, 1]
    cmd_buf[env_ids, 2] = commands[:, 2]


def _clamp_commands(cmds: torch.Tensor, cmd_limit: torch.Tensor) -> torch.Tensor:
    """
    Clamp commands elementwise to symmetric limits.

    Args:
        cmds: Tensor of shape (..., 3).
        cmd_limit: Tensor of shape (3,).

    Returns:
        Clamped tensor with the same shape as cmds.
    """
    return torch.max(torch.min(cmds, cmd_limit), -cmd_limit)


def main() -> None:
    """
    Launch Isaac Lab and run command-space MPPI with a lower-level velocity policy.
    """
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to a TorchScript velocity policy checkpoint.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device to use.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of outer MPPI iterations.")
    parser.add_argument("--num-rollouts", type=int, default=32,
                        help="Number of rollout environments used by MPPI.")
    parser.add_argument("--horizon", type=int, default=20,
                        help="MPPI planning horizon.")
    parser.add_argument("--sigma-vx", type=float, default=0.05,
                        help="Exploration noise standard deviation for vx residual.")
    parser.add_argument("--sigma-vy", type=float, default=0.05,
                        help="Exploration noise standard deviation for vy residual.")
    parser.add_argument("--sigma-wz", type=float, default=0.10,
                        help="Exploration noise standard deviation for yaw-rate residual.")
    parser.add_argument("--lambda-mppi", type=float, default=1.0,
                        help="Temperature parameter for MPPI.")
    parser.add_argument("--target-vx", type=float, default=0.20,
                        help="Nominal desired forward velocity.")
    parser.add_argument("--target-vy", type=float, default=0.00,
                        help="Nominal desired lateral velocity.")
    parser.add_argument("--target-wz", type=float, default=0.00,
                        help="Nominal desired yaw rate.")
    parser.add_argument("--cmd-limit-vx", type=float, default=0.60,
                        help="Absolute clamp for commanded vx.")
    parser.add_argument("--cmd-limit-vy", type=float, default=0.40,
                        help="Absolute clamp for commanded vy.")
    parser.add_argument("--cmd-limit-wz", type=float, default=1.20,
                        help="Absolute clamp for commanded yaw rate.")
    parser.add_argument("--target-height", type=float, default=0.32,
                        help="Desired body height for stability cost.")
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from thomas_MBRL.controllers.torch_policy_wrapper import TorchVelocityPolicy
    from thomas_MBRL.envs.go2_velocity_env_cfg import ThomasGo2VelocityEnvCfg

    device = args.device
    k_rollouts = args.num_rollouts
    horizon = args.horizon

    base_cmd = torch.tensor(
        [args.target_vx, args.target_vy, args.target_wz],
        device=device,
        dtype=torch.float32,
    )
    cmd_limit = torch.tensor(
        [args.cmd_limit_vx, args.cmd_limit_vy, args.cmd_limit_wz],
        device=device,
        dtype=torch.float32,
    )
    sigma = torch.tensor(
        [args.sigma_vx, args.sigma_vy, args.sigma_wz],
        device=device,
        dtype=torch.float32,
    )

    cfg = ThomasGo2VelocityEnvCfg()
    cfg.scene.num_envs = 1 + k_rollouts
    cfg.sim.device = device

    env = gym.make("Thomas-Go2-Velocity-v0", cfg=cfg)
    obs, _ = env.reset()

    policy = TorchVelocityPolicy(args.policy_path, device)

    robot = env.unwrapped.scene["robot"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"Action dim from low-level policy/env interface: {action_dim}")
    print("Starting command-space MPPI control...")

    env0_id = torch.tensor([0], device=device, dtype=torch.long)
    rollout_env_ids = torch.arange(1, k_rollouts + 1, device=device, dtype=torch.long)

    # The optimized variable is not the env action. It is a residual on the
    # commanded base velocity.
    u_nom = torch.zeros((horizon, 3), device=device, dtype=torch.float32)

    for outer in range(args.iters):
        print(f"\n=== MPPI iteration {outer + 1} ===")

        # Set env0 command to the current commanded velocity before snapshotting.
        env0_cmd = _clamp_commands(base_cmd + u_nom[0], cmd_limit)
        _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))

        # Snapshot env0 state before running rollouts.
        root_state_0 = robot.data.root_state_w[0].clone()
        joint_pos_0 = robot.data.joint_pos[0].clone()
        joint_vel_0 = robot.data.joint_vel[0].clone()

        # Clone env0 state into the rollout environments.
        robot.write_root_pose_to_sim(root_state_0[:7].repeat(k_rollouts, 1), env_ids=rollout_env_ids)
        robot.write_root_velocity_to_sim(root_state_0[7:].repeat(k_rollouts, 1), env_ids=rollout_env_ids)
        robot.write_joint_state_to_sim(
            joint_pos_0.repeat(k_rollouts, 1),
            joint_vel_0.repeat(k_rollouts, 1),
            env_ids=rollout_env_ids,
        )

        costs = torch.zeros(k_rollouts, device=device, dtype=torch.float32)
        noise = torch.randn((k_rollouts, horizon, 3), device=device) * sigma.view(1, 1, 3)

        for t in range(horizon):
            # Build rollout commands around the nominal base command.
            cmd_residual_t = u_nom[t].unsqueeze(0) + noise[:, t]
            rollout_cmds = _clamp_commands(base_cmd.unsqueeze(0) + cmd_residual_t, cmd_limit)

            # Keep env0 on the current real command while rollouts explore.
            _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))
            _set_base_velocity_command(env, rollout_env_ids, rollout_cmds)

            # Recompute observations after updating commands, then query the policy.
            fresh_obs = _get_fresh_obs(env, obs)
            actions = policy(fresh_obs)

            obs, reward, terminated, truncated, info = env.step(actions)

            base_vel = robot.data.root_state_w[1:1 + k_rollouts, 7:10]
            base_pos = robot.data.root_state_w[1:1 + k_rollouts, 0:3]
            projected_gravity = robot.data.projected_gravity_b[1:1 + k_rollouts]
            joint_pos = robot.data.joint_pos[1:1 + k_rollouts]
            joint_vel = robot.data.joint_vel[1:1 + k_rollouts]

            vel_error = (base_vel[:, :2] - base_cmd[:2].unsqueeze(0)) ** 2
            yaw_error = (base_vel[:, 2] - base_cmd[2]) ** 2
            height_error = (base_pos[:, 2] - args.target_height) ** 2
            upright_cost = projected_gravity[:, 0] ** 2 + projected_gravity[:, 1] ** 2
            pose_cost = ((joint_pos - joint_pos_0.unsqueeze(0)) ** 2).sum(dim=1)
            vel_joint_cost = (joint_vel ** 2).sum(dim=1)
            residual_cost = ((cmd_residual_t / cmd_limit.unsqueeze(0)) ** 2).sum(dim=1)

            fall_penalty = torch.where(base_pos[:, 2] < 0.22, 50.0, 0.0)
            tilt_penalty = torch.where(upright_cost > 0.25, 25.0, 0.0)
            done_penalty = 50.0 * (
                terminated[1:1 + k_rollouts].float() + truncated[1:1 + k_rollouts].float()
            )

            step_cost = (
                8.0 * vel_error.sum(dim=1)
                + 2.0 * yaw_error
                + 2.0 * height_error
                + 2.5 * upright_cost
                + 0.15 * pose_cost
                + 0.01 * vel_joint_cost
                + 0.05 * residual_cost
                + fall_penalty
                + tilt_penalty
                + done_penalty
            )

            costs += step_cost

        weights = torch.softmax(-costs / args.lambda_mppi, dim=0)
        delta = torch.einsum("k,khd->hd", weights, noise)
        u_nom = u_nom + delta

        best_idx = torch.argmin(costs).item()
        best_cost = costs[best_idx].item()

        print(f"Best rollout cost: {best_cost:.6f}")
        print(f"Best rollout index: {best_idx}")

        # Restore env0 to its pre-rollout state so only the chosen command gets applied.
        robot.write_root_pose_to_sim(root_state_0[:7].unsqueeze(0), env_ids=env0_id)
        robot.write_root_velocity_to_sim(root_state_0[7:].unsqueeze(0), env_ids=env0_id)
        robot.write_joint_state_to_sim(
            joint_pos_0.unsqueeze(0),
            joint_vel_0.unsqueeze(0),
            env_ids=env0_id,
        )

        # Apply the chosen command to env0 and let the lower-level policy generate actions.
        env0_cmd = _clamp_commands(base_cmd + u_nom[0], cmd_limit)
        _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))

        # Keep rollout env commands equal to env0 during the real step so they remain benign.
        rollout_cmds_real = env0_cmd.unsqueeze(0).repeat(k_rollouts, 1)
        _set_base_velocity_command(env, rollout_env_ids, rollout_cmds_real)

        fresh_obs = _get_fresh_obs(env, obs)
        real_actions = policy(fresh_obs)

        obs, reward, terminated, truncated, info = env.step(real_actions)

        base_vel0 = robot.data.root_state_w[0, 7:10]
        base_pos0 = robot.data.root_state_w[0, 0:3]
        proj_grav0 = robot.data.projected_gravity_b[0]
        vel_err0 = base_vel0 - base_cmd

        print("env0 terminated:", bool(terminated[0].item()))
        print("env0 truncated:", bool(truncated[0].item()))
        print("env0 commanded vel:", env0_cmd.detach().cpu().numpy())
        print("env0 actual vel:", base_vel0.detach().cpu().numpy())
        print("env0 vel error:", vel_err0.detach().cpu().numpy())
        print("env0 pos:", base_pos0.detach().cpu().numpy())
        print("env0 projected gravity:", proj_grav0.detach().cpu().numpy())

        # Shift the nominal sequence and append zero residual at the tail.
        u_nom = torch.cat([u_nom[1:], torch.zeros_like(u_nom[:1])], dim=0)

        # Keep the residual command modest.
        residual_limit = 0.35 * cmd_limit
        u_nom = _clamp_commands(u_nom, residual_limit)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()