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
    """
    try:
        return env.unwrapped.observation_manager.compute()
    except Exception:
        return last_obs


def _set_base_velocity_command(env: Any, env_ids: torch.Tensor, commands: torch.Tensor) -> None:
    """
    Write base velocity commands into the command buffer for selected envs.
    """
    cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")
    cmd_buf[env_ids, 0] = commands[:, 0]
    cmd_buf[env_ids, 1] = commands[:, 1]
    cmd_buf[env_ids, 2] = commands[:, 2]


def _clamp_commands(cmds: torch.Tensor, cmd_limit: torch.Tensor) -> torch.Tensor:
    """
    Clamp commands elementwise to symmetric limits.
    """
    return torch.max(torch.min(cmds, cmd_limit), -cmd_limit)


def _freeze_reference_robot(
    robot: Any,
    ref_env_id: torch.Tensor,
    ref_root_state: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
) -> None:
    """
    Keep the reference robot frozen at the initial state for visualization.
    """
    robot.write_root_pose_to_sim(
        ref_root_state[:7].unsqueeze(0),
        env_ids=ref_env_id,
    )
    robot.write_root_velocity_to_sim(
        torch.zeros_like(ref_root_state[7:]).unsqueeze(0),
        env_ids=ref_env_id,
    )
    robot.write_joint_state_to_sim(
        ref_joint_pos.unsqueeze(0),
        torch.zeros_like(ref_joint_vel).unsqueeze(0),
        env_ids=ref_env_id,
    )


def main() -> None:
    """
    Launch Isaac Lab and run command-space MPPI with a lower-level velocity policy.
    """
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to a TorchScript velocity policy checkpoint.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of outer MPPI replanning iterations.",
    )
    parser.add_argument(
        "--mppi-hold-steps",
        type=int,
        default=5,
        help="Number of low-level policy steps to hold the chosen MPPI command before replanning.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=32,
        help="Number of rollout environments used by MPPI.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="MPPI planning horizon.",
    )
    parser.add_argument(
        "--sigma-vx",
        type=float,
        default=0.05,
        help="Exploration noise standard deviation for vx residual.",
    )
    parser.add_argument(
        "--sigma-vy",
        type=float,
        default=0.05,
        help="Exploration noise standard deviation for vy residual.",
    )
    parser.add_argument(
        "--sigma-wz",
        type=float,
        default=0.10,
        help="Exploration noise standard deviation for yaw-rate residual.",
    )
    parser.add_argument(
        "--lambda-mppi",
        type=float,
        default=1.0,
        help="Temperature parameter for MPPI.",
    )
    parser.add_argument(
        "--target-vx",
        type=float,
        default=0.20,
        help="Nominal desired forward velocity.",
    )
    parser.add_argument(
        "--target-vy",
        type=float,
        default=0.00,
        help="Nominal desired lateral velocity.",
    )
    parser.add_argument(
        "--target-wz",
        type=float,
        default=0.00,
        help="Nominal desired yaw rate.",
    )
    parser.add_argument(
        "--cmd-limit-vx",
        type=float,
        default=0.60,
        help="Absolute clamp for commanded vx.",
    )
    parser.add_argument(
        "--cmd-limit-vy",
        type=float,
        default=0.40,
        help="Absolute clamp for commanded vy.",
    )
    parser.add_argument(
        "--cmd-limit-wz",
        type=float,
        default=1.20,
        help="Absolute clamp for commanded yaw rate.",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=0.20,
        help="Desired body height for stability cost.",
    )
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )
    from thomas_MBRL.controllers.torch_policy_wrapper import TorchVelocityPolicy

    device = "cuda"
    k_rollouts = args.num_rollouts
    horizon = args.horizon
    hold_steps = args.mppi_hold_steps

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
        [args.sigma_vx, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = 1 + k_rollouts
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = device

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    obs, _ = env.reset()

    print("Initial observation type:", type(obs))
    if isinstance(obs, dict):
        print("Initial observation keys:", obs.keys())

    policy = TorchVelocityPolicy(args.policy_path, device)

    robot = env.unwrapped.scene["robot"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"Action dim from low-level policy/env interface: {action_dim}")
    print(f"Starting command-space MPPI control with hold_steps={hold_steps}...")

    env0_id = torch.tensor([0], device=device, dtype=torch.long)
    ref_env_id = torch.tensor([1], device=device, dtype=torch.long)
    rollout_env_ids = torch.arange(1, k_rollouts + 1, device=device, dtype=torch.long)

    # Save a frozen reference robot at the initial state for visualization.
    ref_root_state = robot.data.root_state_w[0].clone()
    ref_joint_pos = robot.data.joint_pos[0].clone()
    ref_joint_vel = robot.data.joint_vel[0].clone()

    _freeze_reference_robot(
        robot,
        ref_env_id,
        ref_root_state,
        ref_joint_pos,
        ref_joint_vel,
    )

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
        robot.write_root_pose_to_sim(
            root_state_0[:7].repeat(k_rollouts, 1),
            env_ids=rollout_env_ids,
        )
        robot.write_root_velocity_to_sim(
            root_state_0[7:].repeat(k_rollouts, 1),
            env_ids=rollout_env_ids,
        )
        robot.write_joint_state_to_sim(
            joint_pos_0.repeat(k_rollouts, 1),
            joint_vel_0.repeat(k_rollouts, 1),
            env_ids=rollout_env_ids,
        )

        # Immediately restore the reference robot so it stays frozen even though
        # ref_env_id is included inside rollout_env_ids.
        _freeze_reference_robot(
            robot,
            ref_env_id,
            ref_root_state,
            ref_joint_pos,
            ref_joint_vel,
        )

        # Warm up rollout envs so the policy can settle onto the copied state.
        warmup_steps = 5
        warmup_cmds = base_cmd.unsqueeze(0).repeat(k_rollouts, 1)

        # Keep the reference robot command at zero for visualization.
        warmup_cmds[0] = torch.zeros(3, device=device)

        for _ in range(warmup_steps):
            _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))
            _set_base_velocity_command(env, rollout_env_ids, warmup_cmds)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

            fresh_obs = _get_fresh_obs(env, obs)
            warmup_actions = policy(fresh_obs)

            # Keep the reference robot still.
            warmup_actions[1] = 0.0

            obs, reward, terminated, truncated, info = env.step(warmup_actions)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

        costs = torch.zeros(k_rollouts, device=device, dtype=torch.float32)
        noise = torch.randn((k_rollouts, horizon, 3), device=device) * sigma.view(1, 1, 3)

        for t in range(horizon):
            # Build rollout commands around the nominal base command.
            cmd_residual_t = u_nom[t].unsqueeze(0) + noise[:, t]
            rollout_cmds = _clamp_commands(base_cmd.unsqueeze(0) + cmd_residual_t, cmd_limit)

            # Keep the reference robot command at zero for visualization.
            rollout_cmds[0] = torch.zeros(3, device=device)

            # Keep env0 on the current real command while rollouts explore.
            _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))
            _set_base_velocity_command(env, rollout_env_ids, rollout_cmds)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

            # Recompute observations after updating commands, then query the policy.
            fresh_obs = _get_fresh_obs(env, obs)
            actions = policy(fresh_obs)

            # Keep the reference robot still.
            actions[1] = 0.0

            obs, reward, terminated, truncated, info = env.step(actions)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

            # Skip env 1 from rollout scoring because it is the frozen reference robot.
            base_vel = robot.data.root_state_w[2:1 + k_rollouts, 7:10]
            base_pos = robot.data.root_state_w[2:1 + k_rollouts, 0:3]
            projected_gravity = robot.data.projected_gravity_b[2:1 + k_rollouts]
            joint_pos = robot.data.joint_pos[2:1 + k_rollouts]
            joint_vel = robot.data.joint_vel[2:1 + k_rollouts]

            vx_error = (base_vel[:, 0] - base_cmd[0]) ** 2
            vy_cost = base_vel[:, 1] ** 2
            wz_cost = base_vel[:, 2] ** 2
            height_error = (base_pos[:, 2] - args.target_height) ** 2
            upright_cost = projected_gravity[:, 0] ** 2 + projected_gravity[:, 1] ** 2
            pose_cost = ((joint_pos - joint_pos_0.unsqueeze(0)) ** 2).sum(dim=1)
            vel_joint_cost = (joint_vel ** 2).sum(dim=1)
            residual_cost = ((cmd_residual_t[1:] / cmd_limit.unsqueeze(0)) ** 2).sum(dim=1)

            fall_penalty = torch.where(base_pos[:, 2] < 0.22, 50.0, 0.0)
            tilt_penalty = torch.where(upright_cost > 0.04, 40.0, 0.0)
            done_penalty = 50.0 * (
                terminated[2:1 + k_rollouts].float() + truncated[2:1 + k_rollouts].float()
            )

            # env 1 is frozen reference, so fill only costs[1:] from rollouts.
            costs[1:] += (
                12.0 * vx_error
                + 10.0 * vy_cost
                + 4.0 * wz_cost
                + 6.0 * height_error
                + 12.0 * upright_cost
                + 0.15 * pose_cost
                + 0.01 * vel_joint_cost
                + 0.05 * residual_cost
                + fall_penalty
                + tilt_penalty
                + done_penalty
            )

            # Make the reference env impossible to select as best rollout.
            costs[0] = 1e9

        weights = torch.softmax(-costs / args.lambda_mppi, dim=0)
        delta = torch.einsum("k,khd->hd", weights, noise)
        u_nom = u_nom + delta

        best_idx = torch.argmin(costs).item()
        best_cost = costs[best_idx].item()

        print(f"Best rollout cost: {best_cost:.6f}")
        print(f"Best rollout index: {best_idx}")

        # Restore env0 to its pre-rollout state so only the chosen command gets applied.
        robot.write_root_pose_to_sim(
            root_state_0[:7].unsqueeze(0),
            env_ids=env0_id,
        )
        robot.write_root_velocity_to_sim(
            root_state_0[7:].unsqueeze(0),
            env_ids=env0_id,
        )
        robot.write_joint_state_to_sim(
            joint_pos_0.unsqueeze(0),
            joint_vel_0.unsqueeze(0),
            env_ids=env0_id,
        )

        _freeze_reference_robot(
            robot,
            ref_env_id,
            ref_root_state,
            ref_joint_pos,
            ref_joint_vel,
        )

        # Apply the chosen command to env0.
        env0_cmd = _clamp_commands(base_cmd + u_nom[0], cmd_limit)
        rollout_cmds_real = env0_cmd.unsqueeze(0).repeat(k_rollouts, 1)

        # Keep the reference robot command at zero.
        rollout_cmds_real[0] = torch.zeros(3, device=device)

        # Hold the chosen MPPI command for several low-level policy steps before replanning.
        for hold_idx in range(hold_steps):
            _set_base_velocity_command(env, env0_id, env0_cmd.unsqueeze(0))
            _set_base_velocity_command(env, rollout_env_ids, rollout_cmds_real)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

            fresh_obs = _get_fresh_obs(env, obs)
            real_actions = policy(fresh_obs)

            # Keep the reference robot still.
            real_actions[1] = 0.0

            obs, reward, terminated, truncated, info = env.step(real_actions)

            _freeze_reference_robot(
                robot,
                ref_env_id,
                ref_root_state,
                ref_joint_pos,
                ref_joint_vel,
            )

            if hold_idx == hold_steps - 1:
                base_vel0 = robot.data.root_state_w[0, 7:10]
                base_pos0 = robot.data.root_state_w[0, 0:3]
                proj_grav0 = robot.data.projected_gravity_b[0]
                vel_err0 = base_vel0 - env0_cmd
                forward_disp0 = robot.data.root_state_w[0, 0] - ref_root_state[0]

                print("env0 terminated:", bool(terminated[0].item()))
                print("env0 truncated:", bool(truncated[0].item()))
                print("env0 commanded vel:", env0_cmd.detach().cpu().numpy())
                print("env0 actual vel:", base_vel0.detach().cpu().numpy())
                print("env0 vel error:", vel_err0.detach().cpu().numpy())
                print("env0 pos:", base_pos0.detach().cpu().numpy())
                print("env0 projected gravity:", proj_grav0.detach().cpu().numpy())
                print("env0 forward displacement from reference:", float(forward_disp0.item()))

        # Shift the nominal sequence and append zero residual at the tail.
        u_nom = torch.cat([u_nom[1:], torch.zeros_like(u_nom[:1])], dim=0)

        # Keep the residual command modest.
        residual_limit = 0.25 * cmd_limit
        u_nom = _clamp_commands(u_nom, residual_limit)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()