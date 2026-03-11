# scripts/mppi/run_go2_velocity_mppi.py

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
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import thomas_MBRL  # noqa: F401
    import gymnasium as gym
    import torch

    from thomas_MBRL.envs.go2_velocity_env_cfg import ThomasGo2VelocityEnvCfg

    K = 64
    device = "cuda"

    cfg = ThomasGo2VelocityEnvCfg()
    cfg.scene.num_envs = 1 + K
    cfg.sim.device = device

    env = gym.make("Thomas-Go2-Velocity-v0", cfg=cfg)
    obs, _ = env.reset()

    fixed_cmd = torch.tensor([0.2, 0.0, 0.0], device=device)

    robot = env.unwrapped.scene["robot"]
    print("Robot found.")

    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"Action dim: {action_dim}")

    # MPPI settings
    K_test = 32
    H_test = 25
    sigma = 0.12
    lam = 1.0
    target_height = 0.32

    env_ids_test = torch.arange(1, K_test + 1, device=device)
    env0_id = torch.tensor([0], device=device)

    u_nom = torch.zeros((H_test, action_dim), device=device)

    print("Starting continuous MPPI control...")

    for outer in range(100):
        print(f"\n=== MPPI iteration {outer + 1} ===")

        # Keep command visualization consistent
        cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")
        cmd_buf[:, 0] = fixed_cmd[0]
        cmd_buf[:, 1] = fixed_cmd[1]
        cmd_buf[:, 2] = fixed_cmd[2]

        # Snapshot env0 BEFORE rollout
        root_state_0 = robot.data.root_state_w[0].clone()
        joint_pos_0 = robot.data.joint_pos[0].clone()
        joint_vel_0 = robot.data.joint_vel[0].clone()

        # Clone env0 into rollout envs
        robot.write_root_pose_to_sim(
            root_state_0[:7].repeat(K_test, 1),
            env_ids=env_ids_test,
        )
        robot.write_root_velocity_to_sim(
            root_state_0[7:].repeat(K_test, 1),
            env_ids=env_ids_test,
        )
        robot.write_joint_state_to_sim(
            joint_pos_0.repeat(K_test, 1),
            joint_vel_0.repeat(K_test, 1),
            env_ids=env_ids_test,
        )

        costs = torch.zeros(K_test, device=device)
        noise = sigma * torch.randn((K_test, H_test, action_dim), device=device)

        # Rollout phase
        for t in range(H_test):
            actions = torch.zeros((1 + K, action_dim), device=device)

            # rollout envs only
            actions[1:1 + K_test] = u_nom[t] + noise[:, t]

            obs, reward, terminated, truncated, info = env.step(actions)

            base_vel = robot.data.root_state_w[1:1 + K_test, 7:10]
            base_pos = robot.data.root_state_w[1:1 + K_test, 0:3]
            projected_gravity = robot.data.projected_gravity_b[1:1 + K_test]

            cmd_xy = fixed_cmd[:2].unsqueeze(0).repeat(K_test, 1)

            vel_error = (base_vel[:, :2] - cmd_xy) ** 2
            height_error = (base_pos[:, 2] - target_height) ** 2
            upright_cost = projected_gravity[:, 0] ** 2 + projected_gravity[:, 1] ** 2
            control_cost = (actions[1:1 + K_test] ** 2).sum(dim=1)

            step_cost = (
                6.0 * vel_error.sum(dim=1)
                + 0.3 * height_error
                + 0.2 * upright_cost
                + 0.0001 * control_cost
            )

            costs += step_cost

        # MPPI update
        weights = torch.softmax(-costs / lam, dim=0)
        delta = torch.einsum("k,khd->hd", weights, noise)
        u_nom = u_nom + delta

        best_idx = torch.argmin(costs).item()
        best_cost = costs[best_idx].item()

        print(f"Best rollout cost: {best_cost}")
        print(f"Best rollout index: {best_idx}")

        # Restore env0 to PRE-ROLLOUT snapshot
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

        # Apply chosen action to env0 only
        real_actions = torch.zeros((1 + K, action_dim), device=device)
        real_actions[0] = u_nom[0]

        obs, reward, terminated, truncated, info = env.step(real_actions)

        print("env0 terminated:", bool(terminated[0].item()))
        print("env0 truncated:", bool(truncated[0].item()))

        base_vel0 = robot.data.root_state_w[0, 7:10]
        base_pos0 = robot.data.root_state_w[0, 0:3]
        proj_grav0 = robot.data.projected_gravity_b[0]
        vel_err0 = base_vel0[:2] - fixed_cmd[:2]

        print("env0 vel:", base_vel0.detach().cpu().numpy())
        print("target cmd:", fixed_cmd.detach().cpu().numpy())
        print("env0 vel error xy:", vel_err0.detach().cpu().numpy())
        print("env0 pos:", base_pos0.detach().cpu().numpy())
        print("env0 projected gravity:", proj_grav0.detach().cpu().numpy())
        print("Applied MPPI action to env0.")

        # Shift nominal sequence
        u_nom = torch.cat([u_nom[1:], torch.zeros_like(u_nom[:1])], dim=0)
        u_nom = torch.clamp(u_nom, -0.7, 0.7)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()