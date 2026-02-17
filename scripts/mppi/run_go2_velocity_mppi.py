# import gymnasium as gym
# import torch

# TASK = "Thomas-Go2-Velocity-v0"

# device = "cuda"
# K = 64

# env = gym.make(
#     TASK,
#     num_envs=1 + K,
#     device=device,
#     headless=True,
# )

# obs, _ = env.reset()

# robot = env.unwrapped.scene["robot"]

# print("Robot found.")

# # Snapshot real env (env 0)
# root_state = robot.data.root_state_w[0].clone()
# joint_pos = robot.data.joint_pos[0].clone()
# joint_vel = robot.data.joint_vel[0].clone()

# # Clone into rollout envs
# env_ids = torch.arange(1, K+1, device=device)

# robot.write_root_pose_to_sim(
#     root_state[:7].repeat(K, 1),
#     env_ids=env_ids,
# )

# robot.write_root_velocity_to_sim(
#     root_state[7:].repeat(K, 1),
#     env_ids=env_ids,
# )

# robot.write_joint_state_to_sim(
#     joint_pos.repeat(K, 1),
#     joint_vel.repeat(K, 1),
#     env_ids=env_ids,
# )

# print("State cloning successful.")

# actions = torch.zeros((1 + K, env.action_space.shape[0]), device=device)
# env.step(actions)

# print("Step successful.")

import gymnasium as gym
import torch

TASK = "Thomas-Go2-Velocity-v0"
device = "cuda"

K = 128
H = 15
lam = 1.0
sigma = 0.1

env = gym.make(TASK, num_envs=1 + K, device=device, headless=True)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
act_dim = env.action_space.shape[0]

u_nom = torch.zeros((H, act_dim), device=device)

for outer in range(1000):

    # Snapshot real state
    root_state = robot.data.root_state_w[0].clone()
    joint_pos = robot.data.joint_pos[0].clone()
    joint_vel = robot.data.joint_vel[0].clone()

    env_ids = torch.arange(1, K+1, device=device)

    robot.write_root_pose_to_sim(root_state[:7].repeat(K, 1), env_ids)
    robot.write_root_velocity_to_sim(root_state[7:].repeat(K, 1), env_ids)
    robot.write_joint_state_to_sim(joint_pos.repeat(K, 1),
                                   joint_vel.repeat(K, 1),
                                   env_ids)

    noise = sigma * torch.randn((K, H, act_dim), device=device)
    costs = torch.zeros(K, device=device)

    for t in range(H):
        actions = torch.zeros((1 + K, act_dim), device=device)
        actions[1:] = u_nom[t] + noise[:, t]

        obs, _, _, _, _ = env.step(actions)

        # Minimal cost: velocity tracking
        base_vel = robot.data.root_state_w[1:, 7:10]
        cmd = env.unwrapped.command_manager.get_command("base_velocity")[1:]

        vel_error = (base_vel[:, :2] - cmd[:, :2]) ** 2
        costs += vel_error.sum(dim=1)

        costs += 0.001 * (actions[1:] ** 2).sum(dim=1)

    weights = torch.softmax(-costs / lam, dim=0)
    delta = torch.einsum("k,khd->hd", weights, noise)
    u_nom += delta

    real_actions = torch.zeros((1 + K, act_dim), device=device)
    real_actions[0] = u_nom[0]
    env.step(real_actions)

    u_nom = torch.cat([u_nom[1:], torch.zeros_like(u_nom[:1])], dim=0)
