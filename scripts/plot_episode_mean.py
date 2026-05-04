# Example Usage
# python scripts/plot_episode_mean.py

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


ppo_agent_filepath = "logs/skrl_go2_flat_ppo_2026-03-18_11-18-16_ppo_torch.csv"
sac_agent_filepath = "logs/skrl_unitree_sac_events.out.tfevents.1774229031.theo-desktop.172264.csv"



ppo_agent = pd.read_csv(ppo_agent_filepath)
sac_agent = pd.read_csv(sac_agent_filepath)



# Average Return across Iterations
plt.figure(figsize=(8,6))

plt.plot(ppo_agent["Step"], ppo_agent["Value"], label="PPO Agent")
plt.plot(sac_agent["Step"], sac_agent["Value"], label="SAC Agent")

plt.title("Average Episode Length During Training")
plt.legend()
plt.xlabel("Total Timesteps")
plt.ylabel("Average Episode Length")
plt.grid(True)
plt.tight_layout()

plt.savefig("Average Episode Length During Training.png")
plt.show()