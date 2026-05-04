# Example Usage
# python scripts/plot_return.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RAND_AGENT_FILEPATH = "logs/go2_flat_ppo_2026-02-08_14-15-29_ppo_torch_rand.csv"
PPO_AGENT_FILEPATH = "logs/go2_flat_ppo_2026-03-18_11-18-16_ppo_torch.csv"
SAC_AGENT_FILEPATH = "logs/skrl_go2_flat_sac_2026-03-22_21-23-29_sac_torch-total_reward_mean.csv"
GAIT_PPO_FILEPATH = "logs/skrl_go2_hier_gait_ppo_2026-04-22_23-44-38_hier_gait_torch-total_reward_mean.csv"
REWARD_PPO_FILEPATH = "logs/skrl_go2_flat_ppo_2026-04-22_23-59-24_ppo_torch-total_reward_mean.csv"
OUTPUT_FILEPATH = "training_quality_comparison.png"

COLORS = {
    "random": "#b91c1c",
    "ppo": "#2563eb",
    "ppo_gait": "#7c3aed",
    "ppo_reward": "#db2777",
    "sac": "#f97316",
    "mbrl": "#16a34a",
    "mbrl_aux": "#15803d",
}


def latest_mbrl_metrics_filepath() -> str:
    candidates = sorted(Path("logs/mbrl").glob("*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError("No MBRL metrics.csv files found under logs/mbrl")
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def load_tensorboard_csv(filepath: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return pd.DataFrame(
        {
            "method": label,
            "steps": df["Step"],
            "progress": df["Step"] / df["Step"].max(),
            "return": df["Value"],
        }
    )


def load_mbrl_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    completed_episode_lengths = df.loc[df["mean_length_100"] > 0, "mean_length_100"]
    episode_horizon_steps = float(completed_episode_lengths.max()) if not completed_episode_lengths.empty else 1000.0
    if "estimated_return_100" in df:
        dense_return = df["estimated_return_100"]
    else:
        dense_return = df["mean_step_reward_100"] * episode_horizon_steps
    return pd.DataFrame(
        {
            "method": "MBRL",
            "steps": df["env_steps"],
            "progress": df["env_steps"] / df["env_steps"].max(),
            "return": dense_return,
            "completed_return": df["mean_return_100"],
            "step_reward": df["mean_step_reward_100"],
            "episodes_finished": df["episodes_finished"],
            "buffer_size": df["buffer_size"],
        }
    )


def smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_return_curve(ax: plt.Axes, df: pd.DataFrame, label: str, color: str, x_col: str = "steps") -> None:
    ax.plot(df[x_col], smooth(df["return"]), color=color, linewidth=2.2, label=f"{label} smoothed")
    ax.plot(df[x_col], df["return"], color=color, alpha=0.22, linewidth=1.0)


def main() -> None:
    mbrl_agent_filepath = latest_mbrl_metrics_filepath()
    rand_agent = pd.read_csv(RAND_AGENT_FILEPATH)
    ppo_agent = load_tensorboard_csv(PPO_AGENT_FILEPATH, "PPO")
    gait_ppo_agent = load_tensorboard_csv(GAIT_PPO_FILEPATH, "Hierarchical gait modification")
    reward_ppo_agent = load_tensorboard_csv(REWARD_PPO_FILEPATH, "Gait reward modification")
    sac_agent = load_tensorboard_csv(SAC_AGENT_FILEPATH, "SAC")
    mbrl_agent = load_mbrl_csv(mbrl_agent_filepath)

    random_mean = rand_agent["Value"].mean()
    final_scores = pd.DataFrame(
        {
            "method": ["PPO", "Hier gait mod", "Reward mod", "SAC", "MBRL"],
            "final_return": [
                ppo_agent["return"].iloc[-1],
                gait_ppo_agent["return"].iloc[-1],
                reward_ppo_agent["return"].iloc[-1],
                sac_agent["return"].iloc[-1],
                mbrl_agent["return"].iloc[-1],
            ],
            "best_return": [
                ppo_agent["return"].max(),
                gait_ppo_agent["return"].max(),
                reward_ppo_agent["return"].max(),
                sac_agent["return"].max(),
                mbrl_agent["return"].max(),
            ],
        }
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Quality Comparison: PPO Variants vs SAC vs MBRL", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.axhline(random_mean, color=COLORS["random"], linestyle="--", linewidth=1.5, label="Random baseline")
    plot_return_curve(ax, ppo_agent, "PPO", COLORS["ppo"])
    plot_return_curve(ax, gait_ppo_agent, "Hierarchical gait modification", COLORS["ppo_gait"])
    plot_return_curve(ax, reward_ppo_agent, "Gait reward modification", COLORS["ppo_reward"])
    plot_return_curve(ax, sac_agent, "SAC", COLORS["sac"])
    plot_return_curve(ax, mbrl_agent, "MBRL dense return", COLORS["mbrl"])
    ax.step(
        mbrl_agent["steps"],
        mbrl_agent["completed_return"],
        where="post",
        color=COLORS["mbrl"],
        linewidth=1.2,
        alpha=0.35,
        label="MBRL completed return",
    )
    ax.set_title("Return Over Training Steps")
    ax.set_xlabel("Logged training steps")
    ax.set_ylabel("Average return")
    ax.legend()

    ax = axes[0, 1]
    ax.axhline(random_mean, color=COLORS["random"], linestyle="--", linewidth=1.5, label="Random baseline")
    plot_return_curve(ax, ppo_agent, "PPO", COLORS["ppo"], x_col="progress")
    plot_return_curve(ax, gait_ppo_agent, "Hierarchical gait modification", COLORS["ppo_gait"], x_col="progress")
    plot_return_curve(ax, reward_ppo_agent, "Gait reward modification", COLORS["ppo_reward"], x_col="progress")
    plot_return_curve(ax, sac_agent, "SAC", COLORS["sac"], x_col="progress")
    plot_return_curve(ax, mbrl_agent, "MBRL dense return", COLORS["mbrl"], x_col="progress")
    ax.step(
        mbrl_agent["progress"],
        mbrl_agent["completed_return"],
        where="post",
        color=COLORS["mbrl"],
        linewidth=1.2,
        alpha=0.35,
        label="MBRL completed return",
    )
    ax.set_title("Return Over Normalized Run Progress")
    ax.set_xlabel("Fraction of each run completed")
    ax.set_ylabel("Average return")
    ax.legend()

    ax = axes[1, 0]
    x = range(len(final_scores))
    width = 0.36
    ax.bar(
        [i - width / 2 for i in x],
        final_scores["final_return"],
        width=width,
        color=[COLORS["ppo"], COLORS["ppo_gait"], COLORS["ppo_reward"], COLORS["sac"], COLORS["mbrl"]],
        alpha=0.78,
        label="Final return",
    )
    ax.bar(
        [i + width / 2 for i in x],
        final_scores["best_return"],
        width=width,
        color=[COLORS["ppo"], COLORS["ppo_gait"], COLORS["ppo_reward"], COLORS["sac"], COLORS["mbrl"]],
        alpha=0.35,
        label="Best return",
    )
    ax.axhline(random_mean, color=COLORS["random"], linestyle="--", linewidth=1.5, label="Random baseline")
    ax.set_title("Final and Best Observed Return")
    ax.set_xticks(list(x))
    ax.set_xticklabels(final_scores["method"])
    ax.set_ylabel("Average return")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        mbrl_agent["steps"],
        smooth(mbrl_agent["step_reward"], window=9),
        color=COLORS["mbrl_aux"],
        linewidth=2.2,
        label="MBRL mean step reward, smoothed",
    )
    ax.plot(mbrl_agent["steps"], mbrl_agent["step_reward"], color=COLORS["mbrl_aux"], alpha=0.2, linewidth=1.0)
    ax2 = ax.twinx()
    ax2.step(
        mbrl_agent["steps"],
        mbrl_agent["episodes_finished"],
        where="post",
        color="#475569",
        linewidth=1.8,
        alpha=0.8,
        label="Episodes finished",
    )
    ax.set_title("Why MBRL Return Looks Step-Wise")
    ax.set_xlabel("MBRL env steps")
    ax.set_ylabel("Mean step reward")
    ax2.set_ylabel("Completed episodes")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")

    fig.text(
        0.5,
        0.01,
        "Note: PPO/SAC curves come from TensorBoard CSV Step/Value. "
        "MBRL dense return uses rolling step reward on the episode-return scale; the faint step curve is exact completed return.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTPUT_FILEPATH, dpi=200)
    print(f"Saved comparison plot to {OUTPUT_FILEPATH} using MBRL data from {mbrl_agent_filepath}")
    plt.show()


if __name__ == "__main__":
    main()
