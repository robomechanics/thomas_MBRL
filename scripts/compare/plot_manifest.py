#!/usr/bin/env python3
"""Plot PPO/SAC/MBRL comparison curves from a train_all.py manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TENSORBOARD_TAG_CANDIDATES = (
    "Reward / total_reward_mean",
    "Reward/total_reward_mean",
    "total_reward_mean",
    "Reward / completed_total_reward_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot comparison results from a manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_tensorboard_scalar(run_dir: Path) -> pd.DataFrame:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("Install tensorboard to read skrl event files: python3 -m pip install tensorboard") from exc

    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    tag = next((candidate for candidate in TENSORBOARD_TAG_CANDIDATES if candidate in tags), None)
    if tag is None:
        raise KeyError(f"No known return scalar found in {run_dir}. Available scalar tags: {tags}")

    events = accumulator.Scalars(tag)
    return pd.DataFrame({"steps": [event.step for event in events], "return": [event.value for event in events]})


def load_mbrl_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No MBRL metrics.csv found at {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    if "estimated_return_100" in metrics:
        returns = metrics["estimated_return_100"]
    else:
        returns = metrics["mean_return_100"]
    return pd.DataFrame({"steps": metrics["env_steps"], "return": returns})


def load_run(run: dict[str, object]) -> pd.DataFrame:
    method = str(run["method"]).upper()
    run_dir = Path(str(run["run_dir"]))
    if method == "MBRL":
        df = load_mbrl_metrics(run_dir)
    else:
        df = load_tensorboard_scalar(run_dir)
    df["method"] = method
    max_steps = max(float(df["steps"].max()), 1.0)
    df["progress"] = df["steps"] / max_steps
    return df


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output or args.manifest.with_name("comparison.png")

    frames = []
    for run in manifest["runs"]:
        if run.get("returncode") != 0 or not run.get("run_dir"):
            continue
        frames.append(load_run(run))
    if not frames:
        raise RuntimeError("No completed runs with readable log directories found in manifest")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for df in frames:
        label = str(df["method"].iloc[0])
        smooth_return = df["return"].rolling(window=5, min_periods=1).mean()
        axes[0].plot(df["steps"], smooth_return, label=label, linewidth=2.0)
        axes[1].plot(df["progress"], smooth_return, label=label, linewidth=2.0)

    axes[0].set_title("Return by Logged Step")
    axes[0].set_xlabel("Logged step")
    axes[0].set_ylabel("Average return")
    axes[1].set_title("Return by Run Progress")
    axes[1].set_xlabel("Progress")
    axes[1].set_ylabel("Average return")
    for ax in axes:
        ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved comparison plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
