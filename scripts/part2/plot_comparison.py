"""Overlay Custom PPO vs. SB3 PPO evaluation curves across seeds and/or reward types.

Each --run entry is "ALGO:SEED:PATH" or "ALGO:SEED:REWARD_TYPE:PATH". ALGO is
"custom" (JSON eval log from train_custom_ppo.py) or "sb3" (evaluations.npz
from train_sb3.py's EvalCallback). REWARD_TYPE is a free-form label (e.g.
"dense"/"sparse") used to color-group runs and is optional (defaults to "").

Example - dense vs. sparse at the same seed:
    python scripts/part2/plot_comparison.py \
      --run custom:1:dense:results/custom/NoisyGoalReacher-v0__train_custom_ppo__dense__1__.../eval/evaluations.json \
      --run custom:1:sparse:results/custom/NoisyGoalReacher-v0__train_custom_ppo__sparse__1__.../eval/evaluations.json \
      --title "NoisyGoalReacher-v0 - seed 1, dense vs. sparse" \
      --output results/part2/comparison_seed1_dense_vs_sparse.png

Example - Custom PPO (dense) vs. SB3 across seeds:
    python scripts/part2/plot_comparison.py \
      --run custom:1:results/custom/NoisyGoalReacher-v0__train_custom_ppo__1__1784437680/eval/evaluations.json \
      --run custom:12:results/custom/NoisyGoalReacher-v0__train_custom_ppo__dense__12__1784439909/eval/evaluations.json \
      --run sb3:1:results/sb3/NoisyGoalReacher-v0__train_sb3__1__1784367281/eval/evaluations.npz \
      --title "NoisyGoalReacher-v0 - Dense reward, sigma=0.005, two training seeds" \
      --output results/part2/comparison.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALGO_LABELS = {"custom": "Custom PPO", "sb3": "SB3 PPO"}
# Preferred colors for known (algo, reward_type) groups; anything else pulls
# from FALLBACK_PALETTE in first-seen order so new reward_type labels still work.
KNOWN_GROUP_COLORS = {
    ("custom", ""): "tab:orange",
    ("sb3", ""): "tab:blue",
    ("custom", "dense"): "tab:orange",
    ("custom", "sparse"): "tab:red",
    ("sb3", "dense"): "tab:blue",
    ("sb3", "sparse"): "tab:cyan",
}
FALLBACK_PALETTE = ["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]
SEED_LINESTYLES = ["-", "--", ":", "-."]


@dataclass
class RunCurve:
    algo: str
    seed: int
    reward_type: str
    timesteps: np.ndarray
    mean_returns: np.ndarray
    std_returns: np.ndarray
    success_rates: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ALGO:SEED:PATH or ALGO:SEED:REWARD_TYPE:PATH",
        help="Repeatable. ALGO is 'custom' or 'sb3'.",
    )
    parser.add_argument("--title", type=str, default="Custom PPO vs. SB3 PPO")
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def load_custom_json(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        eval_history = json.load(file)

    timesteps = np.asarray([entry["step"] for entry in eval_history], dtype=np.int64)
    mean_returns = np.asarray(
        [entry["mean_return"] for entry in eval_history], dtype=np.float64
    )
    std_returns = np.asarray(
        [entry["return_std"] for entry in eval_history], dtype=np.float64
    )
    success_rates = np.asarray(
        [entry["success_rate"] for entry in eval_history], dtype=np.float64
    )
    return timesteps, mean_returns, std_returns, success_rates


def load_sb3_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)
    mean_returns = np.mean(results, axis=1)
    std_returns = np.std(results, axis=1)
    success_rates = np.mean(np.asarray(data["successes"], dtype=np.float64), axis=1)
    return timesteps, mean_returns, std_returns, success_rates


def parse_run_spec(spec: str) -> RunCurve:
    parts = spec.split(":", 3)
    if len(parts) == 3:
        algo, seed_str, path_str = parts
        reward_type = ""
    elif len(parts) == 4:
        algo, seed_str, reward_type, path_str = parts
    else:
        raise ValueError(
            f"--run must be 'ALGO:SEED:PATH' or 'ALGO:SEED:REWARD_TYPE:PATH', got '{spec}'"
        )

    algo = algo.strip().lower()
    if algo not in ALGO_LABELS:
        raise ValueError(f"Unknown algo '{algo}' in --run '{spec}' (expected 'custom' or 'sb3')")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    if algo == "custom":
        timesteps, mean_returns, std_returns, success_rates = load_custom_json(path)
    else:
        timesteps, mean_returns, std_returns, success_rates = load_sb3_npz(path)

    return RunCurve(
        algo=algo,
        seed=int(seed_str),
        reward_type=reward_type.strip().lower(),
        timesteps=timesteps,
        mean_returns=mean_returns,
        std_returns=std_returns,
        success_rates=success_rates,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs = [parse_run_spec(spec) for spec in args.run]

    # Color groups by (algo, reward_type); linestyle rotates over seeds within a group.
    group_colors: dict[tuple[str, str], str] = {}
    fallback_iter = iter(FALLBACK_PALETTE)
    seeds_per_group: dict[tuple[str, str], list[int]] = {}
    for run in runs:
        group_key = (run.algo, run.reward_type)
        if group_key not in group_colors:
            group_colors[group_key] = KNOWN_GROUP_COLORS.get(group_key) or next(fallback_iter, "tab:gray")
        seeds_per_group.setdefault(group_key, []).append(run.seed)

    figure, (return_axis, success_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for run in runs:
        group_key = (run.algo, run.reward_type)
        color = group_colors[group_key]
        seed_index = seeds_per_group[group_key].index(run.seed)
        linestyle = SEED_LINESTYLES[seed_index % len(SEED_LINESTYLES)]
        reward_suffix = f" {run.reward_type}" if run.reward_type else ""
        label = f"{ALGO_LABELS[run.algo]}{reward_suffix} (seed {run.seed})"

        return_axis.plot(
            run.timesteps, run.mean_returns, color=color, linestyle=linestyle,
            linewidth=2, label=label,
        )
        return_axis.fill_between(
            run.timesteps,
            run.mean_returns - run.std_returns,
            run.mean_returns + run.std_returns,
            color=color, alpha=0.12,
        )

        success_axis.plot(
            run.timesteps, run.success_rates, color=color, linestyle=linestyle,
            linewidth=2, label=label,
        )

    return_axis.set_ylabel("Mean episode return")
    return_axis.grid(True, alpha=0.3)
    return_axis.legend()

    success_axis.set_xlabel("Environment timesteps")
    success_axis.set_ylabel("Success rate")
    success_axis.set_ylim(-0.05, 1.05)
    success_axis.grid(True, alpha=0.3)
    success_axis.legend()

    figure.suptitle(args.title)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    print(f"Comparison plot: {args.output}")


if __name__ == "__main__":
    main()
