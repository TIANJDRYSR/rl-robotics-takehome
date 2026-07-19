"""Plot success_rate / mean_return vs. sensor-noise sigma from play.py --output JSON files.

Unlike plot_comparison.py / plot_reward_type_comparison.py (which plot a metric
against training timesteps for a single fixed evaluation seed), this plots a
metric against noise_sigma for a single fixed (best) checkpoint -- i.e. a
sensitivity sweep, not a learning curve. Each --run entry is "SIGMA:PATH",
where PATH is a JSON file written by `play.py --output ...`.

Example:
    python scripts/part2/plot_noise_sensitivity.py \
      --run 0:results/part2/final_eval/dense_seed1_noise_sigma_0.json \
      --run 0.005:results/part2/final_eval/dense_seed1_noise_sigma_0.005.json \
      --run 0.02:results/part2/final_eval/dense_seed1_noise_sigma_0.02.json \
      --title "Dense PPO (seed 1) - sensitivity to observation noise" \
      --output results/part2/noise_sensitivity_dense_seed1.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NoiseRun:
    sigma: float
    mean_return: float
    std_return: float
    success_rate: float
    n_episodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SIGMA:PATH",
        help="Repeatable. PATH is a JSON file written by play.py --output.",
    )
    parser.add_argument("--title", type=str, default="Sensitivity to observation noise")
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def load_run(spec: str) -> NoiseRun:
    sigma_str, path_str = spec.split(":", 1)
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"play.py output file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return NoiseRun(
        sigma=float(sigma_str),
        mean_return=float(data["mean_return"]),
        std_return=float(data["std_return"]),
        success_rate=float(data["success_rate"]),
        n_episodes=int(data["n_episodes"]),
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs = sorted((load_run(spec) for spec in args.run), key=lambda run: run.sigma)

    sigmas = np.asarray([run.sigma for run in runs], dtype=np.float64)
    success_rates = np.asarray([run.success_rate for run in runs], dtype=np.float64)
    mean_returns = np.asarray([run.mean_return for run in runs], dtype=np.float64)
    std_returns = np.asarray([run.std_return for run in runs], dtype=np.float64)
    n_episodes = runs[0].n_episodes

    figure, (success_axis, return_axis) = plt.subplots(1, 2, figsize=(11, 4.5))

    success_axis.plot(sigmas, success_rates, marker="o", linewidth=2, color="tab:orange")
    success_axis.set_xlabel("Observation noise sigma")
    success_axis.set_ylabel("Success rate")
    success_axis.set_ylim(-0.05, 1.05)
    success_axis.set_xticks(sigmas)
    success_axis.grid(True, alpha=0.3)

    # SEM (std / sqrt(n)) rather than raw per-episode std, since we're
    # comparing means across a fixed number of episodes at each sigma.
    return_sem = std_returns / np.sqrt(n_episodes)
    return_axis.errorbar(
        sigmas, mean_returns, yerr=return_sem, marker="o", linewidth=2, capsize=4, color="tab:blue"
    )
    return_axis.set_xlabel("Observation noise sigma")
    return_axis.set_ylabel("Mean episode return")
    return_axis.set_xticks(sigmas)
    return_axis.grid(True, alpha=0.3)

    figure.suptitle(f"{args.title} (n={n_episodes} episodes per sigma)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    print(f"Noise sensitivity plot: {args.output}")


if __name__ == "__main__":
    main()
