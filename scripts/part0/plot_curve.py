"""Plot the deterministic evaluation learning curve from EvalCallback data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.evaluations.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {args.evaluations}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.evaluations)

    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)

    if results.ndim != 2:
        raise ValueError(
            f"Expected results with shape [evaluations, episodes], "
            f"received {results.shape}"
        )

    mean_returns = np.mean(results, axis=1)
    std_returns = np.std(results, axis=1)

    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timesteps",
                "mean_eval_return",
                "std_eval_return",
                "n_eval_episodes",
            ]
        )

        for timestep, mean_return, std_return in zip(
            timesteps,
            mean_returns,
            std_returns,
            strict=True,
        ):
            writer.writerow(
                [
                    int(timestep),
                    float(mean_return),
                    float(std_return),
                    int(results.shape[1]),
                ]
            )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        timesteps,
        mean_returns,
        linewidth=2,
        label="Deterministic evaluation return",
    )

    axis.fill_between(
        timesteps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.2,
        label="±1 episode std",
    )

    axis.set_title("SB3 PPO on Reacher-v5")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Mean episode return")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    print(f"Learning curve: {args.output}")
    print(f"Curve data:     {args.csv_output}")


if __name__ == "__main__":
    main()
