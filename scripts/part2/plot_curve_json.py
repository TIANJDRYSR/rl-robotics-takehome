"""Plot the deterministic evaluation learning curve from the custom PPO JSON eval log."""

from __future__ import annotations

import argparse
import csv
import json
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

    with args.evaluations.open("r", encoding="utf-8") as file:
        eval_history = json.load(file)

    if not isinstance(eval_history, list) or not eval_history:
        raise ValueError(
            f"Expected a non-empty JSON list of eval records, "
            f"received {type(eval_history)}"
        )

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

    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timesteps",
                "mean_eval_return",
                "std_eval_return",
                "success_rate",
            ]
        )

        for timestep, mean_return, std_return, success_rate in zip(
            timesteps,
            mean_returns,
            std_returns,
            success_rates,
            strict=True,
        ):
            writer.writerow(
                [
                    int(timestep),
                    float(mean_return),
                    float(std_return),
                    float(success_rate),
                ]
            )

    figure, (return_axis, success_axis) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True
    )

    return_axis.plot(
        timesteps,
        mean_returns,
        linewidth=2,
        color="tab:blue",
        label="Deterministic evaluation return",
    )
    return_axis.fill_between(
        timesteps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.2,
        color="tab:blue",
        label="±1 episode std",
    )
    return_axis.set_title("Custom PPO on NoisyGoalReacher-v0")
    return_axis.set_ylabel("Mean episode return")
    return_axis.grid(True, alpha=0.3)
    return_axis.legend()

    success_axis.plot(
        timesteps,
        success_rates,
        linewidth=2,
        color="tab:green",
        label="Success rate",
    )
    success_axis.set_xlabel("Environment timesteps")
    success_axis.set_ylabel("Success rate")
    success_axis.set_ylim(-0.05, 1.05)
    success_axis.grid(True, alpha=0.3)
    success_axis.legend()

    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    print(f"Learning curve: {args.output}")
    print(f"Curve data:     {args.csv_output}")


if __name__ == "__main__":
    main()
