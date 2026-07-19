"""Plot mean ± cross-seed std learning curves, grouped by label (e.g. dense vs. sparse).

Each --group entry is "LABEL:PATH" (repeatable). Runs sharing the same LABEL are
aggregated into one curve: the mean across seeds at each step, shaded by ±1
std across those same seeds. All eval logs within a LABEL must share
identical step values (i.e. the same --eval-freq/--total-timesteps/--num-envs
training config) -- this is the case for runs produced by the same
train_custom_ppo.py invocation pattern with only --seed/--reward-type varied.

Example:
    python scripts/part2/plot_reward_type_comparison.py \
      --group dense:results/custom/NoisyGoalReacher-v0__train_custom_ppo__dense__1__1784437680/eval/evaluations.json \
      --group dense:results/custom/NoisyGoalReacher-v0__train_custom_ppo__dense__12__1784439909/eval/evaluations.json \
      --group dense:results/custom/NoisyGoalReacher-v0__train_custom_ppo__dense__1300__1784441544/eval/evaluations.json \
      --group sparse:results/custom/NoisyGoalReacher-v0__train_custom_ppo__sparse__1__1784445008/eval/evaluations.json \
      --group sparse:results/custom/NoisyGoalReacher-v0__train_custom_ppo__sparse__12__1784444082/eval/evaluations.json \
      --group sparse:results/custom/NoisyGoalReacher-v0__train_custom_ppo__sparse__1300__1784442481/eval/evaluations.json \
      --title "NoisyGoalReacher-v0 - Dense vs. Sparse (mean over 3 seeds)" \
      --output results/part2/main_dense_vs_sparse_mean.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KNOWN_COLORS = {"dense": "tab:orange", "sparse": "tab:red"}
FALLBACK_PALETTE = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


@dataclass
class RunCurve:
    label: str
    seed_path: Path
    steps: np.ndarray
    mean_returns: np.ndarray
    success_rates: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help="Repeatable. Runs sharing the same LABEL are averaged together.",
    )
    parser.add_argument("--title", type=str, default="Mean learning curve by group")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="success_rate",
        choices=["success_rate", "mean_return", "both"],
        help=(
            "Which curve(s) to plot. Defaults to success_rate since it's the "
            "one metric that's directly comparable between dense and sparse "
            "reward runs -- mean_return is not (different reward scales)."
        ),
    )

    return parser.parse_args()


def load_run(spec: str) -> RunCurve:
    label, path_str = spec.split(":", 1)
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        eval_history = json.load(file)

    return RunCurve(
        label=label.strip().lower(),
        seed_path=path,
        steps=np.asarray([entry["step"] for entry in eval_history], dtype=np.int64),
        mean_returns=np.asarray([entry["mean_return"] for entry in eval_history], dtype=np.float64),
        success_rates=np.asarray([entry["success_rate"] for entry in eval_history], dtype=np.float64),
    )


def aggregate(runs: list[RunCurve]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack same-label runs and return (steps, mean_return_mean, mean_return_std, success_mean, success_std)."""
    reference_steps = runs[0].steps
    for run in runs[1:]:
        if not np.array_equal(run.steps, reference_steps):
            raise ValueError(
                f"Step mismatch within group '{runs[0].label}': "
                f"{runs[0].seed_path} has {len(reference_steps)} evals, "
                f"{run.seed_path} has {len(run.steps)} -- runs being averaged "
                "must share the same --eval-freq/--total-timesteps/--num-envs."
            )

    returns_stack = np.stack([run.mean_returns for run in runs])
    success_stack = np.stack([run.success_rates for run in runs])
    ddof = 1 if len(runs) > 1 else 0

    return (
        reference_steps,
        returns_stack.mean(axis=0),
        returns_stack.std(axis=0, ddof=ddof),
        success_stack.mean(axis=0),
        success_stack.std(axis=0, ddof=ddof),
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs = [load_run(spec) for spec in args.group]

    runs_by_label: dict[str, list[RunCurve]] = {}
    for run in runs:
        runs_by_label.setdefault(run.label, []).append(run)

    fallback_iter = iter(FALLBACK_PALETTE)
    label_colors = {
        label: KNOWN_COLORS.get(label) or next(fallback_iter, "tab:gray") for label in runs_by_label
    }

    aggregated = {label: aggregate(label_runs) for label, label_runs in runs_by_label.items()}

    if args.metric == "both":
        figure, (return_axis, success_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        axes = [return_axis, success_axis]
    else:
        figure, single_axis = plt.subplots(1, 1, figsize=(8, 5))
        axes = [single_axis]
        return_axis = single_axis if args.metric == "mean_return" else None
        success_axis = single_axis if args.metric == "success_rate" else None

    for label, label_runs in runs_by_label.items():
        color = label_colors[label]
        steps, return_mean, return_std, success_mean, success_std = aggregated[label]
        plot_label = f"{label} (n={len(label_runs)} seeds)"

        if return_axis is not None:
            return_axis.plot(steps, return_mean, color=color, linewidth=2, label=plot_label)
            return_axis.fill_between(
                steps, return_mean - return_std, return_mean + return_std, color=color, alpha=0.2
            )

        if success_axis is not None:
            success_axis.plot(steps, success_mean, color=color, linewidth=2, label=plot_label)
            success_axis.fill_between(
                steps, success_mean - success_std, success_mean + success_std, color=color, alpha=0.2
            )

    if return_axis is not None:
        return_axis.set_ylabel("Mean episode return")
        return_axis.grid(True, alpha=0.3)
        return_axis.legend()

    if success_axis is not None:
        success_axis.set_ylabel("Success rate")
        success_axis.set_ylim(-0.05, 1.05)
        success_axis.grid(True, alpha=0.3)
        success_axis.legend()

    axes[-1].set_xlabel("Environment timesteps")

    figure.suptitle(args.title)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    print(f"Mean comparison plot: {args.output}")


if __name__ == "__main__":
    main()
