"""Deterministically evaluate a trained PPO policy on Reacher-v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


ENV_ID = "Reacher-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make(ENV_ID))
    model = PPO.load(args.model, device=args.device)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            observation, info = env.reset(seed=episode_seed)

            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_return += float(reward)
                episode_length += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            print(
                f"episode={episode_index:03d} "
                f"seed={episode_seed} "
                f"return={episode_return:.4f} "
                f"length={episode_length}"
            )

    finally:
        env.close()

    returns_array = np.asarray(episode_returns, dtype=np.float64)
    lengths_array = np.asarray(episode_lengths, dtype=np.int64)

    result = {
        "env_id": ENV_ID,
        "model_path": str(args.model),
        "deterministic": True,
        "base_seed": args.seed,
        "episode_seeds": [
            args.seed + index for index in range(args.episodes)
        ],
        "n_episodes": args.episodes,
        "mean_return": float(np.mean(returns_array)),
        "std_return": float(np.std(returns_array, ddof=1))
        if args.episodes > 1
        else 0.0,
        "min_return": float(np.min(returns_array)),
        "max_return": float(np.max(returns_array)),
        "mean_episode_length": float(np.mean(lengths_array)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print()
    print("=" * 70)
    print(f"Episodes:    {args.episodes}")
    print(f"Mean return: {result['mean_return']:.4f}")
    print(f"Std return:  {result['std_return']:.4f}")
    print(f"Saved to:    {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
