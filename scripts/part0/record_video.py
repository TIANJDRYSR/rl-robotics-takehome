"""Record one deterministic Reacher-v5 rollout as an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO


ENV_ID = "Reacher-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        width=640,
        height=480,
    )

    model = PPO.load(args.model, device=args.device)

    observation, info = env.reset(seed=args.seed)
    frames = [env.render()]

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0

    try:
        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            frame = env.render()
            frames.append(frame)

            episode_return += float(reward)
            episode_length += 1

    finally:
        env.close()

    imageio.mimsave(
        args.output,
        frames,
        fps=args.fps,
        codec="libx264",
        macro_block_size=None,
    )

    print("=" * 70)
    print(f"Video:         {args.output}")
    print(f"Seed:          {args.seed}")
    print(f"Episode return:{episode_return:.4f}")
    print(f"Episode length:{episode_length}")
    print(f"Frames:        {len(frames)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
