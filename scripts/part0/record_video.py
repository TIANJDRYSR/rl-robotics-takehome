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
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record back-to-back into a single video (each reset with seed + episode_index).",
    )

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

    frames = []
    episode_returns = []
    episode_lengths = []

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            frames.append(env.render())

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

                frame = env.render()
                frames.append(frame)

                episode_return += float(reward)
                episode_length += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

    finally:
        env.close()

    suffix = args.output.suffix.lower()

    if suffix == ".webm":
        imageio.mimsave(
            args.output,
            frames,
            fps=args.fps,
            codec="libvpx-vp9",
            pixelformat="yuv420p",
            output_params=[
                "-crf",
                "30",
                "-b:v",
                "0",
            ],
            macro_block_size=16,
        )
    elif suffix == ".mp4":
        imageio.mimsave(
            args.output,
            frames,
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=[
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-movflags",
                "+faststart",
            ],
            macro_block_size=16,
        )
    else:
        raise ValueError(
            f"Unsupported video extension: {suffix}. "
            "Use .webm or .mp4."
        )

    print("=" * 70)
    print(f"Video:          {args.output}")
    print(f"Seed:           {args.seed}")
    print(f"Episodes:       {args.episodes}")
    for idx, (ep_return, ep_length) in enumerate(zip(episode_returns, episode_lengths)):
        print(f"  Episode {idx:03d}: return={ep_return:9.4f}, length={ep_length}")
    print(f"Frames:         {len(frames)}")
    print(f"Duration (s):   {len(frames) / args.fps:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
