"""Render a random-action rollout in Reacher-v5."""

from __future__ import annotations

import gymnasium as gym


def main() -> None:
    env = gym.make("Reacher-v5", render_mode="human")
    observation, info = env.reset(seed=0)
    env.action_space.seed(0)

    for _ in range(500):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
