"""Basic smoke test for Gymnasium Reacher-v5."""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def main() -> None:
    env = gym.make("Reacher-v5")

    # Seed both reset randomness and random action sampling.
    observation, info = env.reset(seed=0)
    env.action_space.seed(0)

    print("=" * 60)
    print("Environment: Reacher-v5")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation dtype: {observation.dtype}")
    print(f"Initial observation: {observation}")
    print(f"Initial info: {info}")
    print("=" * 60)

    assert env.observation_space.contains(observation), (
        "Initial observation is outside the declared observation space."
    )

    total_reward = 0.0
    episode_count = 0

    for step in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        assert env.action_space.contains(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)

        total_reward += float(reward)

        if step < 3:
            print(
                f"step={step:03d} "
                f"action={action} "
                f"reward={reward:.4f} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

        if terminated or truncated:
            episode_count += 1
            observation, info = env.reset()

    env.close()

    print("=" * 60)
    print("Smoke test passed.")
    print(f"Completed episodes: {episode_count}")
    print(f"Total reward across 100 random steps: {total_reward:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
