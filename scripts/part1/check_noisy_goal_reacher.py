"""Smoke test for the custom NoisyGoalReacher environment."""

import gymnasium as gym

# Importing the package performs environment registration.
import noisy_goal_reacher


def main() -> None:
    env = gym.make("NoisyGoalReacher-v0")

    observation, info = env.reset(seed=0)
    env.action_space.seed(0)

    print(f"Wrapped environment: {type(env)}")
    print(f"Base environment:    {type(env.unwrapped)}")
    print(f"Observation space:   {env.observation_space}")
    print(f"Action space:        {env.action_space}")
    print(f"Observation shape:   {observation.shape}")
    print(f"Observation dtype:   {observation.dtype}")
    print(f"Initial info:        {info}")

    assert env.observation_space.contains(observation)

    for step_index in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            print(
                f"Episode ended at loop step {step_index}: "
                f"terminated={terminated}, truncated={truncated}"
            )
            observation, info = env.reset()

    env.close()
    print("Minimal custom environment smoke test passed.")


if __name__ == "__main__":
    main()
