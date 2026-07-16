"""Run Gymnasium's environment checker."""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import noisy_goal_reacher


def main() -> None:
    env = gym.make(
        "NoisyGoalReacher-v0",
        reward_type="dense",
        noise_sigma=0.005,
    )

    # check_env should inspect the base environment rather than TimeLimit.
    check_env(env.unwrapped)

    env.close()
    print("Gymnasium environment check passed.")


if __name__ == "__main__":
    main()
