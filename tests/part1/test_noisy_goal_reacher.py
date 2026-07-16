"""Tests for NoisyGoalReacherEnv."""

import gymnasium as gym
import numpy as np

import noisy_goal_reacher


def make_env(**kwargs):
    return gym.make(
        "NoisyGoalReacher-v0",
        **kwargs,
    )


def test_goal_is_inside_annulus() -> None:
    env = make_env(
        goal_radius_min=0.05,
        goal_radius_max=0.18,
    )

    for seed in range(50):
        env.reset(seed=seed)

        goal = env.unwrapped.goal
        radius = np.linalg.norm(goal)

        assert 0.05 <= radius <= 0.18

    env.close()


def test_observation_has_expected_shape() -> None:
    env = make_env(noise_sigma=0.005)

    observation, info = env.reset(seed=0)

    assert observation.shape == (10,)
    assert env.observation_space.contains(observation)

    env.close()


def test_sparse_reward_is_zero_or_minus_one() -> None:
    env = make_env(
        reward_type="sparse",
        noise_sigma=0.005,
    )

    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    _, reward, _, _, info = env.step(action)

    assert reward in {-1.0, 0.0}
    assert "is_success" in info
    assert reward == (0.0 if info["is_success"] else -1.0)

    env.close()


def test_info_always_contains_is_success() -> None:
    env = make_env()

    env.reset(seed=0)

    for _ in range(10):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)

        assert "is_success" in info
        assert isinstance(info["is_success"], bool)

        if terminated or truncated:
            env.reset()

    env.close()
