"""Registration for the NoisyGoalReacher environment."""

from gymnasium.envs.registration import register, registry


ENV_ID = "NoisyGoalReacher-v0"


if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="noisy_goal_reacher.env:NoisyGoalReacherEnv",
        max_episode_steps=50,
    )
