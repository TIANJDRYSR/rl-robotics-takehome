"""Custom MuJoCo reaching environment."""
from __future__ import annotations
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
import numpy as np

class NoisyGoalReacherEnv(ReacherEnv):
    """Minimal custom environment based on Reacher-v5.

    At this stage, the environment behaves exactly like Reacher-v5.
    Custom goals, rewards, success criteria, and observation noise will
    be added incrementally.
    """
    def __init__(
        self,
        goal_radius_min: float = 0.05,
        goal_radius_max: float = 0.18,
        noise_sigma: float = 0.005,
        success_threshold: float = 0.04,
        reward_type: str = "dense",
        success_bonus: float = 3.0,
        **kwargs,
    ) -> None:

        """Initialize the custom reaching environment."""
        if goal_radius_min < 0.0:
            raise ValueError("goal_radius_min must be non-negative.")
        if goal_radius_max <= goal_radius_min:
            raise ValueError(
                "goal_radius_max must be greater than goal_radius_min."
            )
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative.")
        if success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive.")
        if reward_type not in {"dense", "sparse"}:
            raise ValueError(
                "reward_type must be either 'dense' or 'sparse'."
            )
        if success_bonus < 0.0:
            raise ValueError("success_bonus must be non-negative.")

        self.goal_radius_min = float(goal_radius_min)
        self.goal_radius_max = float(goal_radius_max)
        self.noise_sigma = float(noise_sigma)
        self.success_threshold = float(success_threshold)
        self.reward_type = reward_type
        self.success_bonus = float(success_bonus)

        super().__init__(**kwargs)

 
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal uniformly by area from the goal annulus."""

        # Random direction around the robot base.
        angle = self.np_random.uniform(
            low=-np.pi,
            high=np.pi,
        )

        # Sample radius squared uniformly so points are uniform by area.
        radius_squared = self.np_random.uniform(
            low=self.goal_radius_min**2,
            high=self.goal_radius_max**2,
        )
        radius = np.sqrt(radius_squared)

        goal = np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
            ],
            dtype=np.float64,
        )

        return goal
    
    def reset_model(self) -> np.ndarray:
        """Reset the arm and sample a new goal for the episode."""
        # Randomize the initial MuJoCo position state.
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        # Sample a goal from our custom annulus.
        self.goal = self._sample_goal()
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        theta = self.data.qpos.flatten()[:2]   
        clean_fingertip_pos = self.get_body_com("fingertip")[:2]
        clean_target_pos = self.get_body_com("target")[:2]
        noise = self.np_random.normal(
            loc=0.0, scale=self.noise_sigma, size=2
        )
        noisy_fingertip_pos = clean_fingertip_pos + noise
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flatten()[2:],
                self.data.qvel.flatten()[:2],
                noisy_fingertip_pos - clean_target_pos,
            ]
        )
    def _is_success(self) -> bool:
        """Determine if the current state is a success."""
        clean_fingertip_pos = self.get_body_com("fingertip")[:2]
        clean_target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(clean_fingertip_pos - clean_target_pos)
        return bool(distance < self.success_threshold)
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info
        is_success = self._is_success()
        info = {
            **reward_info,
            "is_success": is_success,
        }
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info
    
    def _get_rew(self, action):
        # get the real fingertip and target positions (without noise)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        distance = float(np.linalg.norm(vec))
        is_success = distance < self.success_threshold
        reward_success_bonus = 0.0

        if self.reward_type == "sparse":
            reward_dist = 0.0 if is_success else -1.0
            reward_ctrl = 0.0
        elif self.reward_type == "dense":
            reward_dist = -distance * self._reward_dist_weight
            reward_ctrl = -np.square(action).sum() * self._reward_control_weight
            if is_success:
                reward_success_bonus = self.success_bonus

        reward = float(reward_dist + reward_ctrl + reward_success_bonus)

        reward_info = {
            "distance": distance,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_success_bonus": reward_success_bonus,
        }

        return reward, reward_info


