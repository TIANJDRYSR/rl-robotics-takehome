# play.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

# 注册自定义环境
import noisy_goal_reacher  # noqa: F401

# 修改成你自己的 Custom PPO Agent 所在文件
from train_custom_ppo import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Custom PPO or Stable-Baselines3 PPO.")
    parser.add_argument("--algo", type=str, required=True, choices=["custom", "sb3"], help="Which implementation produced the model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--env-id", type=str, default="NoisyGoalReacher-v0")
    parser.add_argument("--goal-radius-min", type=float, default=0.05, help="NoisyGoalReacherEnv goal annulus inner radius.")
    parser.add_argument("--goal-radius-max", type=float, default=0.18, help="NoisyGoalReacherEnv goal annulus outer radius.")
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="Stddev of observation noise on the fingertip position.")
    parser.add_argument("--success-threshold", type=float, default=0.04, help="Distance (m) under which an episode counts as a success.")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"], help="Must match how the model was trained -- it changes what mean_return means, not what counts as a success.")
    parser.add_argument("--success-bonus", type=float, default=0.0, help="Dense-reward arrival bonus added once distance < success-threshold. Default 0.0 reproduces the original dense reward (distance + control penalty only); pass 3.0 to evaluate checkpoints trained with --success-bonus 3.0.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "video", "none"],
        help="'human' opens the MuJoCo window, 'video' saves MP4 files, 'none' only evaluates numerically.",
    )
    parser.add_argument("--video-dir", type=str, default="videos/play")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save per-episode results and summary stats as JSON.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stochastic actions instead of deterministic actions.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(device_name)


def make_play_env(env_id: str, render_mode: str, video_dir: str, env_kwargs: dict | None = None) -> gym.Env:
    env_kwargs = env_kwargs or {}

    if render_mode == "human":
        env = gym.make(env_id, render_mode="human", **env_kwargs)
    elif render_mode == "video":
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda episode_id: True,
            name_prefix="play",
            disable_logger=False,
        )
    else:
        env = gym.make(env_id, **env_kwargs)
    return env


class SB3Policy:
    def __init__(self, model_path: str, device: str, stochastic: bool) -> None:
        self.model = PPO.load(model_path, device=device)
        self.deterministic = not stochastic

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action)


class CustomPolicy:
    def __init__(
        self, model_path: str, env: gym.Env, device: torch.device, stochastic: bool, env_kwargs: dict | None = None
    ) -> None:
        self.device = device
        self.stochastic = stochastic
        env_kwargs = env_kwargs or {}

        # Agent 一般需要 observation_space 和 action_space。
        # 为了兼容原本基于 VectorEnv 的 Agent，这里创建一个只有一个环境的临时 SyncVectorEnv。
        env_id = env.spec.id if env.spec is not None else "NoisyGoalReacher-v0"
        self.agent_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id, **env_kwargs)])
        self.agent = Agent(self.agent_env).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.agent.load_state_dict(state_dict)
        self.agent.eval()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.stochastic:
                # 这里假设你的 Agent 提供 get_action_and_value。如果函数签名不同，按你的实现调整这一行。
                action, _, _, _ = self.agent.get_action_and_value(observation_tensor)
            else:
                # 确定性播放使用高斯策略的均值。
                action = self.agent.actor_mean(observation_tensor)

        return action.squeeze(0).cpu().numpy()

    def close(self) -> None:
        self.agent_env.close()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    device = resolve_device(args.device)

    # 只有 NoisyGoalReacher-v0 认识这几个关键字参数；其他 env_id 保持默认构造。
    env_kwargs = {}
    if args.env_id == "NoisyGoalReacher-v0":
        env_kwargs = {
            "goal_radius_min": args.goal_radius_min,
            "goal_radius_max": args.goal_radius_max,
            "noise_sigma": args.noise_sigma,
            "success_threshold": args.success_threshold,
            "reward_type": args.reward_type,
            "success_bonus": args.success_bonus,
        }

    env = make_play_env(env_id=args.env_id, render_mode=args.render_mode, video_dir=args.video_dir, env_kwargs=env_kwargs)

    custom_policy = None
    if args.algo == "sb3":
        policy = SB3Policy(model_path=str(model_path), device=str(device), stochastic=args.stochastic)
    else:
        custom_policy = CustomPolicy(
            model_path=str(model_path), env=env, device=device, stochastic=args.stochastic, env_kwargs=env_kwargs
        )
        policy = custom_policy

    returns = []
    lengths = []
    successes = []

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            episode_success = False

            while not (terminated or truncated):
                action = policy.get_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if bool(info.get("is_success", False)):
                    episode_success = True

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(episode_success))

            end_reason = "terminated" if terminated else "truncated"
            print(
                f"Episode {episode + 1:03d} | "
                f"return={episode_return:9.3f} | "
                f"length={episode_length:4d} | "
                f"success={episode_success} | "
                f"end={end_reason}"
            )
    finally:
        env.close()
        if custom_policy is not None:
            custom_policy.close()

    print("\nEvaluation summary")
    print("-" * 50)
    print(f"Algorithm:       {args.algo}")
    print(f"Model:           {model_path}")
    print(f"Episodes:        {args.episodes}")
    print(f"Mean return:     {np.mean(returns):.3f}")
    print(f"Return std:      {np.std(returns):.3f}")
    print(f"Mean length:     {np.mean(lengths):.1f}")
    print(f"Success rate:    {np.mean(successes) * 100:.1f}%")

    if args.render_mode == "video":
        print(f"Videos saved to: {Path(args.video_dir).resolve()}")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "algo": args.algo,
            "model_path": str(model_path),
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
            "deterministic": not args.stochastic,
            "base_seed": args.seed,
            "episode_seeds": [args.seed + episode for episode in range(args.episodes)],
            "n_episodes": args.episodes,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "episode_returns": returns,
            "episode_lengths": lengths,
            "episode_successes": successes,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
