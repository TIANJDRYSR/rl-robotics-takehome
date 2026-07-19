# train_sb3.py

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

# 注册自定义环境
import noisy_goal_reacher  # noqa: F401

# scripts/part2/train_sb3.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable-Baselines3 PPO on NoisyGoalReacher.")
    parser.add_argument("--env-id", type=str, default="NoisyGoalReacher-v0")
    parser.add_argument("--goal-radius-min", type=float, default=0.05, help="NoisyGoalReacherEnv goal annulus inner radius.")
    parser.add_argument("--goal-radius-max", type=float, default=0.18, help="NoisyGoalReacherEnv goal annulus outer radius.")
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="Stddev of observation noise on the fingertip position.")
    parser.add_argument("--success-threshold", type=float, default=0.04, help="Distance (m) under which an episode counts as a success.")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--exp-name", type=str, default="train_sb3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel training environments.")
    parser.add_argument("--num-steps", type=int, default=512, help="Rollout steps collected by each environment.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size used during PPO optimization.")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of optimization epochs per rollout.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate approximately every this many total environment steps. The script adjusts it for vectorized environments.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Save a checkpoint approximately every this many total environment steps.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-subproc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SubprocVecEnv. Enable this when the environment is computationally expensive and multiprocessing works correctly.",
    )
    parser.add_argument(
        "--check-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the SB3 environment checker before training.",
    )
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_single_env(
    env_id: str, seed: int, rank: int, monitor_dir: Path, env_kwargs: dict | None = None
) -> Callable[[], gym.Env]:
    """Return an environment factory compatible with DummyVecEnv/SubprocVecEnv."""
    env_kwargs = env_kwargs or {}

    def thunk() -> gym.Env:
        env = gym.make(env_id, **env_kwargs)

        # Monitor 负责记录 episodic return 和 episode length。
        monitor_file = monitor_dir / f"env_{rank}"
        env = Monitor(env, filename=str(monitor_file))

        # 为 action_space 单独设种子。
        env.action_space.seed(seed + rank)

        # SB3 会负责调用 reset()。这里不提前 reset，避免初始化时额外开始一个 episode。
        return env

    return thunk


def build_train_env(args: argparse.Namespace, monitor_dir: Path, env_kwargs: dict | None = None):
    env_fns = [
        make_single_env(env_id=args.env_id, seed=args.seed, rank=rank, monitor_dir=monitor_dir, env_kwargs=env_kwargs)
        for rank in range(args.num_envs)
    ]

    if args.use_subproc and args.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    # 汇总多个环境的 Monitor 统计。
    vec_env = VecMonitor(vec_env)
    vec_env.seed(args.seed)
    return vec_env


def build_eval_env(env_id: str, seed: int, eval_monitor_dir: Path, env_kwargs: dict | None = None):
    eval_env = DummyVecEnv(
        [make_single_env(env_id=env_id, seed=seed, rank=0, monitor_dir=eval_monitor_dir, env_kwargs=env_kwargs)]
    )
    eval_env = VecMonitor(eval_env)
    eval_env.seed(seed)
    return eval_env


def validate_arguments(args: argparse.Namespace) -> None:
    rollout_size = args.num_envs * args.num_steps

    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")
    if args.batch_size > rollout_size:
        raise ValueError(f"batch_size={args.batch_size} is larger than the rollout buffer size={rollout_size}.")

    if rollout_size % args.batch_size != 0:
        print(
            "Warning: rollout size is not divisible by batch size.\n"
            f"rollout_size = {args.num_envs} × {args.num_steps} = {rollout_size}\n"
            f"batch_size = {args.batch_size}\n"
            "SB3 can still train, but the final minibatch will be smaller."
        )


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    set_global_seed(args.seed)

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    # 只有 NoisyGoalReacher-v0 认识这几个关键字参数；其他 env_id 保持默认构造。
    env_kwargs = {}
    if args.env_id == "NoisyGoalReacher-v0":
        env_kwargs = {
            "goal_radius_min": args.goal_radius_min,
            "goal_radius_max": args.goal_radius_max,
            "noise_sigma": args.noise_sigma,
            "success_threshold": args.success_threshold,
            "reward_type": args.reward_type,
        }

    # 与 scripts/part0/train_sb3.py 保持同样的目录约定：
    # 模型存 checkpoints/sb3/<run_name>/，日志与配置存 results/sb3/<run_name>/。
    model_dir = REPO_ROOT / "checkpoints" / "sb3" / run_name
    run_dir = REPO_ROOT / "results" / "sb3" / run_name

    periodic_checkpoint_dir = model_dir / "periodic"
    best_model_dir = model_dir / "best"
    tensorboard_dir = run_dir / "tensorboard"
    eval_log_dir = run_dir / "eval"
    train_monitor_dir = run_dir / "monitor" / "train"
    eval_monitor_dir = run_dir / "monitor" / "eval"

    for directory in [
        model_dir,
        periodic_checkpoint_dir,
        best_model_dir,
        run_dir,
        tensorboard_dir,
        eval_log_dir,
        train_monitor_dir,
        eval_monitor_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    rollout_size = args.num_envs * args.num_steps

    print("=" * 60)
    print("Stable-Baselines3 PPO training")
    print("=" * 60)
    print(f"Environment:       {args.env_id}")
    print(f"Run directory:     {run_dir}")
    print(f"Seed:              {args.seed}")
    print(f"Number of envs:    {args.num_envs}")
    print(f"Steps per env:     {args.num_steps}")
    print(f"Rollout size:      {rollout_size}")
    print(f"Minibatch size:    {args.batch_size}")
    print(f"Update epochs:     {args.update_epochs}")
    print(f"Total timesteps:   {args.total_timesteps}")
    print(f"Vector backend:    {'SubprocVecEnv' if args.use_subproc else 'DummyVecEnv'}")
    print(f"Device request:    {args.device}")
    print("=" * 60)

    if args.check_env:
        print("Checking environment compatibility...")
        check_env_instance = gym.make(args.env_id, **env_kwargs)
        try:
            check_env(check_env_instance, warn=True)
        finally:
            check_env_instance.close()
        print("Environment check completed.")

    train_env = build_train_env(args=args, monitor_dir=train_monitor_dir, env_kwargs=env_kwargs)
    eval_env = build_eval_env(
        env_id=args.env_id, seed=args.seed + 10_000, eval_monitor_dir=eval_monitor_dir, env_kwargs=env_kwargs
    )

    # Callback 的 save_freq/eval_freq 是按照 callback 调用次数计算。
    # 使用 n_envs 个并行环境时，每次 env.step() 会增加 n_envs 个 timestep，
    # 因此这里除以 num_envs，使频率近似对应总环境步数。
    adjusted_eval_freq = max(args.eval_freq // args.num_envs, 1)
    adjusted_checkpoint_freq = max(args.checkpoint_freq // args.num_envs, 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=adjusted_checkpoint_freq,
        save_path=str(periodic_checkpoint_dir),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    policy_kwargs = {
        # 与常见 CleanRL PPO MLP 结构接近：Actor 和 Critic 各自使用两个 64 单元隐藏层。
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": torch.nn.Tanh,
    }

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.num_steps,
        batch_size=args.batch_size,
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    # 训练配置快照，保存到 results/sb3/<run_name>/metadata.json，方便复现实验。
    metadata = {
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "algorithm": "Stable-Baselines3 PPO",
        "run_name": run_name,
        "seed": args.seed,
        "total_timesteps_requested": args.total_timesteps,
        "num_envs": args.num_envs,
        "hyperparameters": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
        },
        "device": args.device,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO",
            progress_bar=args.progress_bar,
            reset_num_timesteps=True,
        )

        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))  # SB3 自动添加 .zip 后缀。

        metadata["total_timesteps_actual"] = int(model.num_timesteps)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("\nTraining completed.")
        print(f"Final model: {final_model_path}.zip")
        print(f"Best model:  {best_model_dir / 'best_model.zip'}")
        print(f"TensorBoard: {tensorboard_dir}")
        print(f"Evaluation:  {eval_log_dir}")
        print(f"Config:      {metadata_path}")

    except KeyboardInterrupt:
        interrupted_model_path = model_dir / "interrupted_model"
        model.save(str(interrupted_model_path))
        print("\nTraining interrupted by user.")
        print(f"Interrupted model saved to: {interrupted_model_path}.zip")

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    # Windows/macOS 使用 SubprocVecEnv 时，main guard 必须保留。
    main()
