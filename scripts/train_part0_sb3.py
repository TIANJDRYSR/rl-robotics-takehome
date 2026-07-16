"""Train an SB3 PPO agent on Gymnasium Reacher-v5."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


ENV_ID = "Reacher-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path("results") / "part0" / args.run_name
    checkpoint_dir = Path("checkpoints") / "part0" / args.run_name

    eval_log_dir = run_dir / "eval"
    best_model_dir = checkpoint_dir / "best"
    periodic_checkpoint_dir = checkpoint_dir / "periodic"
    tensorboard_dir = run_dir / "tensorboard"

    for directory in (
        run_dir,
        eval_log_dir,
        best_model_dir,
        periodic_checkpoint_dir,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # make_vec_env applies SB3's Monitor wrapper to each training env.
    train_env = make_vec_env(
        ENV_ID,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(run_dir / "monitor"),
    )

    # Evaluation must use a separate environment.
    eval_env = Monitor(gym.make(ENV_ID))
    eval_env.reset(seed=args.seed + 10_000)
    eval_env.action_space.seed(args.seed + 10_000)

    # EvalCallback counts vector-environment calls rather than individual
    # transitions, so divide the requested transition frequency by n_envs.
    callback_eval_freq = max(args.eval_freq // args.n_envs, 1)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=callback_eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(periodic_checkpoint_dir),
        name_prefix="ppo_reacher",
        verbose=1,
    )

    callbacks = CallbackList(
        [
            eval_callback,
            checkpoint_callback,
        ]
    )

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {
            "pi": [64, 64],
            "vf": [64, 64],
        },
    }

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    metadata = {
        "env_id": ENV_ID,
        "algorithm": "Stable-Baselines3 PPO",
        "run_name": args.run_name,
        "seed": args.seed,
        "total_timesteps_requested": args.total_timesteps,
        "n_envs": args.n_envs,
        "eval_freq_transitions": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "device": args.device,
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "policy_network": [64, 64],
            "value_network": [64, 64],
            "activation": "Tanh",
        },
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=args.run_name,
        )

        final_model_path = checkpoint_dir / "final_model"
        model.save(final_model_path)

        metadata["total_timesteps_actual"] = int(model.num_timesteps)

        with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print()
        print("=" * 70)
        print("Training complete")
        print(f"Final model: {final_model_path}.zip")
        print(f"Best model:  {best_model_dir / 'best_model.zip'}")
        print(f"Eval data:   {eval_log_dir / 'evaluations.npz'}")
        print("=" * 70)

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
