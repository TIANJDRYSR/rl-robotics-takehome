# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# Adapted from CleanRL's ppo_continuous_action.py
# Source: https://github.com/vwxyzjn/cleanrl  (docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)
# GAE (compute_gae) re-implemented from scratch; see "# implemented from scratch" below.
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from gymnasium.vector import AutoresetMode
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import noisy_goal_reacher

# scripts/part2/train_custom_ppo.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `checkpoints/custom/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "NoisyGoalReacher-v0"
    """the id of the environment"""
    goal_radius_min: float = 0.05
    """minimum radius of the goal annulus sampled by NoisyGoalReacherEnv"""
    goal_radius_max: float = 0.18
    """maximum radius of the goal annulus sampled by NoisyGoalReacherEnv"""
    noise_sigma: float = 0.005
    """stddev of the Gaussian noise added to the observed fingertip position"""
    success_threshold: float = 0.04
    """distance (meters) under which an episode counts as a success"""
    reward_type: str = "dense"
    """reward shaping mode used by NoisyGoalReacherEnv: 'dense' or 'sparse'"""
    success_bonus: float = 3.0
    """bonus added to the dense reward on success; set to 0 to reproduce the pre-bonus dense reward"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks (SB3's PPO uses a constant LR, so this defaults off to match)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function (SB3's PPO defaults clip_range_vf=None, i.e. unclipped, so this defaults off to match)"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    debug: bool = False

    # Periodic evaluation & checkpointing
    eval_freq: int = 50_000
    """evaluate the current deterministic policy every this many environment steps"""
    eval_episodes: int = 10
    """number of deterministic episodes run per periodic evaluation"""
    eval_seed: int | None = None
    """seed for the periodic evaluation env; defaults to seed + 10_000 if unset, so dense/sparse runs and different models see identical evaluation conditions"""
    checkpoint_freq: int = 100_000
    """save an intermediate checkpoint every this many environment steps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, env_kwargs=None):
    env_kwargs = env_kwargs or {}

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, str(REPO_ROOT / "videos" / "custom" / run_name))
        else:
            env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # 不做 obs/reward normalization，与 train_sb3.py (只用 Monitor 包装) 保持一致。
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

# implemented from scratch
def compute_gae(rewards,values,terminated,truncated,next_values,gamma,gae_lambda,):
    advantages = torch.zeros_like(rewards)
    T, num_envs = rewards.shape
    for env_idx in range(num_envs):
        last_advantage = 0.0
        for t in reversed(range(T)):
            reward = rewards[t, env_idx]
            value = values[t, env_idx]
            value_next = next_values[t, env_idx]
            # trminaltion
            if terminated[t, env_idx]:
                delta = reward - value
                advantage = delta

            # trauncation
            elif truncated[t, env_idx]:
                delta = (reward+ gamma * value_next- value)
                advantage = delta

            # nromal case
            else:
                delta = (reward+ gamma * value_next- value)
                advantage = (delta+ gamma* gae_lambda* last_advantage)

            advantages[t, env_idx] = advantage
            last_advantage = advantage
    returns = advantages + values
    return advantages, returns


def evaluate_agent(agent, env_id, env_kwargs, eval_episodes, eval_seed, device, gamma):
    """Run `eval_episodes` deterministic episodes on a fixed seed and return summary stats."""
    eval_env = make_env(env_id, 0, False, "eval", gamma, env_kwargs)()

    returns = []
    successes = []
    for episode in range(eval_episodes):
        obs, _ = eval_env.reset(seed=eval_seed + episode)
        terminated = truncated = False
        episode_return = 0.0
        episode_success = False

        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # 确定性播放：使用高斯策略的均值，不采样。
                action = agent.actor_mean(obs_tensor).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_return += float(reward)
            if bool(info.get("is_success", False)):
                episode_success = True

        returns.append(episode_return)
        successes.append(float(episode_success))

    eval_env.close()
    returns = np.array(returns, dtype=np.float64)
    return {
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std()),
        "success_rate": float(np.mean(successes)),
        "episode_returns": returns.tolist(),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.reward_type}__{args.seed}__{int(time.time())}"

    # 只有 NoisyGoalReacher-v0 认识这几个关键字参数；其他 env_id (如 HalfCheetah-v4) 保持默认构造。
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

    # 固定周期评估用的种子：不同 reward_type/模型在相同条件下评估，结果才可比。
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000

    # 与 scripts/part0, scripts/part2/train_sb3.py 保持同样的目录约定：
    # 模型存 checkpoints/custom/<run_name>/{periodic,best,<exp_name>.cleanrl_model}，
    # 日志与周期评估存 results/custom/<run_name>/{tensorboard,eval}/。
    checkpoint_dir = REPO_ROOT / "checkpoints" / "custom" / run_name
    periodic_checkpoint_dir = checkpoint_dir / "periodic"
    best_checkpoint_dir = checkpoint_dir / "best"
    results_dir = REPO_ROOT / "results" / "custom" / run_name
    tensorboard_dir = results_dir / "tensorboard"
    eval_log_dir = results_dir / "eval"
    for directory in (checkpoint_dir, periodic_checkpoint_dir, best_checkpoint_dir, tensorboard_dir, eval_log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(str(tensorboard_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, env_kwargs) for i in range(args.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs),dtype=torch.bool,device=device,)
    truncateds = torch.zeros((args.num_steps, args.num_envs),dtype=torch.bool,device=device,)
    truncated_next_values = torch.zeros_like(values)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # 周期性评估 / checkpoint 状态
    next_eval_step = args.eval_freq
    next_checkpoint_step = args.checkpoint_freq
    best_success_rate = -1.0
    last_eval_metrics = None
    eval_history = []
    eval_log_path = eval_log_dir / "evaluations.json"
    best_model_path = best_checkpoint_dir / f"{args.exp_name}.cleanrl_model"

    def run_periodic_eval(step):
        global best_success_rate, last_eval_metrics
        metrics = evaluate_agent(agent, args.env_id, env_kwargs, args.eval_episodes, eval_seed, device, args.gamma)
        last_eval_metrics = metrics
        print(
            f"eval@{step}: success_rate={metrics['success_rate']:.2f} "
            f"mean_return={metrics['mean_return']:.2f} return_std={metrics['return_std']:.2f}"
        )
        writer.add_scalar("eval/success_rate", metrics["success_rate"], step)
        writer.add_scalar("eval/mean_return", metrics["mean_return"], step)
        writer.add_scalar("eval/return_std", metrics["return_std"], step)

        eval_history.append(
            {
                "step": step,
                "mean_return": metrics["mean_return"],
                "return_std": metrics["return_std"],
                "success_rate": metrics["success_rate"],
            }
        )
        with eval_log_path.open("w", encoding="utf-8") as file:
            json.dump(eval_history, file, indent=2)

        if args.save_model and metrics["success_rate"] > best_success_rate:
            best_success_rate = metrics["success_rate"]
            torch.save(agent.state_dict(), str(best_model_path))
            print(f"new best model (success_rate={best_success_rate:.2f}) saved to {best_model_path}")

        return metrics

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            terminateds[step] = torch.as_tensor(terminations,device=device,dtype=torch.bool,)
            truncateds[step] = torch.as_tensor(truncations,device=device,dtype=torch.bool,)

            if np.any(truncations):
                with torch.no_grad():
                    for env_idx in np.where(truncations)[0]:
                        final_obs = torch.as_tensor(
                            infos["final_obs"][env_idx],
                            device=device,
                            dtype=torch.float32,
                        ).unsqueeze(0)

                        truncated_next_values[step, env_idx] = (
                            agent.get_value(final_obs).item()
                        )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in infos:
                        episode_mask = infos.get(
                            "_episode",
                            np.ones(args.num_envs, dtype=bool),
                        )

                        for env_idx in np.where(episode_mask)[0]:
                            episodic_return = float(infos["episode"]["r"][env_idx])
                            episodic_length = int(infos["episode"]["l"][env_idx])

                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={episodic_return:.2f}, "
                                f"episodic_length={episodic_length}"
                            )

                            writer.add_scalar(
                                "charts/episodic_return",
                                episodic_return,
                                global_step,
                            )

                            writer.add_scalar(
                                "charts/episodic_length",
                                episodic_length,
                                global_step,
                            )

        # bootstrap value if not done
        with torch.no_grad():
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            next_values[-1] = agent.get_value(next_obs).flatten()
            next_values[truncateds] = truncated_next_values[truncateds]
            advantages, returns = compute_gae(rewards, values, terminateds, truncateds, next_values, args.gamma, args.gae_lambda)


        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # 周期性确定性评估：学习曲线 (success_rate / mean_return / return_std vs. global_step)。
        while args.eval_freq > 0 and global_step >= next_eval_step:
            run_periodic_eval(global_step)
            next_eval_step += args.eval_freq

        # 周期性中间 checkpoint。
        while args.save_model and args.checkpoint_freq > 0 and global_step >= next_checkpoint_step:
            checkpoint_path = periodic_checkpoint_dir / f"{args.exp_name}_{global_step}_steps.cleanrl_model"
            torch.save(agent.state_dict(), str(checkpoint_path))
            print(f"checkpoint saved to {checkpoint_path}")
            next_checkpoint_step += args.checkpoint_freq

    # 训练结束后再跑一次确定性评估，确保学习曲线覆盖到 total_timesteps，并检查是否刷新了 best model。
    # 若上面的周期评估恰好已经在这个 global_step 跑过，则跳过，避免重复。
    if args.eval_freq > 0 and (not eval_history or eval_history[-1]["step"] != global_step):
        run_periodic_eval(global_step)

    if args.save_model:
        model_path = str(checkpoint_dir / f"{args.exp_name}.cleanrl_model")
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            video_dir = str(REPO_ROOT / "videos" / "custom" / f"{run_name}-eval")
            episodic_returns = last_eval_metrics["episode_returns"] if last_eval_metrics else []
            push_to_hub(args, episodic_returns, repo_id, "PPO", str(checkpoint_dir), video_dir)

    envs.close()
    writer.close()