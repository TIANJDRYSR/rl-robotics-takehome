# NoisyGoalReacher — RL Robotics Take-home

A custom noisy goal-reaching environment built on MuJoCo `Reacher-v5`, with a custom PPO
implementation (GAE implemented from scratch) trained and compared against a Stable-Baselines3
baseline, plus reward and robustness studies.

The full write-up (environment design, algorithm, experiments, failure modes, future work) is in
the report:

- **`rl_robotics_report_TIANYUXIN.pdf`** — full report (English)

---

## 1. Setup

The project uses Conda to manage the Python environment (Python 3.11).

Recommended (core environment):

```bash
conda env create -f environment.yml
conda activate rl-robotics-takehome
```

To reproduce the exact dependency versions used in the experiments (full snapshot):

```bash
conda env create -f environment-lock.yml
conda activate rl-robotics-takehome
```

> Only Gymnasium + MuJoCo are used for simulation. Isaac Sim / Isaac Gym are not used.
> A GPU is optional; all experiments are small enough to run on CPU.

---

## 2. Repository Structure

```text
.
├── README.md
├── pyproject.toml                      # src-layout package definition
├── environment.yml                     # core dependencies
├── environment-lock.yml                # full pinned snapshot
├── rl_robotics_report_TIANYUXIN.pdf           # report
├── configs/
│   ├── part0_commands.txt              # Part 0 reproducible commands
│   ├── part1_commands.txt              # Part 1 reproducible commands
│   └── part2_commands.txt              # Part 2/3 reproducible commands
├── src/
│   └── noisy_goal_reacher/             # custom environment package
│       └── env.py                      # NoisyGoalReacherEnv
├── scripts/
│   ├── part0/                          # toolchain check (SB3 PPO on Reacher-v5)
│   │   ├── train_sb3.py
│   │   ├── evaluate.py
│   │   ├── plot_curve.py
│   │   ├── record_video.py
│   │   ├── render_random_reacher.py
│   │   └── smoke_test_reacher.py
│   ├── part1/
│   │   ├── check_noisy_goal_reacher.py # custom smoke test (100-step rollout)
│   │   └── check_custom_env.py         # gymnasium env_checker
│   └── part2/
│       ├── train_custom_ppo.py         # custom PPO (GAE implemented from scratch)
│       ├── train_sb3.py                # SB3 baseline
│       ├── play.py
│       ├── plot_comparison.py
│       ├── plot_curve_json.py
│       └── plot_reward_type_comparison.py
├── tests/
│   └── part1/
│       └── test_noisy_goal_reacher.py  # unit tests (pytest)
├── checkpoints/                        # trained model checkpoints
├── results/                            # evaluation JSON, learning-curve data
└── videos/                             # rollout videos
```

---

## 3. Reproducing the Experiments

All full commands are collected in **`configs/part0_commands.txt`**, **`configs/part1_commands.txt`**,
and **`configs/part2_commands.txt`**. Run them from the repository root with the
`rl-robotics-takehome` environment activated. Representative commands are listed below.

### Part 0 — Toolchain check (SB3 PPO on Reacher-v5)

```bash
python scripts/part0/train_sb3.py \
  --run-name ppo_reacher_seed_0 \
  --seed 0 \
  --total-timesteps 500000 \
  --n-envs 4 \
  --eval-freq 10000 \
  --n-eval-episodes 10 \
  --device cuda

python scripts/part0/evaluate.py \
  --model checkpoints/part0/ppo_reacher_seed_0/best/best_model.zip \
  --output results/part0/ppo_reacher_seed_0/final_evaluation.json \
  --episodes 50 \
  --seed 20000 \
  --device cuda

python scripts/part0/plot_curve.py \
  --evaluations results/part0/ppo_reacher_seed_0/eval/evaluations.npz \
  --output results/part0/ppo_reacher_seed_0/learning_curve.png \
  --csv-output results/part0/ppo_reacher_seed_0/learning_curve.csv

python scripts/part0/record_video.py \
  --model checkpoints/part0/ppo_reacher_seed_0/best/best_model.zip \
  --output videos/part0/ppo_reacher_seed_0.mp4 \
  --seed 30000 \
  --device cpu \
  --fps 50
```

### Part 1 — Custom environment check

```bash
# Registers NoisyGoalReacher-v0, resets/steps it for 100 iterations, and
# asserts every observation stays inside the declared observation space.
python scripts/part1/check_noisy_goal_reacher.py

# Runs Gymnasium's official env checker (gymnasium.utils.env_checker.check_env)
# against the unwrapped NoisyGoalReacherEnv (reward_type=dense, noise_sigma=0.005).
python scripts/part1/check_custom_env.py
```

### Part 2 — Custom PPO vs. SB3 baseline (dense reward)

```bash
# custom PPO (dense, seed 1)
python scripts/part2/train_custom_ppo.py --seed 1 --reward-type dense

# SB3 baseline (dense, seed 1)
python scripts/part2/train_sb3.py --seed 1 --reward-type dense
```

### Part 3 — Dense vs. sparse (3 seeds), success-bonus ablation, and noise sensitivity

See `configs/part2_commands.txt` for the full commands. It covers:

- Dense vs. sparse, 3 seeds each (seeds 1, 12, 1300).
- Success-bonus ablation: the same dense reward re-run with `--success-bonus 0` ("old" dense,
  no bonus on success) vs. the default `--success-bonus 3.0` ("new" dense), 3 seeds each, compared
  against sparse in the `dense_old_new_rew_vs_sparse_mean.png` plot.
- "Sensor-noise sweep" section: evaluating the best dense policy at σ ∈ {0, 0.005, 0.02}.

---

## 4. Tests

Unit tests (and the Gymnasium `env_checker`) can be run with:

```bash
pytest
```

Expected: 4 tests pass.

---

## 5. Environment Overview

`NoisyGoalReacher-v0` extends MuJoCo `Reacher-v5` with:

- **Randomized goal** sampled from an annulus (`goal_radius_min=0.05`, `goal_radius_max=0.18`),
  uniform in area; goal position is included in the observation.
- **Sensor noise**: zero-mean Gaussian noise (`noise_sigma=0.005`, configurable) added only to the
  observed end-effector position (actions and physics stay noise-free).
- **Success**: `is_success` when the true end-effector–goal distance `< success_threshold` (0.04 m),
  returned in `info` on every `step()`.
- **Two reward modes**: `reward_type="dense"` (distance + control penalty, plus a success bonus in
  the improved version) and `reward_type="sparse"` (0 / -1).

See the report for the full design rationale and experiments.

---

## 6. Attribution

- The custom PPO training script (`scripts/part2/train_custom_ppo.py`) is **adapted from CleanRL's**
  `ppo_continuous_action.py` — https://github.com/vwxyzjn/cleanrl
- The **GAE computation** (`compute_gae`) is implemented from scratch and marked with
  `# implemented from scratch` in the code.
```