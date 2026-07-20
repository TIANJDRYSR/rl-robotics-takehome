"""Tests for compute_gae in scripts/part2/train_custom_ppo.py.

The real NoisyGoalReacher-v0 rollouts never set terminated=True (Reacher has
no early-termination condition, only a TimeLimit truncation), so the
terminated branch of compute_gae is never exercised by training or eval.
These tests build synthetic terminated/truncated rollouts by hand so that
branch has coverage.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "part2" / "train_custom_ppo.py"

_spec = importlib.util.spec_from_file_location("train_custom_ppo", MODULE_PATH)
_train_custom_ppo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_custom_ppo)
compute_gae = _train_custom_ppo.compute_gae

GAMMA = 0.99
LAMBDA = 0.95


def _tensor(rows):
    return torch.tensor(rows, dtype=torch.float32)


def _bool_tensor(rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_terminated_step_does_not_bootstrap():
    # Single terminated step: advantage must equal reward - value, ignoring
    # next_value entirely (there is no state after a true termination).
    rewards = _tensor([[1.0]])
    values = _tensor([[0.5]])
    next_values = _tensor([[999.0]])  # would blow up the result if used
    terminated = _bool_tensor([[True]])
    truncated = _bool_tensor([[False]])

    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, next_values, GAMMA, LAMBDA
    )

    assert advantages[0, 0].item() == 0.5
    assert returns[0, 0].item() == 1.0


def test_truncated_step_does_bootstrap():
    # Single truncated step: advantage must include gamma * next_value,
    # since the episode was cut off by the time limit, not a real end.
    rewards = _tensor([[1.0]])
    values = _tensor([[0.5]])
    next_values = _tensor([[2.0]])
    terminated = _bool_tensor([[False]])
    truncated = _bool_tensor([[True]])

    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, next_values, GAMMA, LAMBDA
    )

    expected = 1.0 + GAMMA * 2.0 - 0.5
    assert advantages[0, 0].item() == pytest.approx(expected)


def test_boundary_step_blocks_gae_recursion_into_earlier_steps():
    # Two-step episode ending in termination at t=1. The advantage at t=0
    # should use gamma*lambda*advantage[1] (standard recursive GAE), while
    # advantage[1] itself must not have bootstrapped past the terminal step.
    rewards = _tensor([[1.0], [1.0]])
    values = _tensor([[0.5], [0.6]])
    next_values = _tensor([[0.6], [999.0]])
    terminated = _bool_tensor([[False], [True]])
    truncated = _bool_tensor([[False], [False]])

    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, next_values, GAMMA, LAMBDA
    )

    adv1 = 1.0 - 0.6
    delta0 = 1.0 + GAMMA * 0.6 - 0.5
    adv0 = delta0 + GAMMA * LAMBDA * adv1

    assert advantages[1, 0].item() == pytest.approx(adv1)
    assert advantages[0, 0].item() == pytest.approx(adv0)


def test_truncation_also_blocks_gae_recursion():
    # A truncation boundary must cut the backward recursion just like a
    # termination does -- the next episode's advantage should not leak
    # across the reset into the step before the cutoff.
    rewards = _tensor([[1.0], [1.0]])
    values = _tensor([[0.5], [0.6]])
    next_values = _tensor([[0.6], [2.0]])
    terminated = _bool_tensor([[False], [False]])
    truncated = _bool_tensor([[False], [True]])

    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, next_values, GAMMA, LAMBDA
    )

    adv1 = 1.0 + GAMMA * 2.0 - 0.6  # bootstrapped, but not lambda-recursed further
    delta0 = 1.0 + GAMMA * 0.6 - 0.5
    adv0 = delta0 + GAMMA * LAMBDA * adv1

    assert advantages[1, 0].item() == pytest.approx(adv1)
    assert advantages[0, 0].item() == pytest.approx(adv0)


def test_envs_are_independent():
    # Stacking two different single-env rollouts side by side must give the
    # same result as running compute_gae on each column separately -- guards
    # against an indexing bug mixing env_idx columns together.
    rewards = _tensor([[1.0, 2.0], [0.5, -1.0]])
    values = _tensor([[0.5, 1.0], [0.6, 0.3]])
    next_values = _tensor([[0.6, 0.3], [0.0, 5.0]])
    terminated = _bool_tensor([[False, False], [True, False]])
    truncated = _bool_tensor([[False, False], [False, True]])

    joint_advantages, joint_returns = compute_gae(
        rewards, values, terminated, truncated, next_values, GAMMA, LAMBDA
    )

    for env_idx in range(2):
        col = slice(env_idx, env_idx + 1)
        solo_advantages, solo_returns = compute_gae(
            rewards[:, col],
            values[:, col],
            terminated[:, col],
            truncated[:, col],
            next_values[:, col],
            GAMMA,
            LAMBDA,
        )
        assert torch.allclose(joint_advantages[:, col], solo_advantages)
        assert torch.allclose(joint_returns[:, col], solo_returns)
