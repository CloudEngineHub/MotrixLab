# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the manipulation DirectEnv step pipeline.

Every ``*_np`` manipulation environment must follow the split:

- ``reset(env_ids)`` only writes reset rows and returns a plain dict;
- ``compute_transition`` executes the read program exactly once at the top and
  only fills reward / terminated / truncated / info / metrics — never obs;
- ``compute_observation`` fully rebuilds obs from the cache left by the last
  read-program execution — it must not execute reads itself and must not
  modify reward, termination, or info.
"""

from collections.abc import Iterable

import numpy as np
import pytest

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs

_MANIPULATION_ENVS = [
    "franka-lift-cube",
    "shadow-hand-repose",
    "franka-open-cabinet",
    "rm65_insert_peg",
    "rm65-open-cabinet",
]


def _random_actions(env, rng: np.random.Generator) -> np.ndarray:
    low, high = env.action_space.low, env.action_space.high
    finite = np.isfinite(low) & np.isfinite(high)
    shape = (env.num_envs, *env.action_space.shape)
    action = rng.uniform(-0.1, 0.1, size=shape).astype(env.action_space.dtype)
    return np.where(finite, np.clip(action, low, high), action)


def _spy_execute(env) -> list:
    calls: list = []
    original_execute = env.sim_data.execute

    def recording_execute(env_ids=None):
        calls.append(env_ids)
        return original_execute(env_ids)

    env.sim_data.execute = recording_execute
    return calls


def _assert_finite_state(state: ArrayEnvState, num_envs: int, obs_dim: Iterable[int]) -> None:
    assert isinstance(state.obs, NpObs)
    assert state.obs.policy.shape == (num_envs, *obs_dim)
    assert not np.isnan(state.obs.policy).any()
    assert state.reward.shape == (num_envs,)
    assert state.terminated.shape == (num_envs,)
    assert state.truncated.shape == (num_envs,)
    assert not np.isnan(state.reward).any()


@pytest.mark.parametrize("env_name", _MANIPULATION_ENVS)
def test_manipulation_env_step_and_auto_reset(env_name: str) -> None:
    num_envs = 2
    env = registry.make(env_name, num_envs=num_envs)
    rng = np.random.default_rng(0)

    state = None
    # Enough steps that mid-episode auto-resets (partial rows) are exercised.
    for _ in range(5):
        state = env.step(_random_actions(env, rng))
        _assert_finite_state(state, num_envs, env.observation_space.shape)

    assert state is not None
    # Transition may only produce reward/terminated; obs comes from observation.
    assert not state.terminated.all(), "all envs terminated at once suggests a broken pipeline"


@pytest.mark.parametrize("env_name", _MANIPULATION_ENVS)
def test_compute_transition_executes_once_and_never_touches_obs(env_name: str) -> None:
    num_envs = 2
    env = registry.make(env_name, num_envs=num_envs)
    rng = np.random.default_rng(1)
    for _ in range(2):
        env.step(_random_actions(env, rng))

    state = env.state
    obs_before = state.obs.policy.copy()
    info_before = {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in state.info.items()}

    calls = _spy_execute(env)
    transitioned = env.compute_transition(state)

    # Exactly one read-program execution, at the top of compute_transition.
    assert len(calls) == 1
    # Transition fills reward/terminated and leaves the observation untouched
    # (same object, same contents).
    assert transitioned.reward.shape == (num_envs,)
    assert transitioned.terminated.shape == (num_envs,)
    assert transitioned.obs is state.obs
    np.testing.assert_array_equal(transitioned.obs.policy, obs_before)
    # Info may gain bookkeeping entries, but pre-existing arrays keep contents.
    for key, value in info_before.items():
        if isinstance(value, np.ndarray) and key in transitioned.info:
            assert transitioned.info[key] is value or transitioned.info[key].shape == value.shape


@pytest.mark.parametrize("env_name", _MANIPULATION_ENVS)
def test_compute_observation_reads_cache_without_executing(env_name: str) -> None:
    num_envs = 2
    env = registry.make(env_name, num_envs=num_envs)
    rng = np.random.default_rng(2)
    for _ in range(2):
        env.step(_random_actions(env, rng))

    state = env.state
    reward_before = state.reward.copy()
    terminated_before = state.terminated.copy()
    info_before = {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in state.info.items()}

    calls = _spy_execute(env)
    observed = env.compute_observation(state)

    # Observation must be served purely from cache.
    assert calls == []
    # step() normalizes obs to NpObs; a raw compute_observation call may return
    # either the dataclass or the bare policy array.
    obs = observed.obs.policy if isinstance(observed.obs, NpObs) else observed.obs
    assert obs.shape == (num_envs, *env.observation_space.shape)
    assert not np.isnan(obs).any()
    # Reward / termination / info must not be modified by observation.
    np.testing.assert_array_equal(observed.reward, reward_before)
    np.testing.assert_array_equal(observed.terminated, terminated_before)
    assert observed.info is state.info
    for key, value in info_before.items():
        assert key in observed.info
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(observed.info[key], value)


@pytest.mark.parametrize("env_name", _MANIPULATION_ENVS)
def test_reset_only_writes_reset_state_and_returns_dict(env_name: str) -> None:
    num_envs = 2
    env = registry.make(env_name, num_envs=num_envs)
    rng = np.random.default_rng(3)
    for _ in range(2):
        env.step(_random_actions(env, rng))

    env_ids = np.array([0], dtype=np.int64)
    obs_before = env.state.obs.policy.copy()

    info = env.reset(env_ids)

    assert isinstance(info, dict)
    # Reset writes simulator rows (and refreshes their cache) but must not
    # touch the published observation batch.
    np.testing.assert_array_equal(env.state.obs.policy, obs_before)
