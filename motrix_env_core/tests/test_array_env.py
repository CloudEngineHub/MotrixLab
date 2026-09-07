# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ArrayEnv frontend hierarchy (issue #222 MR1)."""

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.numba.manager.env import ManagerEnv
from motrix_env_core.registry import _infer_sim_backend


def test_np_env_inherits_array_env() -> None:
    assert issubclass(DirectEnv, ArrayEnv)


def test_manager_env_is_array_env_sibling_of_np_env() -> None:
    assert issubclass(ManagerEnv, ArrayEnv)
    assert not issubclass(ManagerEnv, DirectEnv)


def test_registry_infers_np_backend_for_manager_env() -> None:
    assert _infer_sim_backend(ManagerEnv) == "np"
    assert _infer_sim_backend(DirectEnv) == "np"
