# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import assert_type

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.numba.kernel import NumbaKernelOutputs, clone_kernel_value


def check_numba_env_types(env: ManagerEnv[EnvCfg], state: ArrayEnvState) -> None:
    assert_type(env.compute_transition(state), ArrayEnvState)
    assert_type(env.compute_observation(state), ArrayEnvState)


def check_clone_kernel_value_type(outputs: NumbaKernelOutputs) -> None:
    assert_type(clone_kernel_value(outputs), NumbaKernelOutputs)
