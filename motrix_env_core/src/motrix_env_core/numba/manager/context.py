# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_env_core.numba.kernel_data import Map, kernel_data
from motrix_env_core.numba.manager.rand import RandValue


@kernel_data
class ManagerContext:
    """Framework-owned manager data available to every compiled term."""

    env_id: np.ndarray
    actions: Map
    commands: Map
    metrics: Map[np.ndarray]
    rand: RandValue
    sim: Map
    dt: np.float32
    # Writable per-lane flag: a command term sets it inside a fused kernel to
    # request simulator-state rematerialization for that lane after the
    # transition. It is not an episode reset: episode bookkeeping and
    # action-term state are untouched (see CommandTerm.advance).
    sim_reset_requested: np.ndarray


__all__ = ["ManagerContext"]
