# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

_DIR = os.path.dirname(__file__)


@configclass
class LqrBaseCfg(DirectEnvCfg):
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.03
    max_episode_seconds: float = None
    control_cost_coef: float = 0.1
    velocity_cost_coef: float = 0.05
    reset_position_norm: float = 2.0**0.5
    boundary_position_limit: float = 1.2
    boundary_velocity_limit: float = 8.0
    success_position_tol: float = 0.06
    success_velocity_tol: float = 0.05
    success_bonus: float = 3.0
    out_of_bounds_penalty: float = 2.0
    expected_nq: int = 0
    expected_nu: int = 0


@registry.envcfg("dm-lqr-2-1")
@configclass
class Lqr21Cfg(LqrBaseCfg):
    """Stabilize a linear system with two state and one control dimensions.

    zh_CN: 稳定具有 2 维状态和 1 维控制的线性系统。
    """

    scene: SceneCfg = SceneCfg(file=os.path.join(_DIR, "lqr_2_1.xml"))
    reset_position_norm: float = 0.8
    control_cost_coef: float = 0.15
    velocity_cost_coef: float = 0.15
    boundary_position_limit: float = 1.15
    boundary_velocity_limit: float = 6.0
    success_position_tol: float = 0.04
    success_velocity_tol: float = 0.03
    success_bonus: float = 4.0
    out_of_bounds_penalty: float = 3.0
    expected_nq: int = 2
    expected_nu: int = 1


@registry.envcfg("dm-lqr-6-2")
@configclass
class Lqr62Cfg(LqrBaseCfg):
    """Stabilize a linear system with six state and two control dimensions.

    zh_CN: 稳定具有 6 维状态和 2 维控制的线性系统。
    """

    scene: SceneCfg = SceneCfg(file=os.path.join(_DIR, "lqr_6_2.xml"))
    reset_position_norm: float = 1.0
    velocity_cost_coef: float = 0.08
    boundary_position_limit: float = 1.2
    boundary_velocity_limit: float = 8.0
    success_position_tol: float = 0.1
    success_velocity_tol: float = 0.06
    success_bonus: float = 5.0
    out_of_bounds_penalty: float = 3.0
    expected_nq: int = 6
    expected_nu: int = 2
