# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSLRL integration module for MotrixLab.

This module provides configuration classes and utilities for using RSLRL
(ETH Zurich's RL library) with MotrixLab.

The configuration structure matches rsl_rl's flat format with separate
actor and critic configs at the top level.
"""

from motrix_rl.rslrl import framework as _framework
from motrix_rl.rslrl.cfg import (
    RslRlActorCfg,
    RslrlCfg,
    RslRlCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslrlRunnerCfg,
)

_framework.register_framework()

__all__ = [
    "RslRlActorCfg",
    "RslrlCfg",
    "RslRlCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslrlRunnerCfg",
    "field_override",
    "inherit_field",
    "configclass",
]
