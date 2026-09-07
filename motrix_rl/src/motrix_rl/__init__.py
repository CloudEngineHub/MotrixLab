# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from . import fastsac, rslrl, skrl  # noqa: F401  (register RL frameworks/trainers)
from .frameworks import register_framework  # noqa: F401
from .rslrl.cfg import (  # noqa: F401
    RslRlActorCfg,
    RslRlCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslrlRunnerCfg,
)
