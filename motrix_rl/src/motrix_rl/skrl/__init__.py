# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.skrl import framework as _framework
from motrix_rl.skrl.config import SkrlCfg

_framework.register_framework()

__all__ = ["SkrlCfg"]
