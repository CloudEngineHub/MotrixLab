# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unified policy deployment runtime."""

from motrix_deploy.runtime.context import PolicyContext
from motrix_deploy.runtime.result import RolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, LoopScheduler, RealtimeScheduler


def __getattr__(name: str):
    if name == "ControlLoop":
        from motrix_deploy.runtime.loop import ControlLoop

        return ControlLoop
    raise AttributeError(name)


__all__ = [
    "ControlLoop",
    "FixedStepScheduler",
    "LoopScheduler",
    "PolicyContext",
    "RealtimeScheduler",
    "RolloutResult",
]
