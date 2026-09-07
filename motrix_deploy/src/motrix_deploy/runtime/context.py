# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-typed per-tick policy context."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from motrix_deploy.errors import ValidationError

CommandT = TypeVar("CommandT")


@dataclass(frozen=True)
class PolicyContext(Generic[CommandT]):
    """Deterministic per-tick inputs that do not come from robot state."""

    step: int
    elapsed_time_s: float
    command: CommandT

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValidationError("context.step", "a non-negative integer", self.step)
        if not np.isfinite(self.elapsed_time_s) or self.elapsed_time_s < 0:
            raise ValidationError("context.elapsed_time_s", "a non-negative finite value", self.elapsed_time_s)


__all__ = ["PolicyContext"]
