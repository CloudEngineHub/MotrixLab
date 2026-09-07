# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fixed-step and absolute-deadline loop schedulers."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from motrix_deploy.errors import ValidationError


class LoopScheduler(ABC):
    """Provide control time and optional wall-clock pacing."""

    def __init__(self, period_s: float) -> None:
        if period_s <= 0:
            raise ValidationError("scheduler.period_s", "a positive value", period_s)
        self.period_s = period_s
        self.overrun_count = 0

    @abstractmethod
    def reset(self) -> None:
        """Reset timing state before a rollout."""

    @abstractmethod
    def wait(self, step: int) -> None:
        """Wait until the tick's deadline if pacing is enabled."""

    def elapsed_time_s(self, step: int) -> float:
        return step * self.period_s


class FixedStepScheduler(LoopScheduler):
    """Advance deterministic control time without sleeping."""

    def reset(self) -> None:
        self.overrun_count = 0

    def wait(self, step: int) -> None:
        del step


class RealtimeScheduler(LoopScheduler):
    """Pace ticks against absolute monotonic deadlines to avoid drift."""

    def __init__(
        self,
        period_s: float,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(period_s)
        self._period_ns = round(period_s * 1e9)
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._start_ns = 0

    def reset(self) -> None:
        self.overrun_count = 0
        self._start_ns = self._clock_ns()

    def wait(self, step: int) -> None:
        deadline_ns = self._start_ns + step * self._period_ns
        remaining_ns = deadline_ns - self._clock_ns()
        if remaining_ns > 0:
            self._sleep(remaining_ns / 1e9)
        elif step > 0:
            self.overrun_count += 1
