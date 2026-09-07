# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead host metrics sampled at training-panel refresh boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpuLoad:
    """CPU execution time observed over one sampling window.

    Utilization excludes idle, I/O-wait, and stolen virtual-CPU time. The
    equivalent logical CPU count makes the normalized percentage unambiguous
    on SMT systems.
    """

    utilization_percent: float
    used_logical_cpus: float
    logical_cpu_count: int
    physical_core_count: int | None
    iowait_percent: float
    steal_percent: float


@dataclass(frozen=True)
class _CpuTimes:
    total: int
    executing: int
    iowait: int
    steal: int


class CpuLoadSampler:
    """Sample Linux CPU counters for the logical CPUs available to this process."""

    def __init__(
        self,
        *,
        stat_path: str | Path = "/proc/stat",
        topology_root: str | Path = "/sys/devices/system/cpu",
        cpu_ids: set[int] | None = None,
    ) -> None:
        self._stat_path = Path(stat_path)
        self._topology_root = Path(topology_root)
        self._cpu_ids = cpu_ids if cpu_ids is not None else self._available_cpu_ids()
        self._physical_core_count = self._read_physical_core_count()
        self._previous = self._read_times()

    def sample(self) -> CpuLoad | None:
        """Return utilization since the previous call, or ``None`` when unavailable."""
        current = self._read_times()
        common_ids = self._previous.keys() & current.keys()
        if not common_ids:
            self._previous = current
            return None

        total = sum(current[cpu].total - self._previous[cpu].total for cpu in common_ids)
        executing = sum(current[cpu].executing - self._previous[cpu].executing for cpu in common_ids)
        iowait = sum(current[cpu].iowait - self._previous[cpu].iowait for cpu in common_ids)
        steal = sum(current[cpu].steal - self._previous[cpu].steal for cpu in common_ids)
        self._previous = current
        if total <= 0:
            return None

        logical_cpu_count = len(common_ids)
        utilization_percent = 100.0 * executing / total
        return CpuLoad(
            utilization_percent=utilization_percent,
            used_logical_cpus=logical_cpu_count * utilization_percent / 100.0,
            logical_cpu_count=logical_cpu_count,
            physical_core_count=self._physical_core_count,
            iowait_percent=100.0 * iowait / total,
            steal_percent=100.0 * steal / total,
        )

    @staticmethod
    def _available_cpu_ids() -> set[int]:
        if hasattr(os, "sched_getaffinity"):
            return set(os.sched_getaffinity(0))
        return set(range(os.cpu_count() or 1))

    def _read_times(self) -> dict[int, _CpuTimes]:
        try:
            lines = self._stat_path.read_text().splitlines()
        except OSError:
            return {}

        times: dict[int, _CpuTimes] = {}
        for line in lines:
            label, *raw_fields = line.split()
            if not label.startswith("cpu") or not label[3:].isdigit():
                continue
            cpu_id = int(label[3:])
            if cpu_id not in self._cpu_ids:
                continue
            fields = [int(value) for value in raw_fields[:8]]
            fields.extend([0] * (8 - len(fields)))
            idle, iowait, steal = fields[3], fields[4], fields[7]
            total = sum(fields)
            times[cpu_id] = _CpuTimes(
                total=total,
                executing=total - idle - iowait - steal,
                iowait=iowait,
                steal=steal,
            )
        return times

    def _read_physical_core_count(self) -> int | None:
        cores: set[tuple[int, int]] = set()
        try:
            for cpu_id in self._cpu_ids:
                topology = self._topology_root / f"cpu{cpu_id}" / "topology"
                package_id = int((topology / "physical_package_id").read_text())
                core_id = int((topology / "core_id").read_text())
                cores.add((package_id, core_id))
        except (OSError, ValueError):
            return None
        return len(cores)
