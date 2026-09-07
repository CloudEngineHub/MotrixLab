# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.system_metrics import CpuLoadSampler


def test_cpu_load_sampler_uses_counter_deltas_for_available_cpus(tmp_path) -> None:
    stat_path = tmp_path / "stat"
    stat_path.write_text(
        "cpu  200 0 0 1800 0 0 0 0\ncpu0 100 0 0 900 0 0 0 0\ncpu1 100 0 0 900 0 0 0 0\ncpu2 100 0 0 900 0 0 0 0\n"
    )
    sampler = CpuLoadSampler(stat_path=stat_path, topology_root=tmp_path, cpu_ids={0, 1})

    stat_path.write_text(
        "cpu  330 0 0 1850 20 0 0 0\ncpu0 150 0 0 950 0 0 0 0\ncpu1 180 0 0 900 20 0 0 0\ncpu2 300 0 0 900 0 0 0 0\n"
    )

    load = sampler.sample()

    assert load is not None
    assert load.utilization_percent == 65.0
    assert load.used_logical_cpus == 1.3
    assert load.logical_cpu_count == 2
    assert load.physical_core_count is None
    assert load.iowait_percent == 10.0
    assert load.steal_percent == 0.0


def test_cpu_load_sampler_returns_none_without_elapsed_cpu_time(tmp_path) -> None:
    stat_path = tmp_path / "stat"
    contents = "cpu0 100 0 0 900 0 0 0 0\n"
    stat_path.write_text(contents)
    sampler = CpuLoadSampler(stat_path=stat_path, topology_root=tmp_path, cpu_ids={0})
    stat_path.write_text(contents)

    assert sampler.sample() is None
