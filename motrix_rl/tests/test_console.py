# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.console import TrainingPanelStats, _format_value, format_training_panel
from motrix_rl.system_metrics import CpuLoad


def test_format_value_uses_fixed_notation_for_regular_floats() -> None:
    assert _format_value(0.0) == "0.000"
    assert _format_value(0.001) == "0.001"
    assert _format_value(-12.3456, signed=True) == "-12.346"


def test_format_value_uses_scientific_notation_for_small_floats() -> None:
    assert _format_value(0.0003623) == "3.623e-04"
    assert _format_value(-0.00002, precision=4, signed=True) == "-2.0000e-05"


def test_format_value_uses_scientific_notation_for_large_floats() -> None:
    assert _format_value(1_000_000.0) == "1.000e+06"
    assert _format_value(-1_234_567.0, signed=True) == "-1.235e+06"


def test_format_training_panel_groups_timing_metrics() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={"collector": {"wait_ms": 4.0, "env_step_ms": 1.5}},
    )

    panel = format_training_panel(stats)

    assert "collector         wait_ms 4.000" in panel
    assert "env_step_ms 1.500" in panel
    assert "collector_wait_ms" not in panel
    assert "collect 2.0ms" not in panel  # standalone timing row suppressed for tree panels


def test_format_training_panel_renders_nested_timing_groups_as_tree() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={
            "learner(ms)": {
                "drain": 0.5,
                "update": {"total": 3.0, "critic_alpha": 2.0, "actor": 1.0},
            },
        },
    )

    panel = format_training_panel(stats)

    assert "drain 0.500" in panel
    assert sum(1 for line in panel.splitlines() if line.strip() == "└─ update 3.000") == 1
    child_lines = [line for line in panel.splitlines() if line.lstrip().startswith("└─ critic_alpha 2.000")]
    assert len(child_lines) == 1
    assert child_lines[0].startswith(" " * 22)
    assert "actor 1.000" in child_lines[0]


def test_format_training_panel_strips_markup_from_group_titles() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={"learner [magenta]3.0[/]ms": {"drain": 0.5}},
    )

    panel = format_training_panel(stats)

    assert "[magenta]" not in panel
    assert "learner 3.0ms" in panel
    assert "drain 0.500" in panel


def test_format_training_panel_shows_cpu_load_with_explicit_topology() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        cpu_load=CpuLoad(
            utilization_percent=50.0,
            used_logical_cpus=16.0,
            logical_cpu_count=32,
            physical_core_count=16,
            iowait_percent=1.5,
            steal_percent=0.25,
        ),
    )

    panel = format_training_panel(stats)

    assert "cpu  50.0% (16.0/32T, 16C)" in panel
    assert "iowait 1.5%" in panel
    assert "steal 0.2%" in panel
