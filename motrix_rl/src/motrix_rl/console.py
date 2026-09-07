# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Console display helpers shared by RL training backends."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from motrix_rl.system_metrics import CpuLoad

try:  # optional pretty console; callers fall back to plain text if unavailable
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


@dataclass(frozen=True)
class TrainingPanelStats:
    """Structured stats consumed by the shared RL training panel.

    Fields required by the panel layout are explicit dataclass attributes.
    Backend-specific values belong in the extension dictionaries.
    """

    iteration: int
    total_iterations: int
    steps_per_second: float
    elapsed_seconds: float
    mean_return: float
    mean_episode_length: float
    episodes: int
    buffer_size: float
    buffer_capacity: float
    collect_ms: float
    learn_ms: float
    learn_percent: float
    warming: bool = False
    training_metrics: Mapping[str, Any] | None = None
    reward_terms: Mapping[str, Any] = field(default_factory=dict)
    env_metrics: Mapping[str, Any] = field(default_factory=dict)
    timing_metrics: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # Group name -> items. Item values are scalars, or a nested mapping rendered
    # as an indented sub-tree (e.g. per-process timing with stage breakdowns).
    timing_groups: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cpu_load: CpuLoad | None = None


def open_training_live():
    """Start a rich live panel on TTYs, otherwise return ``(None, None)``."""
    if not _RICH or not sys.stdout.isatty():
        return None, None
    console = Console()
    live = Live(console=console, auto_refresh=False, vertical_overflow="visible")
    live.start()
    return console, live


def emit_training_panel(live, stats: TrainingPanelStats, *, title: str = "rl") -> None:
    """Render one training stats panel using rich when ``live`` is available."""
    if live is not None:
        live.update(render_training_panel(stats, title=title), refresh=True)
    else:
        print(format_training_panel(stats, title=title))


def _format_value(value: Any, precision: int = 3, signed: bool = False) -> str:
    if isinstance(value, float):
        sign = "+" if signed else ""
        magnitude = abs(value)
        use_scientific = math.isfinite(value) and value != 0.0 and (magnitude < 10**-precision or magnitude >= 1e6)
        notation = "e" if use_scientific else "f"
        return f"{value:{sign}.{precision}{notation}}"
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    return str(value)


def _format_metric_items(items: Mapping[str, Any], *, precision: int = 3, signed: bool = False) -> list[str]:
    return [f"{k} {_format_value(v, precision=precision, signed=signed)}" for k, v in items.items()]


def format_training_panel(stats: TrainingPanelStats, *, title: str = "rl") -> str:
    """Render a plain-text RL training panel from backend-provided scalar stats."""

    def hms(t: float) -> str:
        t = int(t)
        h, m, sec = t // 3600, (t % 3600) // 60, t % 60
        return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"

    def si(n: float) -> str:
        for unit in ("", "k", "M"):
            if abs(n) < 1000:
                return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
            n /= 1000.0
        return f"{n:.1f}G"

    width = 84
    pct = 100.0 * stats.iteration / max(stats.total_iterations, 1)
    header = (
        f" {title} - iter {stats.iteration}/{stats.total_iterations} ({pct:.1f}%) - "
        f"{stats.steps_per_second:.0f} env-steps/s - {hms(stats.elapsed_seconds)}"
    )
    lines = [
        "-" * width,
        header,
        "-" * width,
        f" rollout     return {stats.mean_return:>9.3f}   "
        f"ep_len {stats.mean_episode_length:>7.1f}   episodes {stats.episodes:>6d}",
    ]
    if stats.cpu_load is not None:
        load = stats.cpu_load
        cores = f", {load.physical_core_count}C" if load.physical_core_count is not None else ""
        lines.append(
            f" system      cpu {load.utilization_percent:>5.1f}% "
            f"({load.used_logical_cpus:.1f}/{load.logical_cpu_count}T{cores})   "
            f"iowait {load.iowait_percent:.1f}%   steal {load.steal_percent:.1f}%"
        )
    metrics = stats.training_metrics
    buf = f"{si(stats.buffer_size)}/{si(stats.buffer_capacity)}"
    if metrics is None:
        message = "warmup - filling replay buffer" if stats.warming else "training metrics unavailable"
        lines.append(f" train       {message} ({buf})")
    else:
        toks = _format_metric_items(metrics)
        colw = max(len(t) for t in toks) + 2 if toks else 1
        per_line = max(1, (width - 13) // colw)
        for i in range(0, len(toks), per_line):
            prefix = " train       " if i == 0 else " " * 13
            row = "".join(t.ljust(colw) for t in toks[i : i + per_line])
            lines.append((prefix + row).rstrip())
        lines.append(f" buffer      {buf}")

    terms = stats.reward_terms
    if terms:
        sorted_terms = dict(sorted(terms.items(), key=lambda kv: -abs(float(kv[1]))))
        toks = _format_metric_items(sorted_terms, precision=4, signed=True)
        colw = max(len(t) for t in toks) + 2
        per_line = max(1, (width - 13) // colw)
        for i in range(0, len(toks), per_line):
            prefix = " reward(dt)  " if i == 0 else " " * 13
            row = "".join(t.ljust(colw) for t in toks[i : i + per_line])
            lines.append((prefix + row).rstrip())

    env_metrics = stats.env_metrics
    if env_metrics:
        toks = _format_metric_items(dict(sorted(env_metrics.items())), signed=True)
        lines.append(" metrics     " + "   ".join(toks))

    # The standalone collect/learn row is for panels without timing groups;
    # tree panels fold those values into their group titles instead.
    if not stats.timing_groups:
        lines.append(f" timing      collect {stats.collect_ms:.1f}ms   learn {stats.learn_ms:.1f}ms")
    if stats.timing_metrics:
        lines.append(" timing(detail) " + "   ".join(_format_metric_items(stats.timing_metrics)))
    for group, items in stats.timing_groups.items():
        _append_timing_group_lines(lines, group, items, width)
    if stats.diagnostics:
        lines.append(" diagnostics " + "   ".join(_format_metric_items(stats.diagnostics)))
    lines.append("-" * width)
    return "\n".join(lines)


def _strip_markup(text: str) -> str:
    """Drop rich-style ``[tag]`` markup so plain-text output stays clean."""
    return re.sub(r"\[[^\]]*\]", "", text)


def _append_timing_group_lines(lines: list[str], group: str, items: Mapping[str, Any], width: int) -> None:
    """Append one timing group as a tree of aligned ``key value`` token rows.

    Scalar items flow as wrapped tokens on the group's label column. A nested
    mapping renders as a tree branch (``├─``/``└─`` by sibling position) whose
    children form an indented token block, each wrapped row joined with branch
    glyphs so the hierarchy stays visible. A ``total`` key inside a nested
    mapping is the node's own aggregate and renders inline after its name.
    Group titles may carry rich markup; it is stripped for plain text.
    """
    group_name = _strip_markup(group)
    group_width = max(18, len(group_name) + 1)
    row_prefix = " " * (group_width + 1)

    def token(key: str, value: Any) -> str:
        return f"{key} {_format_value(value)}"

    def render_block(
        toks: list[str], first_prefix: str, cont_prefix: str, last_prefix: str, single_prefix: str | None = None
    ) -> None:
        if not toks:
            return
        colw = max(len(t) for t in toks) + 2
        per_line = max(1, (width - len(cont_prefix)) // colw)
        chunks = [toks[i : i + per_line] for i in range(0, len(toks), per_line)]
        for j, chunk in enumerate(chunks):
            if len(chunks) == 1:
                prefix = single_prefix if single_prefix is not None else first_prefix
            elif j == 0:
                prefix = first_prefix
            elif j == len(chunks) - 1:
                prefix = last_prefix
            else:
                prefix = cont_prefix
            lines.append((prefix + "".join(t.ljust(colw) for t in chunk)).rstrip())

    pending: list[str] = []
    lead = f" {group_name:<{group_width}}"
    entries = list(items.items())
    for idx, (key, value) in enumerate(entries):
        if isinstance(value, Mapping):
            render_block(pending, lead, row_prefix, row_prefix)
            pending = []
            lead = row_prefix
            branch, hang = ("└─", "   ") if idx == len(entries) - 1 else ("├─", "│  ")
            below = row_prefix + hang
            children = dict(value)
            node_total = children.pop("total", None)
            label = f"{key} {_format_value(node_total)}" if node_total is not None else f"{key}:"
            lines.append(f"{row_prefix}{branch} {label}")
            render_block(
                [token(k, v) for k, v in children.items()],
                below + "├─ ",
                below + "├─ ",
                below + "└─ ",
                single_prefix=below + "└─ ",
            )
        else:
            pending.append(token(key, value))
    render_block(pending, lead, row_prefix, row_prefix)


def render_training_panel(stats: TrainingPanelStats, *, title: str = "rl"):
    """Build a rich panel for RL training stats."""
    if not _RICH:
        raise RuntimeError("rich is not available")

    def hms(t: float) -> str:
        t = int(t)
        h, m, sec = t // 3600, (t % 3600) // 60, t % 60
        return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"

    def si(n: float) -> str:
        for unit in ("", "k", "M"):
            if abs(n) < 1000:
                return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
            n /= 1000.0
        return f"{n:.1f}G"

    pct = 100.0 * stats.iteration / max(stats.total_iterations, 1)
    head = Table.grid(expand=True, padding=(0, 1))
    head.add_column(ratio=1)
    head.add_column(justify="right")
    head.add_row(
        f"[bold]iter[/] {stats.iteration:,}/{stats.total_iterations:,}  [dim]({pct:.1f}%)[/]",
        f"[cyan]{stats.steps_per_second:.0f}[/] env-steps/s   [dim]{hms(stats.elapsed_seconds)}[/]",
    )
    if stats.cpu_load is not None:
        load = stats.cpu_load
        color = "green" if load.utilization_percent < 70.0 else "yellow" if load.utilization_percent < 90.0 else "red"
        cores = f" · {load.physical_core_count}C" if load.physical_core_count is not None else ""
        head.add_row(
            f"[bold]cpu[/] [{color}]{load.utilization_percent:.1f}%[/]  "
            f"[dim]{load.used_logical_cpus:.1f}/{load.logical_cpu_count}T{cores}[/]",
            f"[dim]iowait {load.iowait_percent:.1f}%   steal {load.steal_percent:.1f}%[/]",
        )

    roll = Table.grid(expand=True, padding=(0, 2))
    for _ in range(3):
        roll.add_column(ratio=1)
    roll.add_row(
        f"return  [bold green]{stats.mean_return:.2f}[/]",
        f"ep_len  [yellow]{stats.mean_episode_length:.1f}[/]",
        f"episodes  {stats.episodes:,}",
    )

    buf = f"buffer [magenta]{si(stats.buffer_size)}[/]/{si(stats.buffer_capacity)}"
    blocks = [head, Rule(style="grey37"), roll]
    metrics = stats.training_metrics
    if metrics is None:
        message = "warmup - filling replay buffer" if stats.warming else "training metrics unavailable"
        blocks += [Rule("training", style="grey37", align="left"), f"[dim]{message}[/]   {buf}"]
    else:
        train_grid = _rich_key_value_grid(metrics)
        blocks += [Rule("training", style="grey37", align="left"), train_grid, buf]

    terms = stats.reward_terms
    if terms:
        sorted_terms = dict(sorted(terms.items(), key=lambda kv: -abs(float(kv[1]))))
        grid = _rich_key_value_grid(sorted_terms, signed=True, precision=4)
        blocks += [Rule("rewards x dt", style="grey37", align="left"), grid]

    env_metrics = stats.env_metrics
    if env_metrics:
        metric_grid = _rich_key_value_grid(dict(sorted(env_metrics.items())), signed=True)
        blocks += [Rule("metrics", style="grey37", align="left"), metric_grid]

    if stats.timing_groups:
        # Tree panels: one umbrella section, per-process groups as bold
        # sub-titles (titles may carry markup for value highlighting).
        blocks += [Rule("timing", style="grey37", align="left")]
        for group, items in stats.timing_groups.items():
            blocks += [f"[bold]{group}[/]", _rich_timing_grid(items)]
    else:
        timing = Table.grid(expand=True, padding=(0, 2))
        timing_cells = [
            f"collect [yellow]{stats.collect_ms:.1f}[/]ms",
            f"learn [magenta]{stats.learn_ms:.1f}[/]ms",
        ]
        for _ in timing_cells:
            timing.add_column(ratio=1)
        timing.add_row(*timing_cells)
        blocks += [Rule("timing", style="grey37", align="left"), timing]
    if stats.timing_metrics:
        blocks.append(_rich_key_value_grid(stats.timing_metrics))
    if stats.diagnostics:
        blocks += [Rule("diagnostics", style="grey37", align="left"), _rich_key_value_grid(stats.diagnostics)]

    return Panel(Group(*blocks), title=f"[bold]{title}[/]", border_style="cyan", padding=(0, 1))


def _rich_timing_grid(items: Mapping[str, Any]):
    """Two-column key/value grid; nested mappings render as indented sub-blocks."""
    grid = Table.grid(expand=True, padding=(0, 3))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    rows: list[list[str]] = []
    pending: list[str] = []

    def flush() -> None:
        while pending:
            row = pending[:2]
            del pending[:2]
            rows.append(row + [""] * (2 - len(row)))

    entries = list(items.items())
    for idx, (key, value) in enumerate(entries):
        if isinstance(value, Mapping):
            flush()
            branch = "└─" if idx == len(entries) - 1 else "├─"
            children = dict(value)
            node_total = children.pop("total", None)
            tail = f" {_format_value(node_total)}" if node_total is not None else ""
            rows.append([f"[dim]{branch} {key}[/]{tail}", ""])
            pending.extend(_rich_key_value_cell(f"  {child}", child_value) for child, child_value in children.items())
        else:
            pending.append(_rich_key_value_cell(key, value))
    flush()
    for row in rows:
        grid.add_row(*row)
    return grid


def _rich_key_value_grid(items: Mapping[str, Any], *, precision: int = 3, signed: bool = False):
    grid = Table.grid(expand=True, padding=(0, 3))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    pairs = list(items.items())
    for i in range(0, len(pairs), 2):
        left = _rich_key_value_cell(*pairs[i], precision=precision, signed=signed)
        right = _rich_key_value_cell(*pairs[i + 1], precision=precision, signed=signed) if i + 1 < len(pairs) else ""
        grid.add_row(left, right)
    return grid


def _rich_key_value_cell(key: str, value: Any, *, precision: int = 3, signed: bool = False) -> str:
    color = ""
    if signed and isinstance(value, (float, int)):
        color = "green" if value >= 0 else "red"
    formatted = _format_value(value, precision=precision, signed=signed)
    return f"{key:<22}[{color}]{formatted}[/]" if color else f"{key:<22}{formatted}"
