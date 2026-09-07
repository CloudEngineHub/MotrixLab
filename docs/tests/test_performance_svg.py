# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_benchmark = _load_script("run_benchmark")
performance = _load_script("generate_performance_svg")


def _benchmark_spec():
    return performance.BenchmarkSpec(
        benchmark_id="demo",
        task="demo/skrl.ppo",
        metric="episode_return",
        metric_label={"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
    )


def _series(
    run_id: int,
    values: list[float],
    curriculum_values: list[float] | None = None,
    survival_values: list[float] | None = None,
):
    return performance.RunSeries(
        run_dir=Path(f"runs/demo/{run_id}"),
        environment_steps=np.asarray([0.0, 10.0, 20.0]),
        elapsed_seconds=np.asarray([0.0, 1.0, 3.0]) + run_id,
        values=np.asarray(values),
        curriculum_values=np.asarray(curriculum_values) if curriculum_values is not None else None,
        survival_values=np.asarray(survival_values) if survival_values is not None else None,
    )


def test_resolve_seeds_defaults_to_single_and_supports_multi_seed():
    benchmark = {"default_seed": 7, "benchmark_seeds": [1, 2, 3]}

    assert run_benchmark.resolve_seeds(benchmark, seed=None, seeds=None, multi_seed=False) == [7]
    assert run_benchmark.resolve_seeds(benchmark, seed=None, seeds=None, multi_seed=True) == [1, 2, 3]
    assert run_benchmark.resolve_seeds(benchmark, seed=9, seeds=None, multi_seed=False) == [9]
    assert run_benchmark.resolve_seeds(benchmark, seed=None, seeds=[4, 5], multi_seed=False) == [4, 5]


def test_parse_seeds_rejects_duplicates():
    assert run_benchmark.parse_seeds("0, 2,5") == [0, 2, 5]
    with pytest.raises(run_benchmark.argparse.ArgumentTypeError, match="duplicates"):
        run_benchmark.parse_seeds("1,1")


def test_aggregate_series_draws_confidence_interval_only_for_multiple_runs():
    single = performance.aggregate_series([_series(0, [1.0, 2.0, 3.0])])
    assert np.array_equal(single.mean, single.lower)
    assert np.array_equal(single.mean, single.upper)

    multiple = performance.aggregate_series([_series(0, [1.0, 2.0, 3.0]), _series(1, [2.0, 4.0, 6.0])])
    assert np.allclose(multiple.mean, [1.5, 3.0, 4.5])
    assert np.allclose(multiple.elapsed_seconds, [0.5, 1.5, 3.5])
    assert np.all(multiple.lower < multiple.mean)
    assert np.all(multiple.upper > multiple.mean)
    svg = performance.render_svg(_benchmark_spec(), multiple, 2)
    assert 'class="band"' in svg
    assert "Environment steps (total)" in svg
    assert "Elapsed wall time" in svg
    assert ">demo</text>" not in svg
    assert "2 runs" not in svg
    assert "总环境步数" not in svg
    assert "训练耗时" not in svg
    assert "Episode 回报" not in svg


def test_curriculum_template_draws_secondary_axis_and_serializes_values():
    spec = performance.BenchmarkSpec(
        benchmark_id="demo",
        task="demo/motrix.fastsac",
        metric="rollout/mean_return",
        metric_label={"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
        template="curriculum",
        curriculum_metric="metrics/penalty_scale",
        curriculum_metric_label={"en": "Penalty scale", "zh_CN": "惩罚缩放"},
    )
    series = [_series(0, [1.0, 2.0, 3.0], [0.5, 0.7, 1.0])]
    aggregate = performance.aggregate_series(series)

    svg = performance.render_svg(spec, aggregate, 1)
    snapshot = json.loads(performance.render_data(spec, aggregate, series))

    assert 'class="curriculum-axis"' in svg
    assert 'class="curriculum"' in svg
    assert "Penalty scale" in svg
    assert ">demo</text>" not in svg
    assert "1 run" not in svg
    assert "惩罚缩放" not in svg
    assert "平均 Episode 回报" not in svg
    curriculum_points = re.search(r'<polyline class="curriculum" points="([^"]+)"', svg)
    assert curriculum_points is not None
    point_x = [point.split(",")[0] for point in curriculum_points.group(1).split()]
    point_y = [float(point.split(",")[1]) for point in curriculum_points.group(1).split()]
    assert len(point_x) == 5
    assert point_x[1] == point_x[2]
    assert point_x[3] == point_x[4]
    assert min(point_y) > 90
    assert max(point_y) < 494
    assert snapshot["template"] == "curriculum"
    assert snapshot["curriculum_metric"] == "metrics/penalty_scale"
    assert snapshot["points"][-1]["curriculum_value"] == 1.0


def test_wbt_template_draws_continuous_survival_axis_and_serializes_values():
    spec = performance.BenchmarkSpec(
        benchmark_id="demo",
        task="demo/motrix.fastsac",
        metric="rollout/mean_return",
        metric_label={"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
        template="wbt",
        survival_metric="rollout/mean_ep_len",
        survival_metric_label={"en": "Episode survival (%)", "zh_CN": "Episode 存活率（%）"},
        episode_length_max=500.0,
    )
    series = [_series(0, [1.0, 2.0, 3.0], survival_values=[20.0, 60.0, 100.0])]
    aggregate = performance.aggregate_series(series)

    svg = performance.render_svg(spec, aggregate, 1)
    snapshot = json.loads(performance.render_data(spec, aggregate, series))

    assert 'class="survival-axis"' in svg
    assert 'class="survival"' in svg
    assert "Episode survival (%)" in svg
    assert "Episode 存活率" not in svg
    survival_points = re.search(r'<polyline class="survival" points="([^"]+)"', svg)
    assert survival_points is not None
    assert len(survival_points.group(1).split()) == 3
    assert snapshot["template"] == "wbt"
    assert snapshot["survival_metric"] == "rollout/mean_ep_len"
    assert snapshot["episode_length_max"] == 500.0
    assert snapshot["points"][-1]["survival_value"] == 100.0


def test_nice_ticks_use_conventional_intervals():
    assert performance._nice_ticks(0.5, 1.0, 6, expand=True) == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert performance._nice_ticks(0.0, 20_480_000.0, 6, expand=False) == pytest.approx(
        [0.0, 5_000_000.0, 10_000_000.0, 15_000_000.0, 20_000_000.0]
    )
    assert performance._padded_tick_bounds([2.0, 3.0, 4.0]) == pytest.approx((1.9, 4.1))


def test_read_tensorboard_scalar_reads_event_file(tmp_path):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(tmp_path))
    start = time.time()
    for step, value in ((0, 1.5), (10, 3.0)):
        summary = Summary(value=[Summary.Value(tag="episode_return", simple_value=value)])
        writer.add_event(Event(wall_time=start + float(step), step=step, summary=summary))
    writer.close()

    scalar = performance.read_tensorboard_scalar(tmp_path, "episode_return")

    assert scalar is not None
    assert np.array_equal(scalar.steps, [0.0, 10.0])
    assert scalar.elapsed_seconds[0] >= 0
    assert scalar.elapsed_seconds[1] - scalar.elapsed_seconds[0] == pytest.approx(10.0)
    assert np.allclose(scalar.values, [1.5, 3.0])


@pytest.mark.parametrize(
    ("task", "algo_yaml", "expected"),
    [
        ("demo/skrl.ppo", "{}", [0.0, 40.0, 80.0]),
        ("demo/rslrl.ppo", "num_steps_per_env: 8", [32.0, 352.0, 672.0]),
        ("demo/motrix.fastsac", "asynchronous: false", [4.0, 44.0, 84.0]),
        ("demo/motrix.fastsac", "asynchronous: true", [0.0, 40.0, 80.0]),
    ],
)
def test_normalize_environment_steps_uses_run_task_snapshot(tmp_path, task, algo_yaml, expected):
    rllib, algo = task.split("/")[1].split(".")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task_config.yaml").write_text(
        f"""task:
  env: demo
  rllib: {rllib}
  algo: {algo}
num_envs: 4
algo:
  {algo_yaml}
""",
        encoding="utf-8",
    )
    spec = performance.BenchmarkSpec(
        benchmark_id="demo",
        task=task,
        metric="return",
        metric_label={"en": "Return", "zh_CN": "回报"},
    )

    normalized = performance.normalize_environment_steps(spec, run_dir, np.asarray([0.0, 10.0, 20.0]))

    assert np.array_equal(normalized, expected)


def _write_checkpointed_run(run_dir: Path, metadata: dict):
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"best_policy": {"path": "latest.pt"}}}),
        encoding="utf-8",
    )


def test_discover_run_candidates_filters_env_and_orders_newest_first(tmp_path):
    matching = tmp_path / "demo" / "skrl" / "jax" / "ppo" / "new"
    _write_checkpointed_run(
        matching,
        {
            "env_name": "demo",
            "rllib": "skrl",
            "algo": "ppo",
            "train_backend": "jax",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    older = tmp_path / "demo" / "motrix" / "torch" / "fastsac" / "old"
    _write_checkpointed_run(
        older,
        {
            "env_name": "demo",
            "rllib": "motrix",
            "algo": "fastsac",
            "train_backend": "torch",
            "created_at": "2025-01-01T00:00:00+00:00",
        },
    )
    ignored = tmp_path / "demo" / "rslrl" / "torch" / "ppo" / "ignored"
    _write_checkpointed_run(
        ignored,
        {
            "env_name": "another-env",
            "rllib": "rslrl",
            "algo": "ppo",
            "train_backend": "torch",
        },
    )

    candidates = performance.discover_run_candidates("demo", tmp_path)

    assert candidates == [matching, older]


def test_select_metric_prefers_return_and_supports_override():
    tags = ("train/loss", "reward/component", "rollout/mean_return")

    assert performance.select_metric(tags) == "rollout/mean_return"
    assert performance.select_metric(tags, "train/loss") == "train/loss"
    assert performance.select_metric(tags, "missing") is None


@pytest.mark.parametrize(
    "metric",
    [
        "rollout/mean_return",
        "Reward / Total reward (mean)",
        "episode_return",
        "Train/mean_reward",
        "train/mean_reward",
    ],
)
def test_metric_labels_maps_backend_return_metrics_to_one_label(metric):
    assert performance.metric_labels(metric, None) == {
        "en": "Mean episode return",
        "zh_CN": "平均 Episode 回报",
    }


def test_metric_labels_supports_explicit_override():
    config = performance.MetricConfig(
        metric="Train/mean_reward",
        metric_label={"en": "Custom return", "zh_CN": "自定义回报"},
    )

    assert performance.metric_labels("Train/mean_reward", config) == config.metric_label


def test_select_run_series_loads_curriculum_metric_from_latest_checkpoint(tmp_path):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run_dir = tmp_path / "demo" / "motrix" / "torch" / "fastsac" / "latest"
    _write_checkpointed_run(
        run_dir,
        {
            "env_name": "demo",
            "rllib": "motrix",
            "algo": "fastsac",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    (run_dir / "task_config.yaml").write_text(
        """task:
  env: demo
  rllib: motrix
  algo: fastsac
num_envs: 4
algo:
  asynchronous: true
""",
        encoding="utf-8",
    )
    writer = EventFileWriter(str(run_dir))
    start = time.time()
    for step, episode_return, penalty_scale in ((1, 10.0, 0.5), (2, 20.0, 0.75)):
        summary = Summary(
            value=[
                Summary.Value(tag="rollout/mean_return", simple_value=episode_return),
                Summary.Value(tag="metrics/penalty_scale", simple_value=penalty_scale),
            ]
        )
        writer.add_event(Event(wall_time=start + step, step=step, summary=summary))
    writer.close()
    config = performance.MetricConfig(
        metric="rollout/mean_return",
        metric_label={"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
        template="curriculum",
        curriculum_metric="metrics/penalty_scale",
        curriculum_metric_label={"en": "Penalty scale", "zh_CN": "惩罚缩放"},
    )

    spec, series = performance.select_run_series(
        "demo",
        tmp_path,
        requested_metric=None,
        metric_config=config,
        strict=True,
    )

    assert spec is not None and spec.template == "curriculum"
    assert len(series) == 1
    assert np.array_equal(series[0].environment_steps, [4.0, 8.0])
    assert np.allclose(series[0].curriculum_values, [0.5, 0.75])


def test_select_run_series_normalizes_wbt_episode_length_to_survival_percentage(tmp_path):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run_dir = tmp_path / "demo" / "motrix" / "torch" / "fastsac" / "latest"
    _write_checkpointed_run(
        run_dir,
        {
            "env_name": "demo",
            "rllib": "motrix",
            "algo": "fastsac",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    (run_dir / "task_config.yaml").write_text(
        """task:
  env: demo
  rllib: motrix
  algo: fastsac
num_envs: 4
algo:
  asynchronous: true
""",
        encoding="utf-8",
    )
    writer = EventFileWriter(str(run_dir))
    start = time.time()
    for step, episode_return, episode_length in ((1, 10.0, 250.0), (2, 20.0, 600.0)):
        summary = Summary(
            value=[
                Summary.Value(tag="rollout/mean_return", simple_value=episode_return),
                Summary.Value(tag="rollout/mean_ep_len", simple_value=episode_length),
            ]
        )
        writer.add_event(Event(wall_time=start + step, step=step, summary=summary))
    writer.close()
    config = performance.MetricConfig(
        metric="rollout/mean_return",
        metric_label={"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
        template="wbt",
        survival_metric="rollout/mean_ep_len",
        survival_metric_label={"en": "Episode survival (%)", "zh_CN": "Episode 存活率（%）"},
        episode_length_max=500.0,
    )

    spec, series = performance.select_run_series(
        "demo",
        tmp_path,
        requested_metric=None,
        metric_config=config,
        strict=True,
    )

    assert spec is not None and spec.template == "wbt"
    assert len(series) == 1
    assert np.array_equal(series[0].environment_steps, [4.0, 8.0])
    assert np.allclose(series[0].survival_values, [50.0, 100.0])


def test_generate_writes_only_static_assets_and_preserves_manual_docs(tmp_path, monkeypatch):
    spec = _benchmark_spec()
    static_dir = tmp_path / "docs" / "source" / "_static"
    monkeypatch.setattr(performance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(performance, "STATIC_DIR", static_dir)
    manual_page = tmp_path / "docs" / "source" / "en" / "user_guide" / "envs" / "basic" / "demo.md"
    manual_page.parent.mkdir(parents=True)
    manual_content = "# Demo\n\n## Performance\n\nMaintained manually.\n"
    manual_page.write_text(manual_content, encoding="utf-8")
    selected = [_series(0, [1.0, 2.0, 3.0]), _series(1, [2.0, 4.0, 6.0])]
    selected = [
        performance.RunSeries(
            tmp_path / item.run_dir,
            item.environment_steps,
            item.elapsed_seconds,
            item.values,
        )
        for item in selected
    ]
    monkeypatch.setattr(performance, "select_run_series", lambda *args, **kwargs: (spec, selected))

    stale = performance.generate(
        ["demo"],
        metric_configs={},
        runs_root=tmp_path / "runs",
        requested_metric=None,
        strict=False,
        refresh=True,
        check=False,
    )

    assert static_dir / "images" / "performance" / "demo.svg" in stale
    assert static_dir / "data" / "performance" / "demo.json" in stale
    assert manual_page.read_text(encoding="utf-8") == manual_content
    snapshot = json.loads((static_dir / "data" / "performance" / "demo.json").read_text(encoding="utf-8"))
    assert snapshot["points"][0]["environment_steps"] == 0.0
    assert snapshot["points"][0]["elapsed_seconds"] == 0.5
    assert snapshot["runs"] == ["0", "1"]
    assert "seeds" not in snapshot

    def fail_if_runs_are_read(*args, **kwargs):
        raise AssertionError("committed snapshot should be preferred")

    monkeypatch.setattr(performance, "select_run_series", fail_if_runs_are_read)
    assert (
        performance.generate(
            ["demo"],
            metric_configs={},
            runs_root=tmp_path / "missing-runs",
            requested_metric=None,
            strict=False,
            refresh=False,
            check=True,
        )
        == []
    )
