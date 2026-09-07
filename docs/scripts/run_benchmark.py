# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run a documentation benchmark with one or more random seeds."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "docs" / "performance.yaml"
TASK_DIR = REPO_ROOT / "configs" / "task"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def load_benchmarks(path: Path = DEFAULT_CONFIG) -> dict[str, dict[str, Any]]:
    """Load and validate the benchmark definitions."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RuntimeError(f"{path} must contain performance schema version 1")
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise RuntimeError(f"{path} must contain a benchmarks mapping")

    result = {}
    for benchmark_id, value in benchmarks.items():
        if not isinstance(benchmark_id, str) or not isinstance(value, dict):
            raise RuntimeError(f"Invalid benchmark definition: {benchmark_id!r}")
        for key in ("task", "metric", "default_seed", "benchmark_seeds"):
            if key not in value:
                raise RuntimeError(f"Benchmark {benchmark_id!r} is missing {key!r}")
        validate_task(str(value["task"]))
        default_seed = value["default_seed"]
        benchmark_seeds = value["benchmark_seeds"]
        if not isinstance(default_seed, int):
            raise RuntimeError(f"Benchmark {benchmark_id!r} default_seed must be an integer")
        if (
            not isinstance(benchmark_seeds, list)
            or not benchmark_seeds
            or not all(isinstance(seed, int) for seed in benchmark_seeds)
        ):
            raise RuntimeError(f"Benchmark {benchmark_id!r} benchmark_seeds must be a non-empty integer list")
        if len(set(benchmark_seeds)) != len(benchmark_seeds):
            raise RuntimeError(f"Benchmark {benchmark_id!r} benchmark_seeds contains duplicates")
        if default_seed not in benchmark_seeds:
            raise RuntimeError(f"Benchmark {benchmark_id!r} default_seed must be included in benchmark_seeds")
        result[benchmark_id] = value
    return result


def validate_task(task: str) -> None:
    """Validate a Hydra task selection against the repository config tree."""
    if task.count("/") != 1:
        raise RuntimeError(f"Invalid task {task!r}; expected <env>/<rllib>.<algo>[.<backend>]")
    env_id, recipe = task.split("/", maxsplit=1)
    parts = recipe.split(".")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise RuntimeError(f"Invalid task {task!r}; expected <env>/<rllib>.<algo>[.<backend>]")
    task_path = TASK_DIR / env_id / f"{recipe}.yaml"
    if not task_path.is_file():
        raise RuntimeError(f"Benchmark task config does not exist: {task_path.relative_to(REPO_ROOT)}")


def parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated list of unique integer seeds."""
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def resolve_seeds(
    benchmark: dict[str, Any],
    *,
    seed: int | None,
    seeds: list[int] | None,
    multi_seed: bool,
) -> list[int]:
    """Resolve CLI seed selection with single-seed local runs as the default."""
    if seed is not None:
        return [seed]
    if seeds is not None:
        return seeds
    if multi_seed:
        return list(benchmark["benchmark_seeds"])
    return [benchmark["default_seed"]]


def build_train_command(task: str, seed: int, overrides: list[str]) -> list[str]:
    """Build one training subprocess command."""
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"task={task}",
        f"seed={seed}",
        "logging.backend=tensorboard",
        *overrides,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="benchmark ID from docs/performance.yaml")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="performance config path")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="run exactly one seed")
    seeds.add_argument("--seeds", type=parse_seeds, help="run an explicit comma-separated seed list")
    seeds.add_argument("--multi-seed", action="store_true", help="use benchmark_seeds from the config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="append a Hydra training override; may be passed more than once",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without starting training")
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.config)
    if args.benchmark not in benchmarks:
        parser.error(f"unknown benchmark {args.benchmark!r}; choose from {', '.join(sorted(benchmarks))}")
    benchmark = benchmarks[args.benchmark]
    selected_seeds = resolve_seeds(
        benchmark,
        seed=args.seed,
        seeds=args.seeds,
        multi_seed=args.multi_seed,
    )

    for index, seed in enumerate(selected_seeds, start=1):
        command = build_train_command(str(benchmark["task"]), seed, args.override)
        print(f"[{index}/{len(selected_seeds)}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
