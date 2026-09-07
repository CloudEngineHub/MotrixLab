# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field

import pytest

from motrix_rl import runs
from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskConfig, TaskMeta


@dataclass
class DemoCfg:
    value: int = 1


@dataclass
class DemoAgentCfg:
    learning_rate: float = 1e-3


@dataclass
class NestedDemoCfg:
    agent: DemoAgentCfg = field(default_factory=DemoAgentCfg)


def test_create_run_context_uses_standard_layout(tmp_path):
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="skrl",
        train_backend="torch",
        algo="ppo",
        seed=7,
        checkpoint_format=".PT",
        runs_root=tmp_path,
    )

    assert run.run_dir.parent == tmp_path / "cartpole" / "skrl" / "torch" / "ppo"
    assert run.checkpoint_dir == run.run_dir / "checkpoints"
    assert run.checkpoint_dir.is_dir()
    assert runs.read_metadata(run.run_dir) == run.metadata
    assert run.metadata.checkpoint_format == "pt"


def test_metadata_round_trip_and_policy_lookup(tmp_path):
    run_dir = tmp_path / "runs" / "cartpole" / "skrl" / "torch" / "ppo" / "run-1"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    policy_path = checkpoint_dir / "best_agent.pt"
    policy_path.write_text("checkpoint", encoding="utf-8")

    metadata = runs.make_metadata(
        env_name="cartpole",
        rllib="skrl",
        train_backend="torch",
        algo="ppo",
        seed=42,
        checkpoint_format="pt",
    )
    runs.write_metadata(run_dir, metadata)

    assert runs.read_metadata(run_dir) == metadata
    assert runs.find_metadata_for_policy(policy_path) == (run_dir.resolve(), metadata)


def test_latest_metadata_run_filters_by_method(tmp_path):
    root = tmp_path / "runs"
    older = root / "cartpole" / "skrl" / "torch" / "ppo" / "older"
    newer = root / "cartpole" / "rslrl" / "torch" / "ppo" / "newer"

    runs.write_metadata(
        older,
        runs.make_metadata("cartpole", "skrl", "torch", "ppo", None, 1, "pt"),
    )
    runs.write_metadata(
        newer,
        runs.make_metadata("cartpole", "rslrl", "torch", "ppo", None, 2, "pt"),
    )

    # Make the selection deterministic without sleeping.
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    latest = runs.latest_metadata_run("cartpole", runs_root=root)
    assert latest is not None
    assert latest[0] == newer
    assert latest[1].rllib == "rslrl"

    latest_skrl = runs.latest_metadata_run("cartpole", rllib="skrl", runs_root=root)
    assert latest_skrl is not None
    assert latest_skrl[0] == older
    assert latest_skrl[1].rllib == "skrl"


def test_read_task_config_requires_run_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"task_config\.yaml"):
        runs.read_task_config(tmp_path, DemoCfg)


def test_read_task_config_merges_nested_override(tmp_path):
    task_cfg = TaskConfig(
        task=TaskMeta(env="demo", rllib="demo", algo="nested"),
        num_envs=8,
        play_num_envs=2,
        seed=1,
        logging=LoggingConfig(backend="tensorboard", interval=10),
        checkpoint=CheckpointConfig(interval=20),
        algo=NestedDemoCfg(),
    )
    runs.write_task_config(tmp_path, task_cfg)

    restored = runs.read_task_config(
        tmp_path,
        NestedDemoCfg,
        cfg_override={"agent": {"learning_rate": 5e-4}},
    )

    assert restored.algo.agent.learning_rate == 5e-4
