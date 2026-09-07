# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from motrix_rl import checkpoints, runs
from motrix_rl.result import TrainResult


def test_create_run_context_writes_metadata_and_checkpoint_dir(tmp_path):
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="external",
        train_backend="custom",
        algo="demo",
        seed=42,
        checkpoint_format=".PT",
        runs_root=tmp_path,
    )

    assert run.run_dir.parent == tmp_path / "cartpole" / "external" / "custom" / "demo"
    assert run.checkpoint_dir == run.run_dir / "checkpoints"
    assert run.checkpoint_dir.is_dir()
    assert runs.read_metadata(run.run_dir) == run.metadata
    assert run.metadata.env_name == "cartpole"
    assert run.metadata.train_backend == "custom"
    assert run.metadata.checkpoint_format == "pt"


def test_train_result_finds_best_policy(tmp_path):
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="skrl",
        train_backend="jax",
        algo="ppo",
        seed=None,
        checkpoint_format="pickle",
        runs_root=tmp_path,
    )
    checkpoints_dir = run.checkpoint_dir
    best = checkpoints_dir / "policy.pickle"
    best.touch()
    checkpoints.record_checkpoint_artifact(
        run.run_dir,
        checkpoints.BEST_POLICY,
        best,
        checkpoints.POLICY,
        checkpoint_format="pickle",
    )

    result = TrainResult(run=run)

    assert result.find_best_policy() == best


def test_best_policy_missing_manifest_artifact_raises(tmp_path):
    metadata = runs.make_metadata("cartpole", "motrix", "torch", "fastsac", None, None, "pt")
    runs.write_metadata(tmp_path, metadata)

    with pytest.raises(FileNotFoundError, match="No 'best_policy' or 'latest_training_state' artifact found"):
        checkpoints.best_policy(metadata, tmp_path)


def test_train_result_finds_motrix_fastsac_policy(tmp_path):
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="motrix",
        train_backend="torch",
        algo="fastsac",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    expected = run.checkpoint_dir / "policy.pt"
    expected.touch()
    checkpoints.record_checkpoint_artifact(
        run.run_dir,
        checkpoints.BEST_POLICY,
        expected,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )

    result = TrainResult(run=run)

    assert result.find_best_policy() == expected


def test_best_policy_prefers_manifest_best_policy(tmp_path):
    metadata = runs.make_metadata("cartpole", "external", "custom", "demo", None, 1, "pt")
    runs.write_metadata(tmp_path, metadata)
    legacy = tmp_path / "checkpoints" / "model.pt"
    legacy.parent.mkdir()
    legacy.touch()
    expected = tmp_path / "checkpoints" / "policy.pt"
    expected.touch()

    checkpoints.record_checkpoint_artifact(
        tmp_path,
        checkpoints.BEST_POLICY,
        expected,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )

    assert checkpoints.best_policy(metadata, tmp_path) == expected


def test_resolve_resume_checkpoint_path_prefers_latest_training_state(tmp_path):
    metadata = runs.make_metadata("cartpole", "external", "custom", "demo", None, 1, "pt")
    runs.write_metadata(tmp_path, metadata)
    best = tmp_path / "checkpoints" / "best.pt"
    latest = tmp_path / "checkpoints" / "latest.pt"
    latest.parent.mkdir()
    best.touch()
    latest.touch()
    checkpoints.record_checkpoint_artifact(
        tmp_path,
        checkpoints.BEST_POLICY,
        best,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )
    checkpoints.record_checkpoint_artifact(
        tmp_path,
        checkpoints.LATEST_TRAINING_STATE,
        latest,
        checkpoints.TRAINING_STATE,
        checkpoint_format="pt",
    )

    assert checkpoints.resolve_resume_checkpoint_path(tmp_path) == latest


def test_train_result_uses_manifest_metadata(tmp_path):
    metadata = runs.make_metadata("cartpole", "external", "custom", "demo", None, 1, "pt")
    runs.write_metadata(tmp_path, metadata)
    expected = tmp_path / "checkpoints" / "policy.pt"
    expected.parent.mkdir()
    expected.touch()
    checkpoints.record_checkpoint_artifact(
        tmp_path,
        checkpoints.BEST_POLICY,
        expected,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )

    result = TrainResult(run=runs.open_run_context(tmp_path, metadata))

    assert result.find_best_policy() == expected


def test_record_checkpoint_artifact_normalizes_kind_and_format(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "policy.PT"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    checkpoints.record_checkpoint_artifact(
        tmp_path,
        checkpoints.BEST_POLICY,
        checkpoint,
        checkpoints.CheckpointArtifactKind.POLICY,
    )

    artifact = checkpoints.read_manifest(tmp_path).artifacts[checkpoints.BEST_POLICY]
    assert artifact.path == Path("policy.PT")
    assert artifact.kind is checkpoints.CheckpointArtifactKind.POLICY
    assert artifact.format == "pt"
