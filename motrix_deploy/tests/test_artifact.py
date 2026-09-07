# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Artifact schema, safety and checksum tests."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from motrix_deploy.artifact import DeploymentManifest, inspect_artifact, read_artifact, write_artifact
from motrix_deploy.errors import ArtifactError, ValidationError

POLICY_BYTES = b"deterministic ONNX placeholder"


def test_artifact_round_trip_and_inspect(
    tmp_path: Path,
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    output = tmp_path / "fixture.deploy"
    artifact = write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
    summary = inspect_artifact(output)

    assert artifact.manifest.to_dict() == read_artifact(output).manifest.to_dict()
    assert summary["valid"] is True
    assert summary["task"]["name"] == "test/v1"
    assert summary["robot"]["joint_names"] == ["left_joint", "right_joint"]
    assert summary["policy"]["input"] == {"name": "observation", "shape": [1, 4], "dtype": "float32"}


def test_cli_inspect_outputs_json(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})

    result = subprocess.run(
        [sys.executable, "-m", "motrix_deploy.cli", "inspect", f"artifact={output}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True


def test_corrupt_checksum_is_rejected(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
    (output / "policy/model.onnx").write_bytes(b"corrupt")

    with pytest.raises(ArtifactError, match="policy.sha256 mismatch"):
        read_artifact(output)


@pytest.mark.parametrize("payload_path", ["../model.onnx", "/tmp/model.onnx", "policy\\model.onnx"])
def test_unsafe_payload_path_is_rejected(
    tmp_path: Path,
    manifest_factory: Callable[[], DeploymentManifest],
    payload_path: str,
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["policy"]["payload_path"] = payload_path
    output = tmp_path / "unsafe.deploy"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="policy.payload_path"):
        read_artifact(output)


def test_unknown_schema_version_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["schema_version"] = "motrix-deploy/v999"

    with pytest.raises(ValidationError, match="schema_version"):
        DeploymentManifest.from_dict(manifest)


@pytest.mark.parametrize("name", ["test", "test/v0", "test/v01", "test/v1/extra", 1])
def test_invalid_task_name_is_rejected(
    manifest_factory: Callable[[], DeploymentManifest],
    name: object,
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["task"]["name"] = name

    with pytest.raises(ValidationError, match="task.name"):
        DeploymentManifest.from_dict(manifest)


def test_joint_count_mismatch_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["robot"]["joint_names"] = ["left_joint"]

    with pytest.raises(ValidationError, match="robot.default_joint_position.shape"):
        DeploymentManifest.from_dict(manifest)


def test_reversed_limit_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["robot"]["position_lower"][0] = 2.0

    with pytest.raises(ValidationError, match="robot.position_range"):
        DeploymentManifest.from_dict(manifest)


def test_writer_is_create_only(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
