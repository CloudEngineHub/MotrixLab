# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from motrix_rl.runs import CHECKPOINTS_DIRNAME, RunMetadata, read_metadata

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
BEST_POLICY = "best_policy"
LATEST_TRAINING_STATE = "latest_training_state"


class CheckpointArtifactKind(str, Enum):
    """Built-in checkpoint artifact kinds understood by MotrixLab."""

    POLICY = "policy"
    TRAINING_STATE = "training_state"


TRAINING_STATE = CheckpointArtifactKind.TRAINING_STATE
POLICY = CheckpointArtifactKind.POLICY


@dataclass(frozen=True)
class CheckpointArtifact:
    """A checkpoint artifact recorded in checkpoints/manifest.json.

    Attributes:
        path: Checkpoint file path. Relative paths are stored from the
            directory containing ``manifest.json``.
        kind: Artifact semantics, such as ``policy`` for playback/export or
            ``training_state`` for resume.
        format: Optional checkpoint storage format, such as ``pt`` or ``pickle``.
    """

    path: Path
    kind: CheckpointArtifactKind
    format: str | None = None


@dataclass(frozen=True)
class CheckpointManifest:
    """Run-local checkpoint manifest.

    Artifact paths are stored relative to checkpoints/manifest.json.
    """

    version: int = 1
    artifacts: dict[str, CheckpointArtifact] = field(default_factory=dict)


def manifest_path(run_dir: Path | str) -> Path:
    """Return the standard checkpoint manifest path for a run directory.

    Args:
        run_dir: Root directory of a training run.
    """
    return Path(run_dir) / CHECKPOINTS_DIRNAME / MANIFEST_FILENAME


def standard_checkpoint_dir(run_dir: Path | str) -> Path:
    """Return the standard checkpoint artifact directory for a run directory.

    Args:
        run_dir: Root directory of a training run.
    """
    return Path(run_dir) / CHECKPOINTS_DIRNAME


def read_manifest(run_dir: Path | str) -> CheckpointManifest:
    """Read checkpoints/manifest.json from a run directory.

    Args:
        run_dir: Root directory of a training run.
    """
    data = json.loads(manifest_path(run_dir).read_text(encoding="utf-8"))
    artifacts = {name: _artifact_from_dict(artifact) for name, artifact in data.get("artifacts", {}).items()}
    return CheckpointManifest(version=data.get("version", 1), artifacts=artifacts)


def write_manifest(run_dir: Path | str, manifest: CheckpointManifest) -> Path:
    """Write checkpoints/manifest.json for a run directory.

    Args:
        run_dir: Root directory of a training run.
        manifest: Manifest content to serialize.
    """
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": manifest.version,
        "artifacts": {name: _artifact_to_dict(artifact) for name, artifact in manifest.artifacts.items()},
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_checkpoint_artifact(
    run_dir: Path | str,
    name: str,
    path: Path | str,
    kind: CheckpointArtifactKind,
    checkpoint_format: str | None = None,
) -> Path:
    """Record a checkpoint artifact in the run-local manifest.

    Args:
        run_dir: Root directory of a training run.
        name: Logical artifact name, such as ``best_policy`` or ``latest_training_state``.
        path: Checkpoint file path to record.
        kind: Artifact kind, such as ``CheckpointArtifactKind.POLICY``.
        checkpoint_format: Optional storage format. If omitted, the file suffix is used.
    """
    manifest = _read_manifest_or_empty(run_dir)
    artifacts = dict(manifest.artifacts)
    artifacts[name] = CheckpointArtifact(
        path=_relative_to_manifest(run_dir, path),
        kind=kind,
        format=_normalize_checkpoint_format(checkpoint_format or Path(path).suffix),
    )
    return write_manifest(run_dir, CheckpointManifest(version=manifest.version, artifacts=artifacts))


def final_checkpoint_path(checkpoint_format_or_metadata: RunMetadata | str, run_dir: Path | str) -> Path:
    """Return the standard final training-state checkpoint path for a run.

    Args:
        checkpoint_format_or_metadata: Checkpoint format suffix, or run metadata
            providing the checkpoint format.
        run_dir: Root directory of a training run.
    """
    if isinstance(checkpoint_format_or_metadata, RunMetadata):
        checkpoint_format = checkpoint_format_or_metadata.checkpoint_format
    else:
        checkpoint_format = checkpoint_format_or_metadata
    return standard_checkpoint_dir(run_dir) / f"latest.{checkpoint_format.lstrip('.')}"


def best_policy(metadata: RunMetadata, run_dir: Path | str) -> Path:
    """Resolve the best playable policy for a run.

    Args:
        metadata: Run metadata for the run being queried.
        run_dir: Root directory of a training run.
    """
    run_dir = Path(run_dir)
    manifest_policy = _artifact_path(run_dir, BEST_POLICY) or _artifact_path(run_dir, LATEST_TRAINING_STATE)
    if manifest_policy is not None:
        return manifest_policy

    raise FileNotFoundError(
        f"No '{BEST_POLICY}' or '{LATEST_TRAINING_STATE}' artifact found in {manifest_path(run_dir)} "
        f"for {metadata.rllib}.{metadata.algo} ({metadata.train_backend})."
    )


def resume_checkpoint(run_dir: Path | str, metadata: RunMetadata | None = None) -> Path:
    """Resolve the checkpoint to use when resuming from a run directory.

    Args:
        run_dir: Root directory of a training run.
        metadata: Optional run metadata.
    """
    run_dir = Path(run_dir)
    resume_path = _artifact_path(run_dir, LATEST_TRAINING_STATE) or _artifact_path(run_dir, BEST_POLICY)
    if resume_path is not None:
        return resume_path

    metadata = metadata or read_metadata(run_dir)
    raise FileNotFoundError(
        f"No '{LATEST_TRAINING_STATE}' or '{BEST_POLICY}' artifact found in {manifest_path(run_dir)} "
        f"for {metadata.rllib}.{metadata.algo} ({metadata.train_backend})."
    )


def resolve_checkpoint_path(checkpoint_or_run_dir: Path | str, metadata: RunMetadata | None = None) -> Path:
    """Resolve a policy checkpoint file or run directory to a concrete policy file.

    Args:
        checkpoint_or_run_dir: Existing checkpoint file path, or run directory to resolve.
        metadata: Optional run metadata used when ``checkpoint_or_run_dir`` is a run directory.
    """
    path = Path(checkpoint_or_run_dir)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a run directory: {path}")

    run_metadata = metadata or read_metadata(path)
    return best_policy(run_metadata, path)


def resolve_resume_checkpoint_path(checkpoint_or_run_dir: Path | str, metadata: RunMetadata | None = None) -> Path:
    """Resolve a resume checkpoint file or run directory to a concrete training-state file.

    Args:
        checkpoint_or_run_dir: Existing checkpoint file path, or run directory to resolve.
        metadata: Optional run metadata used when ``checkpoint_or_run_dir`` is a run directory.
    """
    path = Path(checkpoint_or_run_dir)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a run directory: {path}")
    return resume_checkpoint(path, metadata=metadata)


def _read_manifest_or_empty(run_dir: Path | str) -> CheckpointManifest:
    try:
        return read_manifest(run_dir)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return CheckpointManifest()


def _artifact_path(run_dir: Path, name: str) -> Path | None:
    try:
        manifest = read_manifest(run_dir)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    artifact = manifest.artifacts.get(name)
    if artifact is None:
        return None

    path = _artifact_abs_path(run_dir, artifact)
    if path.exists():
        return path
    logger.warning("Checkpoint artifact %s points to missing file: %s", name, path)
    return None


def _artifact_abs_path(run_dir: Path | str, artifact: CheckpointArtifact) -> Path:
    path = artifact.path
    if path.is_absolute():
        return path
    return manifest_path(run_dir).parent / path


def _relative_to_manifest(run_dir: Path | str, path: Path | str) -> Path:
    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = (Path.cwd() / artifact_path).resolve()
    manifest_dir = manifest_path(run_dir).parent.resolve()
    try:
        return artifact_path.relative_to(manifest_dir)
    except ValueError:
        return Path(os.path.relpath(artifact_path, manifest_dir))


def _artifact_from_dict(data: dict) -> CheckpointArtifact:
    return CheckpointArtifact(
        path=Path(data["path"]),
        kind=CheckpointArtifactKind(data["kind"]),
        format=data.get("format"),
    )


def _artifact_to_dict(artifact: CheckpointArtifact) -> dict:
    return {
        "path": str(artifact.path),
        "kind": artifact.kind.value,
        "format": artifact.format,
    }


def _normalize_checkpoint_format(checkpoint_format: str | None) -> str | None:
    if checkpoint_format is None:
        return None
    normalized = checkpoint_format.lstrip(".").lower()
    return normalized or None
