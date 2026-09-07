# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Versioned, self-contained deployment artifacts."""

from motrix_deploy.artifact.io import Artifact, inspect_artifact, read_artifact, sha256_bytes, write_artifact
from motrix_deploy.artifact.schema import (
    ControlSpec,
    DeploymentManifest,
    PolicySpec,
    SourceSpec,
    TaskSpec,
)

__all__ = [
    "Artifact",
    "ControlSpec",
    "DeploymentManifest",
    "PolicySpec",
    "SourceSpec",
    "TaskSpec",
    "inspect_artifact",
    "read_artifact",
    "sha256_bytes",
    "write_artifact",
]
