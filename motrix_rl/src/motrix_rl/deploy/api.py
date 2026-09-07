# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public contracts for training-side policy deployment integration."""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from motrix_deploy.artifact import Artifact
from motrix_rl.config import OnnxParityConfig, TaskConfig
from motrix_rl.runs import RunContext


@dataclass(frozen=True)
class PolicyTensorSpec:
    """One exported ONNX tensor contract."""

    name: str
    shape: tuple[int | None, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy tensor name must be non-empty")
        if not isinstance(self.shape, tuple):
            raise TypeError(f"policy tensor shape must be a tuple, got {type(self.shape).__name__}")
        if not self.shape or any(
            dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0)
            for dimension in self.shape
        ):
            raise ValueError(f"policy tensor shape must contain positive or dynamic dimensions, got {self.shape}")
        if self.dtype != "float32":
            raise ValueError(f"policy tensor dtype must be float32, got {self.dtype}")


@dataclass(frozen=True)
class OnnxParityMetrics:
    """Summary of framework-reference versus ONNX Runtime validation."""

    samples: int
    max_abs_error: float
    max_rel_error: float

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError(f"parity samples must be positive, got {self.samples}")
        if (
            not math.isfinite(self.max_abs_error)
            or not math.isfinite(self.max_rel_error)
            or self.max_abs_error < 0
            or self.max_rel_error < 0
        ):
            raise ValueError("parity errors must be non-negative")


@dataclass(frozen=True)
class OnnxExportRequest:
    """Resolved inputs supplied to one framework-owned exporter."""

    run: RunContext
    checkpoint: Path
    task_config: TaskConfig
    opset: int = 18
    parity: OnnxParityConfig = field(default_factory=OnnxParityConfig)

    def __post_init__(self) -> None:
        if self.opset < 11:
            raise ValueError(f"ONNX opset must be at least 11, got {self.opset}")


@dataclass(frozen=True)
class OnnxExportReport:
    """Tensor contract and numerical parity report shared across export stages."""

    input_spec: PolicyTensorSpec
    output_spec: PolicyTensorSpec
    parity: OnnxParityMetrics


@dataclass(frozen=True)
class OnnxModelArtifact:
    """Validated in-memory ONNX model returned by a framework exporter."""

    model_bytes: bytes
    report: OnnxExportReport

    def __post_init__(self) -> None:
        if not isinstance(self.model_bytes, bytes) or not self.model_bytes:
            raise ValueError("exported ONNX model must be non-empty bytes")


@dataclass(frozen=True)
class OnnxExportResult:
    """Persisted ONNX export returned by the public service."""

    path: Path
    report: OnnxExportReport


@dataclass(frozen=True)
class DeploymentExportResult:
    """Persisted deployment artifact and policy parity summary."""

    artifact: Artifact
    validation_samples: int
    max_abs_error: float


@dataclass(frozen=True)
class SourceRolloutResult:
    """Bounded source-environment rollout with deterministic probes."""

    success: bool
    completed_steps: int
    exit_reason: str
    reset_observation: list[float]
    first_policy_outputs: list[list[float]]
    trace_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnnxPolicyExporter(ABC):
    """Framework adapter that restores and exports one deterministic policy."""

    @abstractmethod
    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        """Restore, export, and validate one deterministic policy."""
