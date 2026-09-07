# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed application configuration for the deployment CLI."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeployRunConfig:
    """Resolved, task-agnostic configuration consumed by one deployment run."""

    artifact: str
    backend: dict[str, Any]
    viewer: bool
    command: dict[str, Any] | None = None
    rollout: dict[str, Any] | None = None
    seed: int | None = None
    realtime: bool | None = None
    hardware: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact:
            raise ValueError("artifact must be a non-empty path")
        if not isinstance(self.viewer, bool):
            raise ValueError(f"viewer must be bool, got {self.viewer!r}")
        if self.realtime is not None and not isinstance(self.realtime, bool):
            raise ValueError(f"realtime must be bool or null, got {self.realtime!r}")
        confirmed = self.hardware.get("confirm", False)
        if not isinstance(confirmed, bool):
            raise ValueError(f"hardware.confirm must be bool, got {confirmed!r}")
        backend_name = self.backend.get("name")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError(f"backend.name must be a non-empty string, got {backend_name!r}")

    @property
    def backend_name(self) -> str:
        return self.backend["name"]

    @property
    def backend_options(self) -> dict[str, Any]:
        return {name: value for name, value in self.backend.items() if name != "name"}


__all__ = ["DeployRunConfig"]
