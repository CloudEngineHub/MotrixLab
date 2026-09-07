# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime contract implemented by concrete deployment-task packages."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from motrix_deploy.artifact import TaskSpec
from motrix_deploy.contracts import FloatArray, RobotCommand, RobotSpec, RobotState
from motrix_deploy.runtime.context import PolicyContext

CommandT = TypeVar("CommandT")


class DeployTask(ABC, Generic[CommandT]):
    """Own task-specific observation, action, and command semantics."""

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """Return the flat policy observation size."""

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Return the flat policy action size."""

    @abstractmethod
    def reset(self, state: RobotState, context: PolicyContext[CommandT]) -> None:
        """Reset episode-local preprocessing and action state."""

    @abstractmethod
    def build_observation(self, state: RobotState, context: PolicyContext[CommandT]) -> FloatArray:
        """Build one flat policy observation."""

    @abstractmethod
    def process_action(self, action: FloatArray) -> RobotCommand:
        """Convert one raw policy action into a canonical robot command."""

    @abstractmethod
    def validate_command(self, command: CommandT) -> None:
        """Validate one external high-level command before using it."""


TaskFactory = Callable[[TaskSpec, RobotSpec], DeployTask[Any]]

_TASK_FACTORIES: dict[str, TaskFactory] = {}
_TASK_NAME_PATTERN = re.compile(r"[^/\s]+/v[1-9][0-9]*")


def register_task(name: str) -> Callable[[TaskFactory], TaskFactory]:
    """Register one concrete task factory when its implementation module is imported."""
    if not isinstance(name, str) or _TASK_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"task name must be a versioned identifier such as go2_walk/v1, got {name!r}")

    def register(factory: TaskFactory) -> TaskFactory:
        if name in _TASK_FACTORIES:
            raise ValueError(f"Deployment task {name} is already registered")
        _TASK_FACTORIES[name] = factory
        return factory

    return register


def create_task(spec: TaskSpec, robot: RobotSpec) -> DeployTask[Any]:
    """Create the task implementation selected by an artifact."""
    try:
        factory = _TASK_FACTORIES[spec.name]
    except KeyError as error:
        supported = ", ".join(sorted(_TASK_FACTORIES)) or "none"
        raise ValueError(f"Unsupported deployment task {spec.name}; supported tasks: {supported}") from error
    return factory(spec, robot)


def registered_tasks() -> tuple[str, ...]:
    """Return versioned task identifiers registered in the current process."""
    return tuple(sorted(_TASK_FACTORIES))


__all__ = ["DeployTask", "create_task", "register_task", "registered_tasks"]
