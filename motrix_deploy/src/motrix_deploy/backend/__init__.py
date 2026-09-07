# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot backend lifecycle contract."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from motrix_deploy.artifact import ControlSpec
from motrix_deploy.contracts import HealthStatus, RobotCapabilities, RobotCommand, RobotSpec, RobotState
from motrix_env_core.input import GamePadDevice, KeyboardDevice


class RobotInterface(ABC):
    """Backend boundary shared by simulators and physical robots."""

    @property
    @abstractmethod
    def capabilities(self) -> RobotCapabilities:
        """Return immutable capabilities used during startup negotiation."""

    @abstractmethod
    def open(self, spec: RobotSpec) -> None:
        """Open resources and bind backend joints to the canonical spec."""

    def enable(self, initial_command: RobotCommand) -> None:
        """Explicitly enable command output when required by the backend."""
        del initial_command
        if self.capabilities.requires_enable:
            raise NotImplementedError("backend declares requires_enable=True but does not implement enable()")

    @abstractmethod
    def read_state(self, timeout_s: float) -> RobotState:
        """Read one state sample or raise on timeout."""

    @abstractmethod
    def write_command(self, command: RobotCommand) -> None:
        """Apply one canonical command."""

    @abstractmethod
    def health(self) -> HealthStatus:
        """Return current backend health."""

    @abstractmethod
    def stop(self) -> None:
        """Enter the backend's declared safe stopped state; must be idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release resources; must be idempotent."""


@runtime_checkable
class KeyboardDeviceProvider(Protocol):
    """Optional backend capability that supplies a focused keyboard device."""

    def get_keyboard_device(self) -> KeyboardDevice:
        """Return a device whose lifecycle is owned by the backend."""


@runtime_checkable
class GamePadDeviceProvider(Protocol):
    """Optional backend capability that supplies a gamepad-like device."""

    def get_gamepad_device(self) -> GamePadDevice:
        """Return a device whose lifecycle is owned by the backend."""


@dataclass(frozen=True)
class BackendCreateContext:
    """Runtime values supplied by the deployment core when constructing a backend."""

    control: ControlSpec
    viewer: bool
    realtime: bool = False
    hardware_confirmed: bool = False


BackendFactory = Callable[[Mapping[str, Any], BackendCreateContext], RobotInterface]
BACKEND_ENTRY_POINT_GROUP = "motrix_deploy.backends"


def _backend_entry_points() -> tuple[metadata.EntryPoint, ...]:
    return tuple(metadata.entry_points(group=BACKEND_ENTRY_POINT_GROUP))


def registered_backends() -> tuple[str, ...]:
    """Return backend names advertised by installed plugin distributions."""
    return tuple(sorted(entry_point.name for entry_point in _backend_entry_points()))


def create_backend(
    name: str,
    config: Mapping[str, Any],
    context: BackendCreateContext,
) -> RobotInterface:
    """Discover and construct one installed backend plugin by name."""
    matches = [entry_point for entry_point in _backend_entry_points() if entry_point.name == name]
    if not matches:
        available = ", ".join(registered_backends()) or "none"
        raise ValueError(f"Unsupported deployment backend {name!r}; installed backends: {available}")
    if len(matches) != 1:
        values = sorted(entry_point.value for entry_point in matches)
        raise ValueError(f"Multiple deployment backend plugins are registered as {name!r}: {values}")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f"Deployment backend plugin {name!r} must load a callable factory")
    backend = factory(config, context)
    if not isinstance(backend, RobotInterface):
        raise TypeError(
            f"Deployment backend plugin {name!r} returned {type(backend).__name__}, expected RobotInterface"
        )
    return backend


__all__ = [
    "BACKEND_ENTRY_POINT_GROUP",
    "BackendCreateContext",
    "BackendFactory",
    "GamePadDeviceProvider",
    "KeyboardDeviceProvider",
    "RobotInterface",
    "create_backend",
    "registered_backends",
]
