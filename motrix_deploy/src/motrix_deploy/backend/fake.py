# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deterministic in-memory backends for runtime contract tests."""

import time

import numpy as np

from motrix_deploy.backend import RobotInterface
from motrix_deploy.contracts import HealthStatus, RobotCapabilities, RobotCommand, RobotSpec, RobotState
from motrix_deploy.errors import ValidationError


class FakeRobotInterface(RobotInterface):
    """Echo position commands into state without a physics dependency."""

    def __init__(
        self,
        joint_names: tuple[str, ...],
        *,
        sample_period_s: float = 0.02,
        response: float = 1.0,
        fail_read_at: int | None = None,
        fail_write_at: int | None = None,
        unhealthy_at: int | None = None,
    ) -> None:
        self._capabilities = RobotCapabilities(
            control_modes=("joint_pd",),
            state_fields=frozenset(
                {
                    "joint_position",
                    "joint_velocity",
                    "base_orientation_xyzw",
                    "base_angular_velocity",
                    "base_linear_acceleration",
                }
            ),
            max_command_rate_hz=1.0 / sample_period_s,
        )
        self._joint_names = joint_names
        self._period_s = sample_period_s
        self._period_ns = round(sample_period_s * 1e9)
        self._response = response
        self._fail_read_at = fail_read_at
        self._fail_write_at = fail_write_at
        self._unhealthy_at = unhealthy_at
        self._spec: RobotSpec | None = None
        self._position = np.zeros(len(joint_names), dtype=np.float32)
        self._velocity = np.zeros(len(joint_names), dtype=np.float32)
        self._sample_time_ns = 0
        self._read_count = 0
        self._write_count = 0
        self._opened = False
        self._closed = False
        self.events: list[str] = []
        self.commands: list[RobotCommand] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    def open(self, spec: RobotSpec) -> None:
        if self._opened and not self._closed:
            raise RuntimeError("fake backend is already open")
        self.events.append("open")
        if len(self._joint_names) != len(spec.joint_names) or set(self._joint_names) != set(spec.joint_names):
            raise ValidationError("backend.joint_names", str(spec.joint_names), self._joint_names)
        self._spec = spec
        self._position = np.array(spec.default_joint_position, copy=True)
        self._velocity.fill(0)
        self._sample_time_ns = 0
        self._read_count = 0
        self._write_count = 0
        self.commands.clear()
        self._opened = True
        self._closed = False

    def read_state(self, timeout_s: float) -> RobotState:
        self._require_open()
        if timeout_s <= 0:
            raise TimeoutError("state timeout must be positive")
        if self._fail_read_at == self._read_count:
            raise TimeoutError(f"injected state timeout at read {self._read_count}")
        self.events.append("read")
        self._read_count += 1
        return self._state()

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        if self._fail_write_at == self._write_count:
            raise RuntimeError(f"injected write failure at command {self._write_count}")
        previous = self._position.copy()
        self._position += np.float32(self._response) * (command.joint_position - self._position)
        self._velocity = ((self._position - previous) / np.float32(self._period_s)).astype(np.float32)
        self._sample_time_ns += self._period_ns
        self.commands.append(command)
        self.events.append("write")
        self._write_count += 1

    def health(self) -> HealthStatus:
        healthy = self._unhealthy_at is None or self._read_count < self._unhealthy_at
        return HealthStatus(
            healthy=healthy,
            reason="" if healthy else f"injected unhealthy state at read {self._read_count}",
            last_successful_communication_ns=time.monotonic_ns(),
        )

    def stop(self) -> None:
        if "stop" not in self.events:
            self.events.append("stop")

    def close(self) -> None:
        if not self._closed:
            self.events.append("close")
            self._closed = True

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("fake backend is not open")

    def _state(self) -> RobotState:
        now = time.monotonic_ns()
        zeros = np.zeros(3, dtype=np.float32)
        return RobotState(
            sample_time_ns=self._sample_time_ns,
            receive_time_ns=now,
            joint_position=self._position,
            joint_velocity=self._velocity,
            base_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_angular_velocity=zeros,
            base_linear_acceleration=zeros,
            base_position=np.array([0.0, 0.0, 0.3], dtype=np.float32),
            base_linear_velocity=zeros,
        )


class LaggedFakeRobotInterface(FakeRobotInterface):
    """A second adapter whose state moves halfway to each command."""

    def __init__(self, joint_names: tuple[str, ...], *, sample_period_s: float = 0.02) -> None:
        super().__init__(joint_names, sample_period_s=sample_period_s, response=0.5)
