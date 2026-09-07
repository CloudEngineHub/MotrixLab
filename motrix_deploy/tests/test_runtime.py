# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unified control loop and fake backend contract tests."""

from collections.abc import Callable

import numpy as np
import pytest

from motrix_deploy.artifact import DeploymentManifest
from motrix_deploy.backend.fake import FakeRobotInterface, LaggedFakeRobotInterface
from motrix_deploy.contracts import FloatArray, RobotCapabilities, RobotCommand, RobotState, TensorSpec, float32_array
from motrix_deploy.errors import LieDownRequestedError
from motrix_deploy.policy import PolicyRuntime
from motrix_deploy.runtime import (
    ControlLoop,
    FixedStepScheduler,
    PolicyContext,
    RealtimeScheduler,
)
from motrix_deploy.task import DeployTask
from motrix_env_core.input import CommandBinding, ConstantPlanarVelocityBinding, PlanarVelocityCommand


class ConstantPolicy(PolicyRuntime):
    def __init__(self, input_size: int, action: FloatArray) -> None:
        self._input_spec = TensorSpec(name="observation", shape=(1, input_size))
        self._output_spec = TensorSpec(name="action", shape=(1, action.size))
        self._action = float32_array(action, path="test.action", shape=(action.size,))
        self.observations: list[FloatArray] = []

    @property
    def input_spec(self) -> TensorSpec:
        return self._input_spec

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    def infer(self, observation: FloatArray) -> FloatArray:
        value = float32_array(observation, path="test.observation", shape=(self.input_spec.shape[1],))
        self.observations.append(value)
        return self._action


class _TestDeployTask(DeployTask[PlanarVelocityCommand]):
    def __init__(self, manifest: DeploymentManifest) -> None:
        self.robot = manifest.robot
        self._previous_action = np.zeros(manifest.task.action_size, dtype=np.float32)

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> None:
        del state, context
        self._previous_action = np.zeros(2, dtype=np.float32)

    def build_observation(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
        del context
        return np.concatenate((state.joint_position - self.robot.default_joint_position, state.joint_velocity)).astype(
            np.float32
        )

    def process_action(self, action: FloatArray) -> RobotCommand:
        executed = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._previous_action = executed
        zeros = np.zeros(2, dtype=np.float32)
        return RobotCommand(
            joint_position=self.robot.default_joint_position + np.float32(0.25) * executed,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=np.full(2, 35.0, dtype=np.float32),
            kd=np.full(2, 0.5, dtype=np.float32),
        )

    def validate_command(self, command: PlanarVelocityCommand) -> None:
        if not isinstance(command, PlanarVelocityCommand) or command.batch_size != 1:
            raise ValueError("expected one planar velocity command")


def _loop(manifest: DeploymentManifest, backend: FakeRobotInterface) -> ControlLoop:
    return ControlLoop(
        robot=manifest.robot,
        backend=backend,
        task=_TestDeployTask(manifest),
        policy=ConstantPolicy(manifest.task.observation_size, np.array([0.4, -0.4], dtype=np.float32)),
        command_binding=ConstantPlanarVelocityBinding([0.5, 0.0, 0.0]),
        scheduler=FixedStepScheduler(manifest.control.period_s),
        state_timeout_s=manifest.control.state_timeout_s,
    )


@pytest.mark.parametrize("backend_type", [FakeRobotInterface, LaggedFakeRobotInterface])
def test_backend_contract_runs_without_control_loop_changes(
    manifest_factory: Callable[[], DeploymentManifest],
    backend_type: type[FakeRobotInterface],
) -> None:
    manifest = manifest_factory()
    backend = backend_type(manifest.robot.joint_names)

    result = _loop(manifest, backend).run(steps=5)

    assert result.success
    assert result.exit_reason == "completed"
    assert result.completed_steps == 5
    assert result.simulation_time_s == pytest.approx(0.1)
    assert len(backend.commands) == 5
    assert backend.events[-2:] == ["stop", "close"]
    assert set(result.latency) == {"input", "read", "observation", "inference", "action", "write", "loop"}


def test_fixed_step_rollout_is_deterministic(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    first_backend = FakeRobotInterface(manifest.robot.joint_names)
    second_backend = FakeRobotInterface(manifest.robot.joint_names)

    first_loop = _loop(manifest, first_backend)
    second_loop = _loop(manifest, second_backend)
    first = first_loop.run(steps=4)
    second = second_loop.run(steps=4)

    assert (first.success, first.exit_reason, first.completed_steps) == (
        second.success,
        second.exit_reason,
        second.completed_steps,
    )
    for first_command, second_command in zip(first_backend.commands, second_backend.commands, strict=True):
        np.testing.assert_array_equal(first_command.joint_position, second_command.joint_position)
    assert isinstance(first_loop.policy, ConstantPolicy)
    assert isinstance(second_loop.policy, ConstantPolicy)
    for first_observation, second_observation in zip(
        first_loop.policy.observations,
        second_loop.policy.observations,
        strict=True,
    ):
        np.testing.assert_array_equal(first_observation, second_observation)


class _FailingBinding(CommandBinding[PlanarVelocityCommand]):
    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        del batch_size
        raise RuntimeError("input disconnected")


class _TrackingBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        self.batch_sizes.append(batch_size)
        return PlanarVelocityCommand(np.zeros((batch_size, 3), dtype=np.float32))


class _InterruptingBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(self, interrupt_at: int) -> None:
        self._interrupt_at = interrupt_at
        self.read_count = 0

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        if self.read_count == self._interrupt_at:
            raise KeyboardInterrupt
        self.read_count += 1
        return PlanarVelocityCommand(np.zeros((batch_size, 3), dtype=np.float32))


def test_control_loop_reads_one_singleton_command_batch_per_tick(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    binding = _TrackingBinding()
    loop = ControlLoop(
        robot=manifest.robot,
        backend=FakeRobotInterface(manifest.robot.joint_names),
        task=_TestDeployTask(manifest),
        policy=ConstantPolicy(manifest.task.observation_size, np.zeros(2, dtype=np.float32)),
        command_binding=binding,
        scheduler=FixedStepScheduler(manifest.control.period_s),
        state_timeout_s=manifest.control.state_timeout_s,
    )

    result = loop.run(steps=3)

    assert result.success
    assert binding.batch_sizes == [1, 1, 1]


def test_control_loop_runs_without_a_step_bound_until_interrupted(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    binding = _InterruptingBinding(interrupt_at=3)
    loop = ControlLoop(
        robot=manifest.robot,
        backend=FakeRobotInterface(manifest.robot.joint_names),
        task=_TestDeployTask(manifest),
        policy=ConstantPolicy(manifest.task.observation_size, np.zeros(2, dtype=np.float32)),
        command_binding=binding,
        scheduler=FixedStepScheduler(manifest.control.period_s),
        state_timeout_s=manifest.control.state_timeout_s,
    )

    result = loop.run()

    assert not result.success
    assert result.exit_reason == "interrupted"
    assert result.completed_steps == 3


def test_command_binding_failure_returns_input_error(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot.joint_names)
    loop = ControlLoop(
        robot=manifest.robot,
        backend=backend,
        task=_TestDeployTask(manifest),
        policy=ConstantPolicy(manifest.task.observation_size, np.zeros(2, dtype=np.float32)),
        command_binding=_FailingBinding(),
        scheduler=FixedStepScheduler(manifest.control.period_s),
        state_timeout_s=manifest.control.state_timeout_s,
    )

    result = loop.run(steps=1)

    assert not result.success
    assert result.exit_reason == "input_error"
    assert result.error == "input disconnected"
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize(
    ("backend_kwargs", "reason", "completed_steps"),
    [
        ({"fail_read_at": 0}, "state_timeout", 0),
        ({"unhealthy_at": 1}, "backend_unhealthy", 0),
        ({"fail_write_at": 2}, "backend_write_error", 2),
    ],
)
def test_backend_failures_return_structured_results_and_cleanup(
    manifest_factory: Callable[[], DeploymentManifest],
    backend_kwargs: dict[str, int],
    reason: str,
    completed_steps: int,
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot.joint_names, **backend_kwargs)

    result = _loop(manifest, backend).run(steps=5)

    assert not result.success
    assert result.exit_reason == reason
    assert result.completed_steps == completed_steps
    assert backend.events[-2:] == ["stop", "close"]


def test_backend_native_joint_order_can_differ_from_canonical_order(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(tuple(reversed(manifest.robot.joint_names)))

    result = _loop(manifest, backend).run(steps=2)

    assert result.success


def test_missing_backend_joint_fails_before_first_command(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface((*manifest.robot.joint_names[:-1], "missing_joint"))

    result = _loop(manifest, backend).run(steps=2)

    assert result.exit_reason == "validation_error"
    assert result.completed_steps == 0
    assert backend.commands == []
    assert backend.events == ["open", "stop", "close"]


class InterruptingBackend(FakeRobotInterface):
    def read_state(self, timeout_s: float):
        del timeout_s
        raise KeyboardInterrupt


def test_user_interrupt_returns_result_and_cleans_up(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    backend = InterruptingBackend(manifest.robot.joint_names)

    result = _loop(manifest, backend).run(steps=2)

    assert result.exit_reason == "interrupted"
    assert backend.events[-2:] == ["stop", "close"]


class InvalidStateBackend(FakeRobotInterface):
    def read_state(self, timeout_s: float):
        del timeout_s
        return None


def test_invalid_state_fails_and_lifecycle_cleanup_is_idempotent(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = InvalidStateBackend(manifest.robot.joint_names)

    result = _loop(manifest, backend).run(steps=2)
    events_after_run = list(backend.events)
    backend.stop()
    backend.close()

    assert result.exit_reason == "invalid_state"
    assert result.completed_steps == 0
    assert backend.events == events_after_run


def test_realtime_scheduler_uses_absolute_deadlines() -> None:
    now = [1_000_000_000]
    sleeps: list[float] = []

    def clock_ns() -> int:
        return now[0]

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += round(duration * 1e9)

    scheduler = RealtimeScheduler(0.02, clock_ns=clock_ns, sleep=sleep)
    scheduler.reset()
    scheduler.wait(1)
    now[0] += 5_000_000
    scheduler.wait(2)

    assert sleeps == pytest.approx([0.02, 0.015])
    assert scheduler.elapsed_time_s(2) == pytest.approx(0.04)


class EnablingBackend(FakeRobotInterface):
    def __init__(self, joint_names: tuple[str, ...], *, sample_period_s: float = 0.02) -> None:
        super().__init__(joint_names, sample_period_s=sample_period_s)
        self._capabilities = RobotCapabilities(
            control_modes=("joint_pd",),
            state_fields=self._capabilities.state_fields,
            max_command_rate_hz=1.0 / sample_period_s,
            requires_enable=True,
        )
        self.enabled = False
        self.initial_command: RobotCommand | None = None

    def enable(self, initial_command: RobotCommand) -> None:
        assert "write" not in self.events
        self.initial_command = initial_command
        self.events.append("enable")
        self.enabled = True

    def write_command(self, command: RobotCommand) -> None:
        assert self.enabled
        super().write_command(command)


class LieDownBackend(FakeRobotInterface):
    def write_command(self, command: RobotCommand) -> None:
        del command
        raise LieDownRequestedError("remote B requested lie-down shutdown")


def test_lie_down_request_returns_structured_result_and_cleans_up(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = LieDownBackend(manifest.robot.joint_names)

    result = _loop(manifest, backend).run(steps=1)

    assert not result.success
    assert result.exit_reason == "lie_down"
    assert result.completed_steps == 0
    assert backend.events[-2:] == ["stop", "close"]


def test_required_enable_completes_before_first_policy_write(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot.joint_names)

    result = _loop(manifest, backend).run(steps=1)

    assert result.success
    assert backend.events[:6] == ["open", "read", "enable", "read", "read", "write"]
