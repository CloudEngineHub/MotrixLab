# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral policy control loop."""

import hashlib
import time
from typing import Generic, TypeVar

import numpy as np

from motrix_deploy.backend import RobotInterface
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.errors import EmergencyStopError, LieDownRequestedError, ValidationError
from motrix_deploy.policy import PolicyRuntime
from motrix_deploy.runtime.context import PolicyContext
from motrix_deploy.runtime.result import LatencyRecorder, RolloutResult
from motrix_deploy.runtime.scheduler import LoopScheduler
from motrix_deploy.task import DeployTask
from motrix_env_core.input import CommandBinding

CommandT = TypeVar("CommandT")


class _LoopFailure(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ControlLoop(Generic[CommandT]):
    """Execute state -> observation -> policy -> action -> command."""

    def __init__(
        self,
        *,
        robot: RobotSpec,
        backend: RobotInterface,
        task: DeployTask[CommandT],
        policy: PolicyRuntime,
        command_binding: CommandBinding[CommandT],
        scheduler: LoopScheduler,
        state_timeout_s: float,
    ) -> None:
        self.robot = robot
        self.backend = backend
        self.task = task
        self.policy = policy
        self.command_binding = command_binding
        self.scheduler = scheduler
        self.state_timeout_s = state_timeout_s
        if scheduler.period_s <= 0 or state_timeout_s <= 0:
            raise ValidationError(
                "control.timing",
                "positive period and timeout",
                (scheduler.period_s, state_timeout_s),
            )
        if policy.input_spec.shape != (1, task.observation_size):
            raise ValidationError("policy.input.shape", str((1, task.observation_size)), policy.input_spec.shape)
        if policy.output_spec.shape != (1, task.action_size):
            raise ValidationError("policy.output.shape", str((1, task.action_size)), policy.output_spec.shape)

    def run(self, *, steps: int | None = None) -> RolloutResult:
        if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
            raise ValidationError("steps", "a positive integer", steps)
        completed = 0
        recorder = LatencyRecorder()
        trace = hashlib.sha256()
        start_ns = time.monotonic_ns()
        exit_reason = "completed"
        error_message: str | None = None
        previous_sample_time: int | None = None
        cleanup_error: Exception | None = None
        try:
            self.backend.open(self.robot)
            self._validate_capabilities()
            initial_state = self._read_and_validate_state(None)
            previous_sample_time = initial_state.sample_time_ns
            self.policy.reset()
            if self.backend.capabilities.requires_enable:
                initial_context = self._read_context(0)
                self.task.reset(initial_state, initial_context)
                try:
                    initial_command = self.task.process_action(np.zeros(self.task.action_size, dtype=np.float32))
                    self.backend.enable(initial_command)
                except (EmergencyStopError, LieDownRequestedError):
                    raise
                except Exception as error:
                    raise _LoopFailure("backend_enable_error", str(error)) from error
                initial_state = self._read_and_validate_state(previous_sample_time)
                previous_sample_time = initial_state.sample_time_ns
                self.task.reset(initial_state, initial_context)
                self.policy.reset()
            self.scheduler.reset()
            while steps is None or completed < steps:
                step = completed
                loop_start = time.monotonic_ns()
                self.scheduler.wait(step)
                phase_start = time.monotonic_ns()
                context = self._read_context(step)
                recorder.add("input", time.monotonic_ns() - phase_start)
                if step == 0:
                    self.task.reset(initial_state, context)
                phase_start = time.monotonic_ns()
                state = self._read_and_validate_state(previous_sample_time)
                recorder.add("read", time.monotonic_ns() - phase_start)
                previous_sample_time = state.sample_time_ns

                phase_start = time.monotonic_ns()
                try:
                    observation = self.task.build_observation(state, context)
                except Exception as error:
                    raise _LoopFailure("invalid_observation", str(error)) from error
                recorder.add("observation", time.monotonic_ns() - phase_start)

                phase_start = time.monotonic_ns()
                try:
                    raw_action = self.policy.infer(observation)
                except Exception as error:
                    raise _LoopFailure("policy_error", str(error)) from error
                recorder.add("inference", time.monotonic_ns() - phase_start)

                phase_start = time.monotonic_ns()
                try:
                    command = self.task.process_action(raw_action)
                except Exception as error:
                    raise _LoopFailure("invalid_action", str(error)) from error
                recorder.add("action", time.monotonic_ns() - phase_start)

                phase_start = time.monotonic_ns()
                try:
                    self.backend.write_command(command)
                except (EmergencyStopError, LieDownRequestedError):
                    raise
                except Exception as error:
                    raise _LoopFailure("backend_write_error", str(error)) from error
                recorder.add("write", time.monotonic_ns() - phase_start)
                health = self.backend.health()
                if not health.healthy:
                    raise _LoopFailure("backend_unhealthy", health.reason)
                trace.update(step.to_bytes(8, byteorder="little", signed=False))
                for value in (
                    state.joint_position,
                    state.joint_velocity,
                    state.base_orientation_xyzw,
                    observation,
                    raw_action,
                    command.joint_position,
                ):
                    trace.update(value.tobytes(order="C"))
                completed += 1
                recorder.add("loop", time.monotonic_ns() - loop_start)
        except KeyboardInterrupt:
            exit_reason = "interrupted"
            error_message = "rollout interrupted by user"
        except EmergencyStopError as error:
            exit_reason = "emergency_stop"
            error_message = str(error)
        except LieDownRequestedError as error:
            exit_reason = "lie_down"
            error_message = str(error)
        except _LoopFailure as error:
            exit_reason = error.reason
            error_message = str(error)
        except ValidationError as error:
            exit_reason = "validation_error"
            error_message = str(error)
        except Exception as error:
            exit_reason = "backend_error"
            error_message = str(error)
        finally:
            try:
                self.backend.stop()
            except Exception as error:
                cleanup_error = cleanup_error or error
            try:
                self.backend.close()
            except Exception as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None and exit_reason == "completed":
            exit_reason = "cleanup_error"
            error_message = str(cleanup_error)
        wall_time_s = (time.monotonic_ns() - start_ns) / 1e9
        simulation_time_s = completed * self.scheduler.period_s
        real_time_factor = simulation_time_s / wall_time_s if wall_time_s > 0 else 0.0
        return RolloutResult(
            success=exit_reason == "completed" and steps is not None and completed == steps,
            exit_reason=exit_reason,
            completed_steps=completed,
            simulation_time_s=simulation_time_s,
            wall_time_s=wall_time_s,
            real_time_factor=real_time_factor,
            overrun_count=self.scheduler.overrun_count,
            trace_sha256=trace.hexdigest(),
            error=error_message,
            latency=recorder.summarize(),
        )

    def _read_context(self, step: int) -> PolicyContext[CommandT]:
        try:
            command = self.command_binding.read_command(batch_size=1)
            self.task.validate_command(command)
            return PolicyContext(
                step=step,
                elapsed_time_s=self.scheduler.elapsed_time_s(step),
                command=command,
            )
        except KeyboardInterrupt:
            raise
        except Exception as error:
            raise _LoopFailure("input_error", str(error)) from error

    def _validate_capabilities(self) -> None:
        capabilities = self.backend.capabilities
        if "joint_pd" not in capabilities.control_modes:
            raise ValidationError("backend.control_modes", "joint_pd", capabilities.control_modes)
        required_fields = {
            "joint_position",
            "joint_velocity",
            "base_orientation_xyzw",
            "base_angular_velocity",
            "base_linear_acceleration",
        }
        missing = required_fields - capabilities.state_fields
        if missing:
            raise ValidationError(
                "backend.state_fields",
                f"contains {sorted(required_fields)}",
                f"missing {sorted(missing)}",
            )
        if capabilities.max_command_rate_hz is not None:
            requested_rate = 1.0 / self.scheduler.period_s
            if requested_rate > capabilities.max_command_rate_hz + 1e-9:
                raise ValidationError(
                    "control.command_rate_hz",
                    f"<= {capabilities.max_command_rate_hz}",
                    requested_rate,
                )

    def _read_and_validate_state(self, previous_sample_time: int | None) -> RobotState:
        try:
            state = self.backend.read_state(self.state_timeout_s)
        except TimeoutError as error:
            raise _LoopFailure("state_timeout", str(error)) from error
        except (EmergencyStopError, LieDownRequestedError):
            raise
        except Exception as error:
            raise _LoopFailure("backend_read_error", str(error)) from error
        try:
            self._validate_state(state, previous_sample_time)
        except _LoopFailure:
            raise
        except Exception as error:
            raise _LoopFailure("invalid_state", str(error)) from error
        health = self.backend.health()
        if not health.healthy:
            raise _LoopFailure("backend_unhealthy", health.reason)
        return state

    def _validate_state(self, state: RobotState, previous_sample_time: int | None) -> None:
        expected = (self.robot.joint_count,)
        if state.joint_position.shape != expected or state.joint_velocity.shape != expected:
            raise _LoopFailure("invalid_state", f"joint shape must be {expected}")
        if previous_sample_time is not None and state.sample_time_ns < previous_sample_time:
            raise _LoopFailure(
                "invalid_state",
                f"state sample timestamp moved backwards: {state.sample_time_ns} < {previous_sample_time}",
            )
        age_s = (time.monotonic_ns() - state.receive_time_ns) / 1e9
        if age_s > self.state_timeout_s:
            raise _LoopFailure("state_timeout", f"state age {age_s:.6f}s exceeds {self.state_timeout_s:.6f}s")
