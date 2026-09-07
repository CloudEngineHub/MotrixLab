# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Direct observation and action implementation for ``go2_walk/v1``."""

import numpy as np

from motrix_deploy.artifact import TaskSpec
from motrix_deploy.contracts import FloatArray, RobotCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime import PolicyContext
from motrix_deploy.task import DeployTask, register_task
from motrix_env_core.input import PlanarVelocityCommand

TASK_NAME = "go2_walk/v1"
OBSERVATION_SIZE = 49
ACTION_SIZE = 12


@register_task(TASK_NAME)
class Go2WalkDeployTaskV1(DeployTask[PlanarVelocityCommand]):
    """Go2 walking policy I/O semantics frozen as task version 1."""

    def __init__(self, spec: TaskSpec, robot: RobotSpec) -> None:
        if spec.name != TASK_NAME:
            raise ValidationError("task.name", TASK_NAME, spec.name)
        if spec.observation_size != OBSERVATION_SIZE or spec.action_size != ACTION_SIZE:
            raise ValidationError(
                "task.policy_shape",
                f"observation={OBSERVATION_SIZE}, action={ACTION_SIZE}",
                f"observation={spec.observation_size}, action={spec.action_size}",
            )
        if robot.joint_count != ACTION_SIZE:
            raise ValidationError("robot.joint_count", str(ACTION_SIZE), robot.joint_count)
        config = spec.config
        shape = (ACTION_SIZE,)

        def cfg(key: str, shp: tuple[int, ...] = shape) -> FloatArray:
            return _config_array(config[key], f"task.config.{key}", shp)

        self.spec = spec
        self.robot = robot
        self._action_scale = cfg("action_scale")
        self._kp = cfg("kp")
        self._kd = cfg("kd")
        self._raw_lower, self._raw_upper = _config_range(config["raw_clip"], "task.config.raw_clip", shape)
        self._command_lower = cfg("command_lower", (3,))
        self._command_upper = cfg("command_upper", (3,))
        self._command_scale = cfg("command_scale", (3,))
        self._feet_phase_offsets = cfg("feet_phase_offsets", (4,))
        gait_frequency = config["gait_frequency_hz"]
        if isinstance(gait_frequency, bool) or not isinstance(gait_frequency, (int, float)):
            raise ValidationError("task.config.gait_frequency_hz", "a positive finite number", gait_frequency)
        self._gait_frequency_hz = gait_frequency
        if not np.isfinite(self._gait_frequency_hz) or self._gait_frequency_hz <= 0:
            raise ValidationError("task.config.gait_frequency_hz", "a positive finite number", gait_frequency)
        standing_threshold = config["standing_threshold"]
        if isinstance(standing_threshold, bool) or not isinstance(standing_threshold, (int, float)):
            raise ValidationError("task.config.standing_threshold", "a non-negative finite number", standing_threshold)
        self._standing_threshold = float(standing_threshold)
        if not np.isfinite(self._standing_threshold) or self._standing_threshold < 0.0:
            raise ValidationError("task.config.standing_threshold", "a non-negative finite number", standing_threshold)
        if (
            np.any(self._action_scale < 0)
            or np.any(self._kp < 0)
            or np.any(self._kd < 0)
            or np.any(self._raw_lower > self._raw_upper)
            or np.any(self._command_lower > self._command_upper)
            or np.any(self._command_scale < 0.0)
            or np.any(self._command_scale > 1.0)
        ):
            raise ValidationError(
                "task.config",
                "ordered ranges, non-negative control values, and command_scale inside [0, 1]",
                "invalid value",
            )
        self._previous_action: FloatArray = np.zeros(ACTION_SIZE, dtype=np.float32)

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def command_lower(self) -> FloatArray:
        return self._command_lower * self._command_scale

    @property
    def command_upper(self) -> FloatArray:
        return self._command_upper * self._command_scale

    def reset(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> None:
        self._validate_state(state)
        if context.step != 0:
            raise ValidationError("context.step", "0 during task reset", context.step)
        self.validate_command(context.command)
        self._previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def build_observation(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
        self._validate_state(state)
        velocity = self._velocity_command(context)
        phase = np.zeros(4, dtype=np.float32)
        if context.step > 0 and np.linalg.norm(velocity) >= self._standing_threshold:
            phase = np.mod(
                context.elapsed_time_s * self._gait_frequency_hz + self._feet_phase_offsets,
                1.0,
            ).astype(np.float32)
        observation = np.concatenate(
            (
                state.base_angular_velocity,
                _projected_gravity(state.base_orientation_xyzw),
                state.joint_position - self.robot.default_joint_position,
                state.joint_velocity,
                self._previous_action,
                velocity,
                phase,
            )
        ).astype(np.float32)
        return float32_array(observation, path="observation", shape=(OBSERVATION_SIZE,))

    def process_action(self, action: FloatArray) -> RobotCommand:
        raw = float32_array(action, path="action.raw", shape=(ACTION_SIZE,))
        executed = np.clip(raw, self._raw_lower, self._raw_upper).astype(np.float32)
        target = self.robot.default_joint_position + self._action_scale * executed
        target = np.clip(target, self.robot.position_lower, self.robot.position_upper).astype(np.float32)
        self._previous_action = executed
        zeros = np.zeros(ACTION_SIZE, dtype=np.float32)
        return RobotCommand(
            joint_position=target,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=self._kp,
            kd=self._kd,
        )

    def validate_command(self, command: PlanarVelocityCommand) -> None:
        if not isinstance(command, PlanarVelocityCommand):
            raise ValidationError("command.type", "PlanarVelocityCommand", type(command).__name__)
        if command.batch_size != 1:
            raise ValidationError("command.batch_size", "1", command.batch_size)
        velocity = command.values[0, :]
        if np.any(velocity < self._command_lower) or np.any(velocity > self._command_upper):
            raise ValidationError(
                "command.velocity",
                f"inside [{self._command_lower.tolist()}, {self._command_upper.tolist()}]",
                velocity.tolist(),
            )

    def _velocity_command(self, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
        self.validate_command(context.command)
        return context.command.values[0, :]

    def _validate_state(self, state: RobotState) -> None:
        expected = (ACTION_SIZE,)
        if state.joint_position.shape != expected or state.joint_velocity.shape != expected:
            raise ValidationError("state.joints.shape", str(expected), state.joint_position.shape)


def _config_array(value: object, path: str, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:  # broadcast scalar config across the required shape
        array = np.broadcast_to(array, shape)
    return float32_array(array, path=path, shape=shape)


def _config_range(value: object, path: str, shape: tuple[int, ...]) -> tuple[FloatArray, FloatArray]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValidationError(path, "[lower, upper] scalars or vectors", value)
    return _config_array(value[0], f"{path}[0]", shape), _config_array(value[1], f"{path}[1]", shape)


def _projected_gravity(orientation_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = orientation_xyzw
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


__all__ = ["Go2WalkDeployTaskV1"]
