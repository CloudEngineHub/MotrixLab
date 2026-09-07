# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Direct Go2 walking deployment-task tests."""

import numpy as np
import pytest

from motrix_deploy.artifact import TaskSpec
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime import PolicyContext
from motrix_deploy_tasks.go2_walk import Go2WalkDeployTaskV1
from motrix_env_core.input import PlanarVelocityCommand


def _context(step: int, elapsed_time_s: float, velocity: list[float] | np.ndarray) -> PolicyContext:
    return PolicyContext(
        step=step,
        elapsed_time_s=elapsed_time_s,
        command=PlanarVelocityCommand(np.asarray(velocity, dtype=np.float32)[None, :]),
    )


def _robot() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=tuple(f"joint_{index}" for index in range(12)),
        default_joint_position=np.zeros(12, dtype=np.float32),
        position_lower=np.full(12, -2.0, dtype=np.float32),
        position_upper=np.full(12, 2.0, dtype=np.float32),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _state(robot: RobotSpec) -> RobotState:
    return RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=robot.default_joint_position + np.float32(0.1),
        joint_velocity=np.full(12, 0.2, dtype=np.float32),
        base_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_angular_velocity=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def _task_spec() -> TaskSpec:
    return TaskSpec(
        name="go2_walk/v1",
        observation_size=49,
        action_size=12,
        config={
            "action_scale": 0.25,
            "command_lower": [-1.0, -1.0, -1.0],
            "command_upper": [1.0, 1.0, 1.0],
            "command_scale": [0.5, 0.5, 0.5],
            "feet_phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "gait_frequency_hz": 2.0,
            "standing_threshold": 0.05,
            "kp": [35.0] * 12,
            "kd": [0.5] * 12,
            "raw_clip": [[-1.0] * 12, [1.0] * 12],
        },
    )


def test_go2_observation_has_exact_49_element_contract() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTaskV1(_task_spec(), robot)
    context = _context(0, 0.0, [0.5, 0.0, -0.2])
    task.reset(state, context)

    observation = task.build_observation(state, context)

    np.testing.assert_array_equal(observation[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observation[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(observation[6:18], 0.1)
    np.testing.assert_allclose(observation[18:30], 0.2)
    np.testing.assert_array_equal(observation[30:42], 0.0)
    np.testing.assert_allclose(observation[42:45], [0.5, 0.0, -0.2])
    np.testing.assert_array_equal(observation[45:49], 0.0)


def test_previous_action_and_trot_phase_advance_after_first_tick() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTaskV1(_task_spec(), robot)
    reset_context = _context(0, 0.0, [0.5, 0.0, 0.0])
    task.reset(state, reset_context)
    task.process_action(np.ones(12, dtype=np.float32))

    observation = task.build_observation(
        state,
        _context(1, 0.02, [0.5, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(observation[30:42], 1.0)
    np.testing.assert_allclose(observation[45:49], [0.04, 0.54, 0.54, 0.04])


def test_standing_command_freezes_deployment_gait_phase() -> None:
    robot = _robot()
    task = Go2WalkDeployTaskV1(_task_spec(), robot)

    observation = task.build_observation(
        _state(robot),
        _context(10, 0.2, np.zeros(3, dtype=np.float32)),
    )

    np.testing.assert_array_equal(observation[45:49], np.zeros(4, dtype=np.float32))


def test_standing_threshold_is_required() -> None:
    spec = _task_spec()
    config = dict(spec.config)
    del config["standing_threshold"]
    spec = TaskSpec(
        name=spec.name,
        observation_size=spec.observation_size,
        action_size=spec.action_size,
        config=config,
    )

    with pytest.raises(KeyError, match="standing_threshold"):
        Go2WalkDeployTaskV1(spec, _robot())


def test_action_processing_applies_clip_scale_and_position_limits() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTaskV1(_task_spec(), robot)
    task.reset(
        state,
        _context(0, 0.0, np.zeros(3, dtype=np.float32)),
    )

    command = task.process_action(np.full(12, 2.0, dtype=np.float32))
    next_observation = task.build_observation(
        state,
        _context(1, 0.02, np.zeros(3, dtype=np.float32)),
    )

    np.testing.assert_allclose(command.joint_position, 0.25)
    np.testing.assert_array_equal(next_observation[30:42], 1.0)
    np.testing.assert_array_equal(command.kp, 35.0)
    np.testing.assert_array_equal(command.kd, 0.5)


def test_go2_task_rejects_non_singleton_command_batch() -> None:
    task = Go2WalkDeployTaskV1(_task_spec(), _robot())

    with pytest.raises(ValidationError, match="command.batch_size"):
        task.validate_command(PlanarVelocityCommand(np.zeros((2, 3), dtype=np.float32)))


def test_go2_task_scales_command_range_for_deployment_input_mapping() -> None:
    task = Go2WalkDeployTaskV1(_task_spec(), _robot())

    np.testing.assert_array_equal(task.command_lower, [-0.5, -0.5, -0.5])
    np.testing.assert_array_equal(task.command_upper, [0.5, 0.5, 0.5])
    task.validate_command(PlanarVelocityCommand(np.array([[0.75, 0.0, 0.0]], dtype=np.float32)))
