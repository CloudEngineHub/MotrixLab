# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for common robot and tensor contracts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from motrix_deploy.contracts import RobotCommand, RobotSpec, TensorSpec
from motrix_deploy.errors import ValidationError


def robot_spec(**overrides: object) -> RobotSpec:
    values = {
        "base_link_name": "base",
        "joint_names": ("a", "b"),
        "default_joint_position": np.array([0.0, 0.0], dtype=np.float32),
        "position_lower": np.array([-1.0, -1.0], dtype=np.float32),
        "position_upper": np.array([1.0, 1.0], dtype=np.float32),
        "torque_limit": np.array([3.0, 3.0], dtype=np.float32),
    }
    values.update(overrides)
    return RobotSpec(**values)


def test_contracts_preserve_valid_array_references() -> None:
    source = np.array([0.0, 0.0], dtype=np.float32)
    spec = robot_spec(default_joint_position=source)
    zero = np.zeros(2, dtype=np.float32)
    command = RobotCommand(
        joint_position=source,
        joint_velocity=zero,
        feedforward_torque=zero,
        kp=zero,
        kd=zero,
    )

    assert spec.default_joint_position is source
    assert command.joint_position is source


def test_robot_spec_fields_are_frozen() -> None:
    spec = robot_spec()

    with pytest.raises(FrozenInstanceError):
        spec.base_link_name = "other"


@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        ("position_lower", np.array([2.0, -1.0], dtype=np.float32), "robot.position_range"),
        ("torque_limit", np.array([3.0, np.inf], dtype=np.float32), "robot.torque_limit"),
        ("default_joint_position", np.array([0.0], dtype=np.float32), "robot.default_joint_position.shape"),
        ("default_joint_position", np.array([0.0, 0.0], dtype=np.float64), "robot.default_joint_position.dtype"),
    ],
)
def test_robot_spec_rejects_invalid_arrays(field: str, value: np.ndarray, error_path: str) -> None:
    with pytest.raises(ValidationError, match=error_path):
        robot_spec(**{field: value})


def test_tensor_spec_requires_float32_fixed_shape() -> None:
    with pytest.raises(ValidationError, match="tensor.dtype"):
        TensorSpec(name="input", shape=(1, 4), dtype="float64")


def test_robot_command_rejects_negative_gains() -> None:
    zero = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValidationError, match="command.gains"):
        RobotCommand(
            joint_position=zero,
            joint_velocity=zero,
            feedforward_torque=zero,
            kp=np.array([1.0, -1.0], dtype=np.float32),
            kd=zero,
        )
