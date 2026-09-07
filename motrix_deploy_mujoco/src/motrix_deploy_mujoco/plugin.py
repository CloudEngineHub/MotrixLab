# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the MuJoCo deployment backend."""

from collections.abc import Mapping
from typing import Any

from motrix_deploy.backend import BackendCreateContext, RobotInterface
from motrix_deploy_mujoco.config import MujocoBackendConfig
from motrix_deploy_mujoco.interface import MujocoRobotInterface


def create_backend(config: Mapping[str, Any], context: BackendCreateContext) -> RobotInterface:
    """Construct the MuJoCo backend selected by a deployment recipe."""
    return MujocoRobotInterface(
        MujocoBackendConfig.from_mapping(config),
        control_period_s=context.control.period_s,
        render=context.viewer,
    )


__all__ = ["create_backend"]
