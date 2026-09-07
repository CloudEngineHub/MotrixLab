# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the Unitree Go2 deployment backend."""

from collections.abc import Mapping
from typing import Any

from motrix_deploy.backend import BackendCreateContext, RobotInterface
from motrix_deploy.errors import ValidationError
from motrix_deploy_unitree.config import UnitreeGo2BackendConfig
from motrix_deploy_unitree.interface import UnitreeGo2RobotInterface


def create_backend(config: Mapping[str, Any], context: BackendCreateContext) -> RobotInterface:
    """Construct a guarded Unitree Go2 hardware backend."""
    if context.viewer:
        raise ValidationError("viewer", "false for physical hardware", context.viewer)
    if not isinstance(context.realtime, bool):
        raise ValidationError("realtime", "a boolean", context.realtime)
    if not context.realtime:
        raise ValidationError("realtime", "true for physical hardware", context.realtime)
    if not isinstance(context.hardware_confirmed, bool):
        raise ValidationError("hardware.confirm", "a boolean", context.hardware_confirmed)
    if not context.hardware_confirmed:
        raise ValidationError(
            "hardware.confirm",
            "true after the operator completes the physical safety checklist",
            context.hardware_confirmed,
        )

    return UnitreeGo2RobotInterface(
        UnitreeGo2BackendConfig.from_mapping(config),
        control_period_s=context.control.period_s,
        state_timeout_s=context.control.state_timeout_s,
        hardware_confirmed=context.hardware_confirmed,
    )


__all__ = ["create_backend"]
