# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo deployment backend configuration schema."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from motrix_deploy.errors import ValidationError


@dataclass
class MujocoBackendConfig:
    """Application-level scene selection and MuJoCo state mapping."""

    scene: str
    scene_mode: str
    sim_dt: float
    solver_iterations: int
    base_position: tuple[float, float, float]
    base_orientation_xyzw: tuple[float, float, float, float]
    base_body_name: str
    imu_site_name: str
    gyro_sensor_name: str
    accelerometer_sensor_name: str
    global_linear_velocity_sensor_name: str
    fall_height_m: float
    fall_up_z: float

    def __post_init__(self) -> None:
        if not self.scene:
            raise ValidationError("backend.scene", "a registered environment name", self.scene)
        if self.scene_mode not in ("train", "play"):
            raise ValidationError("backend.scene_mode", "'train' or 'play'", self.scene_mode)
        if not np.isfinite(self.sim_dt) or self.sim_dt <= 0:
            raise ValidationError("backend.sim_dt", "a positive finite value", self.sim_dt)
        if not isinstance(self.solver_iterations, int) or isinstance(self.solver_iterations, bool):
            raise ValidationError("backend.solver_iterations", "a positive integer", self.solver_iterations)
        if self.solver_iterations <= 0:
            raise ValidationError("backend.solver_iterations", "a positive integer", self.solver_iterations)
        for name in (
            "base_body_name",
            "imu_site_name",
            "gyro_sensor_name",
            "accelerometer_sensor_name",
            "global_linear_velocity_sensor_name",
        ):
            if not getattr(self, name):
                raise ValidationError(f"backend.{name}", "a non-empty string", getattr(self, name))
        if not np.isfinite(self.fall_height_m) or self.fall_height_m <= 0:
            raise ValidationError("backend.fall_height_m", "a positive finite value", self.fall_height_m)
        if not np.isfinite(self.fall_up_z) or not -1.0 <= self.fall_up_z <= 1.0:
            raise ValidationError("backend.fall_up_z", "a finite value in [-1, 1]", self.fall_up_z)
        if len(self.base_position) != 3 or not np.isfinite(self.base_position).all():
            raise ValidationError("backend.base_position", "three finite values", self.base_position)
        if len(self.base_orientation_xyzw) != 4 or not np.isfinite(self.base_orientation_xyzw).all():
            raise ValidationError("backend.base_orientation_xyzw", "four finite values", self.base_orientation_xyzw)
        quaternion_norm = np.linalg.norm(self.base_orientation_xyzw)
        if not np.isclose(quaternion_norm, 1.0, atol=1e-6):
            raise ValidationError("backend.base_orientation_xyzw", "a unit quaternion", quaternion_norm)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MujocoBackendConfig":
        """Parse the backend section of a resolved deployment configuration."""
        values: dict[str, Any] = dict(value)
        values.pop("name", None)
        expected_fields = set(cls.__dataclass_fields__)
        if set(values) != expected_fields:
            raise ValidationError(
                "backend",
                f"fields {sorted(expected_fields)}",
                f"missing={sorted(expected_fields - set(values))}, unknown={sorted(set(values) - expected_fields)}",
            )
        values["base_position"] = tuple(values["base_position"])
        values["base_orientation_xyzw"] = tuple(values["base_orientation_xyzw"])
        return cls(**values)
