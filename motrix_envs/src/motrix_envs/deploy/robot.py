# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert built-in robot configurations into deployment contracts."""

from collections.abc import Sequence

import motrixsim as mtx
import numpy as np
from omegaconf import MISSING

from motrix_deploy.contracts import RobotSpec
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg, SceneCfg, SceneObjsCfg
from motrix_env_motrixsim.compiler import build_scene_model


@configclass
class _RobotSceneObjsCfg(SceneObjsCfg):
    robot: RobotCfg = MISSING


def build_robot_model(robot_cfg: RobotCfg) -> mtx.SceneModel:
    """Compile one robot configuration into a standalone MotrixSim model."""
    return build_scene_model(SceneCfg(objs=_RobotSceneObjsCfg(robot=robot_cfg)))


def resolve_joint_actuators_in_canonical_order(
    actuators: Sequence[mtx.Actuator],
    joint_names: Sequence[str],
) -> list[mtx.Actuator]:
    """Resolve one joint-targeting actuator per canonical joint, preserving joint order."""
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Canonical joint names must be unique")

    by_target: dict[str, mtx.Actuator] = {}
    for actuator in actuators:
        if actuator.target_type != "joint":
            continue
        if actuator.target_name in by_target:
            raise ValueError(f"Multiple actuators target joint {actuator.target_name!r}")
        by_target[actuator.target_name] = actuator

    missing = set(joint_names) - set(by_target)
    if missing:
        raise ValueError(f"No actuator targets canonical joints: {sorted(missing)}")
    return [by_target[name] for name in joint_names]


def build_robot_spec(
    robot_cfg: RobotCfg,
    *,
    key_pose_name: str,
    model: mtx.SceneModel | None = None,
) -> RobotSpec:
    """Build a canonical deployment robot contract from one robot configuration."""
    if key_pose_name not in robot_cfg.key_pose.poses:
        raise ValueError(f"Robot does not define key pose {key_pose_name!r}")

    if model is None:
        model = build_robot_model(robot_cfg)
    joint_names = tuple(robot_cfg.key_pose.joint_names)
    actuators = resolve_joint_actuators_in_canonical_order(model.actuators, joint_names)
    joint_actuator_count = sum(actuator.target_type == "joint" for actuator in model.actuators)
    if len(actuators) != joint_actuator_count:
        raise ValueError("Robot deployment requires its canonical joints to cover all joint actuators")
    if any(actuator.ctrl_range is None for actuator in actuators):
        raise ValueError("Robot deployment requires a finite control range for every canonical joint actuator")
    if any(actuator.force_range is None for actuator in actuators):
        raise ValueError("Robot deployment requires a finite force range for every canonical joint actuator")

    control_ranges = np.asarray([actuator.ctrl_range for actuator in actuators], dtype=np.float32)
    force_ranges = np.asarray([actuator.force_range for actuator in actuators], dtype=np.float32)
    expected_range_shape = (len(joint_names), 2)
    if control_ranges.shape != expected_range_shape:
        raise ValueError(f"Robot actuator control ranges must have shape {expected_range_shape}, got {control_ranges}")
    if force_ranges.shape != expected_range_shape or not np.all(np.isfinite(force_ranges)):
        raise ValueError(f"Robot actuator force ranges must have shape {expected_range_shape} with finite values")
    if not np.allclose(force_ranges[:, 0], -force_ranges[:, 1], atol=1e-6, rtol=0.0):
        raise ValueError("Robot deployment currently requires symmetric actuator force ranges")

    return RobotSpec(
        base_link_name=robot_cfg.base_link_name,
        joint_names=joint_names,
        default_joint_position=np.asarray(robot_cfg.key_pose.poses[key_pose_name], dtype=np.float32),
        position_lower=control_ranges[:, 0],
        position_upper=control_ranges[:, 1],
        torque_limit=force_ranges[:, 1],
    )


def read_position_servo_gains(
    model: mtx.SceneModel,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read canonical kp/kd vectors from native position-servo actuators."""
    actuators = resolve_joint_actuators_in_canonical_order(model.actuators, joint_names)
    data = mtx.SceneData(model)
    kp = np.empty(len(actuators), dtype=np.float32)
    kd = np.empty(len(actuators), dtype=np.float32)
    for index, actuator in enumerate(actuators):
        if isinstance(actuator, mtx.PositionActuator):
            if actuator.kd is None:
                raise ValueError(f"Robot actuator {actuator.name!r} must use absolute damping")
            kp[index] = actuator.kp
            kd[index] = actuator.kd
            continue
        if isinstance(actuator, mtx.GeneralActuator):
            gain = actuator.get_gain_override(data)
            stiffness = actuator.get_stiffness_override(data)
            bias = actuator.get_bias_override(data)
            if not np.isclose(gain, stiffness, atol=1e-7, rtol=0.0) or not np.isclose(bias, 0.0, atol=1e-7, rtol=0.0):
                raise ValueError(f"Robot actuator {actuator.name!r} must implement an unbiased position servo")
            kp[index] = gain
            kd[index] = actuator.get_damping_override(data)
            continue
        raise TypeError(f"Robot actuator {actuator.name!r} must be a position servo, got {type(actuator).__name__}")
    if np.any(~np.isfinite(kp)) or np.any(~np.isfinite(kd)) or np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("Robot position-servo gains must be finite and non-negative")
    return kp, kd


__all__ = [
    "build_robot_model",
    "build_robot_spec",
    "read_position_servo_gains",
    "resolve_joint_actuators_in_canonical_order",
]
