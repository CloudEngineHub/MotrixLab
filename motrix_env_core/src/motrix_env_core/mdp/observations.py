# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable robot observation terms for manager-based environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.numba.manager.context import ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.numba.math.quaternion import rotate_inverse
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
    SimQueriesCfg,
)

if TYPE_CHECKING:
    from motrix_env_core.base import EnvCfg
    from motrix_env_core.numba.manager.env import ManagerEnv


@configclass(kw_only=True)
class UniformNoiseCfg:
    amplitude: float = 0.0


def _scene_robot_base_link_name(env_cfg: EnvCfg) -> str:
    """Return the scene's primary robot base-link name."""
    scene = env_cfg.scene
    robot = scene.objs.robot if scene is not None else None
    if not isinstance(robot, RobotCfg):
        raise ValueError(
            "Framework robot observation terms derive their default sim queries from "
            "scene.objs.robot (RobotCfg); configure a scene robot to derive the defaults."
        )
    return robot.resolved_base_link_name


@dispatch
def robot_joint_pos_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    dof_pos = ctx.sim["obs.robot_joint_pos"]
    out[:] = dof_pos
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@dispatch
def robot_joint_vel_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    dof_vel = ctx.sim["obs.robot_joint_vel"]
    out[:] = dof_vel
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@dispatch
def robot_base_linear_velocity_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    root_quat = ctx.sim["obs.robot_base_quat"]
    root_lin_vel = ctx.sim["obs.robot_base_linear_velocity"]
    rotate_inverse(root_quat, root_lin_vel, out)
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@dispatch
def robot_base_angular_velocity_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    root_quat = ctx.sim["obs.robot_base_quat"]
    root_ang_vel = ctx.sim["obs.robot_base_angular_velocity"]
    rotate_inverse(root_quat, root_ang_vel, out)
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@configclass(kw_only=True)
class RobotJointPosObsCfg(ObservationTermCfg):
    """Joint-position observation from the ``obs.robot_joint_pos`` simulator query.

    The term owns the ``obs.robot_joint_pos`` data query and contributes the
    scene-robot default; tasks may not redeclare a term-owned key.
    """

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def required_sim_queries(self, env_cfg: EnvCfg) -> SimQueriesCfg:
        base_link = _scene_robot_base_link_name(env_cfg)
        return SimQueriesCfg(data={"obs.robot_joint_pos": BodyJointPositionQuery(body=base_link)})

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        size = env.sim_data["obs.robot_joint_pos"].shape[1]
        return ObsTerm(size, robot_joint_pos_obs, np.float32(self.noise.amplitude))


@configclass(kw_only=True)
class RobotJointVelObsCfg(ObservationTermCfg):
    """Joint-velocity observation from the ``obs.robot_joint_vel`` simulator query.

    The term owns the ``obs.robot_joint_vel`` data query and contributes the
    scene-robot default; tasks may not redeclare a term-owned key.
    """

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def required_sim_queries(self, env_cfg: EnvCfg) -> SimQueriesCfg:
        base_link = _scene_robot_base_link_name(env_cfg)
        return SimQueriesCfg(data={"obs.robot_joint_vel": BodyJointVelocityQuery(body=base_link)})

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        size = env.sim_data["obs.robot_joint_vel"].shape[1]
        return ObsTerm(size, robot_joint_vel_obs, np.float32(self.noise.amplitude))


@configclass(kw_only=True)
class RobotBaseLinearVelocityObsCfg(ObservationTermCfg):
    """Base linear velocity observation in the base-local frame.

    Reads world-frame velocity and quaternion from the standard robot sim
    queries. The term owns those keys and contributes scene-robot defaults;
    tasks may not redeclare a term-owned key.
    """

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def required_sim_queries(self, env_cfg: EnvCfg) -> SimQueriesCfg:
        base_link = _scene_robot_base_link_name(env_cfg)
        return SimQueriesCfg(
            data={
                "obs.robot_base_quat": LinkQuaternionQuery(link=base_link),
                "obs.robot_base_linear_velocity": LinkLinearVelocityQuery(link=base_link),
            }
        )

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        size = env.sim_data["obs.robot_base_linear_velocity"].shape[1]
        return ObsTerm(size, robot_base_linear_velocity_obs, np.float32(self.noise.amplitude))


@configclass(kw_only=True)
class RobotBaseAngularVelocityObsCfg(ObservationTermCfg):
    """Base angular velocity observation in the base-local frame.

    Reads world-frame velocity and quaternion from the standard robot sim
    queries. The term owns those keys and contributes scene-robot defaults;
    tasks may not redeclare a term-owned key.
    """

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def required_sim_queries(self, env_cfg: EnvCfg) -> SimQueriesCfg:
        base_link = _scene_robot_base_link_name(env_cfg)
        return SimQueriesCfg(
            data={
                "obs.robot_base_quat": LinkQuaternionQuery(link=base_link),
                "obs.robot_base_angular_velocity": LinkAngularVelocityQuery(link=base_link),
            }
        )

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        size = env.sim_data["obs.robot_base_angular_velocity"].shape[1]
        return ObsTerm(size, robot_base_angular_velocity_obs, np.float32(self.noise.amplitude))


__all__ = [
    "RobotBaseAngularVelocityObsCfg",
    "RobotBaseLinearVelocityObsCfg",
    "RobotJointPosObsCfg",
    "RobotJointVelObsCfg",
    "UniformNoiseCfg",
]
