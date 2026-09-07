# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim backend: scene compilation, translation surface and live behavior."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

import motrixsim as mtx
import numpy as np
import numpy.typing as npt

from motrix_env_core.config import SimCfg
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.sim.backend import (
    ActuatorSpec,
    ActuatorType,
    GeomSpec,
    RenderConfig,
    SimBackend,
    SimModel,
    SimRenderer,
)
from motrix_env_core.sim.model import SimModelQueryCompiler
from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.renderer import MotrixSimRenderer
from motrix_env_motrixsim.sim_data import compile_read_program
from motrix_env_motrixsim.write_compiler import MotrixSimWriteCompiler

FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]


class MotrixSimModelQueryCompiler(SimModelQueryCompiler):
    """Compile typed model queries against one MotrixSim scene model."""

    def __init__(self, model: mtx.SceneModel) -> None:
        self._model = model
        self._others: dict[str, Any] = {}

    def _begin_compile(self) -> None:
        self._others = {}

    def _build_model(self) -> SimModel:
        core = _build_core(self._model)
        return SimModel(actuators=core.actuators, init_dof_pos=core.init_dof_pos, others=self._others)

    def compile_geom_specs(self, key: str, geom_names: tuple[str, ...]) -> None:
        self._others[key] = _geom_specs(self._model, geom_names)

    def compile_body_joint_position_limits(self, key: str, body: str) -> None:
        self._others[key] = _body_joint_position_limits(self._model, body)

    def compile_dof_position_limits(self, key: str) -> None:
        self._others[key] = _dof_position_limits(self._model)

    def compile_actuator_kp(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        self._others[key] = _nominal_actuator_kp(self._model, actuator_names)

    def compile_actuator_kd(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        self._others[key] = _nominal_actuator_kd(self._model, actuator_names)

    def compile_body_mass(self, key: str, body: str) -> None:
        self._others[key] = float(_named_link(self._model, body).mass)

    def compile_body_center_of_mass(self, key: str, body: str) -> None:
        self._others[key] = np.asarray(_named_link(self._model, body).center_of_mass, dtype=np.float32).reshape(3)

    def compile_geom_friction(self, key: str, geom: str) -> None:
        self._others[key] = np.asarray(_named_geom(self._model, geom).friction, dtype=np.float32)


def _nominal_actuator_kp(model: mtx.SceneModel, actuator_names: tuple[str, ...] | None) -> FloatArray:
    values = []
    actuators = model.actuators if actuator_names is None else (_named_actuator(model, name) for name in actuator_names)
    for actuator in actuators:
        if not isinstance(actuator, mtx.PositionActuator):
            raise ValueError(
                "ActuatorKpQuery requires every actuator to carry kp, "
                f"but {actuator.name!r} ({type(actuator).__name__}) does not."
            )
        values.append(float(actuator.kp))
    return np.asarray(values, dtype=np.float32)


def _nominal_actuator_kd(model: mtx.SceneModel, actuator_names: tuple[str, ...] | None) -> FloatArray:
    values = []
    actuators = model.actuators if actuator_names is None else (_named_actuator(model, name) for name in actuator_names)
    for actuator in actuators:
        if not isinstance(actuator, mtx.PositionActuator):
            raise ValueError(
                "ActuatorKdQuery requires every actuator to carry kd, "
                f"but {actuator.name!r} ({type(actuator).__name__}) does not."
            )
        if actuator.kd is None:
            raise ValueError(f"ActuatorKdQuery requires actuator {actuator.name!r} to define kd.")
        values.append(float(actuator.kd))
    return np.asarray(values, dtype=np.float32)


def _named_link(model: mtx.SceneModel, link_name: str) -> mtx.Link:
    link = model.get_link(link_name)
    if link is None:
        raise KeyError(f"Unknown link {link_name!r}.")
    return link


def _named_geom(model: mtx.SceneModel, geom_name: str) -> mtx.Geom:
    geom = model.get_geom(geom_name)
    if geom is None:
        raise KeyError(f"Unknown geom {geom_name!r}.")
    return geom


def _as_pair(values: Iterable[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    lo, hi = (float(value) for value in values)
    return (lo, hi)


def _build_core(model: mtx.SceneModel) -> SimModel:
    """Snapshot a MotrixSim scene model as the required core model surface."""
    actuators = []
    for actuator in model.actuators:
        if actuator.name is None:
            raise ValueError("Every actuator must have a name.")
        actuators.append(
            ActuatorSpec(
                name=actuator.name,
                actuator_type=ActuatorType(actuator.typ),
                target_name=actuator.target_name,
                ctrl_range=_as_pair(actuator.ctrl_range),
                force_range=_as_pair(actuator.force_range),
            )
        )
    return SimModel(
        actuators=tuple(actuators),
        init_dof_pos=np.asarray(model.compute_init_dof_pos(), dtype=np.float32),
    )


def _dof_position_limits(model: mtx.SceneModel) -> tuple[FloatArray, FloatArray]:
    lower = np.full((model.num_dof_pos,), -np.inf, dtype=np.float32)
    upper = np.full((model.num_dof_pos,), np.inf, dtype=np.float32)
    for joint in model.joints:
        position_slice = slice(joint.dof_pos_index, joint.dof_pos_index + joint.num_dof_pos)
        limits = np.asarray(joint.range, dtype=np.float32).reshape(-1, 2)
        if limits.shape[0] != joint.num_dof_pos:
            raise RuntimeError(
                f"Joint {joint.name!r} limits contain {limits.shape[0]} position DOFs, expected {joint.num_dof_pos}."
            )
        lower[position_slice] = limits[:, 0]
        upper[position_slice] = limits[:, 1]
    return lower, upper


def _body_joint_position_limits(model: mtx.SceneModel, body_name: str) -> tuple[FloatArray, FloatArray]:
    body = _named_body(model, body_name)
    ranges = [np.asarray(joint.range, dtype=np.float32).reshape(-1, 2) for joint in body.joints]
    limits = np.concatenate(ranges, axis=0) if ranges else np.empty((0, 2), dtype=np.float32)
    if limits.shape[0] != body.num_joint_dof_pos:
        raise RuntimeError(
            f"Body {body_name!r} joint limits contain {limits.shape[0]} position DOFs, "
            f"expected {body.num_joint_dof_pos}."
        )
    return limits[:, 0].copy(), limits[:, 1].copy()


def _geom_specs(model: mtx.SceneModel, names: tuple[str, ...]) -> dict[str, GeomSpec]:
    specs = {}
    for name in names:
        geom = _named_geom(model, name)
        specs[name] = GeomSpec(
            size=tuple(float(value) for value in np.asarray(geom.size, dtype=np.float32).reshape(-1)),
            local_pose=tuple(float(value) for value in np.asarray(geom.local_pose, dtype=np.float32).reshape(-1)),
        )
    return specs


class MotrixSimBackend(SimBackend):
    """MotrixSim backend: scene compilation at construction plus live behavior."""

    name = "motrixsim"

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int) -> None:
        self._model: mtx.SceneModel = MotrixSimSceneCompiler().compile(scene, sim)
        self._data: mtx.SceneData = mtx.SceneData(self._model, batch=[num_envs])
        self._num_envs = num_envs
        self._model_query_compiler = MotrixSimModelQueryCompiler(self._model)
        self._write_compiler = MotrixSimWriteCompiler(self._model, self._data, self._masked_rows)

    @property
    def model_query_compiler(self) -> SimModelQueryCompiler:
        return self._model_query_compiler

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        return compile_read_program(self._model, self._data, queries)

    @property
    def num_dof_pos(self) -> int:
        return self._model.num_dof_pos

    @property
    def num_dof_vel(self) -> int:
        return self._model.num_dof_vel

    @property
    def num_actuators(self) -> int:
        return self._model.num_actuators

    def create_renderer(
        self,
        config: RenderConfig,
        *,
        num_envs: int,
        render_spacing: float,
        system_camera: SystemCameraCfg,
    ) -> SimRenderer:
        return MotrixSimRenderer(
            self._model,
            lambda: self._data,
            config,
            num_envs=num_envs,
            render_spacing=render_spacing,
            system_camera=system_camera,
        )

    def step(self, substeps: int) -> None:
        self._model.step_n(self._data, substeps)

    def _masked_rows(self, env_ids: IntArray) -> mtx.SceneData:
        mask = np.zeros((self._data.shape[0],), dtype=bool)
        mask[env_ids] = True
        return self._data[mask]

    def sample_terrain_height(self, geom_name: str, env_ids: IntArray, xy: FloatArray) -> FloatArray:
        geom = self._model.get_geom(geom_name)
        if geom is None:
            raise KeyError(f"Unknown terrain geom {geom_name!r}.")
        if hasattr(geom, "sample_height"):
            mask = np.zeros((self._data.shape[0],), dtype=bool)
            mask[env_ids] = True
            points = np.ascontiguousarray(xy, dtype=np.float32)
            return np.asarray(geom.sample_height(self._data[mask], points), dtype=np.float32)
        return np.full(xy.shape[:-1], float(geom.local_pose[2]), dtype=np.float32)

    @property
    def write_compiler(self) -> MotrixSimWriteCompiler:
        return self._write_compiler


def _named_body(model: mtx.SceneModel, body_name: str) -> mtx.Body:
    body = model.get_body(body_name)
    if body is None:
        raise KeyError(f"Unknown body {body_name!r}.")
    return body


def _named_actuator(model: mtx.SceneModel, actuator_name: str) -> mtx.Actuator:
    for actuator in model.actuators:
        if actuator.name == actuator_name:
            return actuator
    raise KeyError(f"Unknown actuator {actuator_name!r}.")


__all__ = ["MotrixSimBackend"]
