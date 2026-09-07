# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Declared model-metadata queries and typed compiler dispatch."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motrix_env_core.sim.backend import SimModel


class ModelQuery(abc.ABC):
    """Base class for declared static model-metadata queries."""

    @abc.abstractmethod
    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        """Record this declaration on the compiler through its typed hook."""


@dataclass(frozen=True)
class GeomSpecsQuery(ModelQuery):
    """Selected ``GeomSpec`` values keyed by geom name."""

    names: tuple[str, ...]

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_specs(key, self.names)


@dataclass(frozen=True)
class BodyJointPositionLimitsQuery(ModelQuery):
    """``(lower, upper)`` float32 arrays in one body's joint-DOF order."""

    body: str

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_body_joint_position_limits(key, self.body)


@dataclass(frozen=True)
class DofPositionLimitsQuery(ModelQuery):
    """``(lower, upper)`` float32 arrays in global DOF-position order."""

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_dof_position_limits(key)


@dataclass(frozen=True)
class ActuatorKpQuery(ModelQuery):
    """Nominal position gains in declared-name or full model order."""

    names: tuple[str, ...] | None = None

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_actuator_kp(key, self.names)


@dataclass(frozen=True)
class ActuatorKdQuery(ModelQuery):
    """Nominal damping gains in declared-name or full model order."""

    names: tuple[str, ...] | None = None

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_actuator_kd(key, self.names)


@dataclass(frozen=True)
class BodyMassQuery(ModelQuery):
    """Scalar ``float`` nominal mass of one named link."""

    name: str

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_body_mass(key, self.name)


@dataclass(frozen=True)
class BodyCenterOfMassQuery(ModelQuery):
    """``(3,)`` float32 nominal center-of-mass offset of one named link."""

    name: str

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_body_center_of_mass(key, self.name)


@dataclass(frozen=True)
class GeomFrictionQuery(ModelQuery):
    """``(3,)`` float32 nominal friction parameters."""

    name: str

    def compile_with(self, compiler: SimModelQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_friction(key, self.name)


class SimModelQueryCompiler(abc.ABC):
    """Compile neutral model queries against one backend model."""

    def compile(self, queries: Mapping[str, ModelQuery]) -> SimModel:
        """Compile the core model and every named metadata query.

        Args:
            queries: Model queries keyed by their logical result names.

        Returns:
            The backend-neutral simulator model and compiled metadata values.
        """
        self._begin_compile()
        for key, query in queries.items():
            query.compile_with(self, key=key)
        return self._build_model()

    def _begin_compile(self) -> None:
        """Reset per-compile accumulation before dispatch; default no-op."""

    @abc.abstractmethod
    def _build_model(self) -> SimModel:
        """Assemble the model from values recorded during dispatch.

        Returns:
            The backend-neutral simulator model.
        """

    @abc.abstractmethod
    def compile_geom_specs(self, key: str, geom_names: tuple[str, ...]) -> None:
        """Compile geometry specifications for an ordered set of geometries.

        Args:
            key: Logical key under which the result is stored.
            geom_names: Ordered geometry names to inspect.
        """

    @abc.abstractmethod
    def compile_body_joint_position_limits(self, key: str, body: str) -> None:
        """Compile joint-position limits in one body's joint-DOF order.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body whose joint limits are read.
        """

    @abc.abstractmethod
    def compile_dof_position_limits(self, key: str) -> None:
        """Compile position limits in canonical DOF-position order.

        Args:
            key: Logical key under which the result is stored.
        """

    @abc.abstractmethod
    def compile_actuator_kp(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        """Compile nominal proportional gains for selected or all actuators.

        Args:
            key: Logical key under which the result is stored.
            actuator_names: Ordered actuator names, or ``None`` for all actuators.
        """

    @abc.abstractmethod
    def compile_actuator_kd(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        """Compile nominal damping gains for selected or all actuators.

        Args:
            key: Logical key under which the result is stored.
            actuator_names: Ordered actuator names, or ``None`` for all actuators.
        """

    @abc.abstractmethod
    def compile_body_mass(self, key: str, body: str) -> None:
        """Compile the nominal mass of one body link.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body link whose mass is read.
        """

    @abc.abstractmethod
    def compile_body_center_of_mass(self, key: str, body: str) -> None:
        """Compile the nominal center-of-mass offset of one body link.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body link whose center of mass is read.
        """

    @abc.abstractmethod
    def compile_geom_friction(self, key: str, geom: str) -> None:
        """Compile nominal friction parameters for one geometry.

        Args:
            key: Logical key under which the result is stored.
            geom: Name of the geometry whose friction is read.
        """
