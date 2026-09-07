# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import inspect
import numbers
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.kernel_data import canonicalize_kernel_data, is_kernel_data
from motrix_env_core.sim import SimQueriesCfg

if TYPE_CHECKING:
    from motrix_env_core.base import EnvCfg
    from motrix_env_core.numba.manager.env import ManagerBasedEnvCfg, ManagerEnv


@configclass(kw_only=True)
class ObservationTermCfg(abc.ABC):
    """Configuration that creates one environment-local observation term."""

    def required_sim_queries(self, env_cfg: EnvCfg) -> SimQueriesCfg:
        """Return the simulator data and model queries this term requires."""
        del env_cfg
        return SimQueriesCfg()

    @abc.abstractmethod
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        """Resolve environment resources and create the runtime term."""


def _canonicalize_observation_args(args: tuple[Any, ...], *, context: str) -> tuple[Any, ...]:
    """Validate and canonicalize positional Numba-compatible arguments."""
    values: list[Any] = []
    for index, value in enumerate(args):
        if is_kernel_data(value):
            values.append(canonicalize_kernel_data(value, context=f"{context} args[{index}]"))
        elif isinstance(value, (bool, int, float, np.generic)):
            values.append(value)
        else:
            raise TypeError(
                f"{context} args[{index}] must be a scalar or a @kernel_data value; got {type(value).__name__}. "
                f"Raw np.ndarray values are not supported: wrap array data in a @kernel_data type "
                f"(e.g. with SharedArray fields) so it lowers into kernel inputs."
            )
    return tuple(values)


@dataclass(frozen=True, slots=True, init=False)
class BaseTerm:
    """A dispatch function and its static positional arguments."""

    dispatch: Callable[..., Any]
    args: tuple[Any, ...]

    def __init__(self, dispatch: Callable[..., Any], *args: Any) -> None:
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "args", args)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not inspect.isfunction(self.dispatch):
            raise TypeError(f"Term dispatch must be a Python function, got {type(self.dispatch).__name__}.")
        if not getattr(self.dispatch, "__motrix_manager_dispatch__", False):
            raise TypeError(f"Term dispatch {self.dispatch.__qualname__!r} must be decorated with @dispatch.")
        if not isinstance(self.args, tuple):
            raise TypeError("Term invocation args must be a tuple.")


@dataclass(frozen=True, slots=True, init=False)
class ObsTerm(BaseTerm):
    """Observation dispatch term with a fixed output width."""

    size: int

    def __init__(self, size: int, dispatch: Callable[..., Any], *args: Any) -> None:
        object.__setattr__(self, "size", size)
        BaseTerm.__init__(self, dispatch, *args)

    def __post_init__(self) -> None:
        BaseTerm.__post_init__(self)
        if not isinstance(self.size, numbers.Integral) or isinstance(self.size, (bool, np.bool_)):
            raise TypeError("Observation term size must be an integer.")
        if self.size <= 0:
            raise ValueError("Observation term size must be positive.")


@configclass
class ManagerObservationGroupCfg:
    """Typed declaration group for one named observation group's terms."""

    def to_dict(self) -> dict[str, ObservationTermCfg]:
        """Return observation term configs keyed by their declaration names."""
        term_cfgs: dict[str, ObservationTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, ObservationTermCfg):
                raise TypeError(
                    f"Observation term {term_field.name!r} must be an ObservationTermCfg, "
                    f"got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


@configclass
class ManagerObservationsCfg:
    """Typed declaration groups for one environment's observation terms."""

    def to_dict(self) -> dict[str, ManagerObservationGroupCfg]:
        """Return observation groups keyed by their declaration names."""
        group_cfgs: dict[str, ManagerObservationGroupCfg] = {}
        for group_field in fields(self):
            group_name = group_field.name
            group_cfg = getattr(self, group_name)
            if group_name not in {"policy", "value"}:
                raise ValueError(f"Unsupported observation group {group_name!r}; expected 'policy' or 'value'.")
            if not isinstance(group_cfg, ManagerObservationGroupCfg):
                raise TypeError(
                    f"Observation group {group_name!r} must be a ManagerObservationGroupCfg, "
                    f"got {type(group_cfg).__name__}."
                )
            group_cfgs[group_name] = group_cfg
        return group_cfgs


@dataclass(frozen=True)
class ObservationTermEntry:
    name: str
    term: ObsTerm
    size: int


@dataclass(frozen=True)
class ObservationGroupEntry:
    name: str
    terms: tuple[ObservationTermEntry, ...]
    size: int


def create_observation_groups(
    cfg: ManagerBasedEnvCfg,
    env: ManagerEnv,
) -> dict[str, ObservationGroupEntry]:
    """Create and validate runtime observation terms grouped by output buffer."""
    groups: dict[str, ObservationGroupEntry] = {}
    for group_name, term_cfgs in cfg.observation_cfgs().items():
        entries = []
        group_size = 0
        for term_name, term_cfg in term_cfgs.items():
            created = term_cfg(env)
            if not isinstance(created, ObsTerm):
                raise TypeError(
                    f"Observation term {group_name}.{term_name} __call__() must return ObsTerm, "
                    f"got {type(created).__name__}."
                )
            term = ObsTerm(
                int(created.size),
                created.dispatch,
                *_canonicalize_observation_args(created.args, context=f"Observation term {group_name}.{term_name}"),
            )
            resolved_size = int(term.size)
            entries.append(ObservationTermEntry(term_name, term, resolved_size))
            group_size += resolved_size
        if entries:
            groups[group_name] = ObservationGroupEntry(group_name, tuple(entries), group_size)
    if "policy" not in groups:
        raise ValueError("Manager environment config requires a 'policy' observation group.")
    return groups


__all__ = [
    "BaseTerm",
    "ManagerObservationGroupCfg",
    "ManagerObservationsCfg",
    "ObsTerm",
    "ObservationGroupEntry",
    "ObservationTermCfg",
    "ObservationTermEntry",
    "create_observation_groups",
]
