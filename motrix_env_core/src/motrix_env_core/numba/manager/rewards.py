# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.kernel_data import canonicalize_kernel_data, is_kernel_data

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@dataclass(frozen=True, slots=True, init=False)
class RewardTerm:
    """Host-side reward dispatch and its static Numba-compatible arguments."""

    dispatch: Callable[..., float]
    args: tuple[Any, ...]

    def __init__(self, dispatch: Callable[..., float], *args: Any) -> None:
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "args", args)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not inspect.isfunction(self.dispatch):
            raise TypeError(f"Reward dispatch must be a Python function, got {type(self.dispatch).__name__}.")
        if not getattr(self.dispatch, "__motrix_manager_dispatch__", False):
            raise TypeError(f"Reward dispatch {self.dispatch.__qualname__!r} must be decorated with @dispatch.")


def _canonicalize_reward_args(args: tuple[Any, ...], *, context: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for index, value in enumerate(args):
        if is_kernel_data(value):
            values.append(canonicalize_kernel_data(value, context=f"{context} args[{index}]"))
        elif isinstance(value, tuple) and all(isinstance(item, (bool, int, float, np.generic)) for item in value):
            values.append(value)
        elif isinstance(value, (bool, int, float, np.generic)):
            values.append(value)
        else:
            raise TypeError(
                f"{context} args[{index}] must be a scalar, a scalar tuple, or a @kernel_data value; "
                f"got {type(value).__name__}. Raw np.ndarray values are not supported: wrap array data in a "
                f"@kernel_data type (e.g. with SharedArray fields) so it lowers into kernel inputs."
            )
    return tuple(values)


@configclass(kw_only=True)
class RewardTermCfg(abc.ABC):
    weight: float

    @abc.abstractmethod
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        """Create one environment-local reward term."""


@configclass
class ManagerRewardsCfg:
    """Typed declaration group for one environment's reward terms."""

    def to_dict(self) -> dict[str, RewardTermCfg]:
        """Return reward configs keyed by their declaration names."""
        term_cfgs: dict[str, RewardTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Manager reward {term_field.name!r} must be a RewardTermCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


def create_reward_terms(cfg: dict[str, RewardTermCfg], env: ManagerEnv) -> dict[str, RewardTerm]:
    terms = {}
    for name, term_cfg in cfg.items():
        created = term_cfg(env)
        if not isinstance(created, RewardTerm):
            raise TypeError(f"Manager reward {name} __call__() must return RewardTerm, got {type(created).__name__}.")
        terms[name] = RewardTerm(
            created.dispatch,
            *_canonicalize_reward_args(created.args, context=f"Manager term reward.{name}"),
        )
    return terms


__all__ = ["ManagerRewardsCfg", "RewardTerm", "RewardTermCfg", "create_reward_terms"]
