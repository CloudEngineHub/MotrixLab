# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Hydra-based train/play/view command-line entry points."""

import logging
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CfgT = TypeVar("CfgT")


def to_typed_config(cfg: DictConfig, expected_type: type[CfgT]) -> CfgT:
    """Convert a Hydra ``DictConfig`` to its registered dataclass schema.

    Hydra always calls ``@hydra.main`` with a ``DictConfig``. This helper keeps
    that dynamic value at the CLI boundary and gives the rest of the application
    a real dataclass with static field types.
    """
    obj = OmegaConf.to_object(cfg)
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(obj).__name__}")
    return obj
