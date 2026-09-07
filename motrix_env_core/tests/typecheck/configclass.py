# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import TYPE_CHECKING

from typing_extensions import assert_type

from motrix_env_core.config import configclass


@configclass
class BaseCfg:
    name: str
    values: list[int] = []


@configclass(kw_only=True)
class ChildCfg(BaseCfg):
    enabled: bool = False
    labels: dict[str, int] = field(default_factory=dict)


if TYPE_CHECKING:
    cfg = ChildCfg("demo", values=[1], enabled=True, labels={"value": 2})

    assert_type(cfg.name, str)
    assert_type(cfg.values, list[int])
    assert_type(cfg.enabled, bool)
    assert_type(cfg.labels, dict[str, int])

    ChildCfg()  # type: ignore[call-arg]
    ChildCfg("demo", [1], True)  # type: ignore[misc]
    ChildCfg("demo", unknown=True)  # type: ignore[call-arg]
