# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class RlMethod:
    rllib: str
    algo: str


def parse_method(value: str, deprecated_rllib: str | None = None) -> RlMethod:
    """Parse a CLI RL method string.

    The preferred form is "rllib.algo" (for example, "skrl.ppo"). During the
    migration period, a bare algorithm name is accepted only when rllib= is
    also provided.
    """
    parts = value.split(".")
    if len(parts) == 2 and all(parts):
        method = RlMethod(rllib=parts[0], algo=parts[1])
        if deprecated_rllib is not None and method.rllib != deprecated_rllib:
            raise ValueError(f"rllib={deprecated_rllib!r} conflicts with algo={value!r}.")
        return method

    if len(parts) == 1 and parts[0] and deprecated_rllib is not None:
        return RlMethod(rllib=deprecated_rllib, algo=parts[0])

    raise ValueError("RL method must use the 'rllib.algo' form, such as 'skrl.ppo' or 'rslrl.ppo'.")
