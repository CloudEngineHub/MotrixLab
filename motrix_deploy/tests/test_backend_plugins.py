# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment backend plugin discovery contract tests."""

from dataclasses import dataclass
from typing import Any

import pytest

import motrix_deploy.backend as backend_registry
from motrix_deploy.artifact import ControlSpec
from motrix_deploy.backend import BackendCreateContext, create_backend, registered_backends
from motrix_deploy.backend.fake import FakeRobotInterface


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: str
    target: Any

    def load(self) -> Any:
        return self.target


def test_backend_plugin_is_discovered_and_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def factory(config, context):
        calls.append((config, context))
        return FakeRobotInterface(("joint",))

    entry_point = _EntryPoint("fake", "test:factory", factory)
    monkeypatch.setattr(backend_registry, "_backend_entry_points", lambda: (entry_point,))
    context = BackendCreateContext(control=ControlSpec(period_s=0.02, state_timeout_s=0.1), viewer=False)

    backend = create_backend("fake", {"name": "fake", "response": 0.5}, context)

    assert isinstance(backend, FakeRobotInterface)
    assert registered_backends() == ("fake",)
    assert calls == [({"name": "fake", "response": 0.5}, context)]


def test_backend_plugin_discovery_rejects_missing_duplicate_and_invalid_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = BackendCreateContext(control=ControlSpec(period_s=0.02, state_timeout_s=0.1), viewer=False)
    monkeypatch.setattr(backend_registry, "_backend_entry_points", lambda: ())
    with pytest.raises(ValueError, match="installed backends: none"):
        create_backend("missing", {}, context)

    duplicates = (
        _EntryPoint("duplicate", "first:factory", lambda config, context: object()),
        _EntryPoint("duplicate", "second:factory", lambda config, context: object()),
    )
    monkeypatch.setattr(backend_registry, "_backend_entry_points", lambda: duplicates)
    with pytest.raises(ValueError, match="Multiple deployment backend plugins"):
        create_backend("duplicate", {}, context)

    invalid = _EntryPoint("invalid", "invalid:factory", lambda config, context: object())
    monkeypatch.setattr(backend_registry, "_backend_entry_points", lambda: (invalid,))
    with pytest.raises(TypeError, match="expected RobotInterface"):
        create_backend("invalid", {}, context)
