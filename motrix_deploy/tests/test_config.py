# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed deployment application configuration tests."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy.config import DeployRunConfig


def _run_mapping() -> dict[str, object]:
    return {
        "artifact": "artifact.deploy",
        "backend": {"name": "test", "option": 1},
        "viewer": False,
    }


def _typed_config(values: dict[str, object]) -> DeployRunConfig:
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DeployRunConfig), values))
    assert isinstance(config, DeployRunConfig)
    return config


def test_deploy_run_config_parses_resolved_application_tree() -> None:
    config = _typed_config(_run_mapping())

    assert config.artifact == "artifact.deploy"
    assert config.backend_name == "test"
    assert config.backend_options == {"option": 1}
    assert config.viewer is False


def test_deploy_run_config_rejects_unknown_top_level_fields() -> None:
    values = _run_mapping()
    values["unknown"] = True

    with pytest.raises(OmegaConfBaseException, match="unknown"):
        _typed_config(values)
