# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment profile compiler registry tests."""

from typing import cast

import pytest

from motrix_deploy.profile import (
    DeploymentProfile,
    build_deployment_profile,
    register_profile_compiler,
    registered_profile_compilers,
)


def test_profile_compiler_is_selected_by_environment_name() -> None:
    expected = cast(DeploymentProfile, object())

    @register_profile_compiler("test-profile-a", "test-profile-b")
    def compile_profile(env_name: str) -> DeploymentProfile:
        assert env_name in {"test-profile-a", "test-profile-b"}
        return expected

    assert build_deployment_profile("test-profile-a") is expected
    assert {"test-profile-a", "test-profile-b"} <= set(registered_profile_compilers())

    with pytest.raises(ValueError, match="already registered"):
        register_profile_compiler("test-profile-a")(compile_profile)


def test_unknown_profile_reports_registered_environments() -> None:
    with pytest.raises(ValueError, match="No deployment profile compiler.*supported environments"):
        build_deployment_profile("unknown-test-profile")
