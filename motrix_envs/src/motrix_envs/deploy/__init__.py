# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment profile compilers for built-in MotrixLab environments."""

from motrix_deploy.profile import build_deployment_profile, registered_profile_compilers
from motrix_envs.deploy.go2_walk import build_go2_walk_profile

__all__ = ["build_deployment_profile", "build_go2_walk_profile", "registered_profile_compilers"]
