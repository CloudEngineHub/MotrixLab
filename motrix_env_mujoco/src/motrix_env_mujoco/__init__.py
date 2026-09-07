# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo backend for :mod:`motrix_env_core`.

This package binds the backend-neutral scene-compiler boundary of
``motrix_env_core`` to MuJoCo: it compiles a declarative ``SceneCfg`` into an
``mujoco.MjModel`` (sim2sim, preview, conversion tools). It provides no live
simulation. Importing the package is cheap on purpose — the native ``mujoco``
module loads only when a deep module (``compiler``, ``backend``) is imported,
so registry discovery never pulls the simulator. Registration happens through
the ``motrix_env.sim_backends`` entry point (see
:mod:`motrix_env_mujoco.register`).
"""
