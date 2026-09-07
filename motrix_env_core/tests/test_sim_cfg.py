# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config.scene import SceneCfg, SceneCompiler


def test_scene_compiler_is_an_abstract_backend_boundary():
    class _DummySceneCompiler(SceneCompiler):
        def compile(self, scene, sim):
            raise NotImplementedError

    with pytest.raises(TypeError):
        SceneCompiler()

    assert isinstance(_DummySceneCompiler(), SceneCompiler)


def test_env_cfg_uses_sim_config_dt_for_substeps_and_validation():
    cfg = EnvCfg(scene=SceneCfg(), sim=SimCfg(dt=0.005), ctrl_dt=0.02)

    cfg.validate()
    assert cfg.sim_substeps == 4

    with pytest.raises(ValueError, match="sim.dt must be less than or equal to ctrl_dt"):
        EnvCfg(scene=SceneCfg(), sim=SimCfg(dt=0.02), ctrl_dt=0.01).validate()


def test_env_cfg_requires_scene_config():
    with pytest.raises(ValueError, match="EnvCfg.scene must be configured"):
        EnvCfg().validate()
