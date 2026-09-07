# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.asset import MaterialCfg, SkyboxCfg, TextureCfg
from motrix_env_core.config.scene.base import RobotCfg, SceneAssetsCfg, SceneCfg, SceneObjsCfg, SceneVisualCfg
from motrix_env_core.config.scene.geometry import FlatTerrainCfg, GeomCfg
from motrix_env_core.config.scene.light import LightCfg

MOTPHYS_GROUND_TEXTURE = Path(__file__).parents[2] / "common" / "motphys-ground.png"


@configclass
class StandardSceneAssetsCfg(SceneAssetsCfg):
    """Assets used by the standard MotrixLab scene preset."""

    skybox: SkyboxCfg = SkyboxCfg()
    tex_ground: TextureCfg = TextureCfg(file=MOTPHYS_GROUND_TEXTURE)
    mat_ground: MaterialCfg = MaterialCfg(
        texture="tex_ground",
        texture_repeat=(0.4, 0.4),
    )


@configclass
class StandardSceneObjsCfg(SceneObjsCfg):
    """Object slots used by the standard MotrixLab robot-scene template.

    The robot is mandatory in the standard template; robot-less scenes use a
    plain ``SceneObjsCfg`` subclass instead.
    """

    floor: GeomCfg = FlatTerrainCfg(
        material="mat_ground",
    )
    sun: LightCfg = LightCfg(
        color=(0.7, 0.7, 0.7),
        illuminance=10_000.0,
    )
    robot: RobotCfg = MISSING


@configclass
class StandardSceneCfg(SceneCfg):
    """A robot-scene template with standard sky, ground plane, and directional light."""

    assets: StandardSceneAssetsCfg = StandardSceneAssetsCfg()
    visual: SceneVisualCfg = SceneVisualCfg(
        ambient_light_color=(0.3, 0.3, 0.3),
        ambient_light_brightness=1_000.0,
        head_light_color=(0.6, 0.6, 0.6),
        head_light_luminous_power=1_000.0,
        haze=(0.1, 0.1, 0.1, 1.0),
        tone_mapping="none",
    )
    objs: StandardSceneObjsCfg = StandardSceneObjsCfg()
