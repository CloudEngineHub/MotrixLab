# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/hopper.xml"


@registry.envcfg("dm-hopper-stand")
@configclass
class HopperStandCfg(DirectEnvCfg):
    """Keep the one-legged Hopper standing upright.

    zh_CN: 让单腿 Hopper 保持直立站立。
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=7.0, elevation=-25.0, azimuth=90.0),
    )
    max_episode_seconds: float = 20.0
    render_spacing: float = 1.5
    sim: SimCfg = SimCfg(dt=0.02)
    ctrl_dt: float = 0.02
    stand_height: float = 0.6
    hop_speed: float = 0.0


@registry.envcfg("dm-hopper-hop")
@configclass
class HopperHopCfg(HopperStandCfg):
    """Keep the one-legged Hopper upright while hopping forward.

    zh_CN: 让单腿 Hopper 保持直立并向前跳跃。
    """

    hop_speed: float = 2.0
