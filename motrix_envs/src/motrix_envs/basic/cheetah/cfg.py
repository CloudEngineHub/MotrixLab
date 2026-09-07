# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/cheetah.xml"


@registry.envcfg("dm-cheetah")
@configclass
class CheetahEnvCfg(DirectEnvCfg):
    """Drive the planar Cheetah forward as fast as possible.

    zh_CN: 驱动平面 Cheetah 尽可能快速地向前奔跑。
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    max_episode_seconds: float = 10.0
    render_spacing: float = 2.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.025
    run_speed: float = 10.0
