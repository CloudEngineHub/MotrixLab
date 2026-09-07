# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/reacher.xml"


@registry.envcfg("dm-reacher")
@configclass
class ReacherEnvCfg(DirectEnvCfg):
    """Move a two-link arm endpoint to a random target.

    zh_CN: 控制双关节机械臂末端到达随机目标。
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=2.0, elevation=-75.0, azimuth=90.0),
    )
    max_episode_seconds: float = 6.0
    render_spacing: float = 0.5
    sim: SimCfg = SimCfg(dt=0.0125)
    move_speed: float = 1.0
    ctrl_dt: float = 0.025
    target_size: float = 0.02
