# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/walker.xml"

# Link names in canonical link-index order; index 0 is the root link.
_WALKER_LINKS = ("torso", "right_thigh", "right_leg", "right_foot", "left_thigh", "left_leg", "left_foot")


@registry.envcfg("dm-walker")
@configclass
class WalkerEnvCfg(DirectEnvCfg):
    """Control the planar Walker to walk forward at a target speed.

    zh_CN: 控制平面 Walker 以目标速度向前行走。
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-20.0, azimuth=90.0),
    )
    max_episode_seconds: float = 25.0
    render_spacing: float = 2.0
    sim: SimCfg = SimCfg(dt=0.0125)
    move_speed: float = 1.0
    ctrl_dt: float = 0.025
    stand_height: float = 1.2


@registry.envcfg("dm-stander")
@configclass
class StanderEnvCfg(WalkerEnvCfg):
    """Control the planar Walker to remain standing upright.

    zh_CN: 控制平面 Walker 保持直立站立。
    """

    move_speed: float = 0.0


@registry.envcfg("dm-runner")
@configclass
class RunnerEnvCfg(WalkerEnvCfg):
    """Control the planar Walker to run at a high target speed.

    zh_CN: 控制平面 Walker 以较高目标速度奔跑。
    """

    move_speed: float = 5.0
