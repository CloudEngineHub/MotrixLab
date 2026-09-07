# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/point_mass.xml"


@registry.envcfg("point_mass")
@configclass
class PointMassEnvCfg(DirectEnvCfg):
    """Move a two-dimensional point mass to a random target.

    zh_CN: 控制二维质点移动到随机目标位置。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    render_spacing: float = 2.0
    target_radius: float = 0.1
