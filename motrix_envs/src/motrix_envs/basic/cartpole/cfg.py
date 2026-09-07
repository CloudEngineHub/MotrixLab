# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/cartpole.xml"


@configclass
class CartPoleBaseCfg:
    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=9.0, elevation=-30.0, azimuth=90.0),
    )
    max_episode_seconds: float = 10
    render_spacing: float = 2.0


@registry.envcfg("cartpole")
@configclass
class CartPoleEnvCfg(CartPoleBaseCfg, DirectEnvCfg):
    """Move a cart to keep an inverted pendulum upright.

    zh_CN: 移动小车以保持倒立摆直立。
    """

    reset_noise_scale: float = 0.01
