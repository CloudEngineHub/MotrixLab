# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/pendulum.xml"


# -- docs-tag-start: pendulum-env-cfg --
@registry.envcfg("pendulum")
@configclass
class PendulumEnvCfg(DirectEnvCfg):
    """Apply joint torque to swing up and balance a pendulum.

    zh_CN: 施加关节力矩使单摆摆起并保持直立。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 10.0
    sim: SimCfg = SimCfg(dt=0.0125)
    ctrl_dt: float = 0.025
    angle_bound: float = 8.0
    cosing_bound: float = 0.0
    # reset_noise_scale: float = 0.01
    # -- docs-tag-end: pendulum-env-cfg --

    def __post_init__(self):
        self.cosing_bound = float(np.cos(np.deg2rad(self.angle_bound)))
