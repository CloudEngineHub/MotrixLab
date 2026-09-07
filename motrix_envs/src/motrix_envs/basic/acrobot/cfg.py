# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/acrobot.xml"


# -- docs-tag-start: acrobot-env-cfg --
@registry.envcfg("acrobot")
@configclass
class AcrobotEnvCfg(DirectEnvCfg):
    """Swing up and balance an underactuated two-link robot.

    zh_CN: 摆起并平衡欠驱动双连杆机器人。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    reset_noise_scale: float = 0.1
    max_episode_seconds: float = 10.0
    render_spacing: float = 2.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.02
    reward_scale: float = 1.0


# -- docs-tag-end: acrobot-env-cfg --
