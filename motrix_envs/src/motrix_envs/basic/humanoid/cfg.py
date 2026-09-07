# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/humanoid.xml"


@configclass
class InitStateConfig:
    reset_height_factor: float = 0.95
    reset_qvel_range: float = 0.01
    reset_actuator_range: float = 0.02
    hip_yaw_range: tuple[float, float] = (-15.0, 15.0)
    hip_roll_range: tuple[float, float] = (-12.0, 12.0)
    hip_pitch_range: tuple[float, float] = (-12.0, 12.0)
    symmetric_leg_pairs: list[tuple[int, int, tuple[float, float]]] = [
        (10, 16, (-18.0, 2.0)),
        (11, 17, (-25.0, 20.0)),
        (12, 18, (-70.0, 5.0)),
        (13, 19, (-45.0, -25.0)),
        (14, 20, (-40.0, 0.0)),
        (15, 21, (-25.0, 25.0)),
    ]
    symmetric_arm_pairs: list[tuple[int, int]] = [
        (22, 25),
        (23, 26),
        (24, 27),
    ]
    arm_margin_factor: float = 0.1


@configclass
class TerminationConfig:
    head_height_factor: float = 0.5
    torso_upright_threshold: float = 0.2
    extreme_vel_threshold: float = 200.0


@registry.envcfg("dm-humanoid-walk")
@configclass
class HumanoidWalkCfg(DirectEnvCfg):
    """Control the DM Control humanoid to walk forward.

    zh_CN: 控制 DM Control 人形机器人向前行走。
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-20.0, azimuth=90.0),
    )
    max_episode_seconds: float = 25.0
    render_spacing: float = 2.0

    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01
    move_speed: float = 1.0
    stand_height: float = 1.4

    init_state: InitStateConfig = InitStateConfig()
    termination_config: TerminationConfig = TerminationConfig()


@registry.envcfg("dm-humanoid-stand")
@configclass
class HumanoidStandCfg(HumanoidWalkCfg):
    """Control the DM Control humanoid to remain standing.

    zh_CN: 控制 DM Control 人形机器人保持站立。
    """

    move_speed: float = 0.0


@registry.envcfg("dm-humanoid-run")
@configclass
class HumanoidRunCfg(HumanoidWalkCfg):
    """Control the DM Control humanoid to run forward.

    zh_CN: 控制 DM Control 人形机器人高速向前奔跑。
    """

    move_speed: float = 10.0
