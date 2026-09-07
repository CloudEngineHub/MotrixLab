# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

bring_ball_model_file = os.path.join(os.path.dirname(__file__), "manipulator_bring_ball.xml")

_ARM_JOINTS = (
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "finger",
    "fingertip",
    "thumb",
    "thumbtip",
)
_TOUCH_SENSORS = ("palm_touch", "finger_touch", "thumb_touch", "fingertip_touch", "thumbtip_touch")

_HAND_GEOMS = (
    "hand",
    "palm1",
    "palm2",
    "thumb1",
    "thumb2",
    "thumbtip1",
    "thumbtip2",
    "finger1",
    "finger2",
    "fingertip1",
    "fingertip2",
)
# (hand geom, ball) collision pairs in the legacy hand × object order.
_HAND_OBJECT_PAIRS = tuple((name, "ball") for name in _HAND_GEOMS)


@registry.envcfg("dm-manipulator-bring-ball")
@configclass
class BringBallCfg(DirectEnvCfg):
    """Control a manipulator to bring a ball to a target location.

    zh_CN: 控制机械臂把球移动到目标位置。
    """

    # Simulation
    scene: SceneCfg = SceneCfg(
        file=bring_ball_model_file,
        system_camera=SystemCameraCfg(distance=11.0, elevation=-25.0, azimuth=135.0),
    )
    max_episode_seconds: float = 10.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01
    render_spacing: float = 2.5

    # Reset sampling (match dm_control defaults).
    p_in_hand: float = 0.1
    p_in_target: float = 0.1
    randomize_arm: bool = True

    # Target sampling.
    target_x_range: tuple[float, float] = (-0.4, 0.4)
    target_z_range: tuple[float, float] = (0.1, 0.4)
    target_y: float = 0.001
    target_angle_range: tuple[float, float] = (-3.14159265, 3.14159265)

    # Object sampling.
    object_x_range: tuple[float, float] = (-0.4, 0.4)
    object_z_range: tuple[float, float] = (0.0, 0.7)
    object_angle_range: tuple[float, float] = (0.0, 6.28318531)
    object_x_vel_range: tuple[float, float] = (-5.0, 5.0)
    min_object_hand_dist: float = 0.08

    # BringBall reward shaping.
    lift_height_threshold: float = 0.04
    touch_threshold: float = 0.01
    side_penalty_scale: float = 0.05
    side_penalty_tanh_scale: float = 10.0
    hover_penalty_scale: float = 0.02
    hover_close_threshold: float = 0.1
    post_grasp_discount: float = 0.7
    lift_height_weight: float = 0.3
    transport_weight: float = 0.7
    transport_progress_scale: float = 0.0
    transport_progress_clip: float = 0.02
    precision_weight: float = 0.0
    precision_margin: float = 0.02
    precision_value_at_margin: float = 0.1

    # BringBall-specific overrides.
    p_in_hand: float = 0.0
    p_in_target: float = 0.0
    randomize_arm: bool = False
    object_z_range: tuple[float, float] = (0.2, 0.7)
    object_x_vel_range: tuple[float, float] = (0.0, 0.0)
    hover_penalty_scale: float = 0.03
    post_grasp_discount: float = 0.0
    lift_height_weight: float = 0.1
    transport_weight: float = 2.0
    transport_progress_scale: float = 2.0
    transport_progress_clip: float = 0.02
    precision_weight: float = 1.0
    precision_margin: float = 0.01

    # Reward component weights (total reward mixing).
    reach_weight: float = 1.0
    orient_weight: float = 1.5
    pause_weight: float = 0.5
    close_weight: float = 2.0
    lift_reward_weight: float = 6.0
