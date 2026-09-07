# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg

_DIR = os.path.dirname(__file__)
_WALK_MODEL = os.path.join(_DIR, "quadruped_walk.xml")
_ESCAPE_MODEL = os.path.join(_DIR, "quadruped_escape.xml")
_FETCH_MODEL = os.path.join(_DIR, "quadruped_fetch.xml")

# Leg geom names in the env's leg order (front-left, front-right, back-right,
# back-left; thigh/shin/foot/toe per leg). Every quadruped scene defines them.
_LEG_BODY_GEOM_NAMES = (
    ("thigh_front_left", "shin_front_left", "foot_front_left", "toe_front_left"),
    ("thigh_front_right", "shin_front_right", "foot_front_right", "toe_front_right"),
    ("thigh_back_right", "shin_back_right", "foot_back_right", "toe_back_right"),
    ("thigh_back_left", "shin_back_left", "foot_back_left", "toe_back_left"),
)
_LEG_GEOM_NAMES = tuple(name for leg in _LEG_BODY_GEOM_NAMES for name in leg)


@configclass
class QuadrupedBaseCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg(
        file=_WALK_MODEL,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    max_episode_seconds: float = 20.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01
    render_spacing: float = 2.0

    # Task parameters
    desired_speed: float = 0.0
    deviation_angle: float = 0.0
    fix_heading: bool = False
    clip_env_actions: bool = True

    # Observation toggles
    include_origin: bool = False
    include_rangefinder: bool = False
    include_ball: bool = False
    include_target: bool = False

    # Task geometry (fallbacks when sites/geom metadata are unavailable)
    target_radius: float = 0.7
    terrain_size: float = 30.0

    # Gait shaping (walk/run)
    stand_height: float = 0.55
    stand_height_margin: float = 0.25
    height_reward_weight: float = 0.1

    lateral_velocity_limit: float = 0.2
    lateral_velocity_margin: float = 0.5
    lateral_reward_weight: float = 0.05

    heading_reward_margin: float = 1.0
    heading_reward_weight: float = 0.0

    action_smoothness_margin: float = 5.0
    action_smoothness_weight: float = 0.08

    lin_vel_z_weight: float = 0.05
    ang_vel_xy_weight: float = 0.02
    similar_to_default_weight: float = 0.025

    # Forward motion shaping (walk/run)
    backward_penalty_weight: float = 0.2

    # Radial motion shaping (escape)
    radial_velocity_weight: float = 0.2

    # Fetch shaping
    fetch_reward_margin: float = 10.0
    fetch_reward_weight: float = 1.0
    fetch_behind_distance: float = 0.7
    fetch_ahead_distance: float = 0.2
    fetch_side_stage_offset: float = 0.6
    fetch_side_stage_ball_distance: float = 1.0
    fetch_side_stage_gate_threshold: float = 0.5
    fetch_side_stage_align_threshold: float = 0.8
    fetch_stage_radius: float = 0.5
    fetch_stage_speed: float = 0.5
    fetch_stage_reward_weight: float = 0.5
    fetch_corridor_width: float = 0.3
    fetch_behind_align_margin: float = 1.0
    fetch_heading_margin: float = 1.0
    fetch_heading_weight: float = 0.2
    fetch_ready_ball_distance: float = 0.5
    fetch_ready_weight: float = 0.3
    fetch_ready_threshold: float = 0.7
    fetch_push_speed: float = 0.2
    fetch_push_reward_weight: float = 0.3
    fetch_backward_penalty_weight: float = 0.2
    fetch_away_penalty_weight: float = 0.4
    fetch_leg_ball_penalty_weight: float = 0.1
    fetch_leg_ball_penalty_margin = 0.03
    fetch_stability_upright_min: float = 0.8
    fetch_stability_upright_margin: float = 0.5
    fetch_stability_height_min: float = 0.45
    fetch_stability_height_margin: float = 0.2
    fetch_fall_upright_min: float = 0.2
    fetch_fall_height_min: float = 0.25


@registry.envcfg("dm-quadruped-walk")
@configclass
class QuadrupedWalkCfg(QuadrupedBaseCfg):
    """Control the DM Control quadruped to walk forward.

    zh_CN: 控制 DM Control 四足机器人向前行走。
    """

    scene: SceneCfg = SceneCfg(
        file=_WALK_MODEL,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    desired_speed: float = 0.5
    fix_heading: bool = True
    heading_reward_weight: float = 0.2


@registry.envcfg("dm-quadruped-run")
@configclass
class QuadrupedRunCfg(QuadrupedBaseCfg):
    """Control the DM Control quadruped to run forward.

    zh_CN: 控制 DM Control 四足机器人高速向前奔跑。
    """

    scene: SceneCfg = SceneCfg(
        file=_WALK_MODEL,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    desired_speed: float = 5.0
    fix_heading: bool = True
    heading_reward_weight: float = 0.2


@registry.envcfg("dm-quadruped-escape")
@configclass
class QuadrupedEscapeCfg(QuadrupedBaseCfg):
    """Control the DM Control quadruped to escape from a bowl-shaped area.

    zh_CN: 控制 DM Control 四足机器人逃离碗形区域。
    """

    scene: SceneCfg = SceneCfg(
        file=_ESCAPE_MODEL,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    render_camera_name: str = "global"
    desired_speed: float = 3.0
    include_origin: bool = True
    include_rangefinder: bool = False
    deviation_angle: float = 20.0
    fix_heading: bool = True
    heading_reward_weight: float = 0.2
    radial_velocity_weight: float = 0.5
    similar_to_default_weight: float = 0.01


@registry.envcfg("dm-quadruped-fetch")
@configclass
class QuadrupedFetchCfg(QuadrupedBaseCfg):
    """Control the DM Control quadruped to find and fetch a ball.

    zh_CN: 控制 DM Control 四足机器人寻找并取回球。
    """

    scene: SceneCfg = SceneCfg(
        file=_FETCH_MODEL,
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
    desired_speed: float = 2.0
    include_ball: bool = True
    include_target: bool = True
