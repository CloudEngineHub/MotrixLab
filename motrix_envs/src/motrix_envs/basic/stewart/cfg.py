# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

_DIR = os.path.dirname(__file__)

# Stewart leg link names in the env's leg order (i: 3 anchor angles, j: 2 per angle).
_STEWART_LEGS = tuple(f"leg{j}{i}" for i in range(3) for j in range(2))
_STEWART_TOP_CONNECTS = tuple(f"top_connect{j}{i}" for i in range(3) for j in range(2))


@configclass
class StewartBaseEnvCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg(file=os.path.join(_DIR, "model.xml"))
    # The XML model is relatively stiff. It is not recommended to increase dt above 0.005.
    sim: SimCfg = SimCfg(dt=0.004)
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 24.0
    render_spacing: float = 4.5

    target_rotation_limit_deg: float = 6.0

    platform_radius: float = 0.8
    top_surface_offset: float = 0.1
    ball_radius: float = 0.10
    init_tilt_deg: float = 3.5
    min_init_tilt_deg: float = 0.8
    init_ball_radius_ratio: float = 0.18

    still_xy: float = 0.12
    still_vel: float = 0.07
    still_xy_hysteresis: float = 1.15
    still_vel_hysteresis: float = 1.20
    zero_vel_thresh: float = 0.07
    still_steps_needed: int = 5

    vel_smooth: float = 0.25
    action_smooth: float = 0.60
    center_control_radius: float = 0.25
    center_control_min_gain: float = 0.15

    k_center: float = 0.7
    k_progress: float = 0.6
    k_still: float = 3.0
    fall_penalty: float = -6.0

    disturbance_enabled: bool = False
    disturbance_include_obs: bool = False
    disturbance_scale: float = 1.0
    disturb_pos_xy_min: float = 0.0
    disturb_pos_xy_max: float = 0.010
    disturb_pos_z_max: float = 0.006
    disturb_rot_max_deg: float = 1.2
    disturb_freq_min_hz: float = 0.25
    disturb_freq_max_hz: float = 0.85
    disturb_ramp_seconds: float = 0.80
    disturb_rot_limit_deg: float = 18.0
    disturb_ang_vel_obs_scale_deg_per_s: float = 30.0

    def validate(self):
        super().validate()
        if self.min_init_tilt_deg < 0.0:
            raise ValueError("min_init_tilt_deg must be non-negative")
        if self.init_tilt_deg < self.min_init_tilt_deg:
            raise ValueError("init_tilt_deg must be greater than or equal to min_init_tilt_deg")
        if not 0.0 <= self.init_ball_radius_ratio <= 1.0:
            raise ValueError("init_ball_radius_ratio must be in [0, 1]")
        if not 0.0 <= self.action_smooth <= 1.0:
            raise ValueError("action_smooth must be in [0, 1]")
        if not 0.0 <= self.center_control_min_gain <= 1.0:
            raise ValueError("center_control_min_gain must be in [0, 1]")
        if self.disturb_rot_limit_deg <= 0.0:
            raise ValueError("disturb_rot_limit_deg must be positive")
        if self.disturb_ang_vel_obs_scale_deg_per_s <= 0.0:
            raise ValueError("disturb_ang_vel_obs_scale_deg_per_s must be positive")


@registry.envcfg("stewart")
@registry.envcfg("stewart-static")
@configclass
class StewartEnvCfg(StewartBaseEnvCfg):
    """Keep a Stewart platform level without external disturbances.

    zh_CN: 控制 Stewart 平台在无外部扰动时保持水平稳定。
    """

    pass


@registry.envcfg("stewart-disturb-xy")
@configclass
class StewartDisturbXYEnvCfg(StewartBaseEnvCfg):
    """Recover and level a Stewart platform under XY disturbances.

    zh_CN: 控制 Stewart 平台在 XY 扰动下恢复并保持水平。
    """

    disturbance_enabled: bool = True
    disturbance_include_obs: bool = True
    disturb_pos_xy_min: float = 0.010
    disturb_pos_xy_max: float = 0.050
    disturb_pos_z_max: float = 0.0
    disturb_rot_max_deg: float = 0.0
    disturb_freq_min_hz: float = 0.005
    disturb_freq_max_hz: float = 0.020
