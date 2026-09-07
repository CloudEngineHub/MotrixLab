# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/finger.xml"
spin_model_file = os.path.dirname(__file__) + "/finger_spin.xml"
turn_easy_model_file = os.path.dirname(__file__) + "/finger_turn_easy.xml"
turn_hard_model_file = os.path.dirname(__file__) + "/finger_turn_hard.xml"

# Collidable geoms shared by every finger model variant (the distal capsule
# and the freejoint-backed target_geom set contype/conaffinity=0 and never
# collide). All cross pairs of this set reproduce the legacy
# ``get_contact_query(data).num_contacts > 0`` reset rejection check exactly.
_FINGER_COLLIDABLE_GEOMS = (
    "ground",
    "proximal_decoration",
    "proximal",
    "fingertip",
    "cap1",
    "cap2",
    "spinner_decoration",
)
_FINGER_COLLISION_PAIRS = tuple(
    (first, second)
    for index, first in enumerate(_FINGER_COLLIDABLE_GEOMS)
    for second in _FINGER_COLLIDABLE_GEOMS[index + 1 :]
)


@configclass
class FingerBaseCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 10.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.02

    # Task setup
    task: str = "spin"  # "spin" | "turn"
    target_radius: float = 0.07

    # Reward thresholds (match dm_control defaults)
    spin_velocity_threshold: float = 15.0

    # Reward mode
    # - "sparse": match dm_control (1 if hinge_velocity <= -threshold else 0)
    # - "shaped": dense reward to make training easier
    reward_mode: str = "sparse"
    shaped_reward_beta: float = 1.0

    # Extra shaping for Spin tasks (helps reduce "no contact" failures)
    spin_touch_bonus_scale: float = 0.0
    spin_touch_bonus_tanh_scale: float = 50.0
    spin_approach_reward_scale: float = 0.0
    spin_approach_sigma: float = 0.15
    # Turn shaping: reward falls linearly to 0 at margin = scale * target_radius
    turn_reward_margin_scale: float = 4.0
    turn_reward_min_margin: float = 0.0
    turn_shaped_reward_beta: float = 1.0
    # Turn shaping mode:
    # - "linear": clip(1 - max(dist,0)/margin, 0..1)
    # - "exp": exp(-max(dist,0)/sigma)
    turn_reward_shape: str = "linear"
    turn_reward_sigma_scale: float = 1.0
    turn_reward_sigma_min: float = 0.05
    # Extra shaping for Turn tasks (to reduce jitter and help contact)
    turn_touch_bonus_scale: float = 0.05
    turn_touch_bonus_tanh_scale: float = 50.0
    # Encourage approaching the spinner (helps avoid "no contact" deadlock)
    turn_approach_reward_scale: float = 0.3
    turn_approach_sigma: float = 0.15
    turn_action_l2_penalty_scale: float = 0.002
    turn_action_delta_l2_penalty_scale: float = 0.01

    # Reset sampling
    reset_collision_free_attempts: int = 200


@registry.envcfg("dm-finger-spin")
@configclass
class FingerSpinCfg(FingerBaseCfg):
    """Use the robotic finger to continuously spin a free object.

    zh_CN: 使用机械手指持续旋转自由物体。
    """

    scene: SceneCfg = SceneCfg(file=spin_model_file)
    task: str = "spin"
    reward_mode: str = "shaped"
    spin_approach_reward_scale: float = 0.15
    spin_touch_bonus_scale: float = 0.03


@registry.envcfg("dm-finger-turn-easy")
@configclass
class FingerTurnEasyCfg(FingerBaseCfg):
    """Turn the finger object to a target angle with an easy tolerance.

    zh_CN: 将机械手指上的物体转到宽容差目标角度。
    """

    scene: SceneCfg = SceneCfg(file=turn_easy_model_file)
    task: str = "turn"
    target_radius: float = 0.07
    reward_mode: str = "shaped"
    turn_reward_shape: str = "exp"


@registry.envcfg("dm-finger-turn-hard")
@configclass
class FingerTurnHardCfg(FingerTurnEasyCfg):
    """Turn the finger object precisely to a target angle.

    zh_CN: 将机械手指上的物体精确转到目标角度。
    """

    scene: SceneCfg = SceneCfg(file=turn_hard_model_file)
    target_radius: float = 0.03
    reward_mode: str = "shaped"
    turn_reward_shape: str = "exp"
