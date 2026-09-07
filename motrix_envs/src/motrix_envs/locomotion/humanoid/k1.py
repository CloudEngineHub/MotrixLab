# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Booster K1 humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.walk_np import HumanoidVelocityTrackingEnv
from motrix_envs.robot import BoosterK1


def _make_k1_robot() -> BoosterK1:
    return BoosterK1(translation=(0.0, 0.0, -0.46))


# Pinned K1 termination collision inventory, owned by the K1 walk task.
_K1_TERMINATION_GEOMS = (
    "trunk_upper_collision",
    "trunk_lower_collision",
    "head_yaw_collision",
    "head_pitch_collision",
    "left_shoulder_collision",
    "left_elbow_collision",
    "left_hand_collision",
    "right_shoulder_collision",
    "right_elbow_collision",
    "right_hand_collision",
    "left_hip_roll_collision",
    "left_hip_yaw_collision",
    "left_shank_upper_collision",
    "left_shank_lower_collision",
    "right_hip_roll_collision",
    "right_hip_yaw_collision",
    "right_shank_upper_collision",
    "right_shank_lower_collision",
)


@registry.envcfg("k1-walk-flat")
def make_k1_walk_flat_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Booster K1 on flat ground.

    zh_CN: 控制 Booster K1 在平地上跟踪行走指令。
    """

    robot = _make_k1_robot()

    return humanoid_cfg.HumanoidVelocityTrackingEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            objs=StandardSceneObjsCfg(robot=robot),
        ),
        control_config=humanoid_cfg.ControlCfg(action_scale=0.5),
        reward_config=humanoid_cfg.RewardCfg(
            scales=humanoid_cfg.RewardScales(
                tracking_lin_vel=2.0,
                tracking_ang_vel=1.5,
                penalty_action_rate=-2.0,
            ),
            tracking_sigma=0.25,
            close_feet_threshold=0.15,
            pose_weights={
                "AAHead_yaw": 50.0,
                "Head_pitch": 50.0,
                "ALeft_Shoulder_Pitch": 50.0,
                "Left_Shoulder_Roll": 50.0,
                "Left_Elbow_Pitch": 50.0,
                "Left_Elbow_Yaw": 50.0,
                "ARight_Shoulder_Pitch": 50.0,
                "Right_Shoulder_Roll": 50.0,
                "Right_Elbow_Pitch": 50.0,
                "Right_Elbow_Yaw": 50.0,
                "Left_Hip_Pitch": 0.01,
                "Left_Hip_Roll": 1.0,
                "Left_Hip_Yaw": 5.0,
                "Left_Knee_Pitch": 0.01,
                "Left_Ankle_Pitch": 5.0,
                "Left_Ankle_Roll": 5.0,
                "Right_Hip_Pitch": 0.01,
                "Right_Hip_Roll": 1.0,
                "Right_Hip_Yaw": 5.0,
                "Right_Knee_Pitch": 0.01,
                "Right_Ankle_Pitch": 5.0,
                "Right_Ankle_Roll": 5.0,
            },
        ),
        asset=humanoid_cfg.AssetCfg(
            foot_height_site_names=("left_foot", "right_foot"),
            ground_geom_name="floor",
            terminate_contact_geom_names=_K1_TERMINATION_GEOMS,
        ),
        sim=SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4),
        spawn_xy_range=0.0,
    )


@registry.envcfg("k1-walk-rough")
def make_k1_walk_rough_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Booster K1 over uneven terrain.

    zh_CN: 控制 Booster K1 在起伏地形上跟踪行走指令。
    """

    return replace(
        make_k1_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_k1_robot(),
            ),
        ),
        spawn_xy_range=4.0,
        render_spacing=0.0,
    )


registry.env("k1-walk-flat")(HumanoidVelocityTrackingEnv)
registry.env("k1-walk-rough")(HumanoidVelocityTrackingEnv)
