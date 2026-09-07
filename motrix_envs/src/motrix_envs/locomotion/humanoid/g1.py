# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.walk_np import HumanoidVelocityTrackingEnv
from motrix_envs.robot import UnitreeG129Dof

# Pinned G1 termination collision inventory, owned by the G1 walk task.
_G1_TERMINATION_GEOMS = (
    "pelvis_collision",
    "left_thigh",
    "right_thigh",
    "torso_collision1",
    "torso_collision2",
    "torso_collision3",
    "head_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
)


@registry.envcfg("g1-walk-flat")
def make_g129dof_walk_flat_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Unitree G1 on flat ground.

    zh_CN: 控制 Unitree G1 在平地上跟踪行走指令。
    """

    robot = UnitreeG129Dof()
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
                "left_hip_pitch_joint": 0.01,
                "left_hip_roll_joint": 1.0,
                "left_hip_yaw_joint": 5.0,
                "left_knee_joint": 0.01,
                "left_ankle_pitch_joint": 5.0,
                "left_ankle_roll_joint": 5.0,
                "right_hip_pitch_joint": 0.01,
                "right_hip_roll_joint": 1.0,
                "right_hip_yaw_joint": 5.0,
                "right_knee_joint": 0.01,
                "right_ankle_pitch_joint": 5.0,
                "right_ankle_roll_joint": 5.0,
                "waist_yaw_joint": 50.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                "left_shoulder_pitch_joint": 50.0,
                "left_shoulder_roll_joint": 50.0,
                "left_shoulder_yaw_joint": 50.0,
                "left_elbow_joint": 50.0,
                "left_wrist_roll_joint": 50.0,
                "left_wrist_pitch_joint": 50.0,
                "left_wrist_yaw_joint": 50.0,
                "right_shoulder_pitch_joint": 50.0,
                "right_shoulder_roll_joint": 50.0,
                "right_shoulder_yaw_joint": 50.0,
                "right_elbow_joint": 50.0,
                "right_wrist_roll_joint": 50.0,
                "right_wrist_pitch_joint": 50.0,
                "right_wrist_yaw_joint": 50.0,
            },
        ),
        asset=humanoid_cfg.AssetCfg(
            foot_height_site_names=("left_foot_contact_point", "right_foot_contact_point"),
            ground_geom_name="floor",
            terminate_contact_geom_names=_G1_TERMINATION_GEOMS,
        ),
        sim=SimCfg(
            dt=0.005,
            solver_iterations=3,
            solver_tolerance=1e-4,
        ),
        spawn_xy_range=0.0,
    )


@registry.envcfg("g1-walk-rough")
def make_g129dof_walk_rough_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Unitree G1 over uneven terrain.

    zh_CN: 控制 Unitree G1 在起伏地形上跟踪行走指令。
    """

    return replace(
        make_g129dof_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=UnitreeG129Dof(),
            ),
        ),
        spawn_xy_range=4.0,
        render_spacing=0.0,
    )


# Backward-compatible class name retained for callers of the old G1 task.
G129dofWalkTask = HumanoidVelocityTrackingEnv

registry.env("g1-walk-flat")(HumanoidVelocityTrackingEnv)
registry.env("g1-walk-rough")(HumanoidVelocityTrackingEnv)
