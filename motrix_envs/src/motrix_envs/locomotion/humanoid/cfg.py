# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration schema for command-conditioned humanoid velocity tracking.

Robot configs provide the scene, model element names, and per-joint pose
weights. :class:`HumanoidVelocityTrackingEnv` reads the default pose from the
scene's robot config and contains no robot-specific names or joint counts.
"""

from omegaconf import MISSING

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import NoiseTerrainGeneratorCfg, ProceduralHFieldAssetCfg, SystemCameraCfg
from motrix_env_core.direct.env import DirectEnvCfg
from motrix_env_core.sim import (
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    GeomPairCollidingQuery,
    JointPositionQuery,
    JointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
    SitePositionQuery,
)
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneCfg


def humanoid_sim_queries(
    *,
    base_link: str,
    foot_links: tuple[str, str],
    sole_sites: tuple[str, str],
    termination_geoms: tuple[str, ...],
    ground_geom: str,
    joints: tuple[str, ...],
) -> dict:
    """Build the shared humanoid walk sim-query set from robot-resolved names.

    ``termination_geoms`` is the explicit collision-geom inventory supplied by
    each robot task configuration.
    """

    return {
        "robot_joint_pos": JointPositionQuery(joints=joints),
        "robot_joint_vel": JointVelocityQuery(joints=joints),
        "base_quat": LinkQuaternionQuery(link=base_link),
        "base_lin_vel": LinkLinearVelocityQuery(link=base_link),
        "base_ang_vel": LinkAngularVelocityQuery(link=base_link),
        "foot_pos": BatchLinkPositionQuery(links=foot_links),
        "foot_quat": BatchLinkQuaternionQuery(links=foot_links),
        "sole_l_pos": SitePositionQuery(site=sole_sites[0]),
        "sole_r_pos": SitePositionQuery(site=sole_sites[1]),
        "termination_colliding": GeomPairCollidingQuery(pairs=tuple((name, ground_geom) for name in termination_geoms)),
    }


@configclass
class TerrainSceneAssetsCfg(StandardSceneAssetsCfg):
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(
            seed=0,
            height_scale=0.05,
            flip_y=True,
        ),
        size=(32.0, 32.0),
        shape=(320, 320),
    )


@configclass
class HumanoidWalkSceneCfg(StandardSceneCfg):
    """Standard humanoid-walk scene with consistent playback framing."""

    system_camera: SystemCameraCfg = SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0)


@configclass
class ControlCfg:
    action_scale: float = 0.5


@configclass
class CommandsCfg:
    # Rows are min/max for [lin_vel_x, lin_vel_y, ang_vel_yaw].
    vel_limit: list[list[float]] = [
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
    stand_prob: float = 0.2
    resampling_time: float = 10.0


@configclass
class NormalizationCfg:
    base_lin_vel: float = 2.0
    base_ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    noise_dof_pos: float = 0.01
    noise_dof_vel: float = 0.1


@configclass
class GaitCfg:
    period: float = 1.0
    swing_height: float = 0.09
    feet_phase_sigma: float = 0.008


@configclass
class CurriculumCfg:
    enabled: bool = True
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001
    penalty_terms: tuple[str, ...] = (
        "penalty_ang_vel_xy",
        "penalty_orientation",
        "penalty_action_rate",
        "pose",
        "penalty_close_feet_xy",
        "penalty_feet_ori",
    )


@configclass
class AssetCfg:
    """Model element names needed by the shared humanoid velocity-tracking environment.

    Attributes:
        foot_height_site_names: Left and right sole-site names used to measure local foot clearance.
        ground_geom_name: Ground geom used for terrain-height lookup and contact termination.
        terminate_contact_geom_names: Robot geoms whose contact with the ground terminates an episode.
    """

    foot_height_site_names: tuple[str, str] = ("", "")
    ground_geom_name: str = ""
    terminate_contact_geom_names: tuple[str, ...] = ()


@configclass
class RewardScales:
    tracking_lin_vel: float = 4.0
    tracking_ang_vel: float = 3.0
    penalty_ang_vel_xy: float = -1.0
    penalty_orientation: float = -10.0
    penalty_action_rate: float = -0.5
    feet_phase: float = 5.0
    pose: float = -0.5
    penalty_close_feet_xy: float = -10.0
    penalty_feet_ori: float = -5.0
    alive: float = 10.0


@configclass
class RewardCfg:
    scales: RewardScales = RewardScales()
    tracking_sigma: float = 0.25
    close_feet_threshold: float = 0.15
    pose_weights: dict[str, float] = {}


@configclass
class HumanoidVelocityTrackingEnvCfg(DirectEnvCfg):
    """Robot-agnostic config consumed by ``HumanoidVelocityTrackingEnv``."""

    max_episode_seconds: float = 20.0
    scene: HumanoidWalkSceneCfg = MISSING
    control_config: ControlCfg = ControlCfg()
    reward_config: RewardCfg = RewardCfg()
    commands: CommandsCfg = CommandsCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    gait: GaitCfg = GaitCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    asset: AssetCfg = AssetCfg()
    sim: SimCfg = SimCfg(dt=0.005)
    ctrl_dt: float = 0.02
    spawn_xy_range: float = 0.0
