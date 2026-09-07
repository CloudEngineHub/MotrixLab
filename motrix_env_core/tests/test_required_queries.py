# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for observation-term required query contribution and merging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg, SceneCfg
from motrix_env_core.config.scene.base import ModelFileCfg
from motrix_env_core.manager import (
    ManagerBasedEnvCfg,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationTermCfg,
)
from motrix_env_core.mdp.observations import (
    RobotBaseAngularVelocityObsCfg,
    RobotBaseLinearVelocityObsCfg,
    RobotJointPosObsCfg,
    RobotJointVelObsCfg,
)
from motrix_env_core.numba.manager.env import (
    observation_required_sim_queries,
)
from motrix_env_core.sim import (
    BodyJointPositionLimitsQuery,
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    DofPositionLimitsQuery,
    JointPositionQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
    SimQueriesCfg,
)

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@configclass(kw_only=True)
class _NoopCfg(ObservationTermCfg):
    def required_sim_queries(self, env_cfg) -> SimQueriesCfg:
        del env_cfg
        return SimQueriesCfg()

    def size(self, env: ManagerEnv) -> int:
        del env
        return 1

    def __call__(self, env: ManagerEnv):
        del env
        raise NotImplementedError


@configclass(kw_only=True)
class _DataRequiredCfg(ObservationTermCfg):
    key: str
    body: str

    def required_sim_queries(self, env_cfg) -> SimQueriesCfg:
        del env_cfg
        return SimQueriesCfg(data={self.key: BodyJointPositionQuery(body=self.body)})

    def size(self, env: ManagerEnv) -> int:
        del env
        return 1

    def __call__(self, env: ManagerEnv):
        del env
        raise NotImplementedError


@configclass(kw_only=True)
class _ModelRequiredCfg(ObservationTermCfg):
    def required_sim_queries(self, env_cfg) -> SimQueriesCfg:
        del env_cfg
        return SimQueriesCfg(model={"obs.limits": DofPositionLimitsQuery()})

    def size(self, env: ManagerEnv) -> int:
        del env
        return 1

    def __call__(self, env: ManagerEnv):
        del env
        raise NotImplementedError


@configclass
class _TwoTerms(ManagerObservationGroupCfg):
    left: ObservationTermCfg = _NoopCfg()
    right: ObservationTermCfg = _NoopCfg()


@configclass
class _ObsCfg(ManagerObservationsCfg):
    policy: _TwoTerms = _TwoTerms()


@configclass
class _RobotTermsGroup(ManagerObservationGroupCfg):
    dof_pos: RobotJointPosObsCfg = RobotJointPosObsCfg()
    dof_vel: RobotJointVelObsCfg = RobotJointVelObsCfg()
    base_lin_vel: RobotBaseLinearVelocityObsCfg = RobotBaseLinearVelocityObsCfg()
    base_ang_vel: RobotBaseAngularVelocityObsCfg = RobotBaseAngularVelocityObsCfg()


@configclass
class _RobotObsCfg(ManagerObservationsCfg):
    policy: _RobotTermsGroup = _RobotTermsGroup()


def _cfg(observations: ManagerObservationsCfg, queries: SimQueriesCfg | None = None) -> ManagerBasedEnvCfg:
    scene = SceneCfg()
    scene.objs.robot = RobotCfg(model=ModelFileCfg(file="robot.xml"), base_link_name="torso", prefix="go2/")
    cfg = (
        ManagerBasedEnvCfg(observations=observations)
        if queries is None
        else (ManagerBasedEnvCfg(observations=observations, queries=queries))
    )
    cfg.scene = scene
    return cfg


def test_robot_terms_contribute_scene_robot_defaults() -> None:
    cfg = _cfg(_RobotObsCfg())
    base_link = "go2/torso"
    assert cfg.sim_query_cfgs() == {
        "obs.robot_joint_pos": BodyJointPositionQuery(body=base_link),
        "obs.robot_joint_vel": BodyJointVelocityQuery(body=base_link),
        "obs.robot_base_quat": LinkQuaternionQuery(link=base_link),
        "obs.robot_base_linear_velocity": LinkLinearVelocityQuery(link=base_link),
        "obs.robot_base_angular_velocity": LinkAngularVelocityQuery(link=base_link),
    }
    assert cfg.model_query_cfgs() == {}


def test_equal_task_and_term_query_declarations_merge() -> None:
    observations = _ObsCfg(policy=_TwoTerms(left=_DataRequiredCfg(key="robot_dof_pos", body="go2/torso")))
    query = BodyJointPositionQuery(body="go2/torso")
    cfg = _cfg(observations, SimQueriesCfg(data={"robot_dof_pos": query}))
    assert cfg.sim_query_cfgs() == {"robot_dof_pos": query}


def test_unequal_task_and_term_query_declarations_fail() -> None:
    observations = _ObsCfg(policy=_TwoTerms(left=_DataRequiredCfg(key="robot_dof_pos", body="go2/torso")))
    queries = SimQueriesCfg(data={"robot_dof_pos": JointPositionQuery(joints=("j1", "j2"))})
    cfg = _cfg(observations, queries)
    with pytest.raises(ValueError, match="unequal declarations"):
        cfg.sim_query_cfgs()


def test_unequal_contributions_fail_with_term_paths() -> None:
    observations = _ObsCfg(
        policy=_TwoTerms(
            left=RobotJointPosObsCfg(),
            right=_DataRequiredCfg(key="obs.robot_joint_pos", body="other"),
        )
    )
    cfg = _cfg(observations)
    with pytest.raises(ValueError, match=r"observations\.policy\.left.*observations\.policy\.right"):
        cfg.sim_query_cfgs()


def test_terms_without_required_sim_queries_declare_nothing() -> None:
    cfg = _cfg(_ObsCfg())
    assert cfg.sim_query_cfgs() == {}
    assert cfg.model_query_cfgs() == {}


def test_model_query_contribution_and_task_key_collision() -> None:
    observations = _ObsCfg(policy=_TwoTerms(left=_ModelRequiredCfg(), right=_NoopCfg()))
    assert _cfg(observations).model_query_cfgs() == {"obs.limits": DofPositionLimitsQuery()}

    declared = {"obs.limits": BodyJointPositionLimitsQuery(body="go2/torso")}
    with pytest.raises(ValueError, match="unequal declarations"):
        _cfg(observations, SimQueriesCfg(model=declared)).model_query_cfgs()


def test_default_queries_require_scene_robot() -> None:
    cfg = ManagerBasedEnvCfg(observations=_RobotObsCfg())
    with pytest.raises(ValueError, match=r"scene\.objs\.robot"):
        cfg.sim_query_cfgs()


def test_observation_required_sim_queries_reports_paths() -> None:
    cfg = _cfg(_ObsCfg(policy=_TwoTerms(left=_ModelRequiredCfg(), right=_NoopCfg())))
    contributions = observation_required_sim_queries(cfg)
    assert [contribution.source for contribution in contributions] == [
        "observations.policy.left",
        "observations.policy.right",
    ]
    assert contributions[0].model == {"obs.limits": DofPositionLimitsQuery()}
