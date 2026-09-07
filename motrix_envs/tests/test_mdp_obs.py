# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core.config.scene import KeyPoseCfg, ModelFileCfg, RobotCfg
from motrix_env_core.mdp.observations import (
    RobotBaseAngularVelocityObsCfg,
    RobotBaseLinearVelocityObsCfg,
    RobotJointPosObsCfg,
    RobotJointVelObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.state import RandValue
from motrix_env_core.numba.manager.rand import initialize_rand_states
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    JointPositionQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
)
from motrix_envs.locomotion.wbt.mdp.action import WbtJointPositionAction
from motrix_envs.locomotion.wbt.mdp.observations import (
    ActionsObsCfg,
    DofPosRelObsCfg,
)


class _FakeSimData:
    """Minimal read-program stand-in: query/view plus key-based reads."""

    def __init__(self, queries: dict, views: dict) -> None:
        self._queries = queries
        self._views = views

    def query(self, key: str):
        return self._queries[key]

    def view(self, key: str):
        return self._views[key]

    def __getitem__(self, key: str):
        return self.view(key)


def _env() -> SimpleNamespace:
    robot = RobotCfg(
        model=ModelFileCfg(file="unused.xml"),
        base_link_name="resolved_robot",
        key_pose=KeyPoseCfg(joint_names=["a", "b", "c"], poses={"default": [0.5, 1.0, 1.5]}),
    )
    layouts = {
        # Task-owned plain key: read by the WBT relative-dof-position term.
        "robot_dof_pos": SimpleNamespace(query=JointPositionQuery(joints=("a", "b", "c")), trailing_shape=(3,)),
        # Term-owned reserved-namespace keys: read by the framework terms.
        "obs.robot_joint_pos": SimpleNamespace(
            query=BodyJointPositionQuery(body="resolved_robot"), trailing_shape=(3,)
        ),
        "obs.robot_joint_vel": SimpleNamespace(
            query=BodyJointVelocityQuery(body="resolved_robot"), trailing_shape=(2,)
        ),
        "obs.robot_base_quat": SimpleNamespace(query=LinkQuaternionQuery(link="resolved_robot"), trailing_shape=(4,)),
        "obs.robot_base_linear_velocity": SimpleNamespace(
            query=LinkLinearVelocityQuery(link="resolved_robot"), trailing_shape=(3,)
        ),
        "obs.robot_base_angular_velocity": SimpleNamespace(
            query=LinkAngularVelocityQuery(link="resolved_robot"), trailing_shape=(3,)
        ),
    }
    queries = {key: value.query for key, value in layouts.items()}
    views = {key: np.zeros((1, *value.trailing_shape), dtype=np.float32) for key, value in layouts.items()}
    sim_data = _FakeSimData(queries, views)
    return SimpleNamespace(
        action_space=SimpleNamespace(shape=(3,)),
        cfg=SimpleNamespace(scene=SimpleNamespace(objs=SimpleNamespace(robot=robot))),
        sim_data=sim_data,
    )


def test_actions_observation_reads_current_actions() -> None:
    action = WbtJointPositionAction(
        current=np.zeros((1, 3), dtype=np.float32),
        previous=np.zeros((1, 3), dtype=np.float32),
        default_angles=np.zeros(3, dtype=np.float32),
        joint_lower=np.zeros(3, dtype=np.float32),
        joint_upper=np.zeros(3, dtype=np.float32),
        action_scales=np.ones(3, dtype=np.float32),
    )
    env = SimpleNamespace(action_terms={"joint_position": action})
    cfg = ActionsObsCfg()
    term = cfg.__call__(env)
    current = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.empty(term.size, dtype=np.float32)

    assert term.size == 3
    ctx = SimpleNamespace(actions={"joint_position": SimpleNamespace(current=current)})
    term.dispatch(ctx, out, *term.args)
    np.testing.assert_array_equal(out, current)


def test_robot_observation_cfg_uses_sim_query_layout_and_robot_key_pose() -> None:
    env = _env()
    position_cfg = RobotJointPosObsCfg()
    relative_position_cfg = DofPosRelObsCfg()
    velocity_cfg = RobotJointVelObsCfg()
    linear_cfg = RobotBaseLinearVelocityObsCfg()
    angular_cfg = RobotBaseAngularVelocityObsCfg()

    position = position_cfg.__call__(env)
    relative_position = relative_position_cfg.__call__(env)
    velocity = velocity_cfg.__call__(env)
    linear = linear_cfg.__call__(env)
    angular = angular_cfg.__call__(env)

    assert position.size == relative_position.size == 3
    assert velocity.size == 2
    assert linear.size == angular.size == 3
    np.testing.assert_array_equal(relative_position.args[0].reference, np.asarray([0.5, 1.0, 1.5], dtype=np.float32))


def test_robot_dof_pos_rel_obs_cfg_requires_key_pose() -> None:
    with pytest.raises(ValueError, match="key pose 'unknown_pose'"):
        DofPosRelObsCfg(reference_key_pose="unknown_pose").__call__(_env())


def test_robot_dof_pos_rel_obs_cfg_requires_scene_robot() -> None:
    env = _env()
    env.cfg = SimpleNamespace(scene=SimpleNamespace(objs=SimpleNamespace(robot=None)))

    with pytest.raises(TypeError, match="RobotCfg"):
        DofPosRelObsCfg().__call__(env)


def test_robot_base_velocities_are_expressed_in_the_base_local_frame() -> None:
    half_sqrt_two = np.float32(np.sqrt(0.5))
    base_quat = np.asarray([0.0, 0.0, half_sqrt_two, half_sqrt_two], dtype=np.float32)
    linear_velocity = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    angular_velocity = np.asarray([0.0, 2.0, 0.0], dtype=np.float32)
    noise = RandValue(np.asarray([1], dtype=np.uint64))
    linear = RobotBaseLinearVelocityObsCfg().__call__(_env())
    angular = RobotBaseAngularVelocityObsCfg().__call__(_env())
    linear_out = np.empty(3, dtype=np.float32)
    angular_out = np.empty(3, dtype=np.float32)

    ctx = SimpleNamespace(
        sim={
            "obs.robot_base_quat": base_quat,
            "obs.robot_base_linear_velocity": linear_velocity,
            "obs.robot_base_angular_velocity": angular_velocity,
        },
        rand=noise,
    )
    linear.dispatch(ctx, linear_out, *linear.args)
    angular.dispatch(ctx, angular_out, *angular.args)

    np.testing.assert_allclose(linear_out, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(angular_out, [2.0, 0.0, 0.0], atol=1e-6)


def test_robot_observation_noise_uses_one_stateful_sequence_per_environment() -> None:
    dof_pos = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    term = RobotJointPosObsCfg(noise=UniformNoiseCfg(amplitude=0.25)).__call__(_env())

    def evaluate(rng_states: np.ndarray) -> np.ndarray:
        out = np.empty((2, dof_pos.shape[0]), dtype=np.float32)
        for env_id in range(2):
            ctx = SimpleNamespace(sim={"obs.robot_joint_pos": dof_pos}, rand=RandValue(rng_states[env_id]))
            term.dispatch(ctx, out[env_id], *term.args)
        return out

    rng_states = initialize_rand_states(2, rand_seed=7)
    first = evaluate(rng_states)
    continued = evaluate(rng_states)
    reproduced = evaluate(initialize_rand_states(2, rand_seed=7))
    different_seed = evaluate(initialize_rand_states(2, rand_seed=8))

    np.testing.assert_array_equal(reproduced, first)
    assert np.any(continued != first)
    assert np.any(different_seed != first)
    assert np.all(first >= dof_pos - 0.25)
    assert np.all(first < dof_pos + 0.25)
    assert not np.array_equal(first[0], first[1])


def test_robot_dof_position_and_relative_position_have_distinct_semantics() -> None:
    dof_pos = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    noise = RandValue(np.asarray([1], dtype=np.uint64))
    raw = RobotJointPosObsCfg().__call__(_env())
    relative = DofPosRelObsCfg().__call__(_env())
    raw_out = np.empty(dof_pos.shape[0], dtype=np.float32)
    relative_out = np.empty(dof_pos.shape[0], dtype=np.float32)

    ctx = SimpleNamespace(sim={"robot_dof_pos": dof_pos, "obs.robot_joint_pos": dof_pos}, rand=noise)
    raw.dispatch(ctx, raw_out, *raw.args)
    relative.dispatch(ctx, relative_out, *relative.args)

    np.testing.assert_array_equal(raw_out, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(relative_out, [0.5, 1.0, 1.5])
