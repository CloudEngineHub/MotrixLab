# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Go2 adapter contract tests."""

import logging
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from motrix_deploy.backend import KeyboardDeviceProvider
from motrix_deploy.contracts import RobotCommand, RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy_mujoco import (
    MujocoBackendConfig,
    MujocoRobotInterface,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from motrix_env_core.config.scene import SystemCameraCfg

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs/deploy/sim2sim/go2_walk_sim2sim.yaml"
FLAT_CONFIG_PATH = ROOT / "configs/deploy/sim2sim/go2_walk_flat_sim2sim.yaml"
JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
CONTROL_PERIOD_S = 0.02


def _config(config_path: Path = CONFIG_PATH) -> MujocoBackendConfig:
    return MujocoBackendConfig.from_mapping(OmegaConf.load(config_path).backend)


def _backend(
    *,
    render: bool = False,
    control_period_s: float = CONTROL_PERIOD_S,
    viewer_factory: Callable[[Any, SystemCameraCfg], Any] | None = None,
) -> MujocoRobotInterface:
    kwargs = {} if viewer_factory is None else {"viewer_factory": viewer_factory}
    return MujocoRobotInterface(
        _config(),
        control_period_s=control_period_s,
        render=render,
        **kwargs,
    )


def _spec(joint_names: tuple[str, ...] = JOINT_NAMES) -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=joint_names,
        default_joint_position=np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=np.float32,
        ),
        position_lower=np.tile(np.array([-0.9472, -1.4, -2.6227], dtype=np.float32), 4),
        position_upper=np.tile(np.array([0.9472, 2.5, -0.84776], dtype=np.float32), 4),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _command(position: np.ndarray, *, kp: float = 35.0, kd: float = 0.5) -> RobotCommand:
    zeros = np.zeros(12, dtype=np.float32)
    return RobotCommand(
        joint_position=position,
        joint_velocity=zeros,
        feedforward_torque=zeros,
        kp=np.full(12, kp, dtype=np.float32),
        kd=np.full(12, kd, dtype=np.float32),
    )


def test_backend_control_period_must_match_scene_config() -> None:
    with pytest.raises(ValidationError, match="backend.scene.ctrl_dt"):
        _backend(control_period_s=0.015)


def test_quaternion_order_conversion_is_explicit_and_round_trips() -> None:
    wxyz = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)

    xyzw = wxyz_to_xyzw(wxyz)

    np.testing.assert_array_equal(xyzw, [0.5, -0.5, 0.5, 0.5])
    np.testing.assert_array_equal(xyzw_to_wxyz(xyzw), wxyz)


def test_config_rejects_ambiguous_wxyz_orientation_key() -> None:
    values = dict(OmegaConf.load(CONFIG_PATH).backend)
    values["base_orientation_wxyz"] = values.pop("base_orientation_xyzw")

    with pytest.raises(ValidationError, match="base_orientation_xyzw"):
        MujocoBackendConfig.from_mapping(values)


def test_initial_state_command_step_contract_is_headless() -> None:
    spec = _spec()
    backend = _backend()
    viewer_modules_before = {name for name in sys.modules if name.startswith("mujoco.viewer")}
    backend.open(spec)

    initial = backend.read_state(timeout_s=0.1)
    assert backend._data is not None
    assert backend._data.ncon == 0
    target = spec.default_joint_position + np.float32(0.01)
    backend.write_command(_command(target))
    stepped = backend.read_state(timeout_s=0.1)
    backend.stop()
    backend.close()
    backend.stop()
    backend.close()

    np.testing.assert_allclose(initial.joint_position, spec.default_joint_position, atol=1e-6)
    np.testing.assert_array_equal(initial.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])
    assert stepped.sample_time_ns == 20_000_000
    assert np.isfinite(stepped.joint_position).all()
    assert backend.capabilities.max_command_rate_hz == 50.0
    assert {name for name in sys.modules if name.startswith("mujoco.viewer")} == viewer_modules_before


def test_glfw_viewer_syncs_without_owning_simulation_step() -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.running = False
            self.open_count = 0
            self.sync_count = 0
            self.close_count = 0
            self.keyboard_device = object()

        def open(self, model: Any, data: Any) -> None:
            del model, data
            self.open_count += 1
            self.running = True

        def is_running(self) -> bool:
            return self.running

        def sync(self) -> None:
            self.sync_count += 1

        def close(self) -> None:
            self.close_count += 1
            self.running = False

    viewer = FakeViewer()
    camera_configs: list[SystemCameraCfg] = []

    def viewer_factory(mujoco: Any, camera_config: SystemCameraCfg) -> FakeViewer:
        del mujoco
        camera_configs.append(camera_config)
        return viewer

    backend = _backend(render=True, viewer_factory=viewer_factory)
    spec = _spec()

    assert backend.get_keyboard_device() is viewer.keyboard_device
    assert camera_configs == [backend._scene.system_camera]
    backend.open(spec)
    initial = backend.read_state(timeout_s=0.1)
    backend.write_command(_command(spec.default_joint_position))
    stepped = backend.read_state(timeout_s=0.1)

    assert backend.capabilities.supports_rendering is True
    assert viewer.open_count == 1
    assert initial.sample_time_ns == 0
    assert stepped.sample_time_ns == 20_000_000
    assert viewer.sync_count == 2

    viewer.running = False
    with pytest.raises(KeyboardInterrupt, match="viewer was closed"):
        backend.read_state(timeout_s=0.1)
    backend.close()
    assert viewer.close_count == 1

    backend.open(spec)
    assert backend.get_keyboard_device() is viewer.keyboard_device
    assert viewer.open_count == 2
    backend.close()
    assert viewer.close_count == 2


def test_headless_backend_does_not_provide_keyboard_input() -> None:
    backend = _backend(render=False)

    assert isinstance(backend, KeyboardDeviceProvider)
    with pytest.raises(RuntimeError, match="viewer=true"):
        backend.get_keyboard_device()


def test_swapped_joint_contract_fails_during_open() -> None:
    swapped = list(JOINT_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    backend = _backend()

    with pytest.raises(ValidationError, match="ctrl_lower"):
        backend.open(_spec(tuple(swapped)))


def test_missing_joint_contract_fails_during_open() -> None:
    joint_names = list(JOINT_NAMES)
    joint_names[0] = "missing_joint"
    backend = _backend()

    with pytest.raises(ValidationError, match="missing_joint"):
        backend.open(_spec(tuple(joint_names)))


def test_actuator_force_range_mismatch_fails_during_open() -> None:
    spec = replace(_spec(), torque_limit=np.full(12, 23.0, dtype=np.float32))
    backend = _backend()

    with pytest.raises(ValidationError, match="force_lower"):
        backend.open(spec)


def test_open_converts_actuators_and_uses_scene_cfg_ground(caplog: pytest.LogCaptureFixture) -> None:
    spec = _spec()
    backend = _backend()
    with caplog.at_level(logging.INFO, logger="motrix_deploy_mujoco.interface"):
        backend.open(spec)

    model = backend._model
    assert model is not None
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.opt.iterations == 100
    floor_id = backend._mj.mj_name2id(model, backend._mj.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    np.testing.assert_allclose(model.geom_friction[floor_id], [0.6, 0.005, 0.0001])
    np.testing.assert_array_equal(
        model.actuator_biastype[backend._actuator_indices],
        backend._mj.mjtBias.mjBIAS_NONE,
    )
    np.testing.assert_allclose(model.actuator_gainprm[backend._actuator_indices, 0], 1.0)
    np.testing.assert_allclose(
        model.actuator_ctrlrange[backend._actuator_indices],
        np.column_stack((-spec.torque_limit, spec.torque_limit)),
    )
    assert "Converting 12 actuators from position servos to torque motors" in caplog.text
    assert "Built MuJoCo deployment model from SceneCfg go2-walk-rough" in caplog.text
    backend.close()


def test_rough_scene_uses_configured_initial_base_position() -> None:
    config = _config()
    backend = MujocoRobotInterface(config, control_period_s=CONTROL_PERIOD_S)
    backend.open(_spec())

    initial = backend.read_state(timeout_s=0.1)

    assert initial.base_position == pytest.approx(config.base_position)
    backend.close()


def test_flat_scene_uses_nonpenetrating_initial_base_position() -> None:
    config = _config(FLAT_CONFIG_PATH)
    backend = MujocoRobotInterface(config, control_period_s=CONTROL_PERIOD_S)
    backend.open(_spec())

    initial = backend.read_state(timeout_s=0.1)

    assert config.scene == "go2-walk-flat"
    assert initial.base_position == pytest.approx(config.base_position)
    assert backend._data is not None
    assert backend._data.ncon == 0
    backend.close()


def test_hybrid_pd_command_is_recomputed_as_torque_each_physics_substep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    backend = _backend()
    backend.open(spec)
    initial = backend.read_state(0.1)
    desired_position = initial.joint_position + np.linspace(0.01, 0.02, 12, dtype=np.float32)
    desired_velocity = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
    feedforward_torque = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    kp = np.linspace(10.0, 21.0, 12, dtype=np.float32)
    kd = np.linspace(0.1, 1.2, 12, dtype=np.float32)
    command = RobotCommand(
        joint_position=desired_position,
        joint_velocity=desired_velocity,
        feedforward_torque=feedforward_torque,
        kp=kp,
        kd=kd,
    )
    snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    mj_step = backend._mj.mj_step

    def record_step(model: object, data: object) -> None:
        snapshots.append(
            (
                data.qpos[backend._joint_qpos_indices].copy(),
                data.qvel[backend._joint_qvel_indices].copy(),
                data.ctrl[backend._actuator_indices].copy(),
            )
        )
        mj_step(model, data)

    monkeypatch.setattr(backend._mj, "mj_step", record_step)

    backend.write_command(command)

    assert backend._data is not None
    assert len(snapshots) == 10
    for position, velocity, applied_torque in snapshots:
        expected_torque = np.clip(
            kp * (desired_position - position) + kd * (desired_velocity - velocity) + feedforward_torque,
            -spec.torque_limit,
            spec.torque_limit,
        )
        np.testing.assert_allclose(applied_torque, expected_torque, atol=1e-6)
    assert not np.array_equal(snapshots[0][2], snapshots[1][2])
    assert backend.read_state(0.1).sample_time_ns == 20_000_000
    backend.stop()
    np.testing.assert_array_equal(backend._data.ctrl[backend._actuator_indices], 0.0)
    backend.close()
