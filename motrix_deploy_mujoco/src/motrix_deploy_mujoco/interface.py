# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo state/command adapter with no policy tensor logic."""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environment configurations
from motrix_deploy.backend import RobotInterface
from motrix_deploy.contracts import HealthStatus, RobotCapabilities, RobotCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.errors import ValidationError
from motrix_deploy_mujoco.config import MujocoBackendConfig
from motrix_deploy_mujoco.transform import convert_position_actuators_to_motors
from motrix_deploy_mujoco.viewer import MujocoGlfwViewer
from motrix_env_core import SimCfg, registry
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.input import KeyboardDevice
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

logger = logging.getLogger(__name__)


def wxyz_to_xyzw(quaternion: object) -> np.ndarray:
    """Convert MuJoCo's explicit ``wxyz`` order to the public ``xyzw`` order."""
    return np.asarray(quaternion, dtype=np.float32)[[1, 2, 3, 0]]


def xyzw_to_wxyz(quaternion: object) -> np.ndarray:
    """Convert the public ``xyzw`` order to MuJoCo's explicit ``wxyz`` order."""
    return np.asarray(quaternion)[[3, 0, 1, 2]]


class MujocoRobotInterface(RobotInterface):
    """Build a MuJoCo scene and apply canonical hybrid-PD commands as torque."""

    def __init__(
        self,
        config: MujocoBackendConfig,
        *,
        control_period_s: float,
        render: bool = False,
        viewer_factory: Callable[[Any, SystemCameraCfg], Any] = MujocoGlfwViewer,
    ) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo backend support requires the 'motrix-deploy-mujoco' package") from error
        env_cfg = registry.make_env_config(config.scene, mode=config.scene_mode)
        if not np.isclose(control_period_s, env_cfg.ctrl_dt, atol=1e-12, rtol=0.0):
            raise ValidationError("backend.scene.ctrl_dt", str(control_period_s), env_cfg.ctrl_dt)
        ratio = control_period_s / config.sim_dt
        if not np.isclose(ratio, round(ratio), atol=1e-9) or round(ratio) < 1:
            raise ValidationError(
                "backend.control_period_s",
                "an integer multiple of scene SimCfg.dt",
                control_period_s,
            )
        self.config = config
        self._mj = mujoco
        self._scene = env_cfg.scene
        self._sim = SimCfg(dt=config.sim_dt, solver_iterations=config.solver_iterations)
        self._control_period_s = control_period_s
        self._physics_substeps = round(ratio)
        self._render = render
        self._model: Any = None
        self._data: Any = None
        self._viewer = viewer_factory(mujoco, self._scene.system_camera) if render else None
        self._spec: RobotSpec | None = None
        self._capabilities: RobotCapabilities | None = None
        self._joint_qpos_indices = np.empty(0, dtype=np.int64)
        self._joint_qvel_indices = np.empty(0, dtype=np.int64)
        self._actuator_indices = np.empty(0, dtype=np.int64)
        self._base_body_id = -1
        self._base_qpos_address = -1
        self._last_communication_ns = 0
        self._health_reason = "backend is not open"
        self._opened = False
        self._closed = False

    @property
    def capabilities(self) -> RobotCapabilities:
        if self._capabilities is None:
            raise RuntimeError("MuJoCo capabilities are available after open()")
        return self._capabilities

    def open(self, spec: RobotSpec) -> None:
        if self._opened and not self._closed:
            raise RuntimeError("MuJoCo backend is already open")
        model = self._build_model(spec)
        data = self._mj.MjData(model)
        self._model = model
        self._data = data
        self._spec = spec
        self._bind_base()
        self._bind_joints_and_actuators(spec)
        self._validate_torque_actuator_contract(spec)
        self._validate_sensors()
        self._mj.mj_resetData(model, data)
        qpos = data.qpos
        qpos[self._base_qpos_address : self._base_qpos_address + 3] = self.config.base_position
        qpos[self._base_qpos_address + 3 : self._base_qpos_address + 7] = xyzw_to_wxyz(
            self.config.base_orientation_xyzw
        )
        qpos[self._joint_qpos_indices] = spec.default_joint_position
        data.qvel.fill(0.0)
        data.ctrl[self._actuator_indices] = 0.0
        self._mj.mj_forward(model, data)
        self._capabilities = RobotCapabilities(
            control_modes=("joint_pd",),
            state_fields=frozenset(
                {
                    "joint_position",
                    "joint_velocity",
                    "base_orientation_xyzw",
                    "base_angular_velocity",
                    "base_linear_acceleration",
                    "base_position",
                    "base_linear_velocity",
                }
            ),
            supports_rendering=True,
            max_command_rate_hz=1.0 / self._control_period_s,
            stop_semantics="zero_torque",
        )
        self._opened = True
        self._closed = False
        self._last_communication_ns = time.monotonic_ns()
        self._health_reason = ""
        if self._render:
            assert self._viewer is not None
            self._viewer.open(model, data)
        self._sync_viewer()

    def get_keyboard_device(self) -> KeyboardDevice:
        """Return keyboard events from the deployment viewer window."""
        if self._viewer is None:
            raise RuntimeError("MuJoCo keyboard input requires viewer=true")
        return self._viewer.keyboard_device

    def read_state(self, timeout_s: float) -> RobotState:
        self._require_open()
        if timeout_s <= 0:
            raise TimeoutError(f"MuJoCo state timeout must be positive, got {timeout_s}")
        self._check_viewer_running()
        self._last_communication_ns = time.monotonic_ns()
        return self._state()

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        assert self._model is not None and self._data is not None and self._spec is not None
        joint_count = self._spec.joint_count
        for field_name in ("joint_position", "joint_velocity", "feedforward_torque", "kp", "kd"):
            value = getattr(command, field_name)
            if value.shape != (joint_count,) or not np.isfinite(value).all():
                raise ValidationError(f"command.{field_name}", f"finite shape ({joint_count},)", value)
        lower = self._spec.position_lower
        upper = self._spec.position_upper
        if np.any(command.joint_position < lower) or np.any(command.joint_position > upper):
            raise ValidationError("command.joint_position", "inside RobotSpec position range", command.joint_position)
        for _ in range(self._physics_substeps):
            position = self._data.qpos[self._joint_qpos_indices]
            velocity = self._data.qvel[self._joint_qvel_indices]
            torque = (
                command.kp * (command.joint_position - position)
                + command.kd * (command.joint_velocity - velocity)
                + command.feedforward_torque
            )
            self._data.ctrl[self._actuator_indices] = np.clip(
                torque,
                -self._spec.torque_limit,
                self._spec.torque_limit,
            )
            self._mj.mj_step(self._model, self._data)
        self._last_communication_ns = time.monotonic_ns()
        self._update_fall_health()
        self._sync_viewer()

    def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._opened and not self._closed and not self._health_reason,
            reason=self._health_reason,
            last_successful_communication_ns=self._last_communication_ns,
        )

    def stop(self) -> None:
        if not self._opened or self._closed or self._data is None:
            return
        if self._joint_qpos_indices.size:
            self._data.ctrl[self._actuator_indices] = 0.0

    def close(self) -> None:
        if self._closed:
            return
        viewer = self._viewer
        try:
            if viewer is not None:
                viewer.close()
                deadline = time.monotonic() + 2.0
                while viewer.is_running() and time.monotonic() < deadline:
                    time.sleep(0.001)
                if viewer.is_running():
                    raise RuntimeError("MuJoCo viewer did not stop within 2 seconds")
        finally:
            self._data = None
            self._model = None
            self._closed = True

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        self._check_viewer_running()
        self._viewer.sync()

    def _check_viewer_running(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            raise KeyboardInterrupt("MuJoCo viewer was closed")

    def _bind_base(self) -> None:
        assert self._model is not None
        body_id = self._name_id(self._mj.mjtObj.mjOBJ_BODY, self.config.base_body_name, "base body")
        joint_address = int(self._model.body_jntadr[body_id])
        if joint_address < 0 or self._model.jnt_type[joint_address] != self._mj.mjtJoint.mjJNT_FREE:
            raise ValidationError("backend.base_body_name", "a body with a free joint", self.config.base_body_name)
        self._base_body_id = body_id
        self._base_qpos_address = int(self._model.jnt_qposadr[joint_address])

    def _build_model(self, robot: RobotSpec) -> Any:
        model_spec = MuJoCoSceneCompiler().create_spec(self._scene, self._sim)
        source_model = model_spec.compile()
        self._model = source_model
        self._bind_joints_and_actuators(robot)
        self._validate_position_actuator_contract(robot)

        actuator_names = convert_position_actuators_to_motors(
            self._mj,
            model_spec,
            source_model,
            self._actuator_indices,
            robot,
        )
        logger.info(
            "Converting %d actuators from position servos to torque motors with MjSpec: %s",
            len(actuator_names),
            ", ".join(actuator_names),
        )
        model = model_spec.compile()
        logger.info("Built MuJoCo deployment model from SceneCfg %s", self.config.scene)
        return model

    def _bind_joints_and_actuators(self, spec: RobotSpec) -> None:
        assert self._model is not None
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        actuator_indices: list[int] = []
        for name in spec.joint_names:
            joint_id = self._name_id(self._mj.mjtObj.mjOBJ_JOINT, name, "joint")
            if self._model.jnt_type[joint_id] != self._mj.mjtJoint.mjJNT_HINGE:
                raise ValidationError(f"backend.joints.{name}", "a one-DoF hinge joint", self._model.jnt_type[joint_id])
            matches = np.flatnonzero(self._model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise ValidationError(f"backend.actuators.{name}", "exactly one actuator", matches.tolist())
            qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self._model.jnt_dofadr[joint_id]))
            actuator_indices.append(int(matches[0]))
        self._joint_qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self._joint_qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        self._actuator_indices = np.asarray(actuator_indices, dtype=np.int64)

    def _validate_position_actuator_contract(self, spec: RobotSpec) -> None:
        assert self._model is not None
        indices = self._actuator_indices
        ctrl_range = self._model.actuator_ctrlrange[indices]
        force_range = self._model.actuator_forcerange[indices]
        kp = self._model.actuator_gainprm[indices, 0]
        stiffness = -self._model.actuator_biasprm[indices, 1]
        kd = -self._model.actuator_biasprm[indices, 2]
        checks = (
            ("ctrl_lower", ctrl_range[:, 0], spec.position_lower),
            ("ctrl_upper", ctrl_range[:, 1], spec.position_upper),
            ("force_lower", force_range[:, 0], -spec.torque_limit),
            ("force_upper", force_range[:, 1], spec.torque_limit),
        )
        for name, actual, expected in checks:
            if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise ValidationError(f"backend.actuators.{name}", expected.tolist(), actual.tolist())
        if not np.allclose(kp, stiffness, atol=1e-7, rtol=0.0):
            raise ValidationError("backend.actuators.kp", "gain equals stiffness", (kp.tolist(), stiffness.tolist()))
        if np.any(kp < 0) or np.any(kd < 0):
            raise ValidationError("backend.actuators.gains", "non-negative kp/kd", (kp.tolist(), kd.tolist()))

    def _validate_torque_actuator_contract(self, spec: RobotSpec) -> None:
        assert self._model is not None
        indices = self._actuator_indices
        checks = (
            ("ctrl_lower", self._model.actuator_ctrlrange[indices, 0], -spec.torque_limit),
            ("ctrl_upper", self._model.actuator_ctrlrange[indices, 1], spec.torque_limit),
            ("force_lower", self._model.actuator_forcerange[indices, 0], -spec.torque_limit),
            ("force_upper", self._model.actuator_forcerange[indices, 1], spec.torque_limit),
            ("gain", self._model.actuator_gainprm[indices, 0], np.ones(spec.joint_count)),
            ("bias", self._model.actuator_biastype[indices], np.zeros(spec.joint_count)),
        )
        for name, actual, expected in checks:
            if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise ValidationError(f"backend.torque_actuators.{name}", expected.tolist(), actual.tolist())

    def _validate_sensors(self) -> None:
        assert self._model is not None
        self._name_id(self._mj.mjtObj.mjOBJ_SITE, self.config.imu_site_name, "IMU site")
        for field_name, expected_dimension in (
            ("gyro_sensor_name", 3),
            ("accelerometer_sensor_name", 3),
            ("global_linear_velocity_sensor_name", 3),
        ):
            sensor_name = getattr(self.config, field_name)
            sensor_id = self._name_id(self._mj.mjtObj.mjOBJ_SENSOR, sensor_name, "sensor")
            actual_dimension = int(self._model.sensor_dim[sensor_id])
            if actual_dimension != expected_dimension:
                raise ValidationError(f"backend.{field_name}", f"a {expected_dimension}D sensor", actual_dimension)

    def _state(self) -> RobotState:
        assert self._model is not None and self._data is not None and self._spec is not None
        orientation_xyzw = wxyz_to_xyzw(self._data.xquat[self._base_body_id])
        return RobotState(
            sample_time_ns=round(self._data.time * 1e9),
            receive_time_ns=self._last_communication_ns,
            joint_position=np.asarray(self._data.qpos[self._joint_qpos_indices], dtype=np.float32),
            joint_velocity=np.asarray(self._data.qvel[self._joint_qvel_indices], dtype=np.float32),
            base_orientation_xyzw=orientation_xyzw,
            base_angular_velocity=self._sensor(self.config.gyro_sensor_name),
            base_linear_acceleration=self._sensor(self.config.accelerometer_sensor_name),
            base_position=np.asarray(self._data.xpos[self._base_body_id], dtype=np.float32),
            base_linear_velocity=self._sensor(self.config.global_linear_velocity_sensor_name),
        )

    def _sensor(self, name: str) -> np.ndarray:
        assert self._model is not None and self._data is not None
        sensor_id = self._name_id(self._mj.mjtObj.mjOBJ_SENSOR, name, "sensor")
        address = int(self._model.sensor_adr[sensor_id])
        dimension = int(self._model.sensor_dim[sensor_id])
        return float32_array(
            np.asarray(self._data.sensordata[address : address + dimension], dtype=np.float32),
            path=f"backend.sensors.{name}",
            shape=(dimension,),
        )

    def _update_fall_health(self) -> None:
        assert self._data is not None
        height = self._data.xpos[self._base_body_id, 2]
        rotation = self._data.xmat[self._base_body_id].reshape(3, 3)
        up_z = rotation[2, 2]
        if not np.isfinite(height) or not np.isfinite(up_z):
            self._health_reason = f"non-finite base pose: height={height}, up_z={up_z}"
        elif height < self.config.fall_height_m:
            self._health_reason = f"base height {height:.6f}m is below {self.config.fall_height_m:.6f}m"
        elif up_z < self.config.fall_up_z:
            self._health_reason = f"base up_z {up_z:.6f} is below {self.config.fall_up_z:.6f}"

    def _name_id(self, object_type: Any, name: str, description: str) -> int:
        assert self._model is not None
        object_id = int(self._mj.mj_name2id(self._model, object_type, name))
        if object_id < 0:
            raise ValidationError(f"backend.{description}", f"existing name {name!r}", "missing")
        return object_id

    def _require_open(self) -> None:
        if not self._opened or self._closed or self._model is None or self._data is None:
            raise RuntimeError("MuJoCo backend is not open")
