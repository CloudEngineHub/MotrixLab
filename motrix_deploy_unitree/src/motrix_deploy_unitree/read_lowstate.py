# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Read Unitree Go2 LowState without creating a LowCmd publisher."""

import sys
import time
from dataclasses import dataclass
from typing import Any

from motrix_deploy_unitree.config import GO2_JOINT_NAME_TO_MOTOR_INDEX
from motrix_deploy_unitree.remote import BUTTON_NAMES, decode_wireless_remote

JOINT_NAMES = tuple(GO2_JOINT_NAME_TO_MOTOR_INDEX)
MOTOR_INDICES = tuple(GO2_JOINT_NAME_TO_MOTOR_INDEX.values())


@dataclass
class ReadStatistics:
    """Counters updated by the DDS callback."""

    received: int = 0
    valid: int = 0
    crc_errors: int = 0
    decode_errors: int = 0


class LowStatePrinter:
    """Validate, throttle, and print canonical Go2 state samples."""

    def __init__(self, crc: Any, *, print_interval_s: float, validate_crc: bool) -> None:
        self.crc = crc
        self.print_interval_ns = round(print_interval_s * 1e9)
        self.validate_crc = validate_crc
        self.statistics = ReadStatistics()
        self._last_print_ns = 0

    def __call__(self, message: Any) -> None:
        self.statistics.received += 1
        try:
            if self.validate_crc and self.crc.Crc(message) != message.crc:
                self.statistics.crc_errors += 1
                return
            if len(message.motor_state) < 12:
                raise ValueError(f"expected at least 12 motor states, got {len(message.motor_state)}")
            position = [float(message.motor_state[index].q) for index in MOTOR_INDICES]
            velocity = [float(message.motor_state[index].dq) for index in MOTOR_INDICES]
            quaternion_wxyz = [float(value) for value in message.imu_state.quaternion]
            gyroscope = [float(value) for value in message.imu_state.gyroscope]
            accelerometer = [float(value) for value in message.imu_state.accelerometer]
            if len(quaternion_wxyz) != 4 or len(gyroscope) != 3 or len(accelerometer) != 3:
                raise ValueError("invalid IMU field shape")
        except Exception as error:
            self.statistics.decode_errors += 1
            if self.statistics.decode_errors <= 3:
                print(f"Invalid LowState: {error}", file=sys.stderr, flush=True)
            return

        self.statistics.valid += 1
        now_ns = time.monotonic_ns()
        if self._last_print_ns and now_ns - self._last_print_ns < self.print_interval_ns:
            return
        self._last_print_ns = now_ns

        try:
            remote = decode_wireless_remote(message.wireless_remote)
            pressed = [name for name in BUTTON_NAMES if remote.pressed(name)]
            remote_text = (
                f"buttons={pressed} "
                f"axes=(lx={remote.lx:.3f}, ly={remote.ly:.3f}, rx={remote.rx:.3f}, ry={remote.ry:.3f})"
            )
        except Exception as error:
            remote_text = f"unavailable ({error})"

        print("\n--- Unitree Go2 LowState ---")
        print(
            f"frames received={self.statistics.received} valid={self.statistics.valid} "
            f"crc_errors={self.statistics.crc_errors} decode_errors={self.statistics.decode_errors}"
        )
        for name, joint_position, joint_velocity in zip(JOINT_NAMES, position, velocity, strict=True):
            print(f"{name:16s} q={joint_position: 9.5f} rad  dq={joint_velocity: 9.5f} rad/s")
        print("imu quaternion wxyz:", quaternion_wxyz)
        print("imu quaternion xyzw:", quaternion_wxyz[1:] + quaternion_wxyz[:1])
        print("imu gyroscope rad/s:", gyroscope)
        print("imu accelerometer m/s^2:", accelerometer)
        print("remote:", remote_text, flush=True)


def _load_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as error:
        raise RuntimeError(
            "LowState reading requires the official unitree_sdk2py package; install "
            "https://github.com/motphys-developers/unitree_sdk2_python"
        ) from error
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, CRC


def run_read_lowstate(
    *,
    network_interface: str,
    domain_id: int,
    topic: str,
    queue_depth: int,
    duration_s: float,
    print_interval_s: float,
    validate_crc: bool,
) -> int:
    """Run a bounded read-only LowState DDS diagnostic."""
    try:
        channel_factory_initialize, channel_subscriber, low_state_type, crc_type = _load_sdk()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    channel_factory_initialize(domain_id, network_interface)
    printer = LowStatePrinter(crc_type(), print_interval_s=print_interval_s, validate_crc=validate_crc)
    subscriber = channel_subscriber(topic, low_state_type)
    subscriber.Init(printer, queue_depth)
    duration = "unlimited time" if duration_s == 0 else f"{duration_s:g} seconds"
    print(f"Listening to {topic!r} on {network_interface!r} for {duration}")
    print("Read-only diagnostic: no LowCmd publisher is created. Press Ctrl+C to stop.", flush=True)

    deadline_ns = None if duration_s == 0 else time.monotonic_ns() + round(duration_s * 1e9)
    try:
        while deadline_ns is None or time.monotonic_ns() < deadline_ns:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        close = getattr(subscriber, "Close", None) or getattr(subscriber, "close", None)
        if close is not None:
            close()

    statistics = printer.statistics
    print(
        f"Summary: received={statistics.received}, valid={statistics.valid}, "
        f"crc_errors={statistics.crc_errors}, decode_errors={statistics.decode_errors}"
    )
    if statistics.valid == 0:
        print("No valid LowState frame was received.", file=sys.stderr)
        return 1
    return 0
