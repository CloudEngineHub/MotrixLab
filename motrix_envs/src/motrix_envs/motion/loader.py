# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixMotion loader — reads and validates MotrixLab motion NPZ v1 files.

This loader is pure numpy and does not depend on any simulation model.
Task-specific views (e.g., ``WbtMotionClip``) wrap this loader, build name-based
indices over the schema fields, and provide tracked-subset / model-order views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motrix_envs.motion.schema import (
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
)


class MotionFormatError(ValueError):
    """Raised when a file does not conform to the MotrixLab motion NPZ schema."""


@dataclass(frozen=True)
class MotionSlice:
    """Per-frame view of a :class:`MotrixMotion` restricted to a body subset.

    Returned by :meth:`MotrixMotion.select_bodies`. Useful for diagnostic /
    visualization tools that only need a few tracked links.
    """

    body_names: tuple[str, ...]
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


class MotrixMotion:
    """Load and validate a MotrixLab motion NPZ v1 file.

    The schema is documented in
    ``wiki/design/motrixlab-motion-npz-schema.md``. All per-frame arrays are
    float32; quaternions follow xyzw convention; world frame is Z-up.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"Motion file does not exist: {self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            files = set(data.files)
            self._validate_required_fields(files)
            arrays: dict[str, np.ndarray] = {k: np.asarray(data[k]) for k in files}

        self.schema_version = self._read_scalar_int(arrays["schema_version"], "schema_version")
        if self.schema_version != SCHEMA_VERSION:
            raise MotionFormatError(
                f"Unsupported schema_version {self.schema_version}; this loader supports v{SCHEMA_VERSION}."
            )

        self.fps = self._read_scalar_int(arrays["fps"], "fps")
        if self.fps <= 0:
            raise MotionFormatError(f"fps must be positive, got {self.fps}")

        self.num_frames = self._read_scalar_int(arrays["num_frames"], "num_frames")
        if self.num_frames < 1:
            raise MotionFormatError(f"num_frames must be >= 1, got {self.num_frames}")

        self.joint_names = self._read_str_array(arrays["joint_names"], "joint_names")
        self.body_names = self._read_str_array(arrays["body_names"], "body_names")

        self.joint_pos = self._read_float_array(
            arrays["joint_pos"], "joint_pos", (self.num_frames, len(self.joint_names))
        )
        self.joint_vel = self._read_float_array(
            arrays["joint_vel"], "joint_vel", (self.num_frames, len(self.joint_names))
        )

        b = len(self.body_names)
        self.body_pos_w = self._read_float_array(arrays["body_pos_w"], "body_pos_w", (self.num_frames, b, 3))
        self.body_quat_w = self._read_float_array(arrays["body_quat_w"], "body_quat_w", (self.num_frames, b, 4))
        self.body_lin_vel_w = self._read_float_array(
            arrays["body_lin_vel_w"], "body_lin_vel_w", (self.num_frames, b, 3)
        )
        self.body_ang_vel_w = self._read_float_array(
            arrays["body_ang_vel_w"], "body_ang_vel_w", (self.num_frames, b, 3)
        )

        self._validate_quaternion_norm(self.body_quat_w)

        self.tracked_body_names = self._read_optional_str_array(arrays, "tracked_body_names")
        self.reference_body_name = self._read_optional_scalar_str(arrays, "reference_body_name")
        self.root_body_name = self._read_optional_scalar_str(arrays, "root_body_name")
        self.clip_name = self._read_optional_scalar_str(arrays, "clip_name")

        self.extensions = {k[len("ext_") :]: v for k, v in arrays.items() if k.startswith("ext_")}

    # ------------------------------------------------------------------
    # Name lookups
    # ------------------------------------------------------------------
    def body_index(self, name: str) -> int:
        try:
            return self.body_names.index(name)
        except ValueError:
            raise KeyError(f"Body name {name!r} not in motion body_names") from None

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError:
            raise KeyError(f"Joint name {name!r} not in motion joint_names") from None

    def body_indices(self, names: list[str]) -> np.ndarray:
        return np.asarray([self.body_index(n) for n in names], dtype=np.int64)

    def joint_indices(self, names: list[str]) -> np.ndarray:
        return np.asarray([self.joint_index(n) for n in names], dtype=np.int64)

    def select_bodies(self, names: list[str]) -> MotionSlice:
        idx = self.body_indices(names)
        return MotionSlice(
            body_names=tuple(names),
            body_pos_w=self.body_pos_w[:, idx],
            body_quat_w=self.body_quat_w[:, idx],
            body_lin_vel_w=self.body_lin_vel_w[:, idx],
            body_ang_vel_w=self.body_ang_vel_w[:, idx],
        )

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------
    def _validate_required_fields(self, files: set[str]) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in files]
        if missing:
            raise MotionFormatError(
                f"Motion file {self.path} missing required fields: {missing}. Present fields: {sorted(files)}"
            )

    @staticmethod
    def _read_scalar_int(arr: np.ndarray, name: str) -> int:
        # Accept both scalar () and 1-dim (1,) storage.
        arr = np.asarray(arr).reshape(-1)
        if arr.size != 1:
            raise MotionFormatError(f"Field {name!r} must be scalar, got shape {arr.shape}")
        return int(arr[0])

    @staticmethod
    def _read_str_array(arr: np.ndarray, name: str) -> list[str]:
        arr = np.asarray(arr).reshape(-1)
        return [str(x) for x in arr.tolist()]

    @staticmethod
    def _read_float_array(arr: np.ndarray, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != expected_shape:
            raise MotionFormatError(f"Field {name!r} shape {arr.shape}, expected {expected_shape}")
        return arr

    @staticmethod
    def _validate_quaternion_norm(quats: np.ndarray, atol: float = 1e-3) -> None:
        norms = np.linalg.norm(quats.reshape(-1, 4), axis=1)
        bad = np.abs(norms - 1.0) > atol
        if np.any(bad):
            bad_idx = np.where(bad)[0]
            raise MotionFormatError(
                f"body_quat_w contains {len(bad_idx)} non-unit quaternions "
                f"(first bad indices: {bad_idx[:5].tolist()}); max deviation="
                f"{float(np.max(np.abs(norms - 1.0))):.4f}."
            )

    @staticmethod
    def _read_optional_str_array(arrays: dict[str, np.ndarray], name: str) -> list[str] | None:
        if name not in arrays:
            return None
        return MotrixMotion._read_str_array(arrays[name], name)

    @staticmethod
    def _read_optional_scalar_str(arrays: dict[str, np.ndarray], name: str) -> str | None:
        if name not in arrays:
            return None
        arr = np.asarray(arrays[name]).reshape(-1)
        if arr.size != 1:
            raise MotionFormatError(f"Optional field {name!r} must be scalar, got shape {arr.shape}")
        return str(arr[0])


# Re-export for type-checkers / static analysis.
__all__ = ["MotrixMotion", "MotionSlice", "MotionFormatError", "OPTIONAL_FIELDS"]
