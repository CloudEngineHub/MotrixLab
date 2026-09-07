# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert Holosoma WBT npz files to MotrixLab motion NPZ v1.

Holosoma schema (input):
    fps               (1,)         int64
    joint_pos         (T, N+7)     float64    # [root_pos(3), root_quat_wxyz(4), joint(N)]
    joint_vel         (T, N+6)     float64    # [root_lin(3), root_ang(3), joint(N)]
    body_pos_w        (T, B, 3)    float64
    body_quat_w       (T, B, 4)    float64    # wxyz
    body_lin_vel_w    (T, B, 3)    float64
    body_ang_vel_w    (T, B, 3)    float64
    joint_names       (N,)         str
    body_names        (B,)         str
    object_*_w        optional     float64    # manipulation objects

MotrixLab v1 (output) — see `motrix_envs/motion/schema.py`:
    Drops the root prefix on joint_pos/joint_vel (root state lives in
    body_pos_w/body_quat_w via root body index).
    Converts quaternions from wxyz to xyzw.
    Adds schema_version, num_frames.
    Preserves joint_names/body_names and ext_object_* extensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from motrix_envs.motion.schema import SCHEMA_VERSION, XYZW_FROM_WXYZ


def convert_holosoma(input_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Convert a Holosoma WBT npz to MotrixLab motion NPZ v1.

    Returns a dict of stats (num_frames, num_joints, num_bodies, has_object)
    for logging/testing.
    """
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as data:
        files = set(data.files)
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
        }
        missing = required - files
        if missing:
            raise ValueError(f"Holosoma motion missing keys {sorted(missing)}: {input_path}")

        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        joint_names = [str(x) for x in data["joint_names"].tolist()]
        body_names = [str(x) for x in data["body_names"].tolist()]
        n_joints = len(joint_names)
        n_bodies = len(body_names)

        joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel_raw = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

        object_keys = [k for k in files if k.startswith("object_") and k.endswith("_w")]
        object_arrays = {k: np.asarray(data[k], dtype=np.float32) for k in object_keys}

    # Validate Holosoma-specific shapes.
    if joint_pos_raw.ndim != 2 or joint_pos_raw.shape[1] != n_joints + 7:
        raise ValueError(f"joint_pos shape {joint_pos_raw.shape}, expected (T, {n_joints + 7})")
    if joint_vel_raw.ndim != 2 or joint_vel_raw.shape[1] != n_joints + 6:
        raise ValueError(f"joint_vel shape {joint_vel_raw.shape}, expected (T, {n_joints + 6})")
    if body_pos_w.ndim != 3 or body_pos_w.shape[1] != n_bodies or body_pos_w.shape[2] != 3:
        raise ValueError(f"body_pos_w shape {body_pos_w.shape} incompatible with body_names")

    num_frames = joint_pos_raw.shape[0]
    if not (joint_vel_raw.shape[0] == body_pos_w.shape[0] == body_quat_w.shape[0] == num_frames):
        raise ValueError(f"Inconsistent first-dim T across Holosoma arrays in {input_path}")

    # Drop root prefix on joint arrays.
    joint_pos = joint_pos_raw[:, 7:].copy()
    joint_vel = joint_vel_raw[:, 6:].copy()

    # wxyz -> xyzw on all body quaternions.
    body_quat_xyzw = body_quat_w[:, :, XYZW_FROM_WXYZ].copy()

    # Object quaternions (if any) also wxyz -> xyzw.
    ext_object: dict[str, np.ndarray] = {}
    for k, v in object_arrays.items():
        out_key = "ext_" + k
        if v.shape[-1] == 4:
            # Treat trailing 4-dim arrays as quaternions; convert wxyz -> xyzw.
            # (object_pos_w shape (T, 3) is unaffected; object_quat_w (T, 4) is converted.)
            flat = v.reshape(-1, 4)
            flat = flat[:, XYZW_FROM_WXYZ]
            ext_object[out_key] = flat.reshape(v.shape).copy()
        else:
            ext_object[out_key] = v.copy()

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(fps),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_xyzw,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    output.update(ext_object)
    np.savez(output_path, **output)

    return {
        "num_frames": num_frames,
        "num_joints": n_joints,
        "num_bodies": n_bodies,
        "has_object": bool(ext_object),
        "output_path": str(output_path),
    }
