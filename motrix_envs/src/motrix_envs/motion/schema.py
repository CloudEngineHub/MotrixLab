# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Schema constants for the MotrixLab motion NPZ format v1.

Quaternion convention: xyzw.
World frame: right-handed Z-up.
One .npz file = one motion clip.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# Required scalar / metadata fields.
REQUIRED_SCALARS = ("schema_version", "fps", "num_frames")

# Required string-array fields (name lists for model binding).
REQUIRED_NAMES = ("joint_names", "body_names")

# Required per-frame robot state fields.
# Shapes (T = num_frames):
#   joint_pos / joint_vel       (T, N)       float32, N = len(joint_names)
#   body_pos_w                  (T, B, 3)    float32, B = len(body_names)
#   body_quat_w                 (T, B, 4)    float32, xyzw
#   body_lin_vel_w              (T, B, 3)    float32
#   body_ang_vel_w              (T, B, 3)    float32
REQUIRED_FIELDS = (
    "schema_version",
    "fps",
    "num_frames",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

# Optional metadata fields the loader recognizes but does not require.
OPTIONAL_FIELDS = (
    "tracked_body_names",
    "reference_body_name",
    "root_body_name",
    "clip_name",
)

# Quaternion xyzw index helper — used by converters and tests.
XYZW_FROM_WXYZ = (1, 2, 3, 0)
