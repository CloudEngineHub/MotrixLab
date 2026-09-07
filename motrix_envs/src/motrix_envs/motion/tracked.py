# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Kernel-compatible, eagerly indexed whole-body motion data."""

from __future__ import annotations

import numpy as np

from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_envs.motion.loader import MotrixMotion


@kernel_data
class WbtMotionClip:
    """Numeric whole-body-tracking views derived from a :class:`MotrixMotion`.

    The factory resolves all string-based joint and body selections once and
    stores only contiguous numeric arrays suitable for sharing with compiled
    manager kernels. ``T`` is the number of frames, ``N`` the selected joint
    count, and ``K`` the tracked-body count.

    Attributes:
        joint_pos: Selected joint positions shaped ``(T, N)``.
        joint_vel: Selected joint velocities shaped ``(T, N)``.
        tracked_bodies_pos_w: Tracked-body world positions shaped ``(T, K, 3)``.
        tracked_bodies_quat_w: Tracked-body world quaternions shaped ``(T, K, 4)``.
        tracked_bodies_lin_vel_w: Tracked-body world linear velocities shaped ``(T, K, 3)``.
        tracked_bodies_ang_vel_w: Tracked-body world angular velocities shaped ``(T, K, 3)``.
        root_body_pos_w: Root-body world positions shaped ``(T, 3)``.
        root_body_quat_w: Root-body world quaternions shaped ``(T, 4)``.
        root_body_lin_vel_w: Root-body world linear velocities shaped ``(T, 3)``.
        root_body_ang_vel_w: Root-body world angular velocities shaped ``(T, 3)``.
        reference_body_pos_w: Reference-body world positions shaped ``(T, 3)``.
        reference_body_quat_w: Reference-body world quaternions shaped ``(T, 4)``.
    """

    joint_pos: SharedArray
    joint_vel: SharedArray
    tracked_bodies_pos_w: SharedArray
    tracked_bodies_quat_w: SharedArray
    tracked_bodies_lin_vel_w: SharedArray
    tracked_bodies_ang_vel_w: SharedArray
    root_body_pos_w: SharedArray
    root_body_quat_w: SharedArray
    root_body_lin_vel_w: SharedArray
    root_body_ang_vel_w: SharedArray
    reference_body_pos_w: SharedArray
    reference_body_quat_w: SharedArray

    @staticmethod
    def create(
        motion: MotrixMotion,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
    ) -> WbtMotionClip:
        """Resolve names and build contiguous numeric WBT motion views.

        Args:
            motion: Loaded source motion in file-defined joint and body order.
            joint_names: Desired output joint order.
            tracked_body_names: Desired tracked-body order.
            reference_body_name: Body used to align the motion to the robot.
            root_body_name: Floating-base root body.

        Returns:
            Kernel-compatible motion data containing no string metadata.
        """
        if motion.num_frames < 2:
            raise ValueError(f"Tracked motion must have at least two frames: {motion.path}")

        joint_motion_idx = motion.joint_indices(joint_names)
        tracked_idx = motion.body_indices(list(tracked_body_names))
        root_index = motion.body_index(root_body_name)
        reference_index = motion.body_index(reference_body_name)

        return WbtMotionClip(
            joint_pos=np.ascontiguousarray(motion.joint_pos[:, joint_motion_idx]),
            joint_vel=np.ascontiguousarray(motion.joint_vel[:, joint_motion_idx]),
            tracked_bodies_pos_w=np.ascontiguousarray(motion.body_pos_w[:, tracked_idx]),
            tracked_bodies_quat_w=np.ascontiguousarray(motion.body_quat_w[:, tracked_idx]),
            tracked_bodies_lin_vel_w=np.ascontiguousarray(motion.body_lin_vel_w[:, tracked_idx]),
            tracked_bodies_ang_vel_w=np.ascontiguousarray(motion.body_ang_vel_w[:, tracked_idx]),
            root_body_pos_w=np.ascontiguousarray(motion.body_pos_w[:, root_index]),
            root_body_quat_w=np.ascontiguousarray(motion.body_quat_w[:, root_index]),
            root_body_lin_vel_w=np.ascontiguousarray(motion.body_lin_vel_w[:, root_index]),
            root_body_ang_vel_w=np.ascontiguousarray(motion.body_ang_vel_w[:, root_index]),
            reference_body_pos_w=np.ascontiguousarray(motion.body_pos_w[:, reference_index]),
            reference_body_quat_w=np.ascontiguousarray(motion.body_quat_w[:, reference_index]),
        )
