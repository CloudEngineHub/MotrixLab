# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Structured target-axis behavior of fixed-width MotrixSim write ops."""

import numpy as np

from motrix_env_motrixsim.write_compiler import _MultiTargetOp


class _Target:
    def __init__(self) -> None:
        self.values = []

    def set_scalar(self, rows, values) -> None:
        self.values.append((rows, values.copy()))

    def set_vector(self, rows, values) -> None:
        self.values.append((rows, values.copy()))


def test_multi_target_scalar_op_routes_declared_target_axis() -> None:
    first = _Target()
    second = _Target()
    op = _MultiTargetOp((first, second), "set_scalar", 1)
    buffers = op.alloc(3)
    assert buffers.shape == (3, 2)
    buffers[:] = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]

    op(buffers, np.asarray([2, 0], dtype=np.int64), "rows")

    np.testing.assert_array_equal(first.values[0][1], [3.0, 1.0])
    np.testing.assert_array_equal(second.values[0][1], [30.0, 10.0])


def test_multi_target_vector_op_routes_declared_target_axis() -> None:
    first = _Target()
    second = _Target()
    op = _MultiTargetOp((first, second), "set_vector", 3)
    buffers = op.alloc(2)
    assert buffers.shape == (2, 2, 3)
    buffers[:] = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]

    op(buffers, slice(None), "rows")

    np.testing.assert_array_equal(first.values[0][1], [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(second.values[0][1], [[4, 5, 6], [10, 11, 12]])
