# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution-level behavior of compiled MotrixSim write programs."""

import numpy as np

from motrix_env_motrixsim.write_compiler import _CompiledWrite, _MotrixSimWriteProgram


class _Model:
    num_dof_pos = 0
    num_dof_vel = 0

    def __init__(self) -> None:
        self.forward_kinematic_rows = []

    def compute_init_dof_pos(self) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    def forward_kinematic(self, rows) -> None:
        self.forward_kinematic_rows.append(rows)


class _Data:
    shape = (3,)


class _ResetData(_Data):
    def __init__(self) -> None:
        self.reset_calls = []

    def reset(self, model, **kwargs) -> None:
        self.reset_calls.append((model, kwargs))


class _Op:
    def __init__(self) -> None:
        self.rows = []

    def __call__(self, buffers, idx, rows) -> None:
        del buffers, idx
        self.rows.append(rows)


def test_write_program_refreshes_kinematics_once_after_all_ops() -> None:
    model = _Model()
    data = _Data()
    first = _Op()
    second = _Op()
    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: ("rows", tuple(env_ids)),
        {},
        [(_CompiledWrite(first), {}), (_CompiledWrite(second), {})],
        reset=False,
        refresh_kinematics=True,
    )

    program.execute(np.asarray([2, 0], dtype=np.int64))

    expected_rows = ("rows", (0, 2))
    assert first.rows == [expected_rows]
    assert second.rows == [expected_rows]
    assert model.forward_kinematic_rows == [expected_rows]


def test_reset_program_passes_compile_time_kinematics_flag_to_native_reset() -> None:
    model = _Model()
    data = _ResetData()
    program = _MotrixSimWriteProgram(model, data, lambda env_ids: env_ids, {}, [], reset=True, refresh_kinematics=False)

    program.execute()

    assert data.reset_calls == [(model, {"forward_kinematic": False})]
    assert model.forward_kinematic_rows == []


def test_reset_program_applies_non_fused_writes_after_native_reset_and_refreshes_once() -> None:
    model = _Model()
    data = _ResetData()
    op = _Op()
    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: env_ids,
        {},
        [(_CompiledWrite(op), {})],
        reset=True,
        refresh_kinematics=True,
    )

    program.execute()

    assert data.reset_calls == [(model, {"forward_kinematic": False})]
    assert op.rows == [data]
    assert model.forward_kinematic_rows == [data]


def test_write_program_skips_kinematic_refresh_when_no_op_requires_it() -> None:
    model = _Model()
    program = _MotrixSimWriteProgram(
        model, _Data(), lambda env_ids: env_ids, {}, [], reset=False, refresh_kinematics=False
    )

    program.execute()

    assert model.forward_kinematic_rows == []
