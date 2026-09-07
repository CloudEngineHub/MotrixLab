# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import numba
import numpy as np
import pytest

from motrix_env_core.numba.kernel_data import (
    KernelDataLowering,
    KernelDataScope,
    Map,
    SharedArray,
    flatten_kernel_data,
    iter_layout_leaves,
    kernel_data,
    map_proxy,
    rebuild_lowered,
    unflatten_kernel_data,
)


@kernel_data
class _Pose:
    position: np.ndarray
    orientation: SharedArray


@kernel_data
class _RobotBase:
    velocity: np.ndarray


@kernel_data
class _Robot(_RobotBase):
    pose: _Pose


@kernel_data
class _Tracking:
    robot: _Robot
    phase: np.float32


@kernel_data
class _MappedValues:
    values: Map


@kernel_data
class _ArrayMapValues:
    values: Map[np.ndarray]


def test_kernel_data_is_frozen_and_compiles_nested_inherited_leaves() -> None:
    value = _Tracking(
        _Robot(np.asarray([[1.0, 2.0]], dtype=np.float32), _Pose(np.asarray([[3.0]]), np.asarray([4.0]))),
        np.float32(0.5),
    )
    with pytest.raises(FrozenInstanceError):
        value.phase = np.float32(1.0)  # type: ignore[misc]

    _, tree_def = flatten_kernel_data(value)
    layout = KernelDataLowering().lower(tree_def, context="test")
    leaves = iter_layout_leaves(layout)
    assert [leaf.path for leaf in leaves] == [
        ("robot", "velocity"),
        ("robot", "pose", "position"),
        ("robot", "pose", "orientation"),
        ("phase",),
    ]
    assert [leaf.scope for leaf in leaves] == [
        KernelDataScope.PER_ENV,
        KernelDataScope.PER_ENV,
        KernelDataScope.SHARED,
        KernelDataScope.SHARED,
    ]


def test_kernel_data_flatten_is_zero_copy_and_rebuilds_both_views() -> None:
    velocity = np.asarray([[1.0]], dtype=np.float32)
    position = np.asarray([[2.0]], dtype=np.float32)
    orientation = np.asarray([3.0], dtype=np.float32)
    value = _Tracking(_Robot(velocity, _Pose(position, orientation)), np.float32(0.25))
    leaves, tree_def = flatten_kernel_data(value)
    assert leaves[:3] == (velocity, position, orientation)
    rebuilt = unflatten_kernel_data(tree_def, leaves)
    layout = KernelDataLowering().lower(tree_def, context="test")
    lowered = rebuild_lowered(layout, leaves)
    assert rebuilt.robot.pose.position is position
    assert lowered.robot.pose.orientation is orientation
    assert lowered.phase == np.float32(0.25)


def test_kernel_data_schema_fingerprint_ignores_values() -> None:
    first_value = _Tracking(
        _Robot(np.asarray([[1.0]], dtype=np.float32), _Pose(np.asarray([[2.0]]), np.asarray([3.0]))),
        np.float32(0.5),
    )
    second_value = _Tracking(
        _Robot(np.asarray([[4.0]], dtype=np.float32), _Pose(np.asarray([[5.0]]), np.asarray([6.0]))),
        np.float32(0.75),
    )
    _, first = flatten_kernel_data(first_value)
    _, second = flatten_kernel_data(second_value)
    assert first is second
    assert first.fingerprint == second.fingerprint
    first_layout = KernelDataLowering().lower(first, context="first")
    second_layout = KernelDataLowering().lower(second, context="second")
    assert first_layout is second_layout
    assert first_layout.lowered_type.__module__ == _Tracking.__module__


def test_map_supports_python_and_numba_literal_lookup() -> None:
    values = Map({"pose": _Pose(np.asarray([[1.0]]), np.asarray([2.0])), "phase": np.float32(0.5)})
    assert values["phase"] == np.float32(0.5)
    with pytest.raises(KeyError, match="available entries"):
        values["missing"]

    proxy_type = map_proxy("test.values", values.keys_tuple, module_name=__name__)
    proxy = proxy_type((np.float32(1.0),), values["phase"])

    @numba.njit
    def read_phase(kernel_values):
        return kernel_values["phase"]

    assert read_phase(proxy) == np.float32(0.5)
    assert read_phase.nopython_signatures


def test_map_participates_in_tree_lowering_and_round_trip() -> None:
    position = np.asarray([[1.0]], dtype=np.float32)
    orientation = np.asarray([2.0], dtype=np.float32)
    value = _MappedValues(Map({"pose": _Pose(position, orientation), "phase": np.float32(0.5)}))

    leaves, tree_def = flatten_kernel_data(value)
    assert leaves == (position, orientation, np.float32(0.5))
    rebuilt = unflatten_kernel_data(tree_def, leaves)
    assert rebuilt.values.keys_tuple == ("pose", "phase")
    assert rebuilt.values["pose"].position is position

    layout = KernelDataLowering().lower(tree_def, context="mapped values")
    lowered = rebuild_lowered(layout, leaves)

    @numba.njit
    def read_values(kernel_values):
        return kernel_values.values["pose"].position[0, 0] + kernel_values.values["phase"]

    assert read_values(lowered) == np.float32(1.5)
    assert read_values.nopython_signatures


def test_generic_array_map_participates_in_numba_lowering() -> None:
    position = np.asarray([[1.0, 2.0]], dtype=np.float32)
    velocity = np.asarray([3.0], dtype=np.float32)
    value = _ArrayMapValues(Map({"position": position, "velocity": velocity}))

    leaves, tree_def = flatten_kernel_data(value)
    layout = KernelDataLowering().lower(tree_def, context="array map")
    lowered = rebuild_lowered(layout, leaves)

    @numba.njit
    def read_values(kernel_values):
        return kernel_values.values["position"][0, 1] + kernel_values.values["velocity"][0]

    assert read_values(lowered) == np.float32(5.0)
    assert read_values.nopython_signatures


def test_map_schema_changes_tree_and_proxy_identity() -> None:
    first = _MappedValues(Map({"value": np.float32(1.0)}))
    second = _MappedValues(Map({"value": np.asarray([[1.0]], dtype=np.float32)}))
    _, first_tree = flatten_kernel_data(first)
    _, second_tree = flatten_kernel_data(second)
    first_layout = KernelDataLowering().lower(first_tree, context="first")
    second_layout = KernelDataLowering().lower(second_tree, context="second")
    assert first_tree.fingerprint != second_tree.fingerprint
    assert first_layout.lowered_type is not second_layout.lowered_type
    assert first_layout.fields[0].child.lowered_type is not second_layout.fields[0].child.lowered_type


def test_map_rejects_dynamic_numba_key() -> None:
    proxy_type = map_proxy("test.dynamic", ("value",), module_name=__name__)
    proxy = proxy_type(np.float32(1.0))

    @numba.njit
    def read_value(kernel_values, name):
        return kernel_values[name]

    with pytest.raises(numba.TypingError):
        read_value(proxy, "value")


def test_kernel_data_rejects_unsupported_leaf() -> None:
    @kernel_data
    class _Invalid:
        label: str

    _, tree_def = flatten_kernel_data(_Invalid("invalid"))
    with pytest.raises(TypeError, match="expected np.ndarray"):
        KernelDataLowering().lower(tree_def, context="invalid")


def test_kernel_data_rejects_inherited_field_override_before_slots_transform() -> None:
    @kernel_data
    class _Base:
        value: int = 1

    with pytest.raises(TypeError, match="must not override inherited fields: value"):

        @kernel_data
        class _AnnotationOverride(_Base):
            value: int = 2

    with pytest.raises(TypeError, match="must not override inherited fields: value"):

        @kernel_data
        class _DefaultOverride(_Base):
            value = 2
