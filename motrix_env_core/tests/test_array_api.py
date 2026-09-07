# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from motrix_env_core.array_api import Array, array_namespace, is_array


class _OtherArrayApiObject:
    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        del api_version
        return np


def _statistics(x: Array, y: Array) -> Array:
    xp = array_namespace(x, y)
    return xp.mean(x, axis=0) + 2 * xp.std(y, axis=0)


def _to_numpy(value: Array) -> npt.NDArray[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def test_is_array_recognizes_supported_runtime_arrays() -> None:
    assert is_array(np.zeros(1, dtype=np.float32))
    assert is_array(torch.zeros(1))
    assert not is_array([0.0])
    assert not is_array(np.float32(0.0))
    assert not is_array(_OtherArrayApiObject())


def test_array_namespace_rejects_other_array_api_backends() -> None:
    with pytest.raises(TypeError, match="only supports NumPy arrays and Torch tensors"):
        array_namespace(_OtherArrayApiObject())


def test_array_namespace_rejects_mixed_supported_backends() -> None:
    with pytest.raises(TypeError, match="Multiple namespaces"):
        array_namespace(np.zeros(1, dtype=np.float32), torch.zeros(1))


@pytest.mark.parametrize(
    ("x", "y", "expected_type"),
    [
        pytest.param(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
            np.ndarray,
            id="numpy",
        ),
        pytest.param(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32),
            torch.Tensor,
            id="torch",
        ),
    ],
)
def test_array_namespace_runs_backend_neutral_statistics(
    x: Array,
    y: Array,
    expected_type: type[object],
) -> None:
    result = _statistics(x, y)

    assert isinstance(result, expected_type)
    np.testing.assert_allclose(_to_numpy(result), np.asarray([6.0, 7.0], dtype=np.float32))
