# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""ONNX policy runtime tests."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from motrix_deploy.contracts import TensorSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import OnnxPolicyRuntime


def _write_linear_model(
    path: Path,
    *,
    output_name: str = "action",
    nan_output: bool = False,
    dynamic_batch: bool = False,
) -> None:
    weight = numpy_helper.from_array(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        name="weight",
    )
    nodes = [helper.make_node("MatMul", ["observation", "weight"], ["linear" if nan_output else output_name])]
    initializers = [weight]
    if nan_output:
        initializers.append(numpy_helper.from_array(np.full((1, 2), np.nan, dtype=np.float32), name="bias"))
        nodes.append(helper.make_node("Add", ["linear", "bias"], [output_name]))
    graph = helper.make_graph(
        nodes,
        "fixture_policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch_size" if dynamic_batch else 1, 4])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch_size" if dynamic_batch else 1, 2])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


def test_onnx_policy_validates_io_and_runs_float32(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_model(path)
    runtime = OnnxPolicyRuntime(
        path,
        TensorSpec(name="observation", shape=(1, 4)),
        TensorSpec(name="action", shape=(1, 2)),
    )

    output = runtime.infer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    np.testing.assert_array_equal(output, [4.0, 5.0])
    assert output.dtype == np.float32


def test_onnx_policy_accepts_dynamic_model_batch_for_single_sample_runtime(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_model(path, dynamic_batch=True)
    runtime = OnnxPolicyRuntime(
        path,
        TensorSpec(name="observation", shape=(1, 4)),
        TensorSpec(name="action", shape=(1, 2)),
    )

    output = runtime.infer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    np.testing.assert_array_equal(output, [4.0, 5.0])


def test_onnx_policy_rejects_manifest_name_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_model(path, output_name="actual_action")

    with pytest.raises(ValidationError, match="policy.output.name"):
        OnnxPolicyRuntime(
            path,
            TensorSpec(name="observation", shape=(1, 4)),
            TensorSpec(name="action", shape=(1, 2)),
        )


def test_onnx_policy_rejects_non_finite_input(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_model(path)
    runtime = OnnxPolicyRuntime(
        path,
        TensorSpec(name="observation", shape=(1, 4)),
        TensorSpec(name="action", shape=(1, 2)),
    )

    with pytest.raises(ValidationError, match="finite values"):
        runtime.infer(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))


def test_onnx_policy_rejects_non_finite_output(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_model(path, nan_output=True)
    runtime = OnnxPolicyRuntime(
        path,
        TensorSpec(name="observation", shape=(1, 4)),
        TensorSpec(name="action", shape=(1, 2)),
    )

    with pytest.raises(ValidationError, match="finite values"):
        runtime.infer(np.zeros(4, dtype=np.float32))
