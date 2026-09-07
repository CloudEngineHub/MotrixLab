# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared ONNX Runtime parity validation."""

from collections.abc import Callable

import numpy as np

from motrix_rl.config import OnnxParityConfig
from motrix_rl.deploy.api import OnnxExportReport, OnnxParityMetrics, PolicyTensorSpec


def validate_onnx_policy(
    model: bytes,
    reference: Callable[[np.ndarray], np.ndarray],
    *,
    observation_size: int,
    action_size: int,
    config: OnnxParityConfig,
    source: str,
) -> OnnxExportReport:
    """Validate one single-input policy against its framework reference."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX validation requires the 'motrix-rl[onnx]' extra") from error

    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{source} exporter requires one ONNX input and output, got {len(inputs)} and {len(outputs)}")

    input_spec = _tensor_spec(inputs[0], kind="input")
    output_spec = _tensor_spec(outputs[0], kind="output")
    expected_input_shape = (None, observation_size)
    expected_output_shape = (None, action_size)
    if input_spec.shape != expected_input_shape:
        raise ValueError(f"{source} ONNX input shape must be {expected_input_shape}, got {input_spec.shape}")
    if output_spec.shape != expected_output_shape:
        raise ValueError(f"{source} ONNX output shape must be {expected_output_shape}, got {output_spec.shape}")
    if input_spec.dtype != "float32" or output_spec.dtype != "float32":
        raise ValueError(
            f"{source} ONNX tensors must be float32, got input={input_spec.dtype}, output={output_spec.dtype}"
        )

    generator = np.random.default_rng(config.seed)
    observations = generator.standard_normal((config.samples, observation_size), dtype=np.float32)
    observations[0].fill(0.0)
    expected = np.asarray(reference(observations), dtype=np.float32)
    actual = np.asarray(session.run([output_spec.name], {input_spec.name: observations})[0], dtype=np.float32)
    expected_shape = (config.samples, action_size)
    if expected.shape != expected_shape:
        raise ValueError(f"{source} reference output shape must be {expected_shape}, got {expected.shape}")
    if actual.shape != expected_shape:
        raise ValueError(f"{source} ONNX output shape must be {expected_shape}, got {actual.shape}")
    if not np.all(np.isfinite(expected)):
        raise ValueError(f"{source} reference output contains NaN or Inf")
    if not np.all(np.isfinite(actual)):
        raise ValueError(f"{source} ONNX output contains NaN or Inf")

    difference = np.abs(actual - expected)
    max_abs_error = float(difference.max(initial=0.0))
    relative = difference / np.maximum(np.abs(expected), np.float32(1e-12))
    max_rel_error = float(relative.max(initial=0.0))
    if not np.allclose(actual, expected, atol=config.atol, rtol=config.rtol):
        raise ValueError(
            f"{source}/ONNX parity failed: max_abs_error={max_abs_error:.8g}, "
            f"max_rel_error={max_rel_error:.8g}, atol={config.atol}, rtol={config.rtol}"
        )
    return OnnxExportReport(
        input_spec=input_spec,
        output_spec=output_spec,
        parity=OnnxParityMetrics(
            samples=config.samples,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
        ),
    )


def _tensor_spec(value, *, kind: str) -> PolicyTensorSpec:
    shape = tuple(dimension if isinstance(dimension, int) else None for dimension in value.shape)
    dtype = {"tensor(float)": "float32"}.get(value.type, value.type)
    if not value.name:
        raise ValueError(f"ONNX {kind} name must be non-empty")
    return PolicyTensorSpec(name=value.name, shape=shape, dtype=dtype)
