# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks, utils
from motrix_rl.method import RlMethod

BACKEND_PREFERENCE = ("jax", "torch")


def resolve_train_backend(
    env_name: str,
    method: RlMethod,
    requested_backend: str | None,
    device_supports: utils.DeviceSupports,
) -> str:
    """Resolve the training backend for an RL method.

    Args:
        env_name: Registered environment name.
        method: Requested RL method.
        requested_backend: Backend requested by the user, or None for auto selection.
        device_supports: Runtime support information for built-in backends.
    """
    if requested_backend is not None:
        _validate_backend(method, requested_backend, device_supports)
        return requested_backend

    framework = frameworks.get_framework(method.rllib)
    candidates = [
        backend
        for backend in framework.supported_train_backends(method.algo)
        if _backend_available(backend, device_supports)
    ]
    if not candidates:
        raise ValueError(
            f"No available train backend for environment '{env_name}', RL method '{method.rllib}.{method.algo}'."
        )
    return _choose_preferred_backend(candidates)


def _validate_backend(
    method: RlMethod,
    backend: str,
    device_supports: utils.DeviceSupports,
) -> None:
    provider = frameworks.get_agent_provider(method.rllib, method.algo, backend)
    if provider is None:
        raise ValueError(
            f"No trainer found for RL framework '{method.rllib}', train backend '{backend}', algorithm '{method.algo}'."
        )
    if not _backend_available(backend, device_supports):
        raise ValueError(f"Train backend '{backend}' is not available on this device.")


def _backend_available(backend: str, device_supports: utils.DeviceSupports) -> bool:
    if backend == "jax":
        return device_supports.jax
    if backend == "torch":
        return device_supports.torch
    return True


def _choose_preferred_backend(candidates: list[str]) -> str:
    for preferred in BACKEND_PREFERENCE:
        if preferred in candidates:
            return preferred
    return sorted(candidates)[0]
