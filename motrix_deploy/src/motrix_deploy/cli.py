# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hydra deployment rollout and artifact inspection entry points."""

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy import __version__
from motrix_deploy.artifact import inspect_artifact, read_artifact
from motrix_deploy.backend import (
    BackendCreateContext,
    GamePadDeviceProvider,
    KeyboardDeviceProvider,
    create_backend,
)
from motrix_deploy.config import DeployRunConfig
from motrix_deploy.errors import ArtifactError, ValidationError
from motrix_deploy.task import create_task


def run(cfg: DeployRunConfig) -> int:
    """Run one deployment rollout from typed application configuration."""
    from motrix_deploy.policy import OnnxPolicyRuntime
    from motrix_deploy.runtime import (
        ControlLoop,
        FixedStepScheduler,
        RealtimeScheduler,
    )
    from motrix_env_core.input import (
        BoundedGamePadPlanarVelocityBinding,
        ConstantPlanarVelocityBinding,
        KeyboardPlanarVelocityBinding,
    )

    artifact = read_artifact(Path(cfg.artifact))
    manifest = artifact.manifest
    task = create_task(manifest.task, manifest.robot)
    viewer = cfg.viewer
    realtime = viewer if cfg.realtime is None else cfg.realtime
    hardware_confirmed = cfg.hardware.get("confirm", False)
    backend = create_backend(
        cfg.backend_name,
        cfg.backend_options,
        BackendCreateContext(
            control=manifest.control,
            viewer=viewer,
            realtime=realtime,
            hardware_confirmed=hardware_confirmed,
        ),
    )
    command_lower = getattr(task, "command_lower", None)
    command_upper = getattr(task, "command_upper", None)
    if command_lower is None or command_upper is None:
        raise ValueError(f"Task {manifest.task.name!r} does not provide planar velocity command bounds")
    command_config = cfg.command or {}
    command_source = command_config.get("source", "constant")
    if viewer and isinstance(backend, KeyboardDeviceProvider):
        command_binding = KeyboardPlanarVelocityBinding(
            backend.get_keyboard_device(),
            command_lower=command_lower,
            command_upper=command_upper,
        )
    elif command_source == "gamepad":
        if not isinstance(backend, GamePadDeviceProvider):
            raise ValueError(f"Backend {cfg.backend_name!r} does not provide gamepad input")
        gamepad = command_config.get("gamepad", {})
        if not isinstance(gamepad, dict):
            raise ValueError("command.gamepad must be a mapping")
        command_binding = BoundedGamePadPlanarVelocityBinding(
            backend.get_gamepad_device(),
            linear_x_axis=gamepad.get("linear_x_axis", "ly"),
            linear_y_axis=gamepad.get("linear_y_axis", "lx"),
            yaw_axis=gamepad.get("yaw_axis", "rx"),
            command_lower=command_lower,
            command_upper=command_upper,
            deadzone=gamepad.get("deadzone", 0.1),
            range_scale=gamepad.get("range_scale", [1.0, 1.0, 1.0]),
            invert_linear_x=gamepad.get("invert_linear_x", False),
            invert_linear_y=gamepad.get("invert_linear_y", False),
            invert_yaw=gamepad.get("invert_yaw", False),
            deadman_button=gamepad.get("deadman_button", "L1"),
        )
    elif command_source == "constant" and "velocity" in command_config:
        command_binding = ConstantPlanarVelocityBinding(command_config["velocity"])
        task.validate_command(command_binding.read_command())
    else:
        raise ValueError(f"Unsupported command source {command_source!r}; use gamepad or configure command.velocity")
    loop = ControlLoop(
        robot=manifest.robot,
        backend=backend,
        task=task,
        policy=OnnxPolicyRuntime(artifact.policy_path, manifest.policy.input, manifest.policy.output),
        command_binding=command_binding,
        scheduler=(
            RealtimeScheduler(manifest.control.period_s) if realtime else FixedStepScheduler(manifest.control.period_s)
        ),
        state_timeout_s=manifest.control.state_timeout_s,
    )
    result = loop.run(steps=_rollout_steps(cfg.rollout, manifest.control.period_s))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _rollout_steps(cfg: dict[str, object] | None, control_period_s: float) -> int | None:
    if cfg is None:
        return None
    steps = cfg.get("steps")
    duration_s = cfg.get("duration_s")
    if steps is None and duration_s is None:
        return None
    if steps is not None and duration_s is not None:
        raise ValueError("Exactly one of rollout.steps and rollout.duration_s must be configured")
    if steps is not None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"rollout.steps must be a positive integer, got {steps!r}")
        return steps
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise ValueError(f"rollout.duration_s must be a positive finite number, got {duration_s!r}")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"rollout.duration_s must be a positive finite number, got {duration_s!r}")
    return math.ceil(duration_s / control_period_s)


@hydra.main(version_base=None, config_path="config", config_name="deploy")
def _hydra_main(cfg: DictConfig) -> None:
    try:
        typed_cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DeployRunConfig), cfg))
        if not isinstance(typed_cfg, DeployRunConfig):
            raise TypeError(f"Expected DeployRunConfig, got {type(typed_cfg).__name__}")
        exit_code = run(typed_cfg)
    except (ArtifactError, OmegaConfBaseException, RuntimeError, TypeError, ValidationError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error
    if exit_code:
        raise SystemExit(exit_code)


def inspect(cfg: DictConfig) -> None:
    """Validate one deployment artifact without opening a backend."""
    print(json.dumps(inspect_artifact(Path(cfg.artifact)), indent=2, sort_keys=True))


@hydra.main(version_base=None, config_path="config", config_name="inspect")
def _hydra_inspect_main(cfg: DictConfig) -> None:
    try:
        inspect(cfg)
    except (ArtifactError, OmegaConfBaseException, RuntimeError, ValidationError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error


def main(argv: Sequence[str] | None = None) -> int | None:
    """Inspect an artifact or run a Hydra-configured deployment rollout."""
    if argv is not None:
        original_argv = sys.argv
        sys.argv = ["motrix-deploy", *argv]
        try:
            return main()
        finally:
            sys.argv = original_argv

    args = sys.argv[1:]
    if args[:1] == ["inspect"]:
        del sys.argv[1]
        _hydra_inspect_main()
        return None
    if args == ["--version"]:
        print(f"motrix-deploy {__version__}")
        return 0
    if not args:
        sys.argv.append("--help")
    elif args[:1] in (["sim2sim"], ["sim2real"]):
        del sys.argv[1]
    _hydra_main()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
