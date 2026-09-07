# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Import concrete tasks and register them with :mod:`motrix_deploy.task`."""

import sys
from collections.abc import Sequence
from pathlib import Path

from motrix_deploy.cli import main as deploy_main
from motrix_deploy_tasks.go2_walk import Go2WalkDeployTaskV1

__version__ = "0.3.0"

_DEFAULT_CONFIG_ROOT = Path("configs/deploy")
_DEFAULT_CONFIG_NAMES = {
    "sim2sim": "go2_walk_sim2sim",
    "sim2real": "go2_walk_flat_sim2real",
}


def main(argv: Sequence[str] | None = None) -> int | None:
    """Register concrete tasks and run the core CLI with the workspace recipe by default."""
    args = list(sys.argv[1:] if argv is None else argv)
    subcommand = args[0] if args[:1] in (["sim2sim"], ["sim2real"]) else "sim2sim"
    default_config_name = _DEFAULT_CONFIG_NAMES[subcommand]
    uses_runtime_config = bool(args) and args[:1] != ["inspect"] and args != ["--version"]
    has_config_path = any(arg == "--config-path" or arg.startswith("--config-path=") for arg in args)
    has_config_name = any(arg == "--config-name" or arg.startswith("--config-name=") for arg in args)
    default_config_dir = Path.cwd() / _DEFAULT_CONFIG_ROOT / subcommand
    default_config = default_config_dir / f"{default_config_name}.yaml"
    if uses_runtime_config and default_config.is_file():
        if not has_config_path:
            args.extend(("--config-path", str(default_config_dir)))
        if not has_config_name:
            args.extend(("--config-name", default_config_name))
    return deploy_main(args)


__all__ = ["Go2WalkDeployTaskV1", "main"]
