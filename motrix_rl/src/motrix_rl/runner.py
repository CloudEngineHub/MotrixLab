# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from motrix_env_core.renderer import RenderConfig
from motrix_rl import backend_runtime, checkpoints, frameworks, runs, utils
from motrix_rl.config import TaskConfig, TrainConfig
from motrix_rl.method import RlMethod
from motrix_rl.result import TrainResult


@dataclass(frozen=True)
class TrainRequest:
    """Framework-level request to create a run and execute training."""

    config: TrainConfig
    render: RenderConfig | None = None
    runs_root: str | Path = runs.LOG_DIR_PREFIX


def train(request: TrainRequest) -> TrainResult:
    """Create a training handle, execute training, and return its result."""
    return create_training_handle(request).train()


def create_training_handle(request: TrainRequest) -> frameworks.TrainerHandle:
    """Create a trainer handle for a new training run."""
    config = request.config
    task = config.task
    method = RlMethod(rllib=task.rllib, algo=task.algo)
    device_supports = utils.get_device_supports()
    train_backend = backend_runtime.resolve_train_backend(
        task.env,
        method,
        task.train_backend,
        device_supports,
    )
    provider = frameworks.get_agent_provider(method.rllib, method.algo, train_backend)
    if provider is None:
        raise ValueError(
            f"No trainer found for RL framework '{method.rllib}', train backend '{train_backend}', "
            f"algorithm '{method.algo}'."
        )
    provider.validate_config(config.algo)

    run = runs.create_run_context(
        env_name=task.env,
        rllib=method.rllib,
        train_backend=provider.train_backend,
        algo=method.algo,
        sim=config.sim,
        seed=config.seed,
        checkpoint_format=provider.checkpoint_format,
        runs_root=request.runs_root,
    )
    runs.write_task_config(run.run_dir, config)
    resume_from = None
    if config.resume is not None:
        resume_from = str(checkpoints.resolve_resume_checkpoint_path(config.resume))
    return _create_handle(
        run,
        provider,
        config,
        render=request.render,
        resume_from=resume_from,
    )


def create_run_handle(
    run: runs.RunContext,
    *,
    cfg_override: dict | None = None,
    play_num_envs: int | None = None,
    seed: int | None = None,
    randomize_seed: bool = False,
    render: RenderConfig | None = None,
    resume_from: str | None = None,
) -> frameworks.TrainerHandle:
    """Restore a typed trainer from an existing metadata-backed run."""
    metadata = run.metadata
    provider = frameworks.get_agent_provider(metadata.rllib, metadata.algo, metadata.train_backend)
    if provider is None:
        raise ValueError(
            f"No trainer found for RL framework '{metadata.rllib}', train backend '{metadata.train_backend}', "
            f"algorithm '{metadata.algo}'."
        )
    task_cfg = runs.read_task_config(
        run.run_dir,
        provider.config_type,
        cfg_override=cfg_override,
    )
    return _create_handle(
        run,
        provider,
        task_cfg,
        play_num_envs=play_num_envs,
        seed=seed,
        randomize_seed=randomize_seed,
        render=render,
        resume_from=resume_from,
    )


def _create_handle(
    run: runs.RunContext,
    provider: frameworks.AgentProvider,
    task_cfg: TaskConfig,
    *,
    play_num_envs: int | None = None,
    seed: int | None = None,
    randomize_seed: bool = False,
    render: RenderConfig | None = None,
    resume_from: str | None = None,
) -> frameworks.TrainerHandle:
    metadata = run.metadata
    if (
        task_cfg.task.env != metadata.env_name
        or task_cfg.task.rllib != metadata.rllib
        or task_cfg.task.algo != metadata.algo
        or task_cfg.task.train_backend not in (None, metadata.train_backend)
    ):
        raise ValueError("task_cfg metadata does not match the run metadata")
    typed_cfg = provider.validate_config(task_cfg.algo)
    effective_seed = None if randomize_seed else (metadata.seed if seed is None else seed)
    return frameworks.create_trainer(
        frameworks.create_trainer_context(
            run,
            num_envs=task_cfg.num_envs,
            play_num_envs=play_num_envs if play_num_envs is not None else task_cfg.play_num_envs,
            seed=effective_seed,
            rl_cfg=typed_cfg,
            logging=task_cfg.logging,
            checkpoint=task_cfg.checkpoint,
            render=render,
            resume_from=resume_from,
        )
    )
