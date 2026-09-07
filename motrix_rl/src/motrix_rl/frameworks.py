# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from hydra.core.config_store import ConfigStore

from motrix_env_core.renderer import RenderConfig
from motrix_rl import runs
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.deploy.api import OnnxExportRequest, OnnxModelArtifact, OnnxPolicyExporter

CfgT = TypeVar("CfgT")
ALGO_GROUP = "algo_base"


class TrainerBase(ABC):
    """Runtime trainer base class implemented by RL framework integrations."""

    @abstractmethod
    def train(self) -> None:
        """Run training for the configured run."""

    @abstractmethod
    def play(self, policy: str) -> None:
        """Run policy playback from a checkpoint path."""


@dataclass(frozen=True)
class TrainerContext(Generic[CfgT]):
    """Minimal runtime context passed to trainer implementations."""

    run: runs.RunContext
    env_name: str
    run_dir: Path
    checkpoint_dir: Path
    sim: str | None
    checkpoint_format: str
    num_envs: int
    play_num_envs: int
    seed: int | None
    rl_cfg: CfgT
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    render: RenderConfig | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class TrainerHandle:
    """Framework-managed handle around a trainer implementation."""

    run: runs.RunContext
    trainer: TrainerBase

    def train(self):
        """Run training and return the framework-owned result view."""
        from motrix_rl.result import TrainResult

        self.trainer.train()
        return TrainResult(run=self.run)

    def play(self, policy: str) -> None:
        """Run policy playback through the wrapped trainer."""
        self.trainer.play(policy)


class AgentProvider(Generic[CfgT], ABC):
    """Framework-scoped provider for one train backend and agent."""

    config_type: type[CfgT]

    @property
    @abstractmethod
    def train_backend(self) -> str:
        """Train backend name handled by this provider, such as ``torch`` or ``jax``."""

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Agent or algorithm name handled by this provider."""

    @property
    @abstractmethod
    def checkpoint_format(self) -> str | None:
        """Preferred checkpoint file format produced by this provider."""

    def validate_config(self, config: object) -> CfgT:
        """Narrow a dynamically composed config to this provider's type."""
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"Provider for '{self.agent_name}' expects {self.config_type.__name__}, got {type(config).__name__}"
            )
        return cast(CfgT, config)

    @abstractmethod
    def create_trainer(self, context: TrainerContext[CfgT]) -> TrainerBase:
        """Create a trainer instance for the supplied run context."""

    def create_policy_exporter(self) -> OnnxPolicyExporter | None:
        """Create this provider's optional ONNX policy exporter."""
        return None


@dataclass(frozen=True)
class AgentRegistration:
    """One algorithm's shared config type and backend-specific providers."""

    config_type: type[Any]
    providers: dict[str, AgentProvider[Any]]


class RlFramework(ABC):
    """Registration entry point for one RL framework integration."""

    def __init__(self, providers: Iterable[AgentProvider[Any]]) -> None:
        self._agents = _index_agents(providers)

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework namespace used as the ``rllib`` component of an RL method."""

    def supported_agents(self) -> tuple[str, ...]:
        """Return agent/algorithm names declared by this framework."""
        return tuple(sorted(self._agents))

    def get_config_type(self, agent_name: str) -> type[Any]:
        """Return the structured config type shared by an algorithm's providers."""
        try:
            return self._agents[agent_name].config_type
        except KeyError as e:
            raise ValueError(f"Agent '{agent_name}' is not registered for RL framework '{self.name}'.") from e

    def supported_train_backends(self, agent_name: str) -> tuple[str, ...]:
        """Return train backends declared for an agent/algorithm name."""
        registration = self._agents.get(agent_name)
        return tuple(sorted(registration.providers)) if registration is not None else ()

    def get_agent_provider(self, agent_name: str, train_backend: str) -> AgentProvider[Any] | None:
        """Return the provider for one agent and train backend, if any."""
        registration = self._agents.get(agent_name)
        return registration.providers.get(train_backend) if registration is not None else None

    def supported_policy_exports(self) -> tuple[tuple[str, str], ...]:
        """Return ``(train_backend, agent)`` combinations with ONNX export support."""
        combinations = []
        for agent_name in self.supported_agents():
            for train_backend in self.supported_train_backends(agent_name):
                provider = self.get_agent_provider(agent_name, train_backend)
                if provider is not None and provider.create_policy_exporter() is not None:
                    combinations.append((train_backend, agent_name))
        return tuple(sorted(combinations))

    def export_policy(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        """Export a metadata-selected policy through its framework provider."""
        metadata = request.run.metadata
        if metadata.rllib != self.name:
            raise ValueError(f"Export request is for RL framework '{metadata.rllib}', but '{self.name}' was selected.")
        provider = self.get_agent_provider(metadata.algo, metadata.train_backend)
        if provider is None:
            supported = ", ".join(
                f"{self.name}/{backend}/{agent}" for backend, agent in self.supported_policy_exports()
            )
            raise ValueError(
                f"No agent provider found for ONNX export {self.name}/{metadata.train_backend}/{metadata.algo}; "
                f"supported combinations: {supported or 'none'}."
            )
        provider.validate_config(request.task_config.algo)
        exporter = provider.create_policy_exporter()
        if exporter is None:
            supported = ", ".join(
                f"{self.name}/{backend}/{agent}" for backend, agent in self.supported_policy_exports()
            )
            raise ValueError(
                f"ONNX policy export is not supported for {self.name}/{metadata.train_backend}/{metadata.algo}; "
                f"supported combinations: {supported or 'none'}."
            )
        return exporter.export(request)


def _index_agents(
    providers: Iterable[AgentProvider[Any]],
) -> dict[str, AgentRegistration]:
    """Index providers by agent and validate their shared config type."""
    providers_by_agent: dict[str, list[AgentProvider[Any]]] = {}
    for provider in providers:
        config_type = getattr(provider, "config_type", None)
        if not isinstance(config_type, type):
            raise TypeError(f"{type(provider).__name__} must declare a config_type class")
        if not is_dataclass(config_type):
            raise TypeError(f"{type(provider).__name__}.config_type must be a dataclass")
        if not provider.agent_name:
            raise ValueError(f"{type(provider).__name__} must declare a non-empty agent_name")
        if not provider.train_backend:
            raise ValueError(f"{type(provider).__name__} must declare a non-empty train_backend")
        providers_by_agent.setdefault(provider.agent_name, []).append(provider)

    index: dict[str, AgentRegistration] = {}
    for agent_name, agent_providers in providers_by_agent.items():
        config_types = {provider.config_type for provider in agent_providers}
        if len(config_types) != 1:
            details = ", ".join(
                f"{provider.train_backend}={provider.config_type.__name__}"
                for provider in sorted(agent_providers, key=lambda item: item.train_backend)
            )
            raise ValueError(f"Providers for agent '{agent_name}' declare inconsistent config types: {details}")

        providers_by_backend: dict[str, AgentProvider[Any]] = {}
        for provider in agent_providers:
            if provider.train_backend in providers_by_backend:
                raise ValueError(
                    f"Duplicate agent provider for agent '{agent_name}' and train backend '{provider.train_backend}'."
                )
            providers_by_backend[provider.train_backend] = provider
        index[agent_name] = AgentRegistration(
            config_type=next(iter(config_types)),
            providers=providers_by_backend,
        )
    return index


_frameworks: dict[str, RlFramework] = {}


def install_algo_schema(rllib: str, algo: str, config_type: type[Any]) -> None:
    """Install one framework-owned algorithm config type in Hydra's ConfigStore."""
    ConfigStore.instance().store(
        group=ALGO_GROUP,
        name=f"_{rllib}_{algo}_schema",
        node=config_type,
    )


def register_framework(framework: RlFramework) -> RlFramework:
    """Register a framework, its providers, and their Hydra config schemas."""
    if not isinstance(framework.name, str) or not framework.name:
        raise ValueError(f"{type(framework).__name__} must declare a non-empty name")
    existing = _frameworks.get(framework.name)
    if existing is framework:
        return framework
    if existing is not None:
        raise ValueError(f"RL framework '{framework.name}' is already registered.")

    for agent_name in framework.supported_agents():
        install_algo_schema(framework.name, agent_name, framework.get_config_type(agent_name))
    _frameworks[framework.name] = framework
    return framework


def get_framework(name: str) -> RlFramework:
    """Return a registered RL framework integration."""
    if name not in _frameworks:
        raise ValueError(f"RL framework '{name}' is not registered.")
    return _frameworks[name]


def iter_frameworks() -> tuple[RlFramework, ...]:
    """Return registered frameworks in stable order."""
    return tuple(_frameworks[name] for name in sorted(_frameworks))


def agent_exists(framework_name: str, agent_name: str, train_backend: str) -> bool:
    """Return whether a framework has an agent for a train backend."""
    return get_agent_provider(framework_name, agent_name, train_backend) is not None


def supported_train_backends(framework_name: str, agent_name: str) -> tuple[str, ...]:
    """Return train backends declared by a registered framework."""
    return get_framework(framework_name).supported_train_backends(agent_name)


def get_config_type(framework_name: str, agent_name: str) -> type[Any]:
    """Return an algorithm config type without maintaining a parallel registry."""
    return get_framework(framework_name).get_config_type(agent_name)


def get_agent_provider(framework_name: str, agent_name: str, train_backend: str) -> AgentProvider[Any] | None:
    """Return a framework-scoped agent provider, if any."""
    try:
        framework = get_framework(framework_name)
    except ValueError:
        return None
    return framework.get_agent_provider(agent_name, train_backend)


def supported_agents(framework_name: str) -> tuple[str, ...]:
    """Return agent/algorithm names declared by a registered framework."""
    return get_framework(framework_name).supported_agents()


def iter_agent_providers(
    framework_name: str | None = None,
    train_backend: str | None = None,
    agent_name: str | None = None,
) -> tuple[AgentProvider[Any], ...]:
    """Iterate providers selected by framework, backend, and agent filters."""
    framework_names = (
        (framework_name,) if framework_name is not None else tuple(framework.name for framework in iter_frameworks())
    )
    items = []
    for name in framework_names:
        framework = get_framework(name)
        for candidate_agent in framework.supported_agents():
            if agent_name is not None and candidate_agent != agent_name:
                continue
            for backend in framework.supported_train_backends(candidate_agent):
                if train_backend is not None and backend != train_backend:
                    continue
                provider = framework.get_agent_provider(candidate_agent, backend)
                if provider is not None:
                    items.append((name, backend, candidate_agent, provider))
    return tuple(
        provider
        for _, _, _, provider in sorted(
            items,
            key=lambda item: (item[0], item[1], item[2]),
        )
    )


def exists(name: str) -> bool:
    return name in _frameworks


def create_trainer(context: TrainerContext[object]) -> TrainerHandle:
    metadata = context.run.metadata
    provider = get_agent_provider(metadata.rllib, metadata.algo, metadata.train_backend)
    if provider is None:
        raise ValueError(
            f"No agent provider found for RL framework '{metadata.rllib}', train backend "
            f"'{metadata.train_backend}', agent '{metadata.algo}'."
        )
    typed_cfg = provider.validate_config(context.rl_cfg)
    typed_context = replace(context, rl_cfg=typed_cfg)
    typed_context = _validated_trainer_context(provider, typed_context)
    trainer = provider.create_trainer(typed_context)
    return TrainerHandle(run=context.run, trainer=trainer)


def create_trainer_context(
    run: runs.RunContext,
    *,
    num_envs: int,
    play_num_envs: int,
    seed: int | None,
    rl_cfg: CfgT,
    logging: LoggingConfig,
    checkpoint: CheckpointConfig,
    render: RenderConfig | None = None,
    resume_from: str | None = None,
) -> TrainerContext[CfgT]:
    _validate_runtime_config(logging, checkpoint)
    metadata = run.metadata
    return TrainerContext(
        run=run,
        env_name=metadata.env_name,
        run_dir=run.run_dir,
        checkpoint_dir=run.checkpoint_dir,
        sim=run.sim,
        checkpoint_format=metadata.checkpoint_format,
        num_envs=num_envs,
        play_num_envs=play_num_envs,
        seed=seed,
        rl_cfg=rl_cfg,
        logging=logging,
        checkpoint=checkpoint,
        render=render,
        resume_from=resume_from,
    )


def _validate_runtime_config(logging: LoggingConfig, checkpoint: CheckpointConfig) -> None:
    if not logging.backend:
        raise ValueError("logging.backend must be non-empty")
    if logging.interval <= 0:
        raise ValueError("logging.interval must be positive")
    if checkpoint.interval < 0:
        raise ValueError("checkpoint.interval must be non-negative")


def _validated_trainer_context(provider: AgentProvider[Any], context: TrainerContext[CfgT]) -> TrainerContext[CfgT]:
    metadata = context.run.metadata
    if provider.train_backend != metadata.train_backend or provider.agent_name != metadata.algo:
        raise ValueError(
            f"AgentProvider ({provider.train_backend}, {provider.agent_name}) does not match run metadata "
            f"({metadata.train_backend}, {metadata.algo})."
        )
    checkpoint_format = context.checkpoint_format or provider.checkpoint_format or ""
    if checkpoint_format != context.checkpoint_format:
        return replace(context, checkpoint_format=checkpoint_format)
    return context
