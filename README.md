**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

`MotrixLab` is a reinforcement learning framework based on the [MotrixSim](https://github.com/Motphys/motrixsim-docs) simulation engine, designed specifically for robot simulation and training. This project provides a complete reinforcement learning development platform that integrates multiple simulation environments and training frameworks.

## Project Overview

The project is divided into ten workspace packages:

- **motrix_deploy** (`motrix-deploy`): Framework-independent artifact, backend, policy, control-loop, registry, and CLI
- **motrix_deploy_mujoco** (`motrix-deploy-mujoco`): MuJoCo deployment backend plugin
- **motrix_deploy_unitree** (`motrix-deploy-unitree`): Unitree SDK2 DDS hardware backend plugin
- **motrix_deploy_tasks** (`motrix-deploy-tasks`): Concrete versioned deployment tasks and executable bootstrap
- **motrix_env_core** (`motrix-env-core`): Environment base classes, configuration, registry, scene construction, NumPy runtime, and rendering. It contains no built-in tasks or robot assets
- **motrix_env_motrixsim** (`motrix-env-motrixsim`): Live MotrixSim backend, renderer, and torch frontend
- **motrix_env_mujoco** (`motrix-env-mujoco`): Compile-only MuJoCo scene backend
- **motrix_envs** (`motrix-envs`): Built-in environments, models, data, and environment-to-deployment-profile compilers
- **motrix_rl** (`motrix-rl`): RL-framework integration built against `motrix-env-core`, with SKRL, RSLRL, and FastSAC support

> Documentation: https://motrixlab.readthedocs.io

## Key Features

- **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
- **Multi-framework Support**: Supports SKRL (JAX/PyTorch), RSLRL (PyTorch), and the built-in FastSAC implementation
- **Rich Environments**: Includes various robot simulation environments such as basic control, locomotion, and manipulation tasks
- **High-performance Simulation**: Built on MotrixSim's high-performance physics simulation engine
- **Visual Training**: Supports real-time rendering and training process visualization

## 🚀 Quick Start

> The following examples use the Python project management tool: [UV](https://docs.astral.sh/uv/)
>
> Before starting, please [install](https://docs.astral.sh/uv/getting-started/installation/) this tool.

### Clone Repository

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### Install Dependencies

Install all dependencies:

```bash
uv sync --all-packages --all-groups --all-extras
```

For an external project that only needs the environment framework, install `motrix-env-core`. Install
`motrix-envs` when the built-in MotrixLab tasks and assets are also required.

SKRL framework supports JAX(Flax) or PyTorch as training backends. You can also choose to install only one training backend based on your hardware environment:

Install JAX as training backend (Linux only):

```bash
uv sync --all-packages --extra skrl-jax
```

Install PyTorch as training backend:

```bash
uv sync --all-packages --extra skrl-torch
```

Install RSLRL framework (PyTorch backend only):

```bash
uv sync --all-packages --extra rslrl
```

### Development Checks

[`dprint-py`](https://pypi.org/project/dprint-py/) is included in the development dependencies. Install
[`prek`](https://github.com/j178/prek), enable the Git pre-commit hook, and run all configured checks:

```bash
uv tool install prek==0.5.2
prek install
prek run --all-files
```

Run dprint directly through uv when needed:

```bash
uv run dprint fmt
```

## 🎯 Usage Guide

### Environment Visualization

View environments without executing training:

```bash
uv run scripts/view.py env=cartpole
```

View a built-in robot in a static standard scene:

```bash
uv run scripts/view.py robot=g1-29dof
```

Available built-in robot names are `dex-evt`, `g1-29dof`, `go1`, `go2`, and `k1`.

### Model Training

Train the default Cartpole SKRL task:

```bash
uv run scripts/train.py task=cartpole/skrl.ppo
```

Train with RSLRL framework:

```bash
uv run scripts/train.py task=cartpole/rslrl.ppo
```

Override runtime settings and algorithm parameters directly through Hydra:

```bash
uv run scripts/train.py task=cartpole/skrl.ppo num_envs=64 algo.agent.learning_rate=1e-3
uv run scripts/train.py task=cartpole/skrl.ppo logging.interval=20 checkpoint.interval=100
```

Training results are saved in the `runs/{env-name}/` directory.

View training data through TensorBoard:

```bash
uv run tensorboard --logdir runs/{env-name}
```

### Model Inference

```bash
uv run scripts/play.py env=cartpole
```

For more usage methods, please refer to the [User Documentation](https://motrixlab.readthedocs.io)

## 📬 Contact

Have questions or suggestions? Feel free to contact us through:

- GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
- Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)

## Project policies

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [License](LICENSE)
