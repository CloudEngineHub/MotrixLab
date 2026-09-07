**语言**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

`MotrixLab` 是一个基于 [MotrixSim](https://github.com/Motphys/motrixsim-docs) 仿真引擎的强化学习框架，专为机器人仿真和训练设计。该项目提供了一个完整的强化学习开发平台，集成了多种仿真环境和训练框架。

## 项目概述

该项目由十个 workspace package 组成：

- **motrix_deploy**（`motrix-deploy`）：独立于训练框架的 artifact、backend、policy、控制循环、registry 与 CLI
- **motrix_deploy_mujoco**（`motrix-deploy-mujoco`）：MuJoCo 部署 backend plugin
- **motrix_deploy_unitree**（`motrix-deploy-unitree`）：Unitree SDK2 DDS 硬件 backend plugin
- **motrix_deploy_tasks**（`motrix-deploy-tasks`）：具体的带版本部署任务实现与可执行入口 bootstrap
- **motrix_env_core**（`motrix-env-core`）：环境基类、配置、registry、场景构建、NumPy runtime 与渲染能力，不包含任何内置任务和机器人资产
- **motrix_env_motrixsim**（`motrix-env-motrixsim`）：MotrixSim 实时仿真 backend、renderer 与 torch frontend
- **motrix_env_mujoco**（`motrix-env-mujoco`）：仅负责编译场景的 MuJoCo backend
- **motrix_envs**（`motrix-envs`）：内置环境、模型、数据及环境到部署 profile 的编译实现
- **motrix_rl**（`motrix-rl`）：基于 `motrix-env-core` 的 RL 框架集成，支持 SKRL、RSLRL 和 FastSAC

> 文档地址：https://motrixlab.readthedocs.io

## 主要特性

- **统一接口**: 提供简洁统一的强化学习训练和评估接口
- **多框架支持**: 支持 SKRL（JAX/PyTorch）、RSLRL（PyTorch）和内置 FastSAC 实现
- **丰富环境**: 包含基础控制、运动、操作等多种机器人仿真环境
- **高性能仿真**: 基于 MotrixSim 的高性能物理仿真引擎
- **可视化训练**: 支持实时渲染和训练过程可视化

## 🚀 快速开始

> 以下示例使用了 Python 项目管理工具：[UV](https://docs.astral.sh/uv/)
>
> 在开始之前，请先[安装](https://docs.astral.sh/uv/getting-started/installation/)该工具。

### 克隆仓库

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### 安装依赖

安装全部依赖：

```bash
uv sync --all-packages --all-groups --all-extras
```

外部项目如果只需要环境 framework，可单独安装 `motrix-env-core`；需要 MotrixLab 内置任务和资产时再安装
`motrix-envs`。

SKRL 框架支持 JAX(Flax)或 PyTorch 作为训练后端，您也可以根据自己的设备环境，选择只安装其中一种训练后端：

安装 JAX 作为训练后端（仅支持 Linux 平台）：

```bash
uv sync --all-packages --extra skrl-jax
```

安装 PyTorch 作为训练后端：

```bash
uv sync --all-packages --extra skrl-torch
```

安装 RSLRL 框架（仅支持 PyTorch 后端）：

```bash
uv sync --all-packages --extra rslrl
```

### 开发检查

开发依赖已包含 [`dprint-py`](https://pypi.org/project/dprint-py/)。安装
[`prek`](https://github.com/j178/prek)、启用 Git pre-commit hook，并运行所有已配置的检查：

```bash
uv tool install prek==0.5.2
prek install
prek run --all-files
```

需要单独运行 dprint 时，通过 uv 执行：

```bash
uv run dprint fmt
```

## 🎯 使用指南

### 环境可视化

查看环境而不执行训练：

```bash
uv run scripts/view.py env=cartpole
```

在静态标准场景中查看内置机器人：

```bash
uv run scripts/view.py robot=g1-29dof
```

可用的内置机器人名称包括 `dex-evt`、`g1-29dof`、`go1`、`go2` 和 `k1`。

### 训练模型

使用默认的 Cartpole SKRL 任务训练：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo
```

使用 RSLRL 框架训练：

```bash
uv run scripts/train.py task=cartpole/rslrl.ppo
```

通过 Hydra 直接覆盖运行参数和算法参数：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo num_envs=64 algo.agent.learning_rate=1e-3
uv run scripts/train.py task=cartpole/skrl.ppo logging.interval=20 checkpoint.interval=100
```

训练结果会保存在 `runs/{env-name}/` 目录下。

通过 TensorBoard 查看训练数据：

```bash
uv run tensorboard --logdir runs/{env-name}
```

### 模型推理

```bash
uv run scripts/play.py env=cartpole
```

更多使用方式请参考[用户文档](https://motrixlab.readthedocs.io)

## 📬 联系方式

有问题或建议？欢迎通过以下方式联系我们：

- GitHub Issues: [提交问题](https://github.com/Motphys/MotrixLab/issues)
- Discussions: [加入讨论](https://github.com/Motphys/MotrixLab/discussions)

## 项目规范

- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [支持与提问](SUPPORT.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [更新日志](CHANGELOG.md)
- [引用信息](CITATION.cff)
- [许可证](LICENSE)
