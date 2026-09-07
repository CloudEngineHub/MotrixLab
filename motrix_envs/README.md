# Motrix Environments

`motrix-envs` contains the built-in MotrixLab reinforcement learning environments and their robot models, meshes,
textures, and motion data. It depends on the standalone `motrix-env-core` framework package.

Importing `motrix_envs` registers all built-in environments in the core registry:

```python
from motrix_envs import registry

env = registry.make("cartpole", num_envs=1)
```

The package includes the `basic`, `locomotion`, and `manipulation` environment families. Environment names used by
training and playback commands remain unchanged.

Install the `deploy` extra and import `motrix_envs.deploy` to register the deployment profile compilers owned by the
built-in environments:

```python
from motrix_envs.deploy import build_deployment_profile

profile = build_deployment_profile("go2-walk-rough")
```

## Framework API

`motrix_envs.core` mirrors the complete public API exported by `motrix_env_core`:

```python
from motrix_envs.core import EnvCfg, NpEnv, SceneCfg, configclass, registry
```

Projects that only need the environment framework should install `motrix-env-core` and import it directly:

```python
from motrix_env_core import EnvCfg, NpEnv, SceneCfg, configclass, registry
```

Framework imports such as `motrix_envs.np.env` move to `motrix_env_core` or the `motrix_envs.core` facade. Concrete
environment paths under `motrix_envs.basic`, `motrix_envs.locomotion`, and `motrix_envs.manipulation` remain stable.

## Registration

The framework never imports or discovers concrete environment packages. Each implementation package owns its
registration side effects. For the built-in environments this happens when `motrix_envs` is imported.
