# SimDataQuery Backend 接入指南

## 摘要

本文说明如何为 MotrixLab 接入一个 simulator backend 的读侧。Backend 在
`SimBackend.compile_reads` 中编译出一个可执行的 `PhysicsReadProgram`，runtime 反复执行它：

```text
SimBackend.compile_reads  ->  PhysicsReadProgram.execute
      （编译一次）                （每个 step / reset 执行）
```

- `compile_reads`：将框架预解析后的完整 `Mapping[str, SimDataQuery]` 声明集（可能含重复）
  降低为一个 `PhysicsReadProgram`——先做 backend-private 的名称/handle 解析
  （通常组合一个 `SimQueryResolver`），折叠相等声明，组装出自管内存布局的可执行程序；
- `PhysicsReadProgram`：可执行编译产物，物理内存布局完全由 program 自管：私有 arena
  （连续扁平 `float32` destination storage，不对外暴露，execute 是唯一写入者）、
  `view(key)`（按声明 key 返回 arena 上的稳定只读 logical view）、`query(key)`（该 key
  下被服务的已解析声明）、`keys`（全部被服务的声明 key）、`execute(env_ids)`（把
  simulator state 写入 arena 的指定 rows）和 `arena_bytes`（存储规模的诊断量）。
  Framework 的 inputs 直接来自 `view(key)`，因此 full-batch execute 直写 authoritative
  storage，无中间拷贝。

整体数据布局和 Manager 集成见 [读侧编译设计](./sim-read-program.md)。

## 1. 实现 `SimBackend.compile_reads`

读侧编译是 backend 的一个抽象成员；写入（包括 reset）通过 backend-owned `write_compiler.compile(...)` 完成：

```python
class MySimBackend(SimBackend):
    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram: ...
```

方法内部使用 backend 自己的 model 与 batched data source（构造时已编译好）。可以像
MotrixSim 一样把降级逻辑写成模块函数、方法一行委托，便于测试直接喂假 model/data：

```python
def compile_read_program(model, source, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
    resolver = _MySimQueryResolver(model)
    ...


class MySimBackend(SimBackend):
    def compile_reads(self, queries):
        return compile_read_program(self._model, self._data, queries)
```

`source.shape[0]` 是 backend 的 environment 数量，也是 physical output 的 leading
dimension。

## 2. 实现 query resolver

`SimDataQuery` 通过 `resolve_with()` 调用 `SimQueryResolver` 的 typed 方法。Backend 组合一个
backend-specific resolver：

```python
class _MySimQueryResolver(SimQueryResolver):
    def __init__(self, model) -> None:
        self._model = model

    def resolve_link_position(
        self,
        key: str,
        query: LinkPositionQuery,
    ) -> _ResolvedLinkQuery:
        link = self._model.get_link(query.link)
        if link is None:
            raise KeyError(f"Simulator query {key!r} references unknown link {query.link!r}.")
        return _ResolvedLinkQuery(
            key=key,
            trailing_shape=(3,),
            link_indices=(link.index,),
            field=_LinkField.POSITION,
        )
```

`compile_reads` 的输入已经由框架完成（`resolve_sim_queries`）：

- query key 校验；
- `EnvCfg` callable resolver 执行；
- `SimName` / `SimNames` 类型校验。

**duplicate 折叠由 backend 自己做**：相等声明折叠成 canonical 集后走 resolver，再让每个
声明 key 作为 canonical 视图的别名被服务。Resolver 需要完成名称到 backend
handle/index 的解析，并计算 logical shape。

当前 typed resolver API 包括：

```text
resolve_body_dof_position
resolve_body_dof_velocity
resolve_link_position
resolve_batch_link_position
resolve_link_quaternion
resolve_batch_link_quaternion
resolve_link_linear_velocity
resolve_batch_link_linear_velocity
resolve_link_angular_velocity
resolve_batch_link_angular_velocity
resolve_link_net_contact_force
resolve_batch_link_net_contact_force
resolve_body_link_net_contact_force
resolve_sensor_values
```

未实现的 typed 方法使用默认 unsupported 行为。

### Resolved query

Resolved query 只保存后续 physical planning 所需的信息：

```python
@dataclass(frozen=True)
class _ResolvedQuery:
    key: str
    trailing_shape: tuple[int, ...]


@dataclass(frozen=True)
class _ResolvedLinkQuery(_ResolvedQuery):
    link_indices: tuple[int, ...]
    field: _LinkField
```

例如 batch position：

```python
def resolve_batch_link_position(self, key, query):
    indices = tuple(self._get_link_index(name, key) for name in query.links)
    return _ResolvedLinkQuery(
        key=key,
        trailing_shape=(len(indices), 3),
        link_indices=indices,
        field=_LinkField.POSITION,
    )
```

## 3. 规划 physical program

`compile_reads` 完成名称解析（委托 resolver）后规划 physical program：

```python
def compile_read_program(model, source, queries):
    canonical, canonical_keys = _collapse_duplicates(queries)
    resolved = _MySimQueryResolver(model).resolve_queries(canonical)
    operations = _plan_physical_reads(resolved)
    return _MyReadProgram(source, declared=queries, canonical_keys=canonical_keys)
```

Planner 负责：

1. 按 backend field 和 handle/index 分组；
2. 选择 native 读取方式；
3. 合并可以由一次读取提供的 query；
4. 规划自有 arena 的内存布局与每个 query 的 view。

arena 是 program 私有的连续 destination storage（扁平 `float32[num_envs * 总行宽]`），在构造
program 时创建，不通过任何公共 API 暴露；native 执行能直写自有存储时直接复用该存储
（例如 plan-allocated program buffer），否则用 `np.empty` 分配。`arena_bytes` 属性返回其
字节数，供诊断。

`view(key)` 按声明 key 返回 arena 上 shape 为 `(num_envs, *trailing_shape)` 的只读 view；
同一 key 重复调用返回同一对象，视图在构造 program 时一次性构建并缓存；别名 key 直接映射
到 canonical 视图对象。

例如一次 native getter 同时返回 link position 和 quaternion，program 可以把它们排进同一行并暴露两个 views：

```text
physical row = [position(3), quaternion(4)]
row width     = 7
```

## 4. Arena 与 view（program 自管内存布局）

物理内存布局完全由 program 自己规划：program 在构造时创建自有 arena（扁平
`float32[num_envs * 总行宽]`）、决定每个 query 的数据落在哪一段、并为每个声明 key 构建一个只读 view：

```python
class _MyReadProgram(PhysicsReadProgram):
    def __init__(self, source, declared, canonical_keys) -> None:
        self._source = source
        self._declared = dict(declared)
        self._num_envs = int(source.shape[0])
        row_width = ...  # 由 planner 决定的每行宽度
        self._arena = np.empty((self._num_envs * row_width,), dtype=np.float32)
        self._views: dict[str, np.ndarray] = {}
        for canonical, alias in canonical_keys.items():
            if canonical not in self._views:
                spec = self._plan[canonical]
                view = np.ndarray(
                    shape=(self._num_envs, *spec.shape),
                    dtype=np.float32,
                    buffer=self._arena,
                    offset=spec.offset * 4,
                    strides=tuple(s * 4 for s in (row_width, *spec.strides)),
                )
                view.flags.writeable = False
                self._views[canonical] = view
            self._views[alias] = self._views[canonical]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._views)

    def query(self, key: str) -> SimDataQuery:
        return self._declared[key]

    def view(self, key: str) -> np.ndarray:
        return self._views[key]
```

规则：

- 每个声明 key 必须有且仅有一个 view（缺失 key 在 `view`/`inputs` 访问时报 `KeyError`）；
- view 必须是 float32 且 leading dimension 是 `num_envs`（program 自构造自校验）；
- view 只读（`writeable = False`），arena 的唯一写入者是 `execute`；
- framework 不创建或替换 views——inputs 直接来自 `program.view(key)`。

## 5. 实现 `PhysicsReadProgram.execute`

目的地内在于 program。建议单次调用覆盖 full 与 partial：

```python
class _MyReadProgram(PhysicsReadProgram):
    def execute(self, env_ids=None) -> None:
        if env_ids is not None and (env_ids.dtype != np.int64 or env_ids.ndim != 1):
            raise TypeError("Partial simulator read env_ids must be a one-dimensional int64 ndarray.")
        # env_ids 为 None 时写全部 rows，否则只写选中的 arena rows（env-indexed 直写）。
        for read, view in self._reads:  # program 自管的 (读取器, arena 行视图) 对
            read(self._source, view, env_ids)
```

### Full read

`env_ids is None` 表示读取所有 environments，直接写完整 arena。

### Partial read

`env_ids` 是一维 `int64` ndarray，保存需要更新的 global environment indices。Partial read
只能更新这些 rows：优先使用 native 的 env-indexed 写入（例如 MotrixSim 的
`program.execute(source, env_ids)`，单次 FFI 只 gather 选中 env、只写选中 rows，重复索引
幂等）；native 不支持时退化为 backend-private scratch + scatter 到 `output[env_ids]`。

## 6. 新增或支持一个 SimDataQuery

支持一个已有 query 时：

1. 在 resolver 中实现对应的 `resolve_*` 方法；
2. 定义或复用 resolved query 类型；
3. 在 planner 中生成对应 operation；
4. 在 operation 中声明 physical shape 和 logical views；
5. 在 execute 中调用 native getter；
6. 添加 full read、partial read 和 view 测试。

框架新增 query 时：

1. 定义新的 `SimDataQuery` dataclass；
2. 实现 `resolve_with()`，调用 `SimQueryResolver` 对应的 typed API；
3. 在 `SimQueryResolver` 添加默认 unsupported 方法；
4. 在需要支持的 backend resolver 中实现该方法；
5. 更新 planner、operation 和 execute；
6. 添加 backend 行为测试。

## 7. 接入验收

Backend 接入完成后应满足：

- 实现 `SimBackend.compile_reads`，输入为框架预解析的完整声明集（含重复）；
- 相等声明共享一个 physical read 与 view，所有声明 key 都被服务；
- 使用 resolver 处理 query 到 backend handle 的解析；
- physical outputs 是 program 私有 arena 的视图，program 是唯一写入者；
- full/partial execute 都遵守 environment row mapping；
- logical query views 的 shape、offset 和 stride 与 native output 一致；
- unsupported query、非法名称和 runtime layout 错误都有明确诊断；
- resolution、planning、view 和 full/partial read 的行为符合 framework contract。
