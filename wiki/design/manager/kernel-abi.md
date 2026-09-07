# Manager Kernel ABI

## 摘要

本文定义 Manager fused Numba kernel 的数据 ABI：`@kernel_data` logical tree 如何 flatten，`ManagerContext` 如何按 lane
提供固定 namespace，以及 `PER_ENV`、`SHARED` 和 metric backing 的 ownership 语义。任务开发者只需遵守
[Manager Task API](./task-authoring.md) 中的 term 协议；backend 读侧和 simulator 边界见 [SimBackend](./sim-backend.md)。

## 1. ManagerContext

`ManagerContext` 是 compiled term 的唯一 manager/runtime 输入入口：

```python
@kernel_data
class ManagerContext:
    env_id: np.ndarray
    actions: Map
    observations: Map
    rewards: Map
    terminations: Map
    commands: Map
    rand: RandValue
    sim: Map
    dt: np.float32
```

Context 不包含 config、provider、backend handle、dynamic Python container 或 generic `states` / `resources` store。
Persistent runtime data 直接归属 concrete term；framework-owned random state 通过 `rand` 暴露，`dt` 提供当前
`ctrl_dt` 的固定 float32 标量。

## 2. Lane-local 语义

Host 侧 `env_id` 是：

```python
np.arange(num_envs, dtype=np.int64)
```

Generated kernel 在 `prange` loop 中按当前 environment id 重建 context。普通 step 和 partial reset 都使用原始 environment id，
因此 `ctx.env_id`、action/command state 和 simulator snapshot 始终属于同一 environment，不会把 reset batch row 当作 environment id。

所有 Map 的 key 顺序、concrete value schema、field tree、leaf dtype、scope 和 input slot 在编译前固定。Map lookup 必须使用 literal key：

```python
motion = ctx.commands["motion"]
dof_pos = ctx.sim["robot_dof_pos"]
```

动态字符串 key、运行期插入/删除和 fallback lookup 不属于 ABI；错误 key 应在 typing/lowering 或 warm-up 阶段失败。

## 3. Leaf scope

| 类型 | 默认 scope | lane 访问语义 |
| --- | --- | --- |
| `np.ndarray` | `PER_ENV` | 取当前 environment 的 row/view |
| `SharedArray` | `SHARED` | 所有 lane 访问完整可写 ndarray |
| Python scalar | `SHARED` | 所有 lane 使用同一个标量 |
| 固定 dtype NumPy scalar | `SHARED` | 所有 lane 使用同一个定 dtype 标量 |

`SharedArray` 只描述 lowering scope，不代表 immutable。数组是否可以更新由 concrete term 的 owner/lifecycle 契约决定。
Term receiver 的参数通常按 shared lowering 处理，以免把静态参数误切片。

## 4. Logical tree 与 flat ABI

`@kernel_data` 支持有序、可逆的 record tree：

```python
leaves, tree_def = flatten_kernel_data(value)
rebuilt = unflatten_kernel_data(tree_def, leaves)
```

Record 可以嵌套另一个 KernelData record；不支持 multiple inheritance、覆盖 inherited field、普通 dataclass、递归结构、dict、list、字符串、object array 或 backend handle。

Logical tree 与 physical lowering 分离：同一 concrete value 在 host term 和 context 中保留相同的 tree schema，但 lowering 根据使用位置决定 scope、slot 和 lane proxy。
Lowered ABI 只允许 ndarray/scalar leaves 跨越 `prange` boundary；nested proxy 在当前 lane 重建，不应产生稳态临时 ndarray 分配。

## 5. Ownership 与 metric

- config 只创建 term，不持有 environment-local mutable arrays；
- 每个 concrete term 只 canonicalize 一次，host 和 compiled path 共享 backing identity；
- action/command term 拥有自己的 persistent buffers；
- observation output 由调用方提供，generated kernel 在当前 invocation 写入；
- simulator snapshot 由 `PhysicsReadProgram` 拥有并写入，compiled consumer 只读；
- `metric()` 字段由 owning term 写入自己的当前 lane，host collector 负责聚合 scalar metric。

Term 可以读取其他 term 的数据，但不应调用其他 term 的行为方法。Context store 表达数据依赖，不承担 manager scheduling。

## 6. 编译与 fingerprint

Compiler 负责：

- flatten concrete runtime terms 和 `ManagerContext`；
- 收集 read program 的 sim input map；
- lowering context、term receiver 和 literal Map lookup；
- 校验方法签名、返回类型、dtype 和 layout；
- 生成 observation、reward、termination、command evaluation/step 和 partial-reset（含 command reset）kernels；
- 生成 layout/fingerprint，覆盖 concrete type、Map key、field tree、scope、sim binding 和 invocation schema。

构造期 warm-up 必须确认 generated dispatcher 的 nopython signature。schema rename、term type 变化、scope 变化和 query layout 变化都必须产生新的 fingerprint，避免错误复用 compiled kernel 或旧 Numba cache。

## 7. 核心不变量

- Context schema 在一个 environment 生命周期内固定；
- 每个 key 对应唯一 concrete type 和 stable slot；
- 所有 `PER_ENV` leaves 使用同一个 current lane id；
- `ctx.env_id` 在 step/reset 中与其他 lane-local data 对齐；
- simulator snapshots 只来自 compiled read program；
- 不建立 generic `states`、`resources` 或 `ManagerValue` registry；
- generated kernel 不接收完整 environment、config、info 或 simulator object；
- nopython、buffer identity 和 deterministic parity 属于 CI 验收，绝对性能和 scaling 属于独立 benchmark。
