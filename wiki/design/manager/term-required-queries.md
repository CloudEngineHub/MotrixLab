# Term 自声明 Sim Query 设计（required_queries 协议）

## 摘要

本设计解决通用 MDP term（框架库中的 `RobotDofPosRelObsCfg` 等）与任务侧 sim query 声明之间的跨层 key 冲突：框架 term 的 kernel 以编译期字面量 `ctx.sim["robot_dof_pos"]` 消费 sim 数据，而该 key 目前的唯一声明点在任务的 `queries.data` 里，契约完全隐式。方案是给 term cfg 增加自声明协议 `required_queries()`——term 自己声明需要的 data/model query，构造期与任务声明按"key 单一属主、任务优先、term 贡献兜底、冲突报错"合并；相等 query 由 backend 既有的 exact-equal 折叠自动 alias，不产生重复物理读。kernel 访问方式（literal `ctx.sim[key]`）、numba lowering、codegen 全部不变。

本文描述已落地的设计；`required_queries()` 协议和 data/model query 合并已实现。manager kernel 只为环境自身配置构建；同一 runtime 不支持重建另一份 manager 配置。

## 1. 问题

现状数据流（三个事实）：

1. **key 的声明权在任务层**。`SimDataQueriesCfg` 字段名即 query key，查询参数（关节集合与顺序、link 名）来自任务 authority，如 `WbtSimQueriesCfg.robot_dof_pos = JointPositionQuery(joints=resolve_wbt_joint_names)`。
2. **key 的消费权在框架层，且被编译期钉死**。`motrix_env_core.mdp.observations` 的 kernel 全部使用 `ctx.sim["robot_dof_pos"]` 字面量；`Map.__getitem__` 的 numba lowering（`prefer_literal` overload）要求 key 是 `StringLiteral`，即 key 必须出现在 kernel 源码里，无法参数化。
3. **契约隐式**。框架没有任何声明指出"我需要哪些 key"，任务作者只能阅读框架 kernel 源码发现约定，唯一保障是构造期 query 访问失败。

冲突本质：同一个字符串 key 承担两个角色——读程序中的布线身份（理应任务所有）与通用量的语义标识（框架所有）——而名字的归属被劈成两层。更深一层的结构性缺陷是：契约面随框架 term 数量线性增长，任何"任务侧集中声明框架所需 key"的方案（如框架发布 canonical query group 基类让任务继承）都会把框架每次新增 term 的成本转嫁给所有任务。

## 2. 目标与非目标

目标：

- 框架通用 term 自带输入声明；框架新增 term 不要求任务侧新增任何声明字段。
- 有自身 authority 的任务（如 WBT 的 motion joint 顺序）保持现有声明方式与语义，零迁移。
- 相同 query 在不同 key 下自动共享物理读（alias），无重复读取。
- data query 与 model query 统一支持。
- kernel 访问路径、numba ABI、codegen、编译 fingerprint 机制全部不变。

非目标：

- 不引入 key 参数化/注入路径（不改 [sim-read-program](./sim-read-program.md) §7 "literal `ctx.sim[key]`，不提供参数注入路径"的决策）。
- 不支持单环境多机器人的 per-entity key 命名空间；canonical key 是单数角色（"the robot"）。
- 不改变 "key 等于声明字段名、不维护第二份 registry" 的既有契约，只扩展声明的来源。

## 3. 协议设计

`ObservationTermCfg` 新增一个默认为空的协议方法：

```python
@configclass(kw_only=True)
class ObservationTermCfg(abc.ABC):
    def required_queries(self) -> SimQueriesCfg:
        """本 term 需要的 sim 输入声明。

        data query 的 key 必须与 kernel 中的 ``ctx.sim`` 字面量一致；
        model query 的 key 必须与 ``__call__`` 中 ``env.model`` 的读取 key 一致。
        返回值必须是 cfg 级可计算的（允许 ``lambda cfg:`` 晚绑定字段），不得依赖已构造的 env。
        """
        return SimQueriesCfg()
```

框架通用机器人 term 各自实现（示例）：

```python
def _default_robot_dof_pos_query() -> SimDataQuery:
    return JointPositionQuery(joints=lambda cfg: resolve_robot_joint_names(cfg))  # 模块级共享


class RobotJointPosObsCfg(ObservationTermCfg):
    def required_queries(self) -> SimQueriesCfg:
        return SimQueriesCfg(data={"robot_dof_pos": _default_robot_dof_pos_query()})
```

约束：

- **同一 key 的所有贡献者必须共享同一个模块级默认构造函数**，保证贡献的 query 在解析后精确相等（见 §5）。
- 默认 query 从 `scene.objs.robot` 推导（全关节 / `resolved_base_link_name`），经 `lambda cfg:` 晚绑定求值；机器人 DOF 默认使用 `BodyJointPositionQuery` / `BodyJointVelocityQuery`，由 `resolved_base_link_name` 确定 body，不需要在 `RobotCfg` 中重复维护关节序。
- 协议本期只加在 `ObservationTermCfg`（当前唯一需求方）。收集循环按 observation groups 实现；未来核心层新增通用 reward/termination 时将同一方法加到对应 cfg base，收集点按需扩展，不预留抽象。

## 4. 合并规则：key 单一属主，任务优先

唯一收集点是 `ManagerBasedEnvCfg` 的两个 cfg 级方法（均在 backend 编译之前执行；方法内部先使用现有 resolver 得到 concrete query）：

- data：`sim_query_cfgs()` = `queries.data_dict()` ∪ Σ observation term 的 `required_queries().data_dict()`；
- model：`model_query_cfgs()` = `queries.model` ∪ Σ observation term 的 `required_queries().model`。

两侧都先经现有 resolver 把 `lambda cfg:` 解析为 concrete query，再按键合并：

| 情形 | 结果 |
| --- | --- |
| 任务已声明该 key | 与 term 声明重复，构造期失败；term 的特殊语义校验（例如相对关节位置的 key-pose 对齐）继续在 `__call__` 中执行 |
| 无人声明 | term 贡献自己的默认 query |
| 多个来源对同一 key 贡献不等 query | 构造期 `ValueError`，报错信息含 key、两侧 query repr 与贡献者路径（如 `observations.policy.dof_pos`） |

一个 key 仍然只对应一个 logical query，"rename key 是 compiled interface 变更"的既有契约不变；本设计只增加声明的第二来源，并让任务声明在语义上保持权威。

## 5. Alias 与去重语义

"相同 query 不同 key 自动共享"是 `PhysicsReadProgram` 的既有文档化不变量，本设计直接依赖、不新增机制：

- `compile_reads` 收到完整声明集（含重复），exact-equal 声明折叠到同一物理区域，每个 key 仍是可服务的别名，`view()` 返回同一视图对象（见 [读侧编译](./sim-read-program.md) §3/§5）。
- 典型场景：任务把某 query 声明在自定义名字下（如 `motion_dof_pos`），term 的 kernel 字面量是 `robot_dof_pos` 且无人声明该 key → term 贡献默认 query → 与任务声明相等 → 一次物理读、两个别名 key、零重复。
- 相等是 **lambda 解析后的 dataclass 精确相等**。`joints` 等序敏感字段顺序不同即不相等——那是不同数据（顺序即语义），本来就该各自读取，不算重复。
- model query 没有折叠契约（按 name 逐个编译），但它是构造期一次性 host 读，重复成本可忽略；且 §4 的单属主规则使同一 key 不会出现两条声明。

## 6. Model query 的绑定方式

model query 是静态数据，不经 `ctx.sim`，不参与每步刷新：term 在 `__call__(env)` 时从 `env.model` 读出结果并烘焙进 term 的 `@kernel_data` params（与 `RelativePositionParams.reference` 烘焙 key pose、WBT action 读取 `env.model.others["robot_joint_position_limits"]` 同构）。term 贡献的 model query key 与它烘焙时读取的 key 是同一个类级常量，天然一致。

## 7. Kernel 字面量的所有权翻转

实现本协议后：

- `ctx.sim["robot_dof_pos"]` 字面量不变，但该 key 由框架 term 的 `required_queries()` 声明进 read program——**字面量回到了它的声明者手里**，跨层依赖方向被理顺；
- 任务侧 kernel（WBT rewards/terminations/actions 读取 `robot_dof_pos` 等）行为不变：key 仍存在，数据仍来自同一声明；
- 单一 literal 约束（每 term 类一个 key）保持现状表达力：同一 term 类的两个实例若需要不同 query，仍不支持，与现状一致。

## 8. 与现有任务和文档的关系

- **WBT：零迁移。** `WbtSimQueriesCfg` 现有声明按 §4 任务优先规则继续胜出；可选择删除与框架默认相等的声明以精简，但非必需。`WbtManagerEnvCfg.__post_init__` 对 `queries.data` 具体类型的校验不受影响。
- **[IsaacLab ManagerCfg Adapter](./isaaclab-manager-cfg-adapter.md)：** 适配层把 `SceneEntityCfg` 式 selector 归一化为"去重后的显式 SimDataQuery"，本协议是该归一化在原生侧的落点——适配器生成的 concrete term cfg 携带自己的 `required_queries()`，无需适配器另行维护 key 分配。
- **[Manager Task API](./task-authoring.md) 与 [读侧编译](./sim-read-program.md)：** query 声明来源和合并规则分别在任务开发指南与本篇定义。
- **诊断：** `SimInputLayout.consumers` 可从常量 `"ManagerContext.sim"` 扩展为贡献者 cfg 路径（可选，不阻塞）。

## 9. 边界与已知限制

- **`_scene_robot_cfg` 的场景扫描**（`RobotDofPosRelObsCfg` 为 `BodyJointPositionQuery` 分支寻找 key-pose robot）是同类的隐式"the robot"角色，另行处理，不阻塞本设计。
- **多机器人**：单环境多机器人需要 per-entity key 命名空间或注入式绑定，届时另立设计；本设计明确不支持。

## 验收标准

- 使用框架通用机器人 term 的任务，若其默认 query 适用，`queries.data` / `queries.model` 中无需任何 robot 相关声明，环境可构造并训练。
- 任务对某 key 的显式声明总是胜出；term 行为与该声明一致；缺失 key 或 backend 无法解析 query 时在构造期失败。
- 相同 query 的不同 key 声明在 read program 中折叠为一次物理读，别名 `view()` 为同一对象（既有 backend 测试覆盖 + 新增协议路径用例）。
- 两个 term 对同一 key 贡献不等 query 时构造期报错，报错含 key 与双方来源。
- WBT 全部现有测试在零配置改动下通过。
- kernel 源码、numba lowering、codegen、`plan_key` 指纹机制无结构性变化。
