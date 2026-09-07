# ConfigClass 配置装饰器设计

## 摘要

`@configclass` 是建立在 Python 标准 dataclass 之上的配置类装饰器。它允许配置字段直接使用 list、dict、set 或嵌套 dataclass 实例作为默认值，并保证每次构造配置时都获得独立的对象，从而消除大量 `field(default_factory=...)` 样板代码。

装饰后的类仍然是标准 dataclass，保留 dataclass 的构造、继承、反射和类型检查语义，并可继续作为 Hydra/OmegaConf structured config 使用。`configclass` 只服务于声明式配置，不作为所有数据类的通用替代品。

## 背景

标准 dataclass 不允许直接声明常见的可变默认值。嵌套配置带有参数时，通常需要写成：

```python
@dataclass
class EnvCfg:
    sim: SimCfg = field(
        default_factory=lambda: SimCfg(
            dt=0.005,
            solver_iterations=3,
        )
    )
```

这段 factory 语法的目的只是为每个 `EnvCfg` 创建独立的默认对象，却让真正重要的配置结构变得难以阅读。

期望的声明方式是：

```python
@configclass
class EnvCfg:
    sim: SimCfg = SimCfg(
        dt=0.005,
        solver_iterations=3,
    )
```

两种写法应具有相同的逐实例隔离语义。

## 设计目标

- 配置字段能够直接使用普通 Python 值表达默认配置树。
- 每个配置实例拥有独立的可变容器和嵌套配置对象。
- 装饰结果仍是标准 dataclass，而不是新的配置对象模型。
- 保留 dataclass 的继承、字段覆写、`field()` 和装饰器参数语义。
- 保留静态类型检查器对 dataclass 构造函数的推导能力。
- 保持与 Hydra/OmegaConf structured config 的兼容性。
- 默认值自动处理规则应明确、有限且可预测。

## 非目标

- 不负责配置校验、类型转换、序列化或配置组合。
- 不根据类型标注自动创建调用方没有声明的默认值。
- 不把 list、dict 等字段改成不可变类型。
- 不自动把配置类变成 frozen dataclass。
- 不处理文件句柄、线程锁、设备 buffer 或其他 runtime resource。
- 不取代 Pydantic、attrs、Hydra-Zen 等具有不同建模目标的框架。

## 基本接口

装饰器支持与 `@dataclass` 相同的两种使用形式：

```python
@configclass
class SimCfg:
    dt: float = 0.01


@configclass(frozen=True, kw_only=True)
class AssetCfg:
    name: str
```

`configclass` 接受标准 dataclass 的装饰器参数，包括 `frozen`、`kw_only`、`slots`、`eq` 和 `order`。调用方不应在同一个类上同时叠加 `@configclass` 与 `@dataclass`。

## 默认值语义

### 自动隔离的默认值

当前 class body 中显式声明的下列默认值，会被转换为逐实例生成的默认值：

- `list`
- `dict`
- `set`
- `numpy.ndarray`
- dataclass 实例，包括 frozen dataclass 实例
- 包含上述任意对象的 tuple

其可观察语义等价于：

```python
template = declared_default
field(default_factory=lambda: deepcopy(template))
```

这里的 `template` 只作为默认配置模板。每次实例化都对模板执行深拷贝，因此嵌套容器也不能在不同配置实例之间共享。

例如：

```python
@configclass
class OptimizerCfg:
    betas: list[float] = [0.9, 0.999]
    options: dict[str, float] = {"eps": 1e-8}


a = OptimizerCfg()
b = OptimizerCfg()

assert a.betas is not b.betas
assert a.options is not b.options
```

嵌套配置同样遵循这一规则：

```python
@configclass
class TrainCfg:
    optimizer: OptimizerCfg = OptimizerCfg(
        options={"eps": 1e-6},
    )
```

不同 `TrainCfg` 实例中的 `optimizer`、`optimizer.betas` 和 `optimizer.options` 都是独立对象。

### 保持原样的字段

下列字段不进行自动改写：

- 没有默认值的必填字段
- `None`、bool、int、float、str、bytes、enum 等值语义默认值
- `ClassVar`
- `InitVar`
- 显式声明的 `field(...)`
- class、function 和其他 callable

调用方显式使用 `field()` 时，其声明具有最高优先级：

```python
@configclass
class CacheCfg:
    values: dict[str, float] = field(default_factory=dict, repr=False)
```

`configclass` 不改写这个字段，从而完整保留 `metadata`、`init`、`repr`、`compare` 和自定义 factory 等标准 dataclass 能力。

## 转换模型

`configclass` 在 class body 执行完成后、标准 dataclass 转换之前处理字段：

```text
class body
    │
    ▼
识别当前类显式声明的字段默认值
    │
    ├── 需要逐实例隔离 ──► deepcopy default factory
    └── 其他字段         ──► 保持原声明
    │
    ▼
标准 dataclass 转换
    │
    ▼
标准 dataclass type
```

只处理当前类新声明或重新赋值的字段。继承字段由基类已经完成的 dataclass 定义提供；子类重新声明字段时，按照子类的新默认值重新判断。

## Dataclass 兼容契约

装饰结果必须满足 `dataclasses.is_dataclass()`，并保留标准 dataclass 的可观察行为，包括：

- 自动生成的 `__init__`、比较与表示方法。
- `fields()`、`replace()`、`asdict()` 等标准 API。
- `__post_init__()` 调用。
- dataclass 字段继承、顺序和子类覆写。
- `frozen`、`kw_only`、`slots`、`eq`、`order` 等选项。
- 标准 dataclass 基类与 configclass 子类的组合。

`configclass` 基类的配置子类应继续使用 `@configclass`，否则子类中直接声明的可变默认值不会经过自动处理。

## 静态类型契约

静态类型检查器和 IDE 应将 `@configclass` 视为 dataclass-like decorator，并能够推导：

- annotated fields 对应的构造函数参数。
- 必填字段与带默认值字段。
- keyword-only 参数。
- 继承后的字段集合。
- 显式 `field()` 声明。

运行时对默认值的改写不能改变字段 annotation，也不能让调用侧看到 `Field` 类型。字段仍按其真实配置类型标注：

```python
sim: SimCfg = SimCfg(dt=0.005)
```

## Hydra / OmegaConf 兼容契约

由于最终结果是标准 dataclass，`configclass` schema 应继续支持 Hydra/OmegaConf 的 structured config 能力：

- 注册 dataclass type 作为 structured config schema。
- 从类或实例创建 structured config。
- 覆写标量、嵌套配置、list 和 dict 字段。
- 将配置转换回原始 dataclass 类型。
- 保留 required field、field metadata 和字段类型信息。

自动生成的 factory 与手写 `field(default_factory=...)` 对 Hydra/OmegaConf 应具有相同的可观察结果。

`@configclass(frozen=True)` 保留标准 frozen dataclass 语义。如果配置系统将 frozen structured config 视为只读，`configclass` 不应绕过或解除该限制。

## 错误与边界

- 自动处理的默认值必须能够被 `deepcopy`。
- 深拷贝失败时应抛出包含类名和字段名上下文的错误。
- 显式 `field()` 的 factory 错误保留标准 dataclass 行为。
- 需要共享的 singleton 不应声明为普通配置字段；应使用 `ClassVar` 或由 runtime owner 管理。
- 包含 runtime resource 的对象不属于自动复制的配置默认值。

## 使用范围

应使用 `@configclass` 的对象：

- 用户可配置的 schema。
- 由纯数据组成的嵌套配置。
- 需要通过 Hydra/OmegaConf 或相似系统覆写的 dataclass schema。

应继续使用 `@dataclass` 的对象：

- runtime state、result 和 metadata。
- 持有外部资源或具有明确生命周期的对象。
- 不属于配置模型的普通数据对象。

## 方案取舍

### 相比显式 default factory

`field(default_factory=...)` 仍然是底层语义和显式控制入口。`configclass` 只自动处理配置类中重复、机械的 factory 声明，让默认配置结构直接呈现在 class body 中。

### 相比 `cfg_default(...)` helper

helper 可以隐藏 lambda，但每个字段仍需要额外 wrapper。`configclass` 将逐实例复制规则统一放在类边界，字段只声明真实默认值。

### 相比 frozen 配置树

不可变配置可以从根本上避免共享修改，但会改变 list/dict 类型以及配置覆写行为。`configclass` 保留现有可变配置模型，只保证不同配置实例之间相互隔离。

### 相比新的配置框架

当前问题是 dataclass 默认值声明噪音，而不是缺少校验、解析或模型生成能力。保留标准 dataclass 可以避免引入另一套对象模型和互操作边界。

## 设计不变量

- 调用侧可以直接声明可变容器和嵌套配置默认值。
- 任意两个配置实例不共享自动处理的默认对象及其嵌套可变内容。
- 装饰结果始终是标准 dataclass。
- 显式 `field()` 始终优先于自动处理。
- 自动处理不改变字段类型和公开 schema。
- 非 frozen schema 的 Hydra/OmegaConf 覆写能力不受影响。
- `configclass` 只用于纯数据配置，不扩展到 runtime resource。

实现已完成：`configclass` 位于 `motrix_env_core.config`，相关迁移与 Hydra/OmegaConf 验证已纳入现有测试。
