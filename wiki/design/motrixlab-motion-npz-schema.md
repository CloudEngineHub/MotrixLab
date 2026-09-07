# MotrixLab Motion NPZ Schema 设计

## 摘要

本文定义 MotrixLab 自己的 WBT (whole-body tracking) `.npz` 动作文件格式 v1,作为运动学/动力学 replay、训练 reference 命令、reward 计算的统一数据源。schema 在 BeyondMimic (`HybridRobotics/whole_body_tracking`) 原生 npz 基础上补充 `joint_names / body_names / schema_version` 等字段,并将四元数约定切到 xyzw 与 MotrixSim 内部 API 对齐。其它仓库 (Holosoma、AMP、LAFAN retarget 等) 的 npz 通过一次性 convert 脚本转入本格式。

## 背景与动机

当前 `motrix_envs/src/motrix_envs/locomotion/g1/wbt_np.py` 的 `G1WbtMotionLoader` 直接读取 Holosoma 风格 npz (MuJoCo qpos layout、wxyz 四元数、root 拼在 `joint_pos[:, :7]`),schema 隐式埋在 loader 代码里。这带来几个问题:

- 字段约定 (root prefix、quat 顺序、joint order) 没有文档化,只能读 loader 源码反推
- 来自其它生态 (BeyondMimic 原生、AMP、IsaacGymEnvs) 的 npz 无法直接读
- `scripts/motion/replay.py` 这类诊断工具只能针对 G1 WBT 专用 schema 写
- 想验证 retarget 正确性、对比 mocap 与 sim,缺统一中间表示

本设计为这些问题提供单一来源:**MotrixLab schema v1** + **converter 工具链** + **通用 loader API**。

## 范围

**当前覆盖**:
- G1 29-DoF humanoid 的 WBT 任务 (`g1-29dof-wbt-largebox`, `g1-wbt-dance`)
- Holosoma npz → MotrixLab npz 的一次性 converter
- 通用 `MotrixMotion` loader class,替代当前 `G1WbtMotionLoader` 的核心字段解析职责

**未来扩展槽位 (schema 已留,实现待后续)**:
- 物体 manipulation (object 字段作为 `ext_` extension)
- 多 clip 数据集调度 (一个文件一 clip,dataset 层管多文件)
- 其它来源 converter (AMP / IsaacGymEnvs / 自定义 retarget)

**不在范围**:
- 实时录制 motion (mocap pipeline)
- BVH / FBX / USD 等非 npz 格式 (retarget pipeline 的事)
- 跨机器人 retarget

## Schema 定义

所有数组默认 float32。四元数约定 **xyzw**。世界系:右手系 Z-up。一个 `.npz` 文件 = 一个 motion clip。

### 必需字段

| 字段 | Shape | Dtype | 说明 |
|---|---|---|---|
| `schema_version` | `()` | int32 | 当前 = `1`。schema 不兼容变更时 bump。 |
| `fps` | `()` | int32 | 帧率 (标量,不是 shape=(1,))。 |
| `num_frames` | `()` | int32 | 显式帧数 `T`,必须等于所有 per-frame 数组的第 0 维。 |
| `joint_names` | `(N,)` | str array | Joint 名,用于 name-based 绑定到 model。 |
| `body_names` | `(B,)` | str array | 所有 robot link 名 (含 root)。 |
| `joint_pos` | `(T, N)` | float32 | DOF 角度,**不含 root prefix**。 |
| `joint_vel` | `(T, N)` | float32 | DOF 速度。 |
| `body_pos_w` | `(T, B, 3)` | float32 | 所有 body 的世界系位置。Root pose 在 `[:, root_body_idx]`。 |
| `body_quat_w` | `(T, B, 4)` | float32 | 所有 body 世界系旋转,**xyzw**。 |
| `body_lin_vel_w` | `(T, B, 3)` | float32 | |
| `body_ang_vel_w` | `(T, B, 3)` | float32 | |

### 可选字段

| 字段 | Shape | Dtype | 说明 |
|---|---|---|---|
| `tracked_body_names` | `(K,)` | str array | `body_names` 子集,WBT tracking reward 的目标 body。未提供时由 env cfg 决定。 |
| `reference_body_name` | `()` | str | 相对 pose command 的参考系 body (例如 torso)。 |
| `root_body_name` | `()` | str | Root body 名,默认取 `body_names[0]` 或 env cfg 决定。 |
| `clip_name` | `()` | str | 可读 clip 标签。 |

### 扩展字段

任何 `ext_` 前缀字段都允许。Loader 把它们收集到 `motion.extensions: dict[str, np.ndarray]`,core 不解释。Task subclass 按需消费。

第一个规划的扩展是物体 manipulation:

- `ext_object_pos_w (T, 3)`、`ext_object_quat_w (T, 4)` xyzw
- `ext_object_lin_vel_w (T, 3)`、`ext_object_ang_vel_w (T, 3)`
- `ext_object_name ()` str (绑定到 scene XML 的 geom 名)

### 与 BeyondMimic 原生 npz 的关系

BeyondMimic `whole_body_tracking/scripts/csv_to_npz.py` 输出的 7 个字段 (`fps / joint_pos / joint_vel / body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w`) 在本 schema 里完全保留 (除了 quat 顺序从 wxyz 改成 xyzw)。

本 schema 相对 BeyondMimic 的纯加字段:
- 加 `schema_version / num_frames / joint_names / body_names` (必需)
- 加 `tracked_body_names / reference_body_name / root_body_name / clip_name` (可选)
- 加 `ext_*` 扩展槽

**不兼容的唯一一点:quat 顺序**。BeyondMimic/Holosoma 存 wxyz (跟 Isaac / MuJoCo 习惯一致),我们存 xyzw (跟 MotrixSim `link.get_rotation()`、`motrix_env_core.math.quaternion`、PyTorch、ROS 一致)。Converter 做一次性 `[:, [1, 2, 3, 0]]` 翻转。

## Loader API

新建 `motrix_envs/src/motrix_envs/motion/` 包:

```
motrix_envs/src/motrix_envs/motion/
├── __init__.py            # re-export MotrixMotion
├── schema.py              # REQUIRED_FIELDS、OPTIONAL_FIELDS、SCHEMA_VERSION 常量
└── loader.py              # MotrixMotion class
```

### `MotrixMotion` class

负责 schema 解析、字段校验、name-based index 构建。不依赖任何 sim model,纯 numpy 数据对象。

```python
class MotrixMotion:
    def __init__(self, path: str | Path):
        # 加载 npz,验证 schema_version、必需字段、shape 一致性、quat 单位范数
        ...

    # Per-frame states
    fps: int
    num_frames: int
    joint_names: list[str]
    body_names: list[str]
    joint_pos: np.ndarray      # (T, N) float32
    joint_vel: np.ndarray      # (T, N) float32
    body_pos_w: np.ndarray     # (T, B, 3) float32
    body_quat_w: np.ndarray    # (T, B, 4) float32, xyzw
    body_lin_vel_w: np.ndarray # (T, B, 3) float32
    body_ang_vel_w: np.ndarray # (T, B, 3) float32

    # Optional (None if absent)
    tracked_body_names: list[str] | None
    reference_body_name: str | None
    root_body_name: str | None
    clip_name: str | None

    # Extension catch-all
    extensions: dict[str, np.ndarray]

    # Helpers
    def body_index(self, name: str) -> int
    def joint_index(self, name: str) -> int
    # 返回只含子集 body 的 view (用 dataclass MotionSlice 或 namedtuple 封装)
    def select_bodies(self, names: list[str]) -> MotionSlice
```

### WBT view

`WbtMotion` 通过组合已加载的 `MotrixMotion` 提供 WBT 任务数据，并在初始化时一次性完成所有名称索引与重排：

```python
class WbtMotion:
    """WBT-specific computed properties over a MotrixMotion."""

    def __init__(self, motion, joint_names_model, tracked_body_names, reference_body_name, root_body_name):
        self.motion = motion
        # name-based index 重建:env model 的 joint order -> motion joint order
        joint_idx = motion.joint_indices(joint_names_model)
        tracked_idx = motion.body_indices(tracked_body_names)

        # 派生数组在构造时一次性建立，runtime 只需按 frame 索引
        self.joint_pos = motion.joint_pos[:, joint_idx]
        self.joint_vel = motion.joint_vel[:, joint_idx]
        self.body_pos_w = motion.body_pos_w[:, tracked_idx]
        self.body_quat_w = motion.body_quat_w[:, tracked_idx]
        # body velocity、root/reference 数组同理
        self.num_frames = motion.num_frames
```

`G129dofWbtTask.__init__` 把 `self.motion = G1WbtMotionLoader(...)` 改成 `self.motion = WbtMotion(...)`。task 其它代码通过 motion 的派生数组取数，改动面小。`WbtMotion` 的字段名直接表达消费侧语义：`joint_pos / joint_vel` 已按 model joint order 排列，`body_*_w` 仅包含 tracked-body 子集。原始 schema 数据可通过 `motion.motion` 访问。

**Task callsite 改名清单** (语义不变,只是 property 名变了):

| 原 (`G1WbtMotionLoader`) | 新 (`WbtMotion`) |
|---|---|
| `motion.joint_pos` | `motion.joint_pos`（按 model joint order） |
| `motion.joint_vel` | `motion.joint_vel`（按 model joint order） |
| `motion.body_pos_w` | `motion.body_pos_w`（tracked-body 子集） |
| `motion.body_quat_w` | `motion.body_quat_w`（tracked-body 子集） |
| `motion.body_lin_vel_w` | `motion.body_lin_vel_w`（tracked-body 子集） |
| `motion.body_ang_vel_w` | `motion.body_ang_vel_w`（tracked-body 子集） |
| `motion.root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_w` | 同名,保留 |
| `motion.ref_pos_w / ref_quat_w` | 同名,保留 |
| `motion.time_step_total` | `motion.num_frames` |

`MotrixMotion` 中的 `body_pos_w / body_quat_w / ...` 始终是 **schema 层的全 body 字段** (motion file order,所有 B 个 body)；`WbtMotion` 中的同名字段则是初始化时索引好的 tracked-body 子集。

## Converter 模式

### 目录结构

```
motrix_envs/tools/motion_converters/
├── __init__.py
├── base.py                  # Converter protocol、common helpers
├── holosoma_converter.py    # Holosoma -> MotrixLab
├── beyondmimic_converter.py # BeyondMimic 原生 -> MotrixLab (主要是 quat 翻转 + 加 names)
└── (future) amp_converter.py

scripts/motion/
├── convert.py               # 格式转换 CLI 入口
├── download_lafan.py        # 下载 LAFAN1 G1 动作并 bake 成 MotrixLab v1
└── replay.py                # kinematic replay 诊断工具
```

### CLI

```bash
uv run scripts/motion/convert.py \
    --from holosoma \
    --input /path/to/source.npz \
    --output /path/to/dest.npz \
    [--robot g1-29dof]          # 提供 joint_names/body_names 模板,某些来源没存 names 时用
```

`--from` 决定调用哪个 converter module。每个 converter 实现 `convert(input_path, output_path, **opts)` 函数,做以下事情:

1. 读源 npz
2. 拆分 root/joint (Holosoma 的 qpos layout → 拆开)
3. quat wxyz → xyzw
4. 补 `schema_version / num_frames / joint_names / body_names`
5. 保留 `ext_*` 字段
6. 写 dest npz (float32)

### 一次性迁移

转换后的动作文件与对应机器人环境实现放在一起，例如 G1 文件位于
`motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/`。Dex-EVT、K1 等机器人在
`wbt/assets/motion/` 下使用各自的子目录，让 motion 与 WBT 任务实现保持在同一模块边界内。

## Replay 脚本集成

`scripts/motion/replay.py` 更新:

- 不再硬编码 `G1WbtMotionLoader` import
- 直接用 `WbtMotion` API (`motion.root_pos_w / root_quat_w / joint_pos`)
- 支持任何符合 schema 的 npz 文件,不限于 G1 WBT env

## 测试

- `tests/motion/test_schema_validation.py`:schema 校验逻辑 (必需字段、shape 一致性、quat 范数)
- `tests/motion/test_motion_loader.py`:load + 索引 + select_bodies 行为

## 不在范围

- BVH / FBX / USD 等非 npz 格式
- 跨机器人 retarget
- 实时 motion 录制
- 多 clip 数据集调度 (一个文件 = 一个 clip;dataset 层另建)
- 物体字段的具体实现 (只定 schema 槽位,实现等 largebox WBT 任务)
