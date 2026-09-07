# 配置覆盖与调参

`HumanoidVelocityTrackingEnvCfg` 为通用速度跟踪逻辑提供默认参数。新增机器人任务时，通常需要根据机器人尺寸、关节范围、
执行器能力和训练表现覆盖部分子配置。本页介绍覆盖方式和各字段的调节重点。

## 1. 覆盖配置

定义平地环境配置时，可以直接传入需要覆盖的子配置，未传入的字段继续使用默认值：

```python
return HumanoidVelocityTrackingEnvCfg(
    scene=scene,
    asset=asset,
    control_config=ControlCfg(action_scale=0.5),
    commands=CommandsCfg(...),
    normalization=NormalizationCfg(...),
    gait=GaitCfg(...),
    reward_config=RewardCfg(...),
    curriculum=CurriculumCfg(...),
)
```

## 2. `control_config`：动作缩放

策略输出是相对于默认站立姿态的关节位置残差：

$$
q_{target}=q_{default}+action\_scale\cdot a
$$

`control_config.action_scale` 控制策略动作对目标关节角的影响幅度。动作空间边界会根据 actuator 的位置控制范围、默认角和
`action_scale` 自动推导。

```python
control_config = ControlCfg(action_scale=0.5)
```

## 3. `commands`：速度命令分布

`CommandsCfg` 控制训练期间采样的 `[vx, vy, yaw_rate]`：

```python
commands = CommandsCfg(
    vel_limit=[
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    stand_prob=0.2,
    resampling_time=10.0,
)
```

| 字段              | 含义                                                          |
| ----------------- | ------------------------------------------------------------- |
| `vel_limit`       | 形状必须为 `(2, 3)`；第一行为三个命令维度的下限，第二行为上限 |
| `stand_prob`      | 采样全零站立命令的概率                                        |
| `resampling_time` | 重新采样速度命令的时间间隔，单位为秒                          |

接入初期可以缩小速度和转向范围，确认机器人能够稳定站立并响应动作后再逐步扩大。

## 4. `normalization`：观察缩放与噪声

`NormalizationCfg` 定义进入网络前的观察缩放，以及仅施加在 actor 观察上的关节噪声。每个物理量在拼接到观察向量前
按下式缩放：

$$
x_{obs}=scale\cdot x_{raw}
$$

缩放只改变 actor/critic 接收的输入数值，不改变仿真状态、速度命令或奖励。调大 scale 会放大对应观察分量的变化，调小
scale 会压低该分量。通常应让不同量纲的观察在机器人正常运动范围内处于相近数量级，避免数值较大的输入主导网络的早期
优化，也避免数值过小的输入难以被利用。

| 字段            | 作用对象                                   |
| --------------- | ------------------------------------------ |
| `base_lin_vel`  | Critic 中机体局部线速度的缩放乘数          |
| `base_ang_vel`  | Actor 和 Critic 中机体局部角速度的缩放乘数 |
| `dof_pos`       | 相对默认姿态的关节位置的缩放乘数           |
| `dof_vel`       | 关节速度的缩放乘数                         |
| `noise_dof_pos` | Actor 关节位置观察的均匀噪声幅度           |
| `noise_dof_vel` | Actor 关节速度观察的均匀噪声幅度           |

关节噪声叠加在缩放后的观察上，因此 `noise_dof_pos` 和 `noise_dof_vel` 的大小应相对于对应的缩放后信号设置。噪声不会
加入 critic 的特权观察。调整关节单位、速度范围或策略输入尺度时，应同步检查这些字段。

## 5. `gait`：双脚步态参考

`GaitCfg` 定义共享的双脚周期参考：

```python
gait = GaitCfg(
    period=1.0,
    swing_height=0.09,
    feet_phase_sigma=0.008,
)
```

| 字段               | 含义                                   |
| ------------------ | -------------------------------------- |
| `period`           | 一个步态周期的时长，单位为秒           |
| `swing_height`     | 摆动脚相对局部地面的期望最大高度       |
| `feet_phase_sigma` | 脚高误差转为 `feet_phase` 奖励时的尺度 |

`swing_height` 应结合机器人腿长、脚底 site 位置和地形起伏设置。站立命令会固定相位，不要求脚部摆动。

## 6. `reward_config`：奖励与姿态约束

`RewardCfg` 包含共享奖励项的权重和少量计算参数：

```python
reward_config = RewardCfg(
    scales=RewardScales(
        tracking_lin_vel=...,
        tracking_ang_vel=...,
        penalty_action_rate=...,
        # 其余共享奖励项按需覆盖
    ),
    tracking_sigma=0.25,
    close_feet_threshold=0.15,
    pose_weights={
        "<joint-name>": 1.0,
        # 必须覆盖机器人 body 的全部关节
    },
)
```

| 字段                   | 含义                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `scales`               | `tracking_lin_vel`、`tracking_ang_vel`、稳定性、动作、姿态和双脚步态等奖励项的权重 |
| `tracking_sigma`       | 线速度和偏航角速度跟踪误差的指数核尺度                                             |
| `close_feet_threshold` | 触发双脚横向距离惩罚的阈值                                                         |
| `pose_weights`         | 每个关节偏离默认姿态时的相对惩罚权重                                               |

`pose_weights` 的 key 必须与机器人 body 上的关节名称完全一致，不能缺少关节或包含未知关节；所有权重必须是有限且非负
的数。较大的值会更强地约束对应关节保持默认姿态，较小的值则允许该关节参与步态运动。

## 7. `curriculum`：惩罚课程

`CurriculumCfg` 根据已结束回合的平均长度，逐渐调整指定惩罚项的整体缩放：

| 字段                      | 含义                                     |
| ------------------------- | ---------------------------------------- |
| `enabled`                 | 是否启用惩罚课程；关闭时缩放固定为 `1.0` |
| `initial_scale`           | 训练开始时的惩罚缩放                     |
| `min_scale` / `max_scale` | 缩放允许达到的下限和上限                 |
| `level_down_threshold`    | 平均回合长度低于该值时减小惩罚           |
| `level_up_threshold`      | 平均回合长度高于该值时增大惩罚           |
| `degree`                  | 每次更新缩放时的相对变化比例             |
| `penalty_terms`           | 受课程缩放影响的奖励项名称               |

接入初期可沿用默认配置。若机器人尚不能保持站立，应先排查模型、默认姿态、PD 和终止条件，再判断是否需要放宽惩罚。

## 8. 地形与出生范围

`spawn_xy_range` 控制 reset 时出生点在 X、Y 方向的均匀采样范围；平地通常设为 `0.0`。

平地和起伏地形应共享机器人、动作、观察、奖励和终止配置；起伏地形配置只需改变场景地形和出生范围。

高度场版本会在出生点周围查询局部地面高度并抬高机器人基座，也会在每一步以脚底当前位置的局部地面作为脚高参考。
