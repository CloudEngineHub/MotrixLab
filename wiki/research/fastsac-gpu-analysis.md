# FastSAC GPU 利用率分析与优化研究

## 摘要

本文记录 FastSAC 异步双进程训练（learner CUDA + collector CUDA inference）在单卡
RTX 3090 上"GPU 利用率不满"问题的完整分析与优化过程，沉淀两部分内容：一是
**分层评测方法**（console 面板统计 → microbenchmark → CUDA Event 遥测 → 双进程
Chrome trace → 固定窗口真实训练 A/B），二是**关键结论**。

最终结论：GPU idle 是真实的 kernel timeline 空档，主因是 learner 各训练阶段内部
大量 eager/compiled 边界造成的 CPU launch/copy 碎片（约 76% idle 位于
critic/actor 阶段内部），而非 ring 断粮或拷贝瓶颈；collector 的 CUDA inference
与 learner 存在可测的跨 context 竞争但体量很小。最有效的优化是把 compile 边界
提升到完整 update callable 并使用 `mode="reduce-overhead"`（CUDA Graph 重放）：
learner p50 从 75--77.5 ms 降至 35.5--35.9 ms，UTD 从约 1.75 升至约 3.82，
GPU busy 从约 30.7% 升至约 61.3%，同墙钟学习进度约 4 倍，代价是
env-steps/s 约 −6%。同时纠正了一个关键评测误区：GPU busy ratio 与训练速度是
两个独立指标，回归判定应以 UTD / learner 圈长为主 KPI。

被测环境：g1-wbt-dance 任务（num_envs=2048、num_atoms=501、batch=8192、
num_updates=4、utd_mode=learner_bound、AMP bf16 + fused AdamW）、RTX 3090、
PyTorch 2.7.0+cu128、远端无 `nsys`/`ncu`、CUDA MPS 在该驱动环境启动失败。

算法背景见 [SAC 与 FastSAC 算法原理入门](./sac-and-fastsac-primer.md)，训练器
架构见 [FastSAC 异步异构训练器](../design/fastsac-async-heterogeneous-trainer.md)。

## 稳态机制模型

理解性能数据的前提是正确的稳态模型（learner_bound 模式）：

```text
collector:  自由全速 ~36ms/批（受 CPU 物理步进限制），ring 常空
learner:    热循环 ~83ms/圈 = update(4) ~76ms + drain(H2D 入 buffer) ~7ms + publish ~1.5--9.7ms
UTD:        = num_updates × collector周期 / 圈长，是结果不是上限
GPU busy:   每圈约 44ms kernel 时间，其余为 CPU 串行段与提交间隙
```

要点：稳态下 learner **从不等待 collector**；`drain` 是把 ring 数据搬进 replay
buffer 的工作时间，不是等待；ring/gate wait 统计的样本几乎全部来自启动期
（collector 建环境与 warmup 期间）。

## 评测方法

问题定位依靠从粗到细的五层评测手段，各层回答不同问题，逐层收敛。

### 1. 真实训练 console 面板

`run_learner_process` 每个 logging window 输出按进程分组的 timing 树
（collector 各阶段均值、learner drain/ring wait/gate wait 均值、update 各阶段
均值），每窗口重置。用于：确认 collector/learner 节奏、发现启动期 ring 断粮、
观察各优化候选对 learner 圈长与 critic_alpha 阶段的影响。数据同时写入
TensorBoard（`perf/collector_*_ms` 等）。

### 2. 固定 microbenchmark

隔离 C51 projection 单路径做固定条件压测：batch/hidden/atoms、AMP bf16、CUDA、
compile、steady-state 采样，输出 mean/p50/p90/p99 与吞吐（本研究基线
5.088 ms → 优化后约 4.8 ms、1.71M samples/s）。用途：低成本筛掉无稳定收益的
kernel 级改写。注意它只隔离单条路径，收益不等价于完整训练收益。

### 3. CUDA Event 遥测

在 learner update 的各阶段（采样/归一化、critic+alpha、actor、soft update）埋
CUDA Event，测 `cuda_sample_normalize` / `cuda_critic_alpha` / `cuda_actor` /
`cuda_soft_update` 各阶段 stream 时间。读数纪律：Event elapsed 是 stream 上的
阶段时间，不等同于 SM active time，也不能当作 `nvidia-smi` 利用率。

### 4. 双进程 Chrome trace + 阶段标注

learner 与 collector 各自用 `torch.profiler` 输出共享 `baseTimeNanoseconds`
的 Chrome trace，可直接合并对齐；训练阶段加 `record_function` annotation。
产出的核心指标是 **GPU busy/idle timeline**：kernel
busy、idle gap 分布（p50/p90/p99/max）、各阶段 wall vs GPU busy vs busy ratio、
`cudaLaunchKernel` / `aten::to` / `copy_` / `linear` 等 CPU operator 计数。
低开销 CUDA-only 模式用于长窗口，CPU+CUDA 模式用于归因。

### 5. 固定窗口真实训练 A/B（最终裁决）

所有候选最终必须在真实训练命令（不覆盖 Hydra 配置；诊断性 override 除外）下
做固定窗口对照，同时记录：learner/collector p50、各阶段 p50、UTD、
env-steps/s、训练指标（return/qf_loss）、GPU SM/功耗 5 秒采样。验收纪律：
预注册主 KPI 与撤回标准，单次 trace 窗口受 update 数量与异步相位影响，
不允许凭单窗口宣布收益。

### 指标读数注意事项（踩坑记录）

- `nvidia-smi` SM 利用率是采样值；GPU idle 必须用 trace kernel timeline 证实。
- profiler 的 `cudaMemcpyAsync` runtime 是 CPU API 阻塞/调用时间，不能与 GPU
  copy engine duration 相加；copy 归因用 trace 的 copy engine 事件。
- trace 的 `queued` 字段恒为 0，不可当作 CPU enqueue 等待时长；enqueue 归因用
  同时间轴 launch/operator 计数。
- 两进程各自 trace 中 context 均被 profiler 映射为 `context=1`，不能用量化
  "context-id gap"的方式归因跨进程竞争，只能靠隔离 A/B 间接证明。
- CUDA Event 替换 `current_stream().synchronize()` 的对照
  （3.64 ms vs 7.49 ms/次）表明临时 Event 创建/同步更慢。
- microbenchmark 有收益 ≠ 真实训练有收益；两者必须分别验证。

## 关键发现

### 已排除的原因（负结果同样有效）

- **ring 断粮**：ring/gate wait 样本全部来自启动期，稳态不存在。
- **projection 内部空转**：单独 projection 的 kernel active/span 约 93.7%，
  问题是 kernel 数量多、elementwise/copy 临时 tensor 多，不是 GPU 空转。
- **H2D 拷贝**：shared-memory 直接 H2D（约 1.10 ms/batch）与 pinned staging
  （约 0.95 ms）差距约 0.15 ms/batch，相对 80--90 ms 的 update 非主瓶颈；
  learner `non_blocking=True` 在 benchmark 与真实训练均无稳定收益。
- **显式同步 API 时间**：细粒度窗口中 learner `cudaStreamSynchronize` 合计
  6.66 ms、collector 14.79 ms，相对约 734 ms 的 learner GPU idle 不是主要体量。
- **collector kernel 体量**：collector 独占 GPU kernel 时间仅约 0.28%（窗口内
  6.1 ms），不是 60% idle 的直接体量来源。
- collector 每步的 `current_stream().synchronize()` 是**必要的正确性边界**
  （CPU 环境必须等 action D2H 完成；CUDA Graph 可能复用输出存储），不能删除。

### collector/learner GPU 竞争

隔离 A/B（仅临时 override `collector_inference_device=cpu`）：

| 指标 | CUDA collector 基线 | 诊断性 CPU collector |
| --- | ---: | ---: |
| env-steps/s | 47.2k--51.3k | 25.5k--25.8k |
| learner p50 | 83.0--92.3 ms | 72.9--74.3 ms |
| critic_alpha p50 | 51.1--55.9 ms | 40.5--41.1 ms |
| GPU SM | 约 51--54% | 约 67--73% |

结论：collector 的 H2D → actor graph → D2H → stream synchronize 短 burst 与
learner 在单卡两个独立 CUDA context 上竞争，使 learner latency 上升约 15%、
critic 主阶段上升约 25%。但 CPU collector 使 env 吞吐减半，不能直接采用。

由此尝试的**learner 单 context inference service 原型**（collector 通过共享内存
mailbox 请求 learner 侧高优先级 stream 执行推理）失败：单 context 不自动消除
资源竞争，inference thread 与训练主线程在 SM/内存带宽/allocator 上互抢，
learner 反而恶化 12--18%，已完整撤回。教训：竞争的本质是 GPU 计算与内存带宽
的争用，不是 context 数量本身；跨 context 迁移不能替代降低 collector 推理的
GPU 成本或调度频率。

### GPU idle 根因定位

双进程 trace 定量（约 1.9 s 窗口）：GPU busy 约 40%、idle 约 60%；idle gap
p50/p90/p99 = 5.6 us / 39.3 us / 288 us。阶段分解显示 **约 76% 的 idle 位于
critic/actor 阶段内部**，而非 ring 等待或阶段之间：

| 阶段 | busy ratio | idle/call 排名 |
| --- | ---: | ---: |
| target actor | 17.5% | 约 2.60 ms/update |
| critic backward | 50.5% | 约 2.16 ms/update |
| target projection | 67.9% | 约 1.88 ms/update |
| alpha update | 14.3% | 约 1.77 ms/update |
| online critic + loss | 77.6% | 约 0.58 ms/update（非问题） |
| target value | 98.2% | 非问题 |

CPU dispatch 证据：target actor 36 次调用含 2376 次 `cudaLaunchKernel`、792 次
`aten::to`；target projection 含 4068 次 launch、1512 次 `aten::to`。直接根因
是：compiled module 只覆盖 `forward`，实际算法路径（target actor sampling、
projection、loss、backward、optimizer）在多个 compiled region 与 eager region
之间往返，产生数千次 launch/cast/copy 与 autograd 调度空档。

## 优化实验记录

### 保留

- **projection kernel 改写**：`arange().mul_()` 替代 `linspace().long()`，移除
  两个冗余 index clamp（边界修正已保证索引合法）。projection 5.088 ms →
  约 4.8 ms，吞吐 1.61M → 1.71M samples/s。
- **大 batch 采样/归一化**：一次迭代合并采样并整批更新归一化统计。
  sample_normalize p50 6.0--8.9 ms → 2.0--2.3 ms，learner p50 76--89 ms →
  75--77.5 ms，UTD +0.005 左右，无回退。
- **阶段 B3：完整 update callable + `mode="reduce-overhead"`**：对
  `_update_main` / `_update_pol` 整体 compile 并启用 CUDA graph trees，每次
  迭代前调用 `torch.compiler.cudagraph_mark_step_begin()`（防止跨步存活的输出
  被覆写，缺此步会崩溃）。

  | 指标 | 对照（default compile） | reduce-overhead |
  | --- | ---: | ---: |
  | learner p50/update call | 75--77.5 ms | 35.5--35.9 ms |
  | critic_alpha p50 | 44--48 ms | 20.6--22.3 ms |
  | UTD | 约 1.75 | 约 3.82 |
  | GPU busy ratio | 30.7% | 61.3% |
  | env-steps/s | 55.4--56.6k | 52.1--53.3k（−6%） |

  同墙钟（约 4.5 分钟）学习进度约 4 倍（iter 6000 return 4.77 vs iter 6900
  return 19.85）。代价：learner 提交加密后 collector 的跨 context 策略推理
  1.65 ms → 3.4 ms，env-steps/s −6%。

### 撤回（含教训）

| 候选 | 结果 | 教训 |
| --- | --- | --- |
| 临时环境变量诊断开关（CUDA Event 计时 / 阶段标注 / 双进程 trace） | 已撤出训练代码 | 观测代码不得常驻训练热路径、不得进入 compiled callable；由独立 profiling 系统承接（#239） |
| `torch.addcmul`、删 critic LayerNorm、twin critic 共享 cat | 无稳定收益 | microbenchmark 无收益的一律撤回 |
| learner `non_blocking=True` H2D | 无稳定收益 | 已测得的理论 0.15 ms/batch 差距太小 |
| CUDA Event 替换 stream synchronize | 更慢（7.49 vs 3.64 ms） | 临时 Event 创建开销 |
| 单独 compile `.projection` bound method | GPU busy 39.6%→30.6% | 对 bound method 再套 compile 只增加 graph 边界 |
| critic forward+loss 融合 compile | busy 持平，p50 恶化 | 小融合消不掉 target/backward 大边界 |
| 单独 compile target actor callable | 单窗口略好，无稳定收益 | 仍是独立 compiled graph |
| 完整 target path（actor+projection+value）绑定 compile | busy 40%→31.7%，回归 | bound method 直拼产生动态 graph 边界；正确单位确认了，但实现要无副作用 wrapper + 固定形状 + 独立 warmup |
| 阶段 A：完整 `_update_main/_update_pol` default-mode compile | p50 略降、UTD 1.72--1.75，但 busy 恶化至 30.4% | CPU enqueue 变快 ≠ GPU busy 提升；由此纠正 KPI |
| mailbox + high-priority stream 推理服务 | learner 恶化 12--18% | 见上文竞争结论 |
| CUDA MPS | 环境不可用 | `CUDA_ERROR_MPS_CONNECTION_FAILED` |

### 评测 KPI 的纠正

阶段 A 的插曲暴露了一个评测误区：仅以 GPU busy ratio 判定回归，会把
"learner 更快但 GPU 更闲"误判为倒退。正确的指标体系：

- **主 KPI**：UTD 与 learner 圈长（直接决定训练速度）；训练指标
  （return/qf_loss）必须正常。
- **约束**：env-steps/s 由 collector 物理步进决定，优化 learner 不应改变它。
- **参考**：GPU busy ratio 仅作效率观察，不作为回归判据。

圈长收益换算：圈长每省 1 ms，UTD 约 +0.021（4 × 36.5 ms collector 周期）。

## 遗留方向与风险

- 恢复 env-steps/s：降低 collector 推理的 context 竞争（MPS 当前不可用），
  或接受 −6% 代价。
- publish 异步化/摊薄（侧线程 + 现有 seqlock 快照），预期圈长 −2~6 ms、
  UTD +0.05~0.13；需保持权重版本节奏语义。
- drain H2D 批量搬运或 pinned staging，预期圈长 −3~5 ms；`commit_read` 必须在
  拷贝完成后以保持语义。
- B3 结论基于单次运行，未做种子重复；CUDA graph RNG 与 eager RNG 流不同但
  统计等价；正式合入前应跑更长时训练确认收敛稳定性。
- 相关实现计划已从 `wiki/plan/` 清理；如需继续推进，请基于本分析重新建立计划。
