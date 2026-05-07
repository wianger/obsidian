---
type: paper
authors:
  - Hao Kang
  - Ziyang Li
  - Xinyu Yang
  - Weili Xu
  - Yinfang Chen
  - Junxiong Wang
  - Beidi Chen
  - Tushar Krishna
  - Chenfeng Xu
  - Simran Arora
publish: 2026-01-01
venue: arXiv
url: https://arxiv.org/abs/2602.13692
zotero: zotero://open-pdf/library/items/DUMNNEZZ
created: 2026-05-07 17:17
abstract: 大型语言模型（LLM）现被用于驱动复杂的多轮智能体工作流。现有系统通过松散组合独立组件（如LLM推理引擎vLLM与工具编排器Kubernetes）来运行智能体推理。尽管智能体工作流涉及多个LLM和工具请求，但这些系统仍基于逐个请求独立调度和分配资源，缺乏对工作流的端到端认知，导致KV缓存和工具执行环境的管理效率低下。为解决上述问题，我们提出ThunderAgent——一种快速、简单且感知程序的智能体推理系统。我们首先将智能体工作流抽象为LLM程序（LLM Program），从而实现对异构资源的统一视图，包括KV缓存、系统状态以及磁盘内存和网络端口等外部工具资产。基于这一抽象，ThunderAgent引入了感知程序的调度器和工具资源管理器，旨在最大化KV缓存命中率、缓解内存不平衡并支持异步环境准备。在编码、路由和科学发现智能体上的评估表明，与最先进的推理系统相比，ThunderAgent在服务场景中实现了1.5-3.6倍的吞吐量提升，在RL rollout中实现1.8-3.9倍提升，并节省了高达4.2倍的磁盘内存。为促进可重复性并支持未来发展，我们在https://github.com/HaoKang-Timmy/ThunderAgent 开源了ThunderAgent的完整系统实现。
tags:
  - llm
  - agent
  - kv-cache
  - tool
  - program-aware-scheduling
---
### 📖 一、论文核心概述
**论文标题**：`ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System`  
**核心目标**：解决当前多轮智能体（Agent）工作流在推理服务（Serving）和强化学习 rollout 阶段吞吐量严重下降的问题。  
**核心思想**：打破现有系统“按单次请求调度”的局限，提出**程序感知（Program-Aware）** 架构，将整个多轮 Agent 工作流抽象为一个统一的调度单元，实现 GPU KV Cache、节点内存与外部工具环境的端到端协同管理。   
**主要成果**：相比 vLLM、Continuum 等 SOTA 系统，ThunderAgent 在服务场景实现 `1.5~3.6×` 吞吐提升，在 RL rollout 场景实现 `1.8~3.9×` 提升，并节省最高 `4.2×` 的磁盘内存。系统已开源，且只需极小改动即可接入现有推理引擎。

---
### 🔍 二、现有系统的三大痛点
当前主流方案通常将 **LLM 推理引擎（如 vLLM/SGLang）** 与 **工具编排器（如 Kubernetes）** 松散拼接，按“单次请求”独立调度，导致以下系统性瓶颈：

| 痛点                            | 现象与后果                                                                            | 根本原因                                                               |     |
| :---------------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------- | --- |
| **1. KV Cache 抖动（Thrashing）** | 工具执行期间，系统为容纳新请求提前驱逐 KV Cache；工具返回后需重新 prefill 全部历史上下文，端到端延迟暴增最高 `7.14×`，吞吐断崖式下跌。 | 请求级调度缺乏对工作流未来复用 KV 的预判，盲目驱逐。                                       |     |
| **2. 跨节点内存不均衡**               | 多节点部署时，部分 GPU 节点内存爆满触发暂停，其他节点却大量闲置（实测峰值差异达 `51%`）。                               | 现有 KV-aware 路由为追求命中率，将同一工作流的所有请求硬性绑定到固定节点，无法适应 Agent 上下文长度的不可预测增长。 |     |
| **3. 工具生命周期无感知**              | 磁盘占用随处理工作流数量线性增长直至系统崩溃；环境准备时间随并发数急剧上升。                                           | 推理引擎与工具编排器状态不同步，已结束的沙箱/端口未被回收；且环境初始化与 LLM 推理串行执行，阻塞整体流水线。          |     |

---
### 🛠️ 三、ThunderAgent 的核心创新
针对上述问题，ThunderAgent 提出三大核心贡献：
1. **程序抽象（Program Abstraction）**：将多轮 Agent 工作流封装为一级调度对象 `P = ⟨ID, c, T, L, τ, s⟩`，统一跟踪上下文长度 `c`、所需工具环境 `T`、节点位置 `L`、执行阶段 `τ`（Reasoning/Acting）、调度状态 `s`（Active/Paused/Terminated）。
2. **程序感知调度器**：基于成本建模，设计全局等待队列、状态感知暂停/恢复、动态跨节点迁移机制，从根源上抑制 KV 抖动与内存倾斜。
3. **程序感知工具资源管理器**：通过生命周期钩子实现即时垃圾回收，并利用异步预加载隐藏工具环境初始化延迟。

---
### ⚙️ 四、关键技术详解

#### 1. 成本模型（Cost Model）
系统采用**空间-时间积（STP）** 量化 GPU 资源消耗，将总成本分解为：
`Cost_total ≈ Cost_decode + Cost_prefill + Cost_recompute + Cost_unused + Cost_caching`
- **有效成本**：`decode` 与 `prefill`（直接贡献吞吐）
- **浪费成本**：`recompute`（KV 驱逐后重计算）、`unused`（节点内存闲置）、`caching`（工具执行期间空占 KV 内存）
**优化目标**：最小化三项浪费成本，从而最大化系统吞吐。

#### 2. 调度策略（Scheduling Policy）
- **周期性抖动检测**：每隔 `Δt` 检查后端内存水位。当超过高水位 `λ_max` 时触发 `Pause`，低于低水位 `λ_min` 时触发 `Restore`，形成迟滞窗口稳定调度。
- **时间衰减函数 `f(t)`**：对处于 `Acting`（工具执行）阶段的程序，其 KV Cache 的内存优先级随等待时间 `t` 衰减。公式：`有效内存权重 = c_q × f(t_q)`。此举动态权衡“缓存占用成本”与“重计算成本”，避免长尾工具调用无限期占用显存。
- **最短优先驱逐（Shortest-First Eviction）**：理论证明重计算成本与上下文长度平方成正比（`Cost_recompute ∝ c_i²`）。因此，当需要释放内存 `ΔC` 时，**优先暂停上下文最短的程序**可全局最小化重计算代价。
- **调度评分公式**：
  `S_restore(P) = 1/c_P + I(τ=R)`  
  `S_pause(P) = 1/c_P + I(τ=A)`  
  指示函数 `I(·)` 确保**优先暂停 Acting 程序、优先恢复 Reasoning 程序**；同状态下按上下文长度最短优先。
- **全局等待队列**：打破“单工作流绑定单节点”的限制。程序一旦被 Pause，其 KV 即被驱逐，恢复时可调度至任意有空闲内存的节点，显著降低 `Cost_unused`。

#### 3. 工具资源管理（Tool Resource Management）
- **Hook-based GC**：严格绑定程序状态 `s`。当程序进入 `Terminated`，立即触发 teardown 回收 Docker 沙箱、网络端口、计算槽位，杜绝资源泄漏。
- **异步环境准备**：监控全局队列，当高优先级程序接近恢复阈值时，提前在 CPU 集群异步拉取镜像、安装依赖、构建仓库。将 I/O 密集型初始化与 GPU 推理重叠，大幅降低端到端延迟。

---
### 📊 五、实验评估与结果
| 维度 | 设置 | 结果 |
|:---|:---|:---|
| **工作负载** | 代码智能体（SWE-Agent, OpenHands）、路由智能体（ToolOrchestra）、科学发现智能体；覆盖确定性 & 随机性工具调用 | 全面验证泛化性 |
| **硬件与模型** | 单卡 RTX 5090 到 2×8×H100 集群；GLM-4.6 (355B)、Qwen3-235B/8B (FP8/FP16) | 覆盖消费级到数据中心级 |
| **Serving 吞吐** | 对比 vLLM、Continuum | 提升 `1.48~3.58×`；高并发下吞吐不崩塌，基线系统则严重退化 |
| **RL Rollout** | 对比 vLLM + SGLang Gateway | 提升 `1.79~3.92×`，有效缓解策略滞后（policy lag） |
| **KV Cache 命中率** | 确定性工具场景 ≈100%；随机性工具场景动态下降 | 命中率下降是**主动权衡**：牺牲部分命中率换取更低 `Cost_caching`，最终吞吐反而更高 |
| **资源开销** | 磁盘占用、延迟分解 | 磁盘节省 `4.2×`；延迟降低主要来自 prefill/decode 优化，工具管理贡献约 `10%` 延迟优化 |
| **超参敏感性** | `Δt`（检测周期）、`f(t)=x^{-t}`（衰减底数） | 在合理范围内吞吐稳定，系统鲁棒性强 |

---
### 📐 六、理论贡献（Appendix 核心证明）
1. **时间衰减函数的最优形式**：在工具执行时间“无记忆性”假设下，严格证明满足边界条件 `f(0)=1, lim f(t)=0` 的衰减函数只能是**连续时间指数衰减 `e^{-λt}`** 或**离散时间几何衰减 `x^{-k}`**。
2. **最短优先驱逐的全局最优性**：基于 `x²` 的严格凸性与超可加性，使用交换论证（exchange argument）证明：将大上下文程序替换为多个小上下文程序，可严格降低 `∑c_i²`，因此贪心选择最短上下文是全局最优解。
3. **工具调用时间的重尾分布**：实证分析表明远程工具（API、检索、沙箱执行）延迟呈重尾分布，p95/p99 远超中位数。这从理论上解释了 Continuum 等基于 TTL 预测的方法为何在真实场景中失效。

---
### 💡 七、总结与实际意义
- **范式转变**：从 `Request-Aware`（请求级）走向 `Program-Aware`（程序级），首次将多轮 Agent 工作流作为完整生命周期对象进行端到端调度。
- **极简接入**：兼容 OpenAI 风格 API，现有系统只需 **3 处改动**（请求附加 `program_id`、工具调用传递 `program_id`、结束时发送 `/programs/release`）即可无缝接入。
- **工程价值**：不修改底层 vLLM/SGLang 内核，以轻量级中间件形态解决 KV 抖动、节点倾斜、资源泄漏三大工业界痛点，特别适合大规模 Agent Serving 与异步 RL 训练。
- **开源地址**：`https://github.com/HaoKang-Timmy/ThunderAgent`

[[CONCUR - High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control]]
[[Sutradhara - An Intelligent Orchestrator-Engine Co-design for Tool-based Agentic Inference]]