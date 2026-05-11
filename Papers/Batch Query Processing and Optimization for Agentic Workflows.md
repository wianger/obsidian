---
type: paper
authors: ["Junyi Shen", "Noppanat Wadlom", "Yao Lu"]
publish: "2025-01-01"
venue: "arXiv"
url: "https://arxiv.org/abs/2509.02121"
zotero: "zotero://open-pdf/library/items/RSNMD2GJ"
created: "2026-05-07 23:16"
abstract: |-
  大型语言模型（LLM）在智能体工作流中结合了多步推理、异构工具使用以及多个专门智能体间的协作。现有的LLM推理引擎孤立地优化单个调用，而多智能体框架侧重于编排，缺乏系统级的性能规划。因此，重复的提示词、重叠的上下文以及碎片化的CPU-GPU执行导致了大量的冗余和较差的硬件利用率，尤其是在批量分析场景中。我们提出了Halo，一个将批量查询处理和优化引入智能体LLM工作流的系统。Halo将每个工作流表示为结构化的查询计划DAG，并为批量查询构建一个暴露共享计算的合并图。在同时考虑异构资源约束、预填充和解码成本、缓存重用以及GPU放置的成本模型指导下，Halo执行计划级优化以最小化冗余执行。处理器集成了自适应批处理、KV缓存共享与迁移，以及细粒度的CPU-GPU流水线，以最大化整体硬件效率。
tags: ["multi-agent", "batch-processing", "kv-cache", "query-optimization", "llm-serving", "cpu-gpu-pipelining"]
---
这是一篇由新加坡国立大学（NUS）研究团队于2025年提出的系统级论文，题为 **《Batch Query Processing and Optimization for Agentic Workflows》**。该论文提出了一个名为 **Halo** 的新型运行时系统，旨在解决大语言模型（LLM）智能体工作流在批量执行时的冗余计算、CPU-GPU资源割裂以及硬件利用率低下等问题。

以下是对该论文的详细结构化解读：

---

### 📖 一、 论文概览

- **核心目标**：将传统数据库中的“批量查询处理与优化”思想引入到异构智能体（Agentic）工作流中，实现跨请求、跨DAG的全局调度与资源协同。
    
- **提出系统**：**Halo**（Holistic scheduling for heterogeneous LLM workflows）
    
- **关键成果**：在6个基准测试中，批量推理延迟最高降低 **3.6×**，在线服务吞吐量提升 **2.6×**，相比朴素逐请求执行（vLLM）最高加速 **400×**，且全程保持输出质量无损。
    
- **定位**：首个将 LLM 推理服务与异构工作流查询优化深度融合的系统，填补了“多智能体编排框架”与“底层LLM推理引擎”之间的系统级优化空白。
    

---

### 🔍 二、 研究背景与核心痛点

当前基于LLM的智能体工作流通常表现为**有向无环图（DAG）**，节点包含：

- **LLM节点**：在GPU上执行推理（Prefill + Decode）
    
- **Tool节点**：在CPU上执行外部调用（SQL查询、HTTP请求、本地函数等）
    

**现有方案的不足**：

|方案类型|代表系统|缺陷|
|---|---|---|
|请求级推理引擎|vLLM, SGLang, TGI|仅优化单个请求，缺乏跨请求/DAG的逻辑感知，无法合并重复计算|
|通用数据平台|Ray, Spark, Dask|支持DAG执行，但将算子视为黑盒，无法利用KV Cache复用、Prefill/Decode调度等LLM特性|
|智能体编排框架|LangGraph, AgentScope|关注控制流与消息传递，与底层推理运行时解耦，导致GPU在CPU工具执行时空闲（Pipeline Bubbles）|

在批量数据分析场景中，数十至数百个相同结构的工作流并发执行，会产生大量**重复Prompt、重叠上下文、相同SQL/API调用**，现有架构无法全局去重与协同调度，造成严重的算力浪费。

---

### ⚠️ 三、 三大核心挑战

1. **C1: 结构感知（Structural Awareness）** 多智能体常遍历相同子图或发出相同工具调用。独立调度会导致重复I/O与计算，需全局识别并合并（Request Coalescing）。
    
2. **C2: 异构流水线气泡（Heterogeneous Pipeline Bubbles）** CPU工具与GPU推理交替执行。若调度不当，GPU会长时间等待CPU结果，产生资源空闲。
    
3. **C3: 状态敏感的LLM算子（Stateful LLM Operators）** 模型切换需重新加载权重（昂贵）；KV Cache是否命中直接影响Prefill延迟。调度必须感知Worker的驻留模型与缓存状态。
    

---

### 🏗️ 四、 Halo 系统架构

Halo 采用经典的 **Parser → Optimizer → Processor** 三段式架构：

#### 1. Parser（解析器）

- 输入：YAML声明式工作流规范 + 批量查询输入
    
- 功能：转换为类型化中间表示 `GraphSpec`。**关键操作是依赖解耦**：将嵌入在Prompt中的非LLM计算（SQL/API/本地函数）提取为独立可调度的CPU节点，使优化器能统一视图。
    

#### 2. Optimizer（优化器）

- **Operator Profiler**：对LLM与Tool节点进行轻量级性能画像（延迟、资源消耗）。
    
- **Solver**：基于成本模型与依赖约束，生成全局 `ExecutionPlan`（节点执行顺序 + GPU Worker分配）。
    

#### 3. Processor（处理器/运行时）

- 协调长期运行的 GPU Worker（托管vLLM实例）与 CPU Worker（执行工具调用）。
    
- 实现动态流水线并行、自适应批处理、KV Cache共享/迁移、请求合并与 Opportunistic Execution（机会执行）。
    

---

### ⚙️ 五、 核心技术创新

#### 🔹 1. 基于 Epoch 的离散化调度与 DP 求解器

- 直接优化连续启动时间是NP-Hard问题。Halo 将时间划分为 **Epoch（决策窗口）**，每个Epoch初根据当前系统状态选择一批就绪的LLM节点分配给GPU。
    
- **目标函数**：最小化总Epoch成本 `C_epoch = μ·max(T_w) + (1-μ)·∑T_w + λ·g(A_e)` （兼顾关键路径延迟、全局负载均衡、调度开销正则化）
    
- **动态规划求解**：利用DAG拓扑前沿（Topological Frontier）剪枝状态空间，结合记忆化递归（Memoization），在保持全局最优性的同时将规划时间从MILP的数小时压缩至 **2秒级**（加速2300×）。
    

#### 🔹 2. 状态感知成本模型（State-Aware Cost Modeling）

延迟估计公式：
`T(w, v, S_e) = T_prep(v) + T_gpu(w, v, H_e_w)` 
`T_gpu = T_model(模型切换惩罚) + T_infer(推理延迟)`

- `T_prep`：上游CPU工具准备时间，显式计入成本以避免GPU饥饿。
    
- `T_model`：若Worker已驻留所需模型则为0，否则计入权重加载/驱逐开销。
    
- `T_infer`：根据KV Cache前缀匹配长度动态折扣Prefill token数，激励局部性分配。
    
- **在线校准**：结合DB `EXPLAIN`、API滑动平均、LLM吞吐量曲线持续更新画像，适应运行时抖动。
    

#### 🔹 3. Processor 运行时优化机制

- **依赖解析与CPU优先级调度**：按DAG深度优先调度解锁GPU前沿的CPU任务，最大化CPU-GPU重叠。
    
- **请求合并（Request Coalescing）**：对相同签名（类型+规范化参数）的工具调用进行物理合并，结果扇出给所有依赖节点；对SQL使用预编译语句复用。
    
- **波前执行与机会调度**：LLM节点完成后立即推进下游；若原计划任务因I/O阻塞，在不破坏依赖与GPU状态的前提下，动态窃取其他就绪任务执行，掩盖长尾延迟。
    
- **语义保持**：严格遵循原始DAG依赖，不修改Prompt，不引入近似推理。
    

---

### 📊 六、 实验评估

- **硬件**：2× AMD EPYC 9755, 2.2TB RAM, 3× NVIDIA H200 (141GB)
    
- **模型**：Qwen3-14B/32B, GPT-OSS-20B
    
- **基线**：vLLM（逐请求）, OpWise（层级同步批处理）, LangGraph, AgentScope, Parrot
    
- **工作负载**：6个典型DAG（W1-W6），涵盖Diamond、Chain、Fanout、Bridge等拓扑，混合SQL/HTTP/LLM节点。数据集：FineWiki, IMDb, TPC-H。
    

**核心结果**：

|指标|Halo 表现|
|---|---|
|批量推理延迟|相比基线降低 1.6×~3.6×，相比朴素vLLM最高加速 400×|
|在线服务吞吐量|提升 1.53×~2.58×（QPS）|
|调度最优性|DP规划结果与MILP理论最优结构一致，规划时间仅2.24s（MILP需1.44h）|
|GPU利用率|消除OpWise的“锯齿状”空闲，GPU-秒消耗降低约 2.0×|
|可扩展性|批量大小256→2048近线性稳定；GPU Worker 1→3弹性加速；适配0.4B~32B模型与A100/H100/H200异构设备|

**消融实验**表明：性能画像、CPU负载引导、机会执行、请求合并四大组件缺一不可，移除任一项在复杂负载（W6）上会导致 8%~154% 的延迟劣化。

---

### 🚧 七、 局限性

1. **部署范围**：目前聚焦**单机多GPU**环境，以实现细粒度显存与局部性控制。扩展到多机集群需解决分布式放置、缓存迁移与网络感知调度。
    
2. **固定逻辑计划**：假设DAG结构固定且严格语义等价，不支持在线重写工作流结构或使用近似/代理模型换取成本优化。
    

---

### 💡 八、 总结与学术/工程意义

- **学术贡献**：首次将数据库查询优化思想（批量处理、代价模型、全局调度）与LLM推理服务（KV Cache、Prefill/Decode、连续批处理）在异构DAG层面统一。提出了一种兼顾最优性与在线可用性的Epoch-DP调度范式。
    
- **工程价值**：为数据分析、决策支持、多智能体仿真等高并发场景提供了可直接落地的运行时底座。无缝对接vLLM/SGLang等后端，不改变用户工作流逻辑即可实现显著的性能与成本优化。
    
- **未来方向**：向多节点分布式扩展、结合逻辑层重写（如Prompt优化/模型路由）、支持动态DAG与流式自适应规划。