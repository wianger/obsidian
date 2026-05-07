---
type: paper
authors: ["Yongtong Wu", "Shaoyuan Chen", "Yinmin Zhong", "Rilin Huang", "Yixuan Tan", "Wentao Zhang", "Liyue Zhang", "Shangyan Zhou", "Yuxuan Liu", "Shunfeng Zhou", "Mingxing Zhang", "Xin Jin", "Panpan Huang"]
publish: "2026-02-26"
venue: "arXiv"
url: "http://arxiv.org/abs/2602.21548"
zotero: "zotero://open-pdf/library/items/M8C4Z6MJ"
created: "2026-05-06 20:32"
abstract: |-
  多轮代理LLM推理的性能越来越受到KV-Cache存储I/O而非计算的限制。在流行的分离式架构中，从外部存储加载大量KV-Cache造成了根本性的不平衡：预填充引擎上的存储NIC带宽饱和，而解码引擎上的NIC却处于空闲状态。这种不对称性严重限制了整体系统吞吐量。我们提出DualPath，一种通过引入双路径KV-Cache加载来打破这一瓶颈的推理系统。除了传统的存储到预填充路径外，DualPath实现了新型的存储到解码路径，其中KV-Cache被加载到解码引擎，然后通过计算网络上的RDMA高效传输到预填充引擎。DualPath将这种优化的数据路径——它本质上避免了网络拥塞并避免干扰对延迟敏感的模型执行通信——与一个全局调度器相结合，该调度器动态平衡预填充和解码引擎之间的负载。我们在三个模型上使用生产代理工作负载的评估表明，DualPath在我们内部推理系统上将离线推理吞吐量提升了高达1.87$\times$。它还能在不违反SLO的前提下将在线服务吞吐量平均提升1.96$\times$。
tags: ["kv-cache", "llm-inference", "agentic-workloads", "dual-path", "rdma", "disaggregated-architecture", "throughput-optimization"]
---
这篇论文《**DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference**》由北京大学、清华大学与 DeepSeek-AI 联合发表，针对当前大模型 **Agent（智能体）多轮推理场景** 中的核心系统瓶颈提出了全新的架构设计与调度机制。以下从问题背景、核心创新、关键技术、实验验证及总结展望五个维度进行详细解读：

---
### 🔍 一、 核心痛点与研究动机
#### 1. Agentic 工作负载的 I/O 密集型特征
- **多轮长上下文+短追加**：Agent 任务通常包含数十至数百轮交互，上下文不断累积（可达数十万 token），但每轮新增输入（如工具输出、用户指令）往往只有几百 token。
- **KV-Cache 命中率极高（≥95%）**：绝大多数 token 的 KV-Cache 可直接从外部存储复用，仅需对少量新增 token 进行 Prefill 计算。
- **缓存/计算比失衡**：论文测算显示，在典型 Agent 负载下，KV-Cache 加载量与计算量的比值高达 `22 GB/PFLOP`（DeepSeek-V3.2），系统性能瓶颈已从 **GPU 计算** 彻底转向 **存储 I/O**。

#### 2. 现有 PD 分离架构的带宽不对称瓶颈
主流推理系统采用 `Prefill-Decode (PD) 分离 + 外部 KV 存储` 架构：
- **Prefill 节点**：负责从分布式存储加载海量 KV-Cache，存储网卡（SNIC）持续饱和。
- **Decode 节点**：仅负责自回归生成，存储网卡大量闲置。
- **结果**：全局存储带宽无法聚合，Prefill 侧 I/O 成为系统吞吐量的绝对天花板。单纯为 Prefill 节点扩容网卡成本高昂且不切实际。

---
### 🛠️ 二、 DualPath 核心架构设计
论文的核心洞察是：**KV-Cache 加载不必以 Prefill 为中心**。DualPath 引入**双路径加载机制**，将 Decode 节点闲置的存储带宽纳入全局调度池。

#### 1. 双路径数据流（Dual-Path Loading）
| 路径                     | 数据流向                                                                          | 适用场景             |
| :--------------------- | :---------------------------------------------------------------------------- | :--------------- |
| **PE Read Path（传统路径）** | 存储 → Prefill DRAM Buffer → 按层流式进 GPU HBM 计算 → 完整 KV 传至 Decode                 | Prefill 侧存储队列较短时 |
| **DE Read Path（新路径）**  | 存储 → Decode DRAM Buffer → Prefill 计算时通过 RDMA 按需拉取对应层 KV → 仅回传新增 miss token KV | Decode 侧存储带宽空闲时  |

- **块布局优化**：与存储交互使用 `Full Block`（包含所有层），GPU 间传输与计算使用 `Layer Block`（单层），完美契合 Layerwise Prefill 范式，避免 HBM 容量瓶颈与内存碎片。
- **无瓶颈理论证明**：论文通过流量建模证明，在合理的 P/D 节点比例下（如 8 卡 1 存储网卡配置下 `1/7 ≤ P/D ≤ 7/2`），双路径设计**不会**导致计算网卡（CNIC）或主机 DRAM 成为新瓶颈。

---
### ⚙️ 三、 两大关键技术组件
实现双路径架构面临三大挑战：细粒度传输开销、流量干扰、动态负载均衡。DualPath 通过以下设计逐一攻克：

#### 1. 以计算网卡为中心的流量管理器（CNIC-Centric Traffic Manager）
- **痛点**：传统 GPUDirect Storage 或 CUDA Copy Engine 走独立 PCIe 路径，无法与模型推理中延迟敏感的集合通信（如 EP AllToAll、TP AllGather）共享 QoS 控制，极易造成推理延迟抖动。
- **反直觉设计**：强制所有进出 GPU 的数据（**包括本地 H2D/D2H 拷贝**）均绕行配对计算网卡（CNIC），通过 GPUDirect RDMA 路径传输。
- **收益**：
  - 利用 InfiniBand 硬件虚拟通道（VL）实现严格流量隔离：高优先级 VL 保留 ~99% 带宽给模型通信，低优先级 VL 承载 KV-Cache 传输，杜绝干扰。
  - RDMA Work Request 提交延迟仅 `~1μs`，显著低于 CUDA 拷贝引擎的 `5-7μs`，对 Layerwise 产生的大量小块数据更友好。

#### 2. 自适应请求调度器（Adaptive Request Scheduler）
采用两级调度策略，同时平衡 **NIC 流量** 与 **GPU 计算负载**：
- **跨引擎调度（Inter-Engine）**：
  - 以 `token 数量` 作为负载代理指标。
  - **PE 调度**：优先分配给磁盘读取队列短且未超载的节点，避免 SNIC 闲置或打满。
  - **DE 调度**：分两步（跨组均衡 token 总数 → 组内根据 HBM 余量与 token 阈值分配），降低 HBM 耗尽与抢占风险。
  - **路径选择**：动态将 KV 读取任务分配给当前读取队列更短的一侧（PE 或 DE）。
- **引擎内调度（Intra-Engine）**：
  - 针对数据并行（DP）下 Attention 层同步等待问题，引入 **Compute Quota（计算配额）** 机制。
  - 预先 Profiling 拟合理论计算量与墙钟时间关系，动态调整 Batch 大小或触发 Chunked Prefill，使各 GPU Attention 执行时间对齐，显著减少同步 Bubble。

---
### 📊 四、 实验评估与结果
#### 1. 实验设置
- **模型**：DeepSeek-V3.2 660B (MoE)、DS 27B（内部缩放版）、Qwen2.5-32B（Dense）
- **负载**：真实生产环境 Agent 轨迹（上下文 32k~64k，平均命中率 98.7%）
- **基线**：SGLang+Mooncake、内部基础框架（Basic）、零 I/O 理想上限（Oracle）

#### 2. 核心结果
| 场景 | 指标 | DualPath 提升 |
|:---|:---|:---|
| **离线批量推理** | 任务完成时间 (JCT) | 最高加速 **1.87×**（DS 660B），性能逼近 Oracle |
| **在线服务** | 满足 SLO 下的吞吐量 (APS) | 平均提升 **1.96×**，TTFT 稳定，TPOT 无额外开销 |
| **P/D 比例敏感性** | 存储带宽验证 | DualPath 1P1D ≈ Basic 2P1D，直接证明瓶颈在于存储带宽而非算力 |
| **大规模扩展** | 1152 GPU (48P96D) | 离线近线性加速；在线吞吐提升 22 倍且延迟稳定，调度器 CPU 开销 <10 核 |

#### 3. 消融实验
- `Layerwise Prefill`：降低 JCT ~17.2%（缓解 HBM 瓶颈，掩盖传输延迟）
- `双路径加载`：降低 JCT ~38.2%（核心贡献，聚合全局存储带宽）
- `动态调度算法`：进一步降低至 ~45.6%，存储网卡负载均衡比从 `1.53` 优化至 `1.18`，Attention 同步 Bubble 显著减少。

---
### 📝 五、 总结、贡献与未来方向
#### ✅ 核心贡献
1. **首次明确界定** Agentic LLM 推理的 I/O 主导特性，并指出 PD 分离架构下存储带宽利用的严重不对称性。
2. 提出 **DualPath 双路径加载架构**，打破 Prefill-centric 传统，通过计算网络 RDMA 中转实现存储带宽全局池化。
3. 设计 **CNIC 中心化流量隔离** 与 **两级自适应调度器**，在保障推理延迟 SLO 的前提下实现 NIC 与 GPU 的双维负载均衡。
4. 在真实生产负载与千卡规模下验证了系统的有效性与可扩展性。

#### 🔮 局限与未来工作
- **动态资源配置**：Agent 负载在训练 Rollout 阶段高度动态（前半段 Prefill 压力极大），需支持 P/D 比例与并行策略的在线自适应调整。
- **工作集（Working Set）膨胀**：实际场景中工具调用延迟与请求间隔会导致 KV 工作集呈平方级增长，可能超出 DRAM 缓存容量，未来需结合分层缓存（DRAM+SSD）协同优化。
- **调度百分位延迟**：超大规模突发请求下的长尾 TTFT 仍有优化空间。

---
### 💡 一句话总结
**DualPath 通过“让 Decode 节点帮忙读缓存 + 计算网卡硬件 QoS 隔离 + 全局动态调度”，将原本卡在 Prefill 侧的存储 I/O 瓶颈转化为可全局调度的带宽池，在 Agent 多轮推理场景下实现近 2 倍吞吐提升，为下一代 I/O 密集型 LLM 服务架构提供了重要范式。**