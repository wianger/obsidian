---
type: paper
authors:
  - Zhuohang Bian
  - Feiyang Wu
  - Chengrui Zhang
  - Hangcheng Dong
  - Yun Liang
  - Youwei Zhuo
publish: 2026-04-03
venue: arXiv
url: http://arxiv.org/abs/2604.03143
zotero: zotero://open-pdf/library/items/U63T7YH6
created: 2026-05-06 18:39
abstract: 多智能体LLM应用以同步回合制方式组织执行，中央调度器收集所有智能体的输出并重新分发合并后的上下文。这种AllGather通信模式造成了大量的KV缓存冗余，因为每个智能体的提示包含相同的共享输出块，但现有重用方法无法有效利用这一点。我们提出TokenDance，一种通过利用All-Gather模式实现集体KV缓存共享来扩展并发智能体数量的系统。TokenDance的KV收集器在单次集体步骤中完成整个回合的KV缓存重用，因此无论智能体数量多少，重用共享块的代价只需支付一次。其差异感知存储将兄弟缓存编码为相对于单个主副本的块稀疏差异，在代表性工作负载上实现了11-17倍的压缩。在GenerativeAgents和AgentSociety上的评估表明，在SLO要求下，TokenDance支持的并发智能体数量比采用前缀缓存的vLLM多2.7倍，每个智能体的KV缓存存储减少高达17.5倍，并且相对于每请求位置无关缓存实现了高达1.9倍的预填充加速。
tags:
  - multi-agent
  - llm
  - kv-cache
  - all-gather
  - difference-aware-storage
  - prefix-caching
---
### 📖 一、论文基本信息
- **标题**：`TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing`
- **作者/单位**：北京大学、上海交通大学（Zhuohang Bian 等）
- **发表年份**：2026（预印本/会议投稿版本）
- **核心目标**：突破多智能体LLM应用中GPU内存墙的限制，显著提升单卡可支持的**并发智能体数量**。

---
### 🔍 二、研究背景与核心痛点
#### 1. 多智能体应用的典型通信模式：All-Gather
当前主流多智能体框架（如 GenerativeAgents、AgentSociety、OpenClaw 等）通常按**同步轮次（Synchronized Rounds）**组织执行：
- 每轮所有智能体生成输出；
- 中央调度器收集所有输出，拼接成共享上下文；
- 将拼接后的上下文分发给每个智能体，作为下一轮的输入。
这种数据流被称为 **All-Gather 模式**。

#### 2. 现有系统的致命瓶颈：KV Cache 冗余爆炸
在 All-Gather 模式下，每个智能体下一轮的 Prompt 都由两部分组成：
- `私有历史（Private History）`：各智能体不同，长度不一
- `共享输出块（Shared Output Blocks）`：所有智能体完全相同

由于私有历史长度不同，**相同的共享块在不同请求中会落在不同的绝对位置上**。这导致：
- **前缀缓存（Prefix Caching，如 vLLM）**：一旦前缀分叉就完全失效。
- **位置无关缓存（PIC，如 CacheBlend/EPIC）**：虽能跨位置复用，但仍是**按请求独立处理**。N个智能体就要对同一份共享块执行N次 RoPE 旋转、N次关键位置筛选，计算开销线性增长。
- **存储浪费**：复用后，各智能体的 KV Cache 相似度高达 91%~97%，但现有系统仍为每个智能体保存一份完整的密集 Cache。内存消耗随智能体数量 `O(N)` 增长，迅速耗尽 GPU 显存，触发频繁换页/抢占，延迟飙升。

**核心矛盾**：多智能体应用天然具有“轮次级共享”结构，但现有 Serving 系统仍以“单个请求”为优化粒度，导致计算与存储双重冗余。

---
### 🛠️ 三、TokenDance 核心设计
TokenDance 的核心思想是：**将优化粒度从“单个请求”提升为“整个 All-Gather 轮次”**。系统由四大模块构成：

#### 1. 轮次感知提示接口（Round-Aware Prompt Interface）
- 在应用层组装 Prompt 时，在逻辑块之间插入保留分隔符 `<TTSEP>`。
- 运行时将传统的“固定大小块哈希”替换为**基于段的哈希（Segment-based Hashing）**。
- **效果**：即使共享块在不同请求中绝对位置不同，也能被识别为同一内容段，为后续集体复用提供结构可见性。

#### 2. 集体 KV Cache 复用（Collective KV Cache Reuse）
- **请求分组**：将同一轮次中长度兼容、槽位不冲突的请求划分为一个 Group。
- **层间锁步执行**：在每一层，将组内所有请求的 Q/K 张量拼接，**仅执行一次**批量 RoPE 旋转和一次关键位置差异分析（Important-Position Selection）。
- **按需刷新**：仅对每个请求中差异显著的位置重新计算 KV，其余位置直接复用旋转后的缓存值。
- **效果**：将原本 `O(N)` 的复用分析开销摊还为 `O(1)`。计算代价不再随智能体数量线性增长。

#### 3. 差分感知存储（Diff-Aware Storage）
- **Master-Mirror 布局**：从组内选出一个与公共结构最接近的请求作为 `Master`（保存完整密集 KV Cache），其余请求作为 `Mirror`。
- **块级稀疏差分（Block-Sparse Diff）**：Mirror 不存完整数据，仅记录与 Master 不同的块索引及对应的 K/V 修正值。差异通常集中在私有历史段或块边界，仅占全量的 10%~20%。
- **效果**：单轮 N 个智能体的存储成本从 `N 份完整 Cache` 降至 `1份 Master + (N-1)份稀疏 Diff`，压缩比达 **11~17.5倍**。

#### 4. 融合差分恢复（Fused DiffRestore）
- 传统做法：读取 Master → 拷贝到新 Buffer → 覆盖 Diff → 写入 GPU，产生额外密集读写。
- TokenDance 做法：在 GPU 层间传输流水线中，使用 **Ping-Pong 双缓冲**。加载 Master 块的同时，直接在 SM 内存中应用稀疏 Diff 并执行 RoPE 位置恢复，随后直接写入 Paged KV Cache。
- **效果**：避免在关键路径上物化完整 Mirror，恢复延迟比密集重建低 **1.3~2.6倍**，且与 FlashAttention 的 Tile 大小对齐，无额外 reshape 开销。

---
### 📊 四、实验评估与关键结果
- **测试环境**：NVIDIA A100 80GB，模型 Qwen2.5-7B / 14B
- **工作负载**：GenerativeAgents（短历史/少智能体）、AgentSociety（长历史/多智能体）
- **基线系统**：vLLM（Prefix Caching）、CacheBlend（普通路径 & 完整PIC恢复）
- **核心指标**：在延迟 SLO（1500ms）下支持的最大并发智能体数、KV Cache 占用、Prefill 吞吐、精度影响。

| 维度 | 关键结果 |
|------|----------|
| **并发扩展性** | 在相同延迟 SLO 下，TokenDance 支持的并发智能体数最高达基线的 **2.7倍**。优势随智能体数量和模型规模增大而显著放大。 |
| **计算加速** | 集体复用相比串行 PIC，Prefill 阶段最高加速 **2.57×**（10 agents, QPS=1）。高 QPS 下仍保持 1.3~1.5× 加速。 |
| **内存压缩** | 7B 模型压缩比 **11.2×**，14B 模型达 **17.5×**。每个 Mirror 平均仅 50~60 个块（32 token/块）与 Master 不同。内存增长从 `O(N)` 逼近 `O(1)`。 |
| **恢复开销** | Fused DiffRestore 比 Dense Restore 快 **1.3~2.6×**，压缩收益在在线推理关键路径上完全保留。 |
| **精度影响** | 与底层 PIC 方法（CacheBlend）输出完全一致。部分场景的微小发散源于 PIC 本身的选择性重计算数值扰动，**TokenDance 未引入额外精度损失**。 |

---
### 💡 五、总结与学术/工程价值
1. **范式转变**：首次指出多智能体 LLM 服务的瓶颈不在单请求调度，而在**轮次级通信结构未被 Serving 栈感知**。将优化单元从 Request 提升到 Round，是系统设计的本质突破。
2. **双管齐下**：同时解决“计算冗余”（Collective Reuse 摊还 RoPE/Diff 分析）与“存储冗余”（Master-Mirror 差分压缩），形成完整闭环。
3. **工程落地友好**：基于 vLLM + LMCache 实现，仅需应用层插入 `<TTSEP>` 分隔符，非 All-Gather 负载自动降级无性能损失。代码增量仅 ~3.5K 行。
4. **未来启示**：论文在 Conclusion 中明确提出：**通信模式（Communication Pattern）应成为 LLM Serving 系统的一等公民**。随着 Multi-Agent、Multi-Modal、Workflow 编排的普及，Pattern-Aware 的服务架构将成为下一代推理引擎的重要演进方向。

[[ForkKV - Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache]]