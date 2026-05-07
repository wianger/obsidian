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
tags: ["agentic-workflows", "multi-agent", "batch-processing", "kv-cache", "query-optimization", "llm-serving", "cpu-gpu-pipelining"]
---
