---
type: paper
authors: ["Shao Wang", "Rui Ren", "Lin Gui"]
publish: "2026-04-07"
venue: "arXiv"
url: "http://arxiv.org/abs/2604.06370"
zotero: "zotero://open-pdf/library/items/NQFKRKB4"
created: "2026-04-28 14:45"
abstract: |-
  大型语言模型（LLM）的服务范式正迅速转向复杂的多智能体工作流，其中专业智能体在庞大的共享上下文上进行协作。虽然低秩适应（LoRA）使得这些专业智能体能够在单个基座模型上高效共置，但它却在服务过程中引入了关键的内存占用瓶颈。具体来说，独特的 LoRA 激活导致智能体之间的键值（KV）缓存出现差异，使得传统的针对共享上下文的前缀缓存失效。这导致了冗余的 KV 缓存维护，迅速饱和 GPU 容量并降低吞吐量。为解决这一挑战，我们引入了 ForkKV，一个面向多 LoRA 智能体工作流的服务系统，其核心是一种新颖的操作系统内存管理范式：带写时复制（CoW）的 fork。通过利用 LoRA 的结构特性，ForkKV 将 KV 缓存物理解耦为一个庞大的共享组件（类似于父进程的内存页）和轻量级的智能体特有组件（子进程的页）。为支持这一机制，我们提出了 DualRadixTree 架构，使新分叉的智能体能够继承庞大的共享缓存，并对其轻量级特有缓存应用 CoW 语义。此外，为确保高效执行，我们设计了 ResidualAttention，一种专用内核，可在片上 SRAM 中直接重建解聚的 KV 缓存。跨多种语言模型及不同任务的实际数据集的全面评估表明，ForkKV 实现了比最先进的多 LoRA 服务系统高达 3.0 倍的吞吐量，且对生成质量的影响可忽略不计。
tags: ["llm", "multi-agent", "serving", "lora", "kv-cache", "copy-on-write", "dualradix-tree", "residual-attention"]
---
