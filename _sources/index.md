# Modern GPU Programming For MLSys

机器学习系统是现代 AI 工作负载的核心。在这些系统中,性能往往取决于少数几个 GPU 内核的质量。注意力内核、LLM 预填充(prefill)与解码(decode)内核、低精度块缩放 GEMM、融合的 MoE 层以及其他大型融合内核,都直接决定了训练与服务两端的端到端速度。

然而,要让这些内核足够快,仅有优化技巧清单是远远不够的。现代 GPU 早已不再是旧设计的简单变体。近年的架构引入了更丰富的存储空间、新的访问模式以及日益特化的执行单元。要把它们编程好,我们既需要对硬件有清晰的心智模型,也需要具备构建高性能内核的实践理解。本书的目标正是同时培养这两者。

本书遵循一条简单的递进脉络:先理解 GPU 硬件,再学习我们将要使用的编程模型,最后一步步构建 SOTA(state-of-the-art)内核。我们的主要目标硬件是 Blackwell 这一代,主要贯穿全书的示例是快速矩阵乘(GEMM)与 FlashAttention。在此过程中,我们还会研究 GPU 优化背后的核心要素:数据布局(data layout)、异步数据搬运以及异步协调(asynchronous coordination)。

这些材料源自卡内基梅隆大学(Carnegie Mellon University)的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列。为了让这些思想更易于学习与运行,本书使用 **TIRx** 这一 Python DSL(领域特定语言)一步步构建真实可运行的 GPU 内核示例。TIRx 贴近硬件,让我们既能推理底层控制,又能通过可运行代码进行学习。

本书是开源的。欢迎通过 [GitHub 仓库](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)贡献内容、提交勘误与补充示例。

## 本书组织结构

- **第一部分,理解 GPU。** 这一部分介绍 GPU 的整体组织结构、编写快速内核的通用套路,以及数据布局、异步内存操作与协调等关键概念。它为全书后续内容奠定了硬件直觉基础。
- **第二部分,TIRx 概览。** 这一部分介绍 TIRx 的关键要素,它们构成了全书代码示例的基础。
- **第三部分,GEMM:从分块到 SOTA。** 一份完整的分块 GEMM 优化指南,通过 TMA 流水线、持久化调度、warp specialization(线程束特化)与 2-CTA 集群逐步搭建而成。
- **第四部分,FlashAttention 4。** 一个基于第三部分技术构建的完整注意力内核:两个 MMA 之间夹着 softmax、在线 softmax 重缩放、因果掩码(causal masking)以及 GQA。
- **参考。** TIRx 语言参考与编译器内部实现。

```{toctree}
:caption: 第一部分,理解 GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: 第二部分,TIRx 概览
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "第三部分,GEMM:从分块到 SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: 第四部分,FlashAttention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: 参考
:maxdepth: 1

appendix/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
tirx_guide/language_reference/index
```
