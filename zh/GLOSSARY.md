# 术语表(GLOSSARY)

所有 subagent 必读并对齐。**缩写与专有名词保留英文,首次出现用半角括号补中文释义;后续沿用英文缩写。** 普通可译术语见「译法」列。

## A. 保留英文(首现括注中文,之后用英文)

| 英文 | 首现释义 | 备注 |
|---|---|---|
| GEMM | 通用矩阵乘(General Matrix Multiply) | 全书核心示例 |
| MMA | 矩阵乘加(Matrix Multiply-Accumulate) | 含 `tcgen05.mma` |
| TMA | 张量内存加速器(Tensor Memory Accelerator) | Blackwell 硬件异步拷贝引擎 |
| TMEM | 张量内存(Tensor Memory) | Blackwell 专用累加存储 |
| SMEM | 共享显存 / 共享内存(Shared Memory) | |
| GMEM | 全局显存(Global Memory / HBM) | |
| CTA | 协作线程阵列(Cooperative Thread Array) | |
| CTA cluster | CTA 集群 | 跨 CTA 协作 |
| warp | 线程束 | 一律用英文 |
| warpgroup | 线程束组(4 个 warp) | 一律用英文 |
| Tensor Core | 张量核 | |
| Blackwell | Blackwell(架构代号) | 不译 |
| mbarrier | 内存屏障 / 屏障 | 保留 |
| swizzle | swizzle | 保留英文;必要时首现说明为地址重映射 |
| MoE | 混合专家(Mixture-of-Experts) | |
| GQA | 分组查询注意力(Grouped-Query Attention) | |
| FlashAttention | FlashAttention | 不译 |
| softmax | softmax | 不译 |
| TIRx | TIRx(Python DSL) | 不译 |
| DSL | 领域特定语言(Domain-Specific Language) | |
| TFLOPS | 每秒万亿次浮点运算 | 单位,保留 |
| GPU / CPU / HBM / DRAM / L1 / L2 | — | 保留 |
| fp16 / fp32 / bf16 / fp8 / int8 | — | 保留 |

## B. 译为中文(普通技术词,按列译法,不再括注英文)

| 英文 | 译法 |
|---|---|
| kernel | 内核(上下文为「算子核」时也可) |
| tile / tiling | 分块 / 分块化 |
| data layout | 数据布局 |
| layout | 布局 |
| scope | 作用域 |
| dispatch | 派发(动词)/ 派发路径(名词) |
| software pipelining | 软件流水线 |
| pipeline / pipeline stage | 流水线 / 流水级 |
| pipelining | 流水化 |
| persistent kernel | 持久化内核 |
| warp specialization | 线程束特化 |
| producer / consumer | 生产者 / 消费者 |
| epilogue | 收尾 |
| tile scheduler | 分块调度器 |
| accumulator | 累加器 |
| register / registers | 寄存器 |
| thread | 线程 |
| lane | lane | 保留英文 |
| bank | bank | 保留英文 |
| streaming multiprocessor (SM) | 流式多处理器 |
| prefetch | 预取 |
| double buffering | 双缓冲 |
| copy / async load | 拷贝 / 异步加载 |
| throughput | 吞吐量 |
| latency | 延迟 |
| bandwidth | 带宽 |
| saturate | 打满 / 饱和 |
| optimization | 优化 |
| lowering | 降级(编译) |
| compiler | 编译器 |
| schedule / scheduling | 调度 |
| coordinate / coordination | 协调 |
| asynchronous coordination | 异步协调 |
| async barrier | 异步屏障 |
| online softmax | 在线 softmax |
| rescaling | 重缩放 |
| causal masking | 因果掩码 |
| prefill | 预填充 |
| decode | 解码 |
| block-scaled | 块缩放 |
| low-precision | 低精度 |
| mental model | 心智模型 / 直观模型 |
| recipe | 套路 / 方法 |
| state-of-the-art (SOTA) | 当前最优 / SOTA |

## C. 易错点

- **warp / warpgroup**:全书保留英文,不要译成「线程束」并替换掉英文(若替换会让与代码/硬件文档脱节)。首现可补释义,正文用英文。
- **warp specialization**:作为固定技术短语时译为「线程束特化」;若与具体 warp/warpgroup 角色并列,可保留英文并括注中文。
- **TMA vs TMEM**:TMA = 异步拷贝引擎(动);TMEM = 张量内存(存储,静)。勿混。
- **swizzle**:保留英文,不要译成「混洗」或「混淆」。需要解释时写作「swizzle 地址重映射」。
- **CTA vs SM**:CTA 是编程/调度单元,SM 是硬件单元。勿互译。
- **tile**:作名词=分块,作动词 tiling=分块化。灵活处理。
- **GEMM 的 $D=AB^\top$**:数学不译,周围叙述里 GEMM 保留英文。
