(chap_performance)=
# 是什么让内核变快

:::{admonition} Overview
:class: overview

- roofline(屋顶线)模型给出一个内核的性能上限。该上限由显存带宽或计算吞吐量决定。
- 算术强度(arithmetic intensity)决定哪一种上限适用,即每搬运一个字节所完成的有用算术运算量。
- 算术强度低意味着内核受限于内存(memory-bound)。主要的出路是搬运更少的字节、更多地复用数据、融合操作,或使用更小的 dtype。
- 算术强度高意味着内核可能受限于计算(compute-bound)。此时的主要任务就是让 Tensor Core(张量核)保持忙碌。
- 在现代 GPU 内核中,主要的杠杆是重叠(overlap)。只要依赖图允许,TMA、Tensor Core、收尾(epilogue)和存储就应该同时运行。
:::

一个内核只有相对于某个上限才谈得上快。一个像 330 TFLOP/s 这样的数字单看可能很大,但放到一块能在稠密 fp16 或 bf16 的 Tensor Core 工作上持续约 2 PFLOP/s 的 GPU 上,含义就完全不同了。如果没有一个上限,很难判断一个内核到底是接近了硬件极限,还是仍让芯片大部分处于空闲。

roofline 模型给出了这个上限。它把内核分解为两类基本活动:搬运字节和执行算术。如果内核无法足够快地搬运数据,显存带宽就成了限制;如果内核有足够的数据复用和足够的算术运算量,计算吞吐量就成了限制。

本章的数字以 NVIDIA B200 作为贯穿示例。沿用 {ref}`chap_background` 的约定,我们用取整后的上限来推理:稠密 fp16 或 bf16 的 Tensor Core 吞吐量约 2 PFLOP/s,HBM3e 带宽约 8 TB/s。具体数值取决于特定器件、时钟、功耗限制和测量设置,因此应将其视为数量级意义上的上限,而非数据手册上的常数。

## Roofline 模型

每个内核都在搬运数据并执行算术。roofline 模型用这两条路径中较慢的那条来约束内核。

计算上限是硬件的最大算术吞吐量。对于 B200 上的 Tensor Core GEMM,相关上限就是 Tensor Core 吞吐量。对于标量或逐元素(elementwise)内核,相关上限则可能是 CUDA core 吞吐量或其他功能单元。

内存上限是带宽乘以算术强度。如果一个内核每搬运一个字节只做很少的算术,显存带宽就会限制性能;如果每个字节上做很多运算,内存就不那么可能成为瓶颈。

基本的 roofline 约束为:

```text
attainable FLOP/s <= min(peak FLOP/s, memory bandwidth * arithmetic intensity)
```

算术强度为:

```text
arithmetic intensity = useful FLOPs / bytes moved
```

必须指明内存层级。对于 HBM roofline,字节数指 HBM 字节;对于 L2 roofline,指 L2 字节;对于 SMEM(共享内存)roofline,指共享内存字节。本章默认使用 HBM roofline。

在 roofline 图上,x 轴是算术强度,单位为每字节 FLOP 数;y 轴是可达性能。内存屋顶是一条斜线:

```text
performance = bandwidth * arithmetic intensity
```

计算屋顶是一条水平线:

```text
performance = peak FLOP/s
```

两者在屋脊点(ridge point)相交:

```text
ridge point = peak FLOP/s / bandwidth
```

用本章采用的 B200 取整数值:

```text
ridge point ≈ 2000 TFLOP/s / 8 TB/s
            ≈ 250 FLOP/byte
```

在 HBM roofline 下,算术强度低于该值的内核是 memory-bound 的。它无法达到峰值 Tensor Core 吞吐量,因为它无法每秒交付足够多的字节来喂饱那么多的算术运算。

算术强度高于该值的内核则可能是 compute-bound 的。此时显存流量不再是第一阶的限制,剩下的工作是把计算单元驱动得足够好,以接近那条水平屋顶。

roofline 模型真正有用的不是那张图本身,而在于它告诉程序员:哪一种资源是绑定(binding)的。一个 memory-bound 的内核不会因为其数学指令略微改善就变快;一个 compute-bound 的内核也不会因为节省了几个无关的字节就变快。第一步是搞清楚内核位于屋脊的哪一侧。

![一张 B200 roofline 图,带有若干示例工作负载,展示了内存屋顶、计算屋顶和屋脊点](../img/zh/roofline.png)

## 常见工作负载的算术强度

算术强度往往首先是一种算法性质,其次才是实现细节。在编写内核之前,通常就能做出一个粗略估计。

### 逐元素与归约

逐元素内核(如 GELU)和归约式内核(如 RMSNorm)会读写大量张量,但每个元素只做很少的 FLOP。

它们的算术强度很低,位于屋脊点左侧很远处。这类内核的最佳版本通常试图逼近内存带宽屋顶,而非 Tensor Core 计算屋顶。

对这些内核而言,重要的问题都很机械:

```text
Are the loads and stores coalesced?
Are bytes moved only once?
Can the operation be fused with a producer or consumer?
Can the dtype be smaller?
Can TMA or vectorized accesses help?
```

如果没有复用、也没有融合机会,内存屋顶就是真正的上限。

### GEMM

GEMM 是相反的情形。它的算术强度随问题规模增长,因为每个被加载的分块(tile)都可以被复用于许多乘加运算。

对于 `M = N = K` 的方阵 fp16 matmul,理想算术强度约为:

```text
AI ≈ 2N^3 / (3 * 2N^2)
   = N / 3 FLOP/byte
```

该估计假设 A 和 B 各读一次、C 写一次、beta 为零、片上复用完美,且没有额外的元数据、padding 或冗余流量。真实内核搬运的数据会多于这一理想模型,但该估计仍然有用。

在 `N = 4096` 时:

```text
AI ≈ 4096 / 3
   ≈ 1365 FLOP/byte
```

这远在 B200 大约 250 FLOP/byte 的屋脊点右侧。因此在 HBM roofline 下,大型 GEMM 是 compute-bound 的。目标不再仅仅是减少 HBM 流量,而是要使用 Tensor Core、让其保持被喂饱,并把数据搬运与计算重叠起来,从而使计算屋顶变得可达。

这就是为什么即便 GEMM 算术强度很高,一个朴素(naive)的 GEMM 仍可能很慢:算法允许高性能,但实现可能让 Tensor Core 空闲。

### Attention

Attention 介于这两个极端之间。它的算术强度取决于序列长度、头维度、分块化(tiling)、掩码,以及中间张量是否被实体化(materialize)。

标准 attention 中的关键问题是得分矩阵(score matrix)。如果内核把得分矩阵写入 HBM、之后又读回来,它就把一个很大的中间量往返搬过了显存。Flash Attention({ref}`chap_flash_attention`)通过把相关分块留在片上、避免这次 HBM 往返,提高了算术强度。

所以 attention 的优化一部分是 roofline 问题,一部分是调度问题。先改变算法以减少进入 HBM 的字节,再调度内核使剩余的搬运与计算相互重叠。

## 当算术强度低时

如果一个内核位于屋脊左侧,它就是 memory-bound 的。Tensor Core 或 CUDA core 可能处于空闲,因为瓶颈在于字节,而不在于算术指令。

有两种应对方式。

第一种应对是提高算术强度。这是杠杆更大的路径,因为它可能把内核推向 compute-bound 区域。

最重要的技术是融合(fusion)。算术强度低的一个常见来源是:把一个中间张量写入 HBM、又在下一个操作里立即读回。把生产者(producer)和消费者(consumer)融合,就能把这个中间量留在寄存器、SMEM 或 TMEM(张量内存)里,HBM 往返随之消失。

例子包括:

```text
GEMM plus elementwise epilogue
normalization folded into a neighboring op
attention computed without materializing the full score matrix
```

第二种技术是为复用而分块(blocking for reuse)。如果一个分块被加载一次、并在被淘汰前被使用多次,每个字节就能支撑更多的算术运算。GEMM 的高算术强度正是源于这种复用。其他工作负载只要对某个分块有重复使用,就可以套用同样的思路。

第三种技术是减少每个值所占的字节数。从 fp32 转向 fp16、fp8 或 fp4,会减少流量并提高每字节的 FLOP 数。当格式需要元数据、缩放因子或额外的转换工作时,实际增益会小于 dtype 本身的比例。块缩放(block-scaled)的 fp8 与 fp4 就是这样的例子。即便如此,更小的 dtype 往往仍是在 roofline 上把内核右移的最直接手段之一。

第二种应对是接受内存屋顶并试图达到它。有些内核没有足够的工作可融合,也没有足够的复用可利用。纯粹的拷贝、简单的逐元素操作,或对大张量的单遍归约,可能本质上就是 memory-bound 的。

在这种情况下,目标不是击穿屋顶,而是打满它。

这意味着:

```text
move each byte once
avoid redundant reads
use coalesced or vectorized accesses
use TMA for regular bulk tiles
keep enough memory requests in flight
use smaller storage dtypes when the algorithm allows it
```

一旦一个 memory-bound 的内核达到了内存屋顶,进一步的计算优化就不再有帮助。要变得更快,唯一的办法是改变算法以搬运更少的字节。

## 优化阶梯

roofline 说的是什么是可能的,而不说是达到该极限有多容易。

一个大型 fp16 GEMM 在理论上可能是 compute-bound 的。这只意味着 HBM 屋顶不是主要限制,并不意味着任何实现都能达到 Tensor Core 屋顶。要弥合这个差距,需要正确的指令、布局(layout)、暂存(staging)、同步与调度。

第三部分中的 GEMM 内核在 B200 上把这一点呈现为一系列步骤({ref}`chap_gemm_advanced`)。每一步都保持相同的基本算法,只改变分块的计算方式或调度方式。

GEMM 阶梯中第一个被测量到的显著跳变,是从线程拷贝的分块路径切换到由 TMA 支撑的路径。TMA 把规则的 GMEM -> SMEM 分块搬运从 CTA(协作线程阵列)线程上卸载下来,让内核通过硬件托管的大块拷贝来喂饱 Tensor Core。

在那第一次跳变之后,主要的改进来自重叠与调度。TMA 把未来的分块搬入共享内存,`tcgen05.mma` 异步运行,收尾把之前的结果排空。软件流水线(software pipelining)和线程束特化(warp specialization)把这些片段编排起来,使硬件引擎同时保持活跃。

也并没有规定每个中间步骤本身都必须更快。像线程束特化这样的步骤,可能会暂时把资源花在一个不能立即改善数字的结构上。但如果它使得更简单的结构无法表达的后续重叠成为可能,它仍然可能是正确的一步。

![B200 上的 GEMM 优化之旅:从同步分块基线,经 TMA、线程束特化、CTA 集群(CTA cluster)和多消费者执行,所测得的各个点](../img/zh/gemm_perf.png)

## 重叠是主要的杠杆

一旦一个 GEMM 已经 compute-bound 并用上了 Tensor Core,剩余的差距通常来自空闲时间。

一个简单的内核可能是这样的:

```text
load tile k
compute tile k
store tile k
load tile k + 1
compute tile k + 1
store tile k + 1
```

这种调度会让硬件空闲。加载运行时,Tensor Core 在等待;Tensor Core 运行时,拷贝引擎可能空闲;存储排空时,两者可能都在等待。

流水线化的内核则试图把相互独立的阶段放在一起运行:

```text
load tile k + 1
compute tile k
store tile k - 1
```

这正是本书后续所用 Blackwell 内核结构背后的核心思想。TMA 负责异步数据搬运,`tcgen05.mma` 负责异步的 Tensor Core 工作,收尾与存储负责输出侧,`mbarrier`(内存屏障)对象把各个阶段连接起来,使得每个消费者只有在真正需要其数据时才等待。

关键不在于消除依赖,而在于围绕依赖来调度。分块 `k` 的 MMA 在分块 `k` 被加载之前无法开始;分块 `k` 的收尾在分块 `k` 的 MMA 完成之前无法读取累加器(accumulator)。但分块 `k + 1` 的加载通常可以在分块 `k` 的 MMA 进行中并行运行,而分块 `k - 1` 的存储通常也可以同时排空。

这就是为什么后续那么多章节聚焦于异步机制:

```text
TMA for global memory to shared memory movement
mbarriers for load completion and resource handoff
tcgen05 for asynchronous Tensor Core compute
TMEM for long-lived accumulators
warp specialization to separate producer and consumer roles
clusters for larger cooperative tiles and multicast
```

它们是不同的机制,但服务于同一个调度目标:让有用的工作同时在多条硬件路径上运行。

## 占用率与资源压力

重叠并不是唯一的延迟隐藏机制。更古老、也更通用的机制是占用率(occupancy)。

占用率是驻留在单个 SM(流式多处理器)上的工作量。如果某个 warp(线程束)停顿,调度器可以运行另一个就绪的 warp。这通过维持一个可用独立 warp 池来隐藏延迟。

占用率受每 SM 资源的限制。主要限制是寄存器、共享内存、warp 槽位和 CTA 槽位。一个每线程使用大量寄存器、或每 CTA 占用大量共享内存的内核,占用率可能很低,因为只有少量 CTA 或 warp 能装得下 SM。

许多现代 Tensor Core 内核会有意以降低占用率的方式消耗资源。多级共享内存流水线消耗 SMEM,大寄存器分片消耗寄存器,TMEM 分配消耗张量内存容量,线程束特化可能为生产者或消费者角色预留整个 warp。

这种权衡是刻意为之。这些内核不靠驻留大量无关 warp 来隐藏延迟,而是通过在较少数量的驻留 CTA 内部进行显式重叠来隐藏延迟。一个低占用率的内核,只要其流水线能让 TMA、Tensor Core 和存储都保持忙碌,就仍然可以很快。

两种方法都不是普遍更优。有些内核需要高占用率,因为它们的内存访问不规则,或显式重叠有限;另一些则需要深度暂存与特化,因为那是高效喂饱 Tensor Core 的唯一途径。正确的问题不是占用率是否高,而是处于活跃的硬件单元是否被保持忙碌。

## 这为后续带来了什么

本书余下部分会不断回到同一个诊断:

```text
Which roof is this kernel under?
What resource is binding?
What change moves the kernel closer to that roof?
```

对 memory-bound 的内核,答案通常是更少的字节和更好的带宽利用。这意味着融合、合并访问(coalescing)、向量化访问、在适用处使用 TMA,以及更小的 dtype。

对 compute-bound 的 GEMM,答案是先上 Tensor Core,再谈重叠。内核必须暂存操作数、派发异步 MMA 工作、让流水线保持填满,并在不阻塞计算路径的前提下排空结果。

对 Flash Attention,第一步是通过把得分与概率分块留在片上来提高算术强度。之后,它使用与 GEMM 相同的重叠工具:分块化的数据搬运、共享内存暂存、异步计算,以及谨慎的资源交接。

这就给出了一个实用的优化工作流:估算算术强度,定位屋顶,判断内核是 memory-bound 还是 compute-bound,然后优化真正设定上限的那项资源。

没有这一步,内核优化就变成了瞎猜。有了它,每一次改动就都有了理由:要么它提高了算术强度,要么它把内存路径推向带宽峰值,要么它减少了计算屋顶下的空闲时间。
