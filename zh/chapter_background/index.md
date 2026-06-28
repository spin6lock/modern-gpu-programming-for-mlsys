(chap_background)=
# GPU 执行模型

:::{admonition} Overview
:class: overview

- 一个 kernel(内核)在一个线程层次结构(thread → warp → warpgroup → CTA → cluster → grid)上运行,跨越多个不同的存储空间(寄存器、SMEM、GMEM、TMEM)。
- 计算分为 CUDA cores 与 Tensor Cores(张量核);TMA(张量内存加速器)等专用引擎负责搬移喂给它们的数据。
- 一个 kernel 是一条流水线,把数据逐级送过这些存储空间,并在相互独立的计算引擎与数据搬运引擎之间交接工作;反复出现的目标,就是让这些引擎同时保持忙碌。
:::

要写出快速的 GPU 程序,重要的是先理解硬件本身,以及代码如何在该硬件上运行。本章概述 GPU 的执行模型:执行工作的线程层次结构、保存与搬运数据的存储空间,以及承担主要工作的计算引擎与数据搬运引擎。我们先把这些部件逐一介绍,然后把它们在一个 GEMM(通用矩阵乘)流水线中组合起来,以便看清数据和执行是如何在硬件中流动的。本书后续几乎每一项优化,都是用某种方式在这些相同部件之间安排工作。

现代 GPU 还包含许多专用硬件单元。为了先给一个直观印象,在深入每个部件之前,下面的交互演示展示了 Blackwell 流式多处理器(streaming multiprocessor,SM)内部的主要元素。你可以点击每个部件查看其细节。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互:Blackwell SM,展示其 warp/warpgroup、共享内存、Tensor Memory,以及 Tensor Core 与 TMA 引擎。*

## 执行层次结构

我们从执行工作的线程讲起。GPU 并不是把它的数千个线程呈现为一个扁平的线程池。相反,它把它们组织成一个嵌套的层次结构,之所以如此,是因为协作会在多个不同规模上同时发生。每一层的存在,都是为了在某一种规模上让协作变得廉价。下图展示了 Blackwell 上的层次结构;你可以点击每个层级来高亮它。

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:点击一个层级:thread → warp → warpgroup → CTA → cluster → grid。*

- **Thread(线程)**:执行的标量单元。每个线程有自己的程序计数器和自己的寄存器,并在其所在的 warp 内由一个 lane ID 标识。
- **Warp**:32 个线程,以 SIMT(*单指令多线程*)方式执行。一个 warp 的各 lane 一起发射同一条指令,但每个 lane 各自保留自己的寄存器,并且可以被单独掩码掉,这正是让单个 warp 的各 lane 能够走上不同分支的机制。
- **Warpgroup(线程束组)**:四个连续的 warp,即 128 个线程。Hopper 引入了 warpgroup,把它作为发射 warpgroup 级 MMA(`wgmma`)的单元,而在 Blackwell 上它又承担了第二个角色:它是 Tensor Memory 访问的协作单元,128 个线程一起把一个 TMEM 分块搬进或搬出寄存器。
- **CTA**(*Cooperative Thread Array*,即 CUDA 中所称的 thread block):硬件调度的基本单元。一个 CTA 运行在单个 SM 上,并在其中拥有一份私有的共享内存分配。多个 CTA 可以同时驻留在同一个 SM 上,此时它们共同瓜分该 SM 的共享内存容量。
- **Cluster(集群)**:一组协作的 CTA,可能分布在不同 SM 上。一个 cluster 内的 CTA 可以相互同步,并读写彼此的共享内存,这一能力被称为分布式共享内存(distributed shared memory)。

这些层级值得反复体会,因为与早期架构不同,Blackwell 的关键操作**并非全由同一组线程发射**。一次 TMA 拷贝由单个线程发起,再由硬件执行。一次 TMEM 到寄存器的加载是 warpgroup 分布式的:四个 warp 协作,各自搬动它那一部分 TMEM 分块。一次 `tcgen05` MMA 由一个被选出的线程提交,而一次集群级 MMA 会一次跨越两个 CTA。因此,每个操作都有它自己的天然粒度,执行该操作的线程集合,就是我们所说的该操作的**作用域(scope)**,这是本书反复回到的三个反复出现的设计要素(作用域、布局、派发)中的第一个。

## 存储空间

该层次结构中的线程,其速度只取决于送达它们的数据有多快,因此接下来我们看数据存放在哪里。不存在一种存储既大又快;物理规律迫使容量与速度之间做出权衡。因此 GPU 提供的是多种存储而非一种,每种都在该权衡的不同点上取得平衡,而 kernel 的工作正是通过这些存储来搬移数据。每个空间都有自己的容量、自己的延迟,以及自己关于谁可以访问它的规则。

| 存储空间 | 归属 | 角色 | 说明 |
|--------|-----------|------|-------|
| **Global (GMEM)** | 设备级 | 持久张量存储 | 大容量 HBM,被所有 SM 共享 |
| **Shared (SMEM)** | 每 CTA(单个 SM) | 分块中转 | 低延迟暂存;B200 上每 SM 最多 228 KB |
| **Tensor Memory (TMEM)** | 每 CTA | MMA 累加器存储 | Blackwell 新增;由 `tcgen05` 使用 |
| **Register File (RF)** | 每线程 | 标量与每线程分块片段 | 速度快;保存收尾/临时值 |

按顺序读下来,这些空间描绘出一条路径。本书中几乎每个 kernel 的数据通路都是 **GMEM → SMEM → (计算) → 寄存器 → SMEM → GMEM**,而对于 Tensor Core kernel,TMEM 位于这条路径的中间,在数学运算进行时保存累加器。

在这四种里,**Tensor Memory (TMEM)** 是唯一在 Blackwell 之前硬件上没有对应物的,它的完整细节留到 {ref}`chap_tensor_cores`。不过,理解它的动机是值得的。早期 GPU 把大型 MMA 累加器保存在寄存器里,在那里它们要竞争一种稀缺资源。Blackwell 则把 `tcgen05` 累加器输出写到 TMEM——一个 CTA 作用域、规模为 128 lane 乘最多 512 个 32位列的二维暂存区(该数组物理上驻留在 SM 上)。然后 kernel 必须在收尾之前,显式地把 TMEM 读回到寄存器。这一额外步骤并非没有代价,它的两个后果会在全书反复出现。其一是 TMEM 读取是**显式且 warpgroup 分布式**的,由 warpgroup 的四个 warp 协作完成。其二是 TMEM 与寄存器不同,必须被**显式分配和释放**。

### 集群内的分布式共享内存

集群是层次结构中唯一其成员可以跨越多个 SM 的层级,而这种跨度换来了其他层级所欠缺的一种存储能力。一个 CTA 运行在一个 SM 上,从该 SM 的共享内存中工作,但单个 CTA 的 SMEM 预算是有限的,而大的分块往往需要比单个 block 所能提供的更多的操作数存储或更多的复用。Hopper 的对策是 **thread block cluster**:一组协作比相互独立的 block 更紧密的 CTA,它们能够一起同步、读写彼此的共享内存,这一能力被称为**分布式共享内存(DSMEM)**。Blackwell 保留了集群并加以增强,加入了动态调度({ref}`chap_clc`)与 2-CTA 协作 MMA。

DSMEM 让一个 CTA 能够直接寻址并访问对端 CTA 的共享内存。一个线程可以命名对端 SMEM 中的一个位置,并把自己的 SMEM 中的一个分块整块拷贝过去,一旦字节落位,就触发一个完成屏障({ref}`chap_async_barriers`)。第三部分中的 2-CTA 集群 GEMM 正是建立在这一机制上,利用它在两个 CTA 之间共享操作数分块,而不必把它们绕回全局内存。

下图展示了一个 CTA 集群所带来的额外 DSMEM 跳转;点击某一块可以看到每个 CTA 各自拥有什么,以及跨 CTA 读取发生在何处。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互:一个 2-CTA 集群,其中每个 CTA 各拥有 A 的一半和 B 的一半,通过集群(DSMEM)读取对方的 B,这对 CTA 一起产出一个 256×256 的输出分块。*

## 计算:CUDA Cores 与 Tensor Cores

线程以及它们搬动的数据,最终必须在一个算术单元上汇合,而一个 SM 提供的是两种截然不同的数学引擎,而非一种。两者之间的分工,几乎决定了每一个 kernel 的写法,它们扮演着互补的角色。

- **CUDA cores** 是通用的 SIMT ALU。它们运行标量与向量指令,处理索引运算、逐元素数学、归约以及控制流,即环绕繁重矩阵运算之外的粘合逻辑。
- **Tensor Cores** 是固定功能单元,以*分块*粒度执行稠密矩阵乘加,在一条指令中计算 $D = AB + C$。

这种划分之所以重要,是因为 Tensor Cores 提供的算术吞吐量远高于 CUDA cores,在 FLOP/s 上大约有一个数量级(10× 或更多),因此稠密线性代数(GEMM、卷积、attention)只有在 Tensor Cores 上运行时才能达到峰值性能。于是,获得性能在很大程度上就是让那些 Tensor Cores 喂饱。从一个 GPU 代际到下一个代际发生变化的是 Tensor Cores *如何*被编程,以及它们的结果*落在何处*。Hopper 引入了异步 warpgroup MMA(`wgmma.mma_async`);Blackwell 的第五代 Tensor Core `tcgen05` 把它的累加器放在 Tensor Memory 而非寄存器中,我们在 {ref}`chap_tensor_cores` 中专门讨论它。

集群以两种方式扩展了这些引擎,这两种方式在 GEMM 各章中反复出现。**2-CTA 协作 MMA** 让两个 CTA 各自把它们的 SMEM 操作数贡献给一个更大的单一 Tensor Core MMA 分块。**TMA multicast(多播)**让数据搬运引擎的一次加载把同一个 GMEM 分块同时送达多个 CTA,消除了各自单独加载本会带来的冗余全局流量。两者都建立在前面介绍的分布式共享内存之上。

## GEMM 数据流水线

到目前为止,我们已经分别介绍了各个硬件单元。要看看它们如何协同,可以用一个典型的 GEMM(通用矩阵乘)流水线作为例子。下面的交互演示展示了一个三段式 GEMM 分块流水线所涉及的单元;点击某个动作(例如 `tma load`)可以高亮它在各硬件单元之间走过的数据通路。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互:Blackwell 上的 load → MMA → epilogue 流水线;点击某个动作,追踪它穿越各硬件单元的数据通路。*

单个 GEMM 分块流经三个阶段。

1. **加载。** 一次 TMA 拷贝({ref}`chap_tma`)把一个 A 或 B 操作数分块从 GMEM 流式加载到 SMEM。单个线程发起拷贝,预先记录预期到达多少字节。随着字节落位,TMA 引擎汇报进度,而一个完成屏障只在所有预期字节都送达时才翻转。
2. **计算。** 一次 `tcgen05` MMA({ref}`chap_tensor_cores`)从 SMEM 读出操作数分块,并把乘积累加进一个 TMEM 分块。由一个被选出的线程发起,并在数学运算完成时向一个屏障发信号。
3. **收尾。** warpgroup 把 TMEM 累加器读回到寄存器,把结果转换到输出 dtype,并存到 GMEM,常常通过经 SMEM 中转并发出一次 TMA store 来完成。

这样写出来,三个阶段看起来是严格顺序的,但慢 kernel 与快 kernel 之间的全部差别就在于**重叠(overlap)**。一个朴素的 kernel 确实按顺序执行各步(加载、等待、计算、等待、存储),于是在等待前一步时让每个引擎都闲置着。快的 kernel 则把它们流水化:当 Tensor Core 正在计算分块 `k` 时,TMA 引擎已经在取分块 `k+1`,而收尾正在忙于排空分块 `k-1`,于是三个引擎同时都保持着占用。让三个异步引擎安全地把工作交接给彼此,正是屏障与相位模型({ref}`chap_async_barriers`)的工作,而第三部分的 GEMM 阶梯正是建立在它之上。

## 接下来读什么

既然我们已看清了高层图景,就可以进入那些深入主要机制的章节:

- {ref}`chap_tensor_cores` 详细讲解 `tcgen05` 计算与 Tensor Memory。
- {ref}`chap_tma` 涵盖基于 TMA 的异步数据搬运。
- {ref}`chap_async_barriers` 介绍协调这些引擎的 mbarrier 与相位模型。
