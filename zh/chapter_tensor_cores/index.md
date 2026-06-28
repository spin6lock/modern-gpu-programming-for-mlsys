(chap_tensor_cores)=
# Tensor Core:`tcgen05`

:::{admonition} Overview
:class: overview

- `tcgen05` 是 Blackwell 的 Tensor Core(张量核)指令族。其 MMA(矩阵乘加)指令以协作方式完成分块矩阵乘加,指令由一个被选中的 thread(线程)提交。
- 累加器存放在 TMEM(张量内存)而非寄存器中。收尾阶段随后通过 `tcgen05.ld` 把它读回寄存器。
- `cta_group::1` 与 `cta_group::2` 控制一次 MMA 由一个 CTA(协作线程阵列)还是两个 CTA 协作完成。该选择还会改变 M 维度到 TMEM 的映射方式。
- 块缩放(block-scaled)MMA 模式(例如 `mxfp8` 和 `nvfp4`)会增加缩放因子操作数。数据操作数位于 SMEM(共享显存),而缩放因子则经 TMEM 暂存。
:::

稠密线性代数是现代 GPU 完成大部分有效计算的地方。普通的 CUDA core(核)矩阵乘无法逼近芯片标称的峰值({ref}`chap_background`)。快速的 GEMM(通用矩阵乘)与 attention(注意力)内核通过以正确的分块形状、布局和同步方式喂给 Tensor Core 来达到该峰值。

其基本思想自 Volta 起并未改变。Tensor Core 消费矩阵分块,把它们相乘并累加结果。代与代之间变化的是:指令如何派发、操作数如何布局、以及累加器存放在何处。

Blackwell 对最后这一点做了大幅改动。`tcgen05` 的累加器不再作为长寿命的寄存器片段保留,而是写入 Tensor Memory,即 TMEM({ref}`chap_tmem`)。这一改动影响整个内核。MMA 写入 TMEM。完成情况以异步方式跟踪。收尾阶段随后把累加器从 TMEM 加载出来,并转换回它进行类型转换与存储所需的寄存器片段。

本章聚焦于计算指令本身。TMA({ref}`chap_tma`)负责把操作数搬入 SMEM。TMEM 负责保存累加器以及部分缩放因子操作数。`tcgen05.mma` 则是介于这两次访存动作之间的 Tensor Core 操作。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示:`tcgen05` 累加器行为。切换 A 或 B 的转置,选择输出宽度 `N`,并逐步推进 `K` 次迭代,观察部分和在 TMEM 中累积的过程。*

## `tcgen05` MMA

`tcgen05` MMA 是 Blackwell 的 Tensor Core 矩阵乘加指令。它是一条协作指令。其工作针对一个 warpgroup(线程束组)完成,在某些模式下还可涉及同一 cluster(集群)中的两个 CTA。该指令并非由每个 thread 独立派发,而是由一个被选中的 thread 代表参与组提交该操作。

把 MMA 拆成三个问题来看会更容易理解。

第一个问题是谁在协作。普通模式使用一个 CTA,写作 `cta_group::1`。更大的模式使用 cluster 中的两个 CTA,写作 `cta_group::2`。两种情形下,该指令都代表针对一个分块的一次 Tensor Core 操作,而不是某个 thread 的一次标量操作。

第二个问题是操作数与结果存放在哪里。数据操作数通常位于 SMEM。某些变体也可以从 TMEM 读入 A 操作数。累加器写入 TMEM。操作数布局必须与 Tensor Core 期望的布局一致,包括数据操作数所使用的经过 swizzle(混排)的共享显存布局({ref}`chap_data_layout`)。

第三个问题是如何观察到完成。`tcgen05.mma` 是异步的。派发 MMA 并不意味着乘加已经完成。该指令在操作提交后即返回,而 Tensor Core 继续运行。内核通过一个 commit group(提交组)和一道 `mbarrier`(内存屏障)来获知结果何时就绪({ref}`chap_async_barriers`)。

正是这种异步行为使重叠成为可能。一个快速的内核不会派发一条 MMA 后立即停顿直到它完成。它可以派发 MMA、开始准备后续的分块,并只在真正需要结果时才等待。代价是每一次交接都必须显式完成。如果收尾阶段在 MMA 完成屏障触发之前读取 TMEM,那就是读得太早了。

## 累加器位于 TMEM

在 Ampere 和 Hopper 上,累加器以寄存器形式暴露给程序。MMA 产生一个按 lane(通道)划分的寄存器片段,收尾阶段直接消费该片段。这种方式简单,但累加器的大小被绑定到每个 thread 的寄存器预算上。

Blackwell 打破了这一绑定。`tcgen05.mma` 把其累加器写入 TMEM——一种 Blackwell 上作用域为 CTA 的存储空间。累加器可以在计算阶段一直留在 TMEM 中,收尾阶段随后用 `tcgen05.ld` 把它加载回寄存器。

这改变了内核的形态。寄存器片段在边界处仍然重要。收尾阶段仍然需要寄存器,以便进行类型转换、逐元素运算以及存储结果。但长寿命的累加器状态不再是寄存器分配问题,而是一个 TMEM 分配与布局问题({ref}`chap_tmem`)。

这就是为什么必须把 `tcgen05` 与 TMEM 放在一起理解。MMA 指令决定计算哪个分块。TMEM 决定累加器落在何处。收尾阶段必须使用与之匹配的加载路径,才能以它期望的寄存器布局恢复出累加器。

## `cta_group::1` 与 `cta_group::2`

`tcgen05` MMA 既可在 `cta_group::1` 模式下运行,也可在 `cta_group::2` 模式下运行。

在 `cta_group::1` 中,一个 CTA 独占该 MMA。其操作数位于该 CTA 的 SMEM,其累加器写入该 CTA 的 TMEM。

在 `cta_group::2` 中,cluster 内的两个 CTA 协作完成一个 MMA 分块。每个 CTA 有自己的 SMEM 和自己的 TMEM。累加器并不存储在一个横跨两个 CTA 的物理 TMEM 区域里,而是被拆分到两个 CTA 之间,每个 CTA 各持有一部分。偶数 CTA 负责派发指令并为这对 CTA 提交完成屏障。

这一选择很关键,因为它改变了逻辑累加器分块 `C(M, N)` 到 TMEM 的映射方式。TMEM 有 128 条硬件 Lane 行以及最多 512 条硬件 Col 列。在 TIRx 的布局记法中,这些轴写作 `TLane` 与 `TCol`。MMA 模式决定 `C` 的行与列如何被放置到这些 TMEM 轴上。

有四种值得记住的有用情形。

下面的图遵循演示的配色约定:紫色标记 SMEM 操作数,橙色标记 TMEM 累加器状态,绿色标记 Tensor Core MMA 路径。CTA 的身份通过标签和位置来区分,而不是改变这些硬件颜色。

### `cta_group::1`,`M = 128`

这是最简单的情形。一个 CTA 计算一个 128 行的分块。TMEM 也有 128 条 Lane 行。因此映射是直接的:累加器的第 `m` 行映射到 Lane `m`,而 N 维度映射到 TMEM 列。

结果填满 128 条 Lane 行乘 N 条 Col 列。这是基准图景。该 CTA 在 SMEM 中持有 A 和 B,并在其 TMEM 中持有完整的累加器分块。

![cta_group::1,M=128:第 m 行直接映射到 TMEM Lane m](../img/mma_cg1_m128.svg)

### `cta_group::1`,`M = 64`

当 `M = 64` 时,累加器只有 64 行,但 TMEM 仍有 128 条 Lane 行。硬件并不会简单地把第 0 到 63 行装进 Lane 0 到 63,而是把它们分散到 128 条 Lane 上,排成四段、每段 16 行。

第 0 到 15 行进入 Lane 0 到 15。第 16 到 31 行进入 Lane 32 到 47。第 32 到 47 行进入 Lane 64 到 79。第 48 到 63 行进入 Lane 96 到 111。

这样在 Lane 16 到 31、48 到 63、80 到 95 以及 112 到 127 处留下了空隙。这些空隙是有意为之。在另一种 Lane 对齐方式下,另一个独立的 `M = 64` MMA 可以占据互补的 Lane。这使得两个较小的 M 分块可以共享 128 条 Lane 的 TMEM 结构而互不踩踏。

N 维度仍然映射到 TMEM 列。不寻常之处仅在于 M 行在 Lane 上的排布方式。

![cta_group::1,M=64:四段 16 行,以 Lane 步长 32 排布,为另一个对齐的 M=64 分块留出空间](../img/mma_cg1_m64.svg)

### `cta_group::2`,`M = 256`

当 M 维度大于一个 CTA 所能自然容纳的大小时,MMA 可以使用 `cta_group::2`。对于 `M = 256`,拆分是直接的。CTA 0 持有第 0 到 127 行。CTA 1 持有第 128 到 255 行。

每个 CTA 使用自己的 TMEM Lane 第 0 到 127 行以及全部 N 列。在物理上,这是两个独立的 128 行 TMEM 区域,分别位于每个 CTA 中。在逻辑上,它们组成一个 256 乘 N 的累加器分块。

每个 CTA 还提供与其 M 行对应的那部分 A。B 按模式要求对两个 CTA 都可见。偶数 CTA 负责派发 MMA 并为这对 CTA 提交完成屏障。

这就是 {ref}`chap_gemm_advanced` 中双 CTA cluster GEMM 所使用的模式。

![cta_group::2,M=256:M 连续拆分到两个 CTA,每个 CTA 128 行](../img/mma_cg2_m256.svg)

### `cta_group::2`,`M = 128`

`cta_group::2`、`M = 128` 模式仍然使用两个 CTA,但 M 维度更短。由于总共只有 128 行,每个 CTA 分得 64 行 M。

剩余的 lane 容量用于打包 N 维度。在每个 CTA 内部,N 的一半占据 Lane 0 到 63,另一半占据 Lane 64 到 127。这样每个 CTA 即便只拥有 64 行 M,也能用满全部 128 条 Lane 行。

因此该拆分包含两部分。M 跨这对 CTA 拆分,每个 CTA 64 行。N 则在每个 CTA 内部,沿 TMEM Lane 行的下半部分和上半部分再各拆一次。

![cta_group::2,M=128:每个 CTA 64 行 M,N 的两半分别叠放在 Lane 的下半与上半](../img/mma_cg2_m128.svg)

在这些模式中,原理是相同的。`tcgen05.mma` 计算的是一个逻辑累加器分块,但该分块必须被放置到物理上的 128 Lane 乘最多 512 Col 的 TMEM 空间中。模式与 M 形状决定了该放置方式。内核其余部分在随后把累加器读回时,必须使用同样的映射。

对于本书中的内核,TMEM 中的累加器通常是 f32。这是常见的高精度路径。它并非唯一的累加器类型。`.kind::f16` 路径可以在 f16 下累加。

## 操作数放置

对于稠密 MMA 模式,A 和 B 在 MMA 运行之前被准备在 SMEM 中。TMA 负责把全局显存分块搬入 SMEM。内核按照 Tensor Core 期望的布局(包括任何必需的 swizzle)来安排这些 SMEM 分块。

累加器 C 写入 TMEM。这是与早前几代的主要区别。收尾阶段不会把累加器直接当作 MMA 指令的输出接收,而是必须显式地用 `tcgen05.ld` 从 TMEM 加载。

在 `cta_group::1` 中,一个 CTA 提供操作数并独占累加器。在 `cta_group::2` 中,每个 CTA 从自己的 SMEM 提供自己一侧的操作数,并且每个 CTA 拥有自己那份 TMEM 中的累加器。当 A 按 M 拆分时,每个 CTA 保留与自身 M 切片对应的 A 行。B 按模式共享,因为两个 M 切片都要与同一个 N 乘 K 的分块相乘。

阅读内核时这种区分很重要。SMEM 放置回答的是 Tensor Core 如何读 A 与 B。TMEM 放置回答的是累加器落在何处。这两种布局通过 MMA 模式相互关联,但它们不是同一个存储空间,不能视为可互换。

## 块缩放 MMA

稠密模式直接从 SMEM 读入其数据操作数,并累加到 TMEM 中。块缩放 MMA 额外增加了两个操作数:A 和 B 的缩放因子张量。

这用于 `mxfp8` 和 `nvfp4` 这类极低精度格式。低精度格式效率高,但其动态范围小。单一的global scale(全局缩放)通常过于粗糙。如果按最大值选取 scale(缩放),较小值会损失精度;如果按小值选取 scale,较大值可能会被截断。

块缩放通过为小的 K 块分配缩放因子来解决这个问题。一组连续的 K 元素共享一个 scale。MMA 在概念上先用各自的 scale 对每个块反量化,再在累加器类型中累加乘积。

对 A 和 B 而言,这引入了两个缩放因子张量:

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK = K / B`,`B` 是沿 K 的块大小。

确切的块大小取决于格式。要点在于缩放轴以更粗的粒度跟随 K。每个缩放因子描述的是一整块 K 值,而不是单个元素,也不是整张矩阵。

其数学形式为:

```text
acc += (Aq * scale_a) * (Bq * scale_b)
```

其中 `Aq` 与 `Bq` 是量化后的低精度值,scale 在累加之前恢复它们的近似量级。

缩放因子的 dtype(数据类型)也很重要。使用 `e8m0` scale 时,每个 scale 实际上是 2 的幂。使用 `e4m3` scale 时(`nvfp4` 即如此),scale 是一个小的浮点值,可以表示介于相邻 2 的幂之间的值。

## 缩放因子存放在何处

块缩放 `tcgen05.mma` 与稠密 MMA 有一处重要的放置规则差异:缩放因子从 TMEM 读取。

数据操作数 A 和 B 仍然暂存在 SMEM 中。缩放因子 SFA 和 SFB 则经 TMEM 暂存。由于 TMA 加载的目标是 SMEM,缩放因子通常需要额外一步。内核先把它们加载到 SMEM,再用 `tcgen05.cp` 从 SMEM 拷贝到 TMEM。只有当缩放因子进入 TMEM 之后,块缩放 MMA 才能读取它们。

这就给缩放因子赋予了与数据操作数不同的移动路径:

```text
A, B:     全局显存到 SMEM,然后 MMA 读 SMEM
SFA, SFB: 全局显存到 SMEM,然后 tcgen05.cp 把 SMEM 拷到 TMEM,然后 MMA 读 TMEM
```

缩放因子的 TMEM 布局很紧凑。一个 128 行的 scale 向量可以打包进 32 条 Lane 行,使用基于 `r % 32` 决定 lane 位置、`r / 32` 沿列排布的映射。随后数据可以广播到读取完整 128 Lane 空间的四个 warp 上({ref}`chap_layout_generations`)。

这是一个说明为何 TMEM 布局必须显式的恰当例子。累加器布局与缩放因子布局都在 TMEM 中,但它们并不是同一种布局。累加器使用 MMA 输出映射;缩放因子使用块缩放 MMA 所期望的紧凑布局。

## `cta_group::2` 中的缩放因子

在双 CTA 情形下,缩放因子跟随它们所缩放的数据。

SFA 缩放 A。由于 A 按 M 拆分到这对 CTA 之间,SFA 也按 M 拆分。每个 CTA 持有与自身 A 行对应的 SFA 行。

SFB 缩放 B。由于两个 CTA 都要与同一个 B 分块相乘,SFB 必须对两个 CTA 都可见。在实践中,这意味着 SFB 在这对 CTA 之间以 multicast(组播)方式传送。

这正是块缩放 cluster GEMM 中常见加载模式的来源。SFA 按 CTA 各自加载,使用该 CTA 自身 M 切片的掩码。SFB 则广播给这对 CTA,因为两个 CTA 都需要相同的 N 侧缩放因子。

![块缩放 MMA 放置:A 与 B 打包在 SMEM;SFA、SFB 与 C 在 TMEM,其中 SFA 按 M 拆分到各 CTA,SFB 在这对 CTA 间组播](../img/mma_block_scaled.svg)

## 保持各 MMA 契约相互匹配

一个 Blackwell GEMM 分块要穿过若干条专用路径。

TMA 把 A 和 B 从全局显存搬入 SMEM。对于块缩放模式,它还把缩放因子搬入 SMEM。`tcgen05.cp` 在需要时把这些缩放因子搬入 TMEM。`tcgen05.mma` 读入其操作数,在 Tensor Core 上异步运行,并累加到 TMEM。完成屏障告诉内核累加器何时就绪。收尾阶段随后用 `tcgen05.ld` 把累加器从 TMEM 加载回寄存器,并存储最终输出。

跨越这些路径,内核必须保持三处契约相互匹配:SMEM 操作数布局、TMEM 累加器或缩放因子布局、以及让下一个消费者得以安全运行的异步完成信号。
