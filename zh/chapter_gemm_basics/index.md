(chap_gemm_basics)=
# 构建分块 GEMM

:::{admonition} Overview
:class: overview

- 从 TIRx 的分块原语出发,以单个输出分块为起点,构建一个正确的分块 GEMM。
- 第 1 步是单分块 GEMM,第 2 步加入 K 维循环累加,第 3 步跨 CTA 做空间分块以覆盖完整矩阵。
- 先求正确;性能是后续两章的任务。
:::

GEMM(通用矩阵乘,General Matrix Multiply)是贯穿整本书的工作负载。它位于线性层、注意力投影和卷积这些占据 GPU 主要计算时间的算子之下,因此一个正确的 GEMM 与一个足够快的 GEMM 之间的差距,就等同于让芯片大部分时间空闲与将其打满之间的差距。

这个差距太大,无法一步跨过。一个能打满硬件的内核会迫使你同时调试数据搬运、累加、分块和 Tensor Core(张量核)调度,而且没有任何可信赖的参照物。更稳妥的做法是:从一个能得到正确结果的最小内核起步,然后每次只做一个决策地逐步扩展。

本章就是要写出这第一个正确的分块 GEMM。前几章抽象地介绍了 TIRx(Python DSL)的作用域 / 布局 / 派发模型,本章则把它应用到一个真实的内核上。我们从一个 128 x 128 的输出分块开始,把它扩展成能够处理全尺寸矩阵的内核,先后加入 K 维累加和跨多个 CTA 的空间分块。

本章是从头到尾走通一条完整 GEMM 优化路径的三章中的第一章。本章我们只构建一个正确的分块内核,就此打住。下一章({ref}`chap_gemm_async`)用 TMA(张量内存加速器,Tensor Memory Accelerator)替换线程拷贝,并通过流水线让数据搬运与计算重叠;再下一章 {ref}`chap_gemm_advanced` 则更进一步,引入线程束特化和 CTA 集群。每一章都建立在前一章之上,因此各个内核逐步累积功能,而不是从头再来。

把每一步理解成对同一份契约的编辑,会有所帮助:这份契约有三项条款——哪一个**作用域**执行操作、操作数分块用哪一种**布局**、哪一条**派发**路径执行它。多数步骤只有一项主要变化,所以我们用一张小卡片开头,点明这一变化,并指出为安全复用所需的任何同步细节。第 1 步确立了基线,之后的整条优化路径都在编辑它。

## GEMM

GEMM 是位于线性层、注意力投影和许多卷积实现之下的稠密矩阵乘,这也是为什么一个快的 GEMM 内核几乎在所有地方都能带来收益。本教程的示例使用 $D = A B^{\top}$:

- $A$ 的形状为 $M \times K$。
- $B$ 的形状为 $N \times K$。
- $D$ 的形状为 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

这里的转置并非我们额外选择执行的操作,它是由数据的存储方式自然产生的。示例中 $B$ 保持为 $N$ 行、每行长度为 $K$,这正是线性层权重通常采用的布局,因此沿 $K$ 做内积时自然读到的就是 $B^{\top}$,无需任何重排。

贯穿整篇教程,我们以吞吐量(单位 TFLOPS)衡量一个内核,用墙钟时间去除每次乘加里的两次浮点运算:

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 数据通路

本教程中的每一项优化最终都归结为数据存放在哪里、如何移动,因此在动手写代码之前把它们梳理清楚是值得的。本质上,一个 Blackwell GEMM 内核只围绕两类活动组织:在各级存储之间搬运分块,以及在这些分块上做计算。下图追踪了一个分块从输入到输出经过的每一级存储:

![*内存数据流*](../img/memory_dataflow.png)

上图展示的是基线路径,后续每一次优化都在编辑它,但从不替换它。从左到右读:操作数分块先从 GMEM 搬到 SMEM;随后 `tcgen05.mma` 消费 SMEM 中的操作数并把累加器写到 TMEM;最后收尾部分把 TMEM 读回到寄存器,再把结果写回 GMEM。请把这条链路记在心里,因为下面每一步改变的只是其中一跳「怎么跳」;它从不改变跳本身。

## 优化路径

上面这条朴素数据通路足以得到正确结果,但它把硬件的大部分都闲置着。教程余下部分通过逐一加入 Blackwell 的特性来弥合这一差距,每个特性都通过一个 TIRx 分块原语表达。我们将要走的路径依次经过这些特性:

- **TMA 异步搬运**通过 Blackwell 的硬件拷贝路径搬运 GMEM <-> SMEM 分块,并用屏障跟踪完成情况。
- **软件流水线**使用多个 SMEM 流水级,使下一个 K 分块的数据搬运能够与当前分块上的 Tensor Core 计算重叠。
- **持久化调度**保持一个固定规模的 CTA 池,每个 CTA 通过分块调度器处理多个输出分块,而不是每个分块启动一个 CTA。
- **线程束特化**把生产者、MMA 消费者和回写三种角色分到不同的 warpgroup 上。
- **CTA 集群**让两个 CTA 合作完成一个更大的 Blackwell MMA 分块。
- **多消费者执行**使用多个消费者 warpgroup 同时计算分块的不同部分,以提高计算密度。

---

(chap_single_tile)=
## 第 1 步:串行单分块 GEMM

仍然能走通完整硬件路径的最简 GEMM,就是只计算单个输出分块的那个。所以我们从这里开始。第 1 步计算一个 128 x 128 的输出分块,K = 64,小到无需任何循环,数据通路里的每个部分都恰好出现一次。没有任何东西重复,我们就能在不得不去推敲一个循环之前,孤立地看清每一跳。

> **这一步确立的内容:基线**
> - 作用域:一个由 128 个线程组成的单一 warpgroup 依次走完整条路径,一级接一级。
> - 布局:A 和 B 分块放在 SMEM 里,累加器放在 TMEM 里,结果经由寄存器暂存后写出。
> - 派发:同步 `Tx.copy` 承担加载,`tcgen05` 执行 MMA。

### 单分块数据流

把基线契约定下来之后,接下来要确定的就是一个分块以何种顺序穿过这条契约。这第一个内核恰好走一遍核心的 GEMM 数据通路,也就是数据流图里那条 GMEM -> SMEM -> TMEM -> 寄存器 -> GMEM 链,外面没有套任何循环。它分配工作内存、加载操作数、计算乘积、写回结果,最后自我清理:

1. **分配**:SMEM(池分配器)、TMEM(`tcgen05.alloc`)、mbarrier
2. **加载**:全部 128 个线程协作地把 A 和 B 分块从 GMEM 拷到 SMEM(同步 `Tx.copy`)
3. **计算**:单个被选中的线程发起 `Tx.gemm_async` + `tcgen05.commit`;所有线程在 mbarrier 上等待
4. **回写**:warpgroup 读 TMEM → 寄存器;每个线程把 fp32→fp16 转换后写回 GMEM
5. **释放**:释放 TMEM

### 第一个内核的四部分

完整内核只有几十行,但分块来看更易消化。我们分四段来读(内存分配、同步加载、MMA 派发、回写),最后再拼装成一个内核。沿途出现的 API 名字,正是第二部分({ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`)介绍过的 TIRx 分块原语词汇。

**内存分配。** 内核开头先为操作数切出共享内存,再留一个存放 TMEM 地址的槽和一个 mbarrier:

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM 地址(4 字节)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier(8 字节)
pool.move_base_to(1024)                           # 跳到偏移 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里有两处细节值得停下来想一想。`pool.move_base_to(1024)` 把 Asmem 和 Bsmem 推到偏移 1024 处,从而把它们上方那些小块元数据的低地址保留下来,让大块的操作数分块落在一个干净的边界上。而 `layout=A_layout` 向 `tma_shared_layout` 请求一种经过 swizzle 的 SMEM 摆放方式,使得 TMA 和 `tcgen05.mma` 都能直接读取——这正是第二部分描述的那种「以布局为契约」的义务。

**同步加载。** 缓冲区就位后,操作数还得真正进入 SMEM。在第一个版本里,我们让 CTA 自己的线程来做这次拷贝:

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

因为这里只有一个分块(M=N=128,K=64),把整个 A 和 B 拷过去就是全部的加载工作。`Tx.cta.copy(...)` 让 CTA 在这次拷贝上协作,每个线程负责自己那一份数据切片。随后的 `T.cuda.cta_sync()` 一身二用:它既等待每个线程完成,又发布它们的共享内存写操作,从而保证 MMA 之后读 `Asmem` 和 `Bsmem` 时看到的是完整分块,而不是只填了一半的缓冲区。这种由线程驱动的拷贝也正是我们最先要替换的东西;下一章({ref}`chap_gemm_async`)会把它换成 TMA。

**MMA 派发。** 操作数现在已落在 SMEM 里,我们可以发起 MMA,并从一个被选中的线程来发起:

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两层嵌套的守卫用两步把发起者收窄下来。外层的 `if warp_id == 0` 只保留 warpgroup 中的 warp 0,内层的 `if T.ptx.elect_sync():` 再从该 warp 中选出一个活跃 lane。两者合起来恰好只剩一个线程来运行 `Tx.gemm_async` 和 `tcgen05.commit`。

关于这一个线程到底意味着什么、不意味着什么,有必要讲清楚,因为最自然的解读是误导性的。单个发起线程*并不*意味着一次单线程的乘法。计算本身仍然是一次完整的分块级 MMA:硬件对由 SMEM 操作数布局和 TMEM 累加器布局共同描述的那个分块执行协作式乘法。关键在于,`Tx.gemm_async` 是一个*分块操作*,而不是一条硬件指令。K = 64 的分块比硬件 MMA 的 K 原子(`MMA_K = 16`)更宽,因此这一次分块操作会降级为沿 K 步进的一小串原始 `tcgen05.mma` 指令,并由 warpgroup 协作地驱动每一条。只让一个线程发起这个分块操作的原因在于,每一条底层 `tcgen05.mma` 本身就是一次*单指令*的协作操作:一次发起就驱动分块 MMA 中那个 K 原子的计算。如果 128 个线程都发起这一串指令,同样的工作只不过被发起 128 次而已。最后,`accum=False` 标志告诉 MMA 去覆写 TMEM 目标,而不是累加进去——这正是我们在此想要的,因为还没有任何先前的部分和可供延续。

**回写。** 乘积现在落在 TMEM 里,但调用方希望它以 fp16 形式回到 GMEM。因此收尾部分必须把结果经由寄存器搬下来,并在途中做类型转换:

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # 每线程一行的 fp32 寄存器
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # 同一行,转换成 fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA 在 TMEM 中留下一个 128 x 128 的 fp32 累加器分块。这里用 fp32 是刻意为之:GEMM 沿 K 对许多乘积求和,以更高精度保存这个累加和可以压住本会累积起来的舍入误差。但 `D` 是 fp16,所以这些值不能直接送出去。它们先落到寄存器,在那里收窄成 fp16,然后才进入 GMEM。

两个寄存器缓冲区各司其职。`Dreg` 是一个每线程 `BLK_N` 个元素的缓冲区,而 `Dreg_wg` 则是在某种选定布局下,对同一批寄存器的一个 warpgroup 级别*视图*:

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这一布局把分块的第一维映射到 warpgroup 的线程上:线程 0 拥有第 0 行,线程 1 拥有第 1 行,依此类推到第 127 行。第二维则留在每个线程自己的寄存器缓冲区内,因此单个线程持有自己那一行的所有列。warpgroup 有 128 个线程,分块有 128 行,这个 128 x 128 的输出就整整齐齐地划分为每线程一行。

在这个视图下读出累加器,正是 `Tx.wg.copy_async(Dreg_wg, tmem)` 所做的事,它会降级为 Blackwell 的 TMEM 加载路径 `tcgen05.ld`。由于该加载是异步的,任何线程在触碰 `Dreg` 之前,都必须先让 `T.ptx.tcgen05.wait.ld()` 完成;否则线程会读到加载尚未填满的寄存器。

等待返回之后,每个线程私有的 `Dreg[:]` 里就持有了它所负责的那一逻辑输出行的 fp32 值。线程把这些值在 `Dreg_f16` 里收窄成 fp16,算出自己负责的全局行号,

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

再写 `D[m_thr, n_st:n_st + BLK_N]`。这些行在四个 warp 之间干净地划分:warp 0 写第 0–31 行,warp 1 写第 32–63 行,warp 2 写第 64–95 行,warp 3 写第 96–127 行。

### 完整内核

现在我们把四部分缝合回一个可运行的内核(M=N=128,K=64)。导入语句排在最前面:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

内核包在后续步骤同样使用的 `hgemm_vX(M, N, K)` 风格里。第 1 步以 `M=N=128, K=64` 运行,所以启动配置里恰好只有一个输出分块:

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录底层硬件 MMA 分块的大小;它们不会被
    # 传给 gemm_async(后者会从操作数和累加器分块推导出 MMA 形状),
    # 因此后续步骤省略了它们。
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 第 1 步是一个单分块内核:M = BLK_M 且 N = BLK_N,所以 grid
        # 是 1x1。从 1x1 的 grid 出发,可使每个 CTA 的分块偏移
        # (m_st, n_st) 平凡地为零;第 3 步及以后会把它推广到更大的 M / N。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # 单个 warpgroup,所以 wg_id 恒为 0(下方未使用)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- 屏障 + TMEM 初始化(仅 warp 0)---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0
    
        # --- 加载:所有线程协作把 global -> shared(同步)。
        # 由于 M=BLK_M 且 N=BLK_N,下面的切片覆盖了整个矩阵;
        # 保留切片形式,使与第 3 步(多分块)的 diff 最小。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- 计算:单个被选中的线程发起 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- 回写:TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

之后每一个 GEMM 步骤都以同样的方式编译、运行并自检,所以我们把这套脚手架在这里完整地讲一遍,从此之后就只展示内核。要运行后面的某个步骤,只需把下面的 `hgemm_vX` 换成对应的版本,再配上匹配的问题规模即可。有一点值得记住:每个新步骤请在一个全新的 Python 会话里单独编译,在尝试下一个之前重启,因为示例复用了内部名字,而编译器持有按会话维度的状态。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) 直接接收 torch 张量,与每一章使用的调用形式一致。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# 相对容差,与线程束特化和 Flash Attention 的单元一致:
# 输出幅值随 K 增长,所以一个固定的绝对上界会在更大的 K 下失效。
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# 可选:对更大的内核做计时。
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

第 1 到第 3 步刻意采用很小的规模(这里是 128×128,第 3 步是 256³),以让这几段最初的走读保持简单易跟。{ref}`chap_gemm_advanced` 末尾那张跨步骤的*端到端结果*表则采取相反的做法:它在单一的 M=N=K=4096 规模下测量每一个步骤(包括这个第 1 步算法),从而使其加速比能够直接比较。

### 单分块内核的局限

这个内核是正确的,这正是第 1 步的全部意义所在,但它只在一种非常狭窄的设定下正确。有四点局限是刻意为之的,优化路径的其余部分会逐一消除它们:

- 它只处理单个 K 分块,因此无法在一个大的 K 上做内积。
- 它只处理单个输出分块,所以 M 和 N 都被钉死在 128。
- 它使用同步的 GMEM -> SMEM 拷贝,而不是 TMA。
- 它不让数据搬运与计算重叠,所以两者从不同时运行。

---

(chap_k_loop)=
## 第 2 步:K 维循环累加

第一个要去除的局限是最小的那个。第 1 步只处理一个 64 宽的 K 分块,而真实矩阵上要做的内积远不止于此。第 2 步里我们仍保留单个输出分块,但让 K 跨越许多个 64 宽的块。

思路很直接:对每个块重复一次 加载 -> MMA -> 等待 序列,并让每次 MMA 累加到同一个 TMEM 槽里。真正的工作量其实在于同步。在多次迭代间复用同一个 mbarrier,引出了本章第一个真正意义上的正确性陷阱。如果代码跟踪错了相位,一次等待可能在其 MMA 实际完成*之前*就返回,悄无声息地破坏结果。下面的机制说明了这究竟是怎么出错的,以及如何避免。

> **这一步改变的是:布局复用**
> - 作用域:不变,仍是一个单一的 warpgroup。
> - 布局/复用:同一对 SMEM 分块和同一个 TMEM 累加器槽在 K 循环间被复用。不分配新的存储;操作数分块流过一对固定的缓冲区,累加器状态停留在一个 TMEM 槽里。
> - 同步:被复用的 MMA 屏障必须在每一个 K 块上推进到正确的相位,否则后续的某次等待可能观测到的是更早一次的完成。
> - 派发:不变。

### K 循环机制

第 1 步在单个 64 宽的 K 分块上做内积;这里我们保留它的单个输出分块,但让 K 想跑多长就跑多长。为了覆盖大于 64 的 K,我们以 `BLK_K=64` 为步长逐块地走完 K。每次迭代把下一个 A 和 B 的 K 切片加载进 SMEM,并发起 `Tx.gemm_async`。`accum` 标志正是把这些块缝合进同一次点积的关键:第一个块上 `accum=False` 初始化 TMEM 累加器,其后每个块上 `accum=True` 把该块的乘积加到已经在 TMEM 里的累加和上。

同步是这里需要小心的地方。我们为每一次 MMA 完成复用同一个 mbarrier,而安全复用它归结为跟踪我们正在等待的是哪一个屏障相位。mbarrier 持有一个 1 比特的相位,取 0 或 1,每当预期的到达落下来时就翻转到另一个值。微妙之处在于等待条件本身:`try_wait(bar, phase)` 会一直阻塞,直到屏障内部相位与传入的 `phase` *不同*。所以我们传入的实参,必须命名我们期望「离开」的那个相位,而不是我们正「等待到达」的那个相位:

| K 迭代 | 等待前本地 `phase_mma` | `try_wait` 等待什么 | 等待后本地更新 |
|---|---:|---|---:|
| 0 | 0 | 屏障翻转到 1 | `phase_mma = 1` |
| 1 | 1 | 屏障翻转到 0 | `phase_mma = 0` |
| 2 | 0 | 屏障翻转到 1 | `phase_mma = 1` |

`phase_mma ^= 1` 这一行正是让上表成立的关键。把它去掉,第二次迭代仍会调用 `try_wait(bar, 0)`,但屏障在第一次 MMA 之后已经翻到相位 1,于是这次等待看到不匹配就立即返回,赶在第二次 MMA 完成之前。内核随之读到一个算了一半的累加器,在没有任何报错的情况下给出错误答案。这种 bug 编译和运行起来都完美无瑕,这也正是相位翻转值得如此重视的原因。

### 完整内核

下面的完整内核,无非就是把第 1 步折入 K 循环和相位翻转。导入语句与之前相同:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

它包在 `hgemm_v2(M, N, K)` 里。grid 仍是 `[1, 1]`,因为我们仍在计算单个输出分块;增长的只是它的 K 范围:

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # 仍是单个输出分块(M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K 循环:以 BLK_K 为步长在 K 上迭代 ===
        for i in T.serial(K_TILES):   # 串行 device 循环(保持全 K 的 A/B 参数形状正确)
            # 加载第 i 个 K 块
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA:首个分块 accum=False,其余为 True
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # 等待 MMA,然后翻转相位
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === 回写(与第 1 步相同)===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## 第 3 步:空间分块(多 CTA)

K 循环处理了内积维度,但 M 和 N 仍被钉死在单个 128 x 128 分块上。真实的输出远大于一个分块,所以基本内核的最后一块拼图,是用许多分块同时覆盖 M 和 N。第 3 步启动一个二维的 CTA grid,每个输出分块对应一个 CTA,让 GPU 并行地计算所有分块。示例采用 M=N=K=256,得到一个 2x2 的分块网格,刚好足以让下标变得不再平凡,又不至于把它埋没。

> **这一步改变的是:作用域**
> - 作用域:一个二维的 CTA grid,每个 CTA 拥有一个 128 x 128 的输出分块。
> - 布局:不变;在一个 CTA 内部,与第 2 步一样是同一条 SMEM/TMEM/寄存器路径。
> - 派发:不变。

### Grid 映射

grid 的形状直接由分块方式决定:每个 128 x 128 输出分块对应一个 CTA,总共需要 `[M // BLK_M, N // BLK_N]` 个 CTA。与第 2 步相比,唯一真正全新的工作,就是教会每个 CTA 矩阵里的哪一片是*它自己*要计算的那一片。

CTA `(bx, by)` 拥有这个输出区域:

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

为产出它,该 CTA 的 K 循环会反复加载它自己的 A 行带和 B 列带所对应的那些 K 切片:

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

这里的下标直接源自 `D = A @ B.T` 的约定:`bx` 选出 A 和 D 的行,而 `by` 选出 B 的行,这些行在施加转置后就成了 D 的列。

每个 CTA 一个分块是最简单可行的映射,但也很浪费。同一行里的每个 CTA 都会从 GMEM 重新加载相同的 A 分块,同一列里的每个 CTA 都会重新加载相同的 B 分块,所以完全谈不上复用相邻 CTA 已经拉进来的数据。我们暂且把这份浪费搁置在原地;持久化调度({ref}`chap_gemm_async` 的第 6 步)会回过头来处理它,把这些共享的操作数保持在 L2 里热着。

**交给你的 agent 试一试**:在 `M=N=K=256`、`BLK_M=BLK_N=128`、`BLK_K=64` 下,让它追踪 CTA `(1, 0)` 和 CTA `(0, 1)`。对每个 CTA,列出 `m_st`、`n_st`、每次 K 迭代加载的 A 和 B 切片,以及所写的 D 区域。因为内核计算的是 `D = A @ B.T`,哪些 B 行会变成 D 的列?

### 完整内核

这个内核再一次是第 2 步,这次只有两处改动:grid 形状和每个 CTA 的偏移。内部的 K 循环和回写原封不动。导入语句相同:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

grid 变成 `[M // BLK_M, N // BLK_N]` 而不是 `[1, 1]`,加载和存储都按 CTA 自己的 `m_st` 与 `n_st` 偏移:

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 二维 grid:每个 128x128 输出分块对应一个 CTA
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # 每个 CTA 的分块偏移
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # 带偏移 A、B 切片的 K 循环
        for i in T.serial(K_TILES):   # 串行 device 循环(保持全 K 的 A/B 参数形状正确)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # 回写到正确的输出分块
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 在第 1–3 步中,`Tx.copy` 在 MMA 之前把 A 和 B 分块搬进 SMEM。为什么内核在 `Tx.gemm_async` 读取这些 SMEM 分块之前需要 `T.cuda.cta_sync()`?
2. 在第 2 步中,如果把 `phase_mma ^= 1` 从 K 循环里去掉会怎样?内核还会等待每一次 MMA 吗,还是某次后续的等待会过早通过?
3. 对于 M=N=4096、BLK_M=BLK_N=128,第 3 步会启动多少个 CTA?哪些操作数分块在相邻 CTA 之间被逻辑上复用,而第 3 步是否利用了这种复用?
