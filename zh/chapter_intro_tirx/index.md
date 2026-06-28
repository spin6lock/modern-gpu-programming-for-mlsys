(chap_tirx_primer)=
# TIRx 简介

:::{admonition} Overview
:class: overview

- TIRx 是一个用于在 IR 层编写 GPU 内核的 Python DSL(领域特定语言):你直接命名硬件,但通过结构化的 IR 来命名。
- 每一个分块操作都由三个设计要素控制:*scope*(作用域,哪些线程)、*layout*(布局,分块位于何处)和 *dispatch*(派发,走哪条硬件路径)。
- 一个可运行的单 MMA GEMM(通用矩阵乘)就展示了全部三者;本书余下部分就是这些设计要素在更大尺度上的展开。
:::

:::{admonition} Running the examples
:class: note

这些示例需要一块 Blackwell GPU(`sm_100a`,例如 B200)。TIRx 编译器以 Apache TVM wheel 包的
`tvm.tirx` 模块形式发布;请与 PyTorch 的 CUDA 构建一起安装:

```bash
pip install apache-tvm
```

用 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认能正常导入。同样的这套环境
可以运行书中每一个可运行的示例。
:::

第一部分讲解了硬件是什么。要让硬件真正算出东西,我们需要一种编程方式。

我们可以直接写裸 CUDA 或 PTX,许多快速的内核正是这样写出来的。问题在于,真正决定内核行为的那些决策在那里很难看清:哪些线程执行一个操作、每一块数据放在哪里、由哪条硬件路径执行。这些选择被埋没在内联函数参数、地址算术和约定之中。

TIRx(Tensor IR neXt)是一个 Python DSL(领域特定语言),它把上述三个决策提升到明面上:**scope**(作用域,哪些线程执行一个操作)、**layout**(布局,操作数分块放在哪里)和 **dispatch**(派发,由哪条硬件路径执行)。它仍然直接命名硬件概念,包括线程、共享内存和张量内存、屏障,以及 `tcgen05` MMA。区别在于,这些选择现在变成了编译器可以降级、检查和调度的结构化 IR。

我们不抽象地引入这些概念,而是从一个完整的内核出发:一个最小的单 MMA GEMM。我们先让它跑起来,然后再逐行回读,看 scope、layout 和 dispatch 各自如何塑造它,以及内核如何被编译。该内核所依赖的张量布局模型将在 {ref}`chap_tirx_layout_api` 中独立展开,完整的语言特性集则在 {ref}`chap_language_reference` 中给出;这里我们只聚焦于这一个内核和三个设计要素。

## 第一个内核:单 MMA GEMM

我们承诺要给出的内核是一个最小化的 GEMM,精简到仍能用到 Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 x 128 输出分块,K = 64。整个计算从头到尾被表达为一个 `Tx.gemm_async` 分块操作。(那一个分块操作并不对应单条硬件指令:因为硬件 MMA 的 K-atom 是 16,K=64 的分块会降级成一小段沿 K 步进的 `tcgen05.mma` 指令序列。这个 DSL 的要点恰恰在于我们写的是分块,而不是序列。)围绕这个操作,内核还做些常规杂活:分配共享内存(SMEM,Shared Memory)和张量内存(TMEM,Tensor Memory),把 A 和 B 从全局内存拷贝到共享内存,把分块 MMA 派发进一个 TMEM 累加器,通过寄存器把该累加器读回,再存储结果。这个内核虽小,却是我们在 {ref}`chap_gemm_basics` 中攀登的 GEMM 阶梯的第 1 步,在那里它会有一次完整的走读。

每个 TIRx 内核都从同样的若干导入开始,所以值得先一次性把它们看清楚:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们把内核包在一个小型构造器 `hgemm_v1(M, N, K)` 里,它接受问题形状并返回一个 `PrimFunc`。对于我们选定的形状 `M=N=128, K=64`,启动恰好只包含一个输出分块,这正是让这第一个版本简单到能一口气读完的原因:

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录底层硬件 MMA 分块的大小;它们不会
    # 传给 gemm_async(gemm_async 会从操作数和累加器分块推导出 MMA 形状),
    # 因此后续步骤中略去了它们。
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
        # Step 1 是一个单分块内核:M = BLK_M 且 N = BLK_N,所以 grid
        # 是 1x1。从 1x1 grid 出发能让每个 CTA 的分块偏移
        # (m_st, n_st) 平凡地为零;Step 3 及以后会把它推广到更大的 M / N。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # 单个 warpgroup,所以 wg_id 恒为 0(下面未使用)
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
    
        # --- 屏障 + TMEM 初始化(仅 warp 0) ---
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
    
        # --- 加载:所有线程把 global -> shared 拷贝(同步)。
        # M=BLK_M 且 N=BLK_N 时,下面的切片覆盖了完整矩阵;
        # 保留切片形式是为了让与 Step 3(多分块)的 diff 最小。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- 计算:单个被选中的线程派发 MMA ---
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

在读这个内核之前,我们先确保它能用。我们编译它,并把输出与一个 torch 参考实现对照检查。我们不必写出确切的架构:架构(例如 `sm_100a`)会从设备自动探测,所以 target 写成 `"cuda"` 就够了,而 `tir_pipeline="tirx"` 正是用来选择 TIRx 降级流水线的。编译完成后,`ex.mod(...)` 直接接受 torch 张量,中间无需任何手动转换。

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

# ex.mod(...) 直接接受 torch 张量,每一章用的都是这种调用形式。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## Scope、Layout、Dispatch

既然内核能跑了,我们就可以回过头读它,追问它的每一行到底决定了什么。这样看,整个内核就是沿三个设计要素做出的一组选择。其中的每一个操作都回答同样三个问题:*谁*运行它、它的数据*在哪*、它*怎么*执行,而这三个回答正是 scope、layout 和 dispatch。本节余下部分逐个讨论这些设计要素;下方的交互演示可以让你看到每个设计要素分别控制了哪些行。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示:点击 Scope / Layout / Dispatch,高亮显示每个设计要素所控制的内核行。*

在使用这个演示时,留意三个问题:

- **Scope:谁运行这个操作?** `Tx.cta.copy(...)` 是 CTA(协作线程阵列)作用域的,所以全部 128 个线程都参与 GMEM -> SMEM 的拷贝。`Tx.gemm_async(...)` 由一个被选中的线程派发一次,因为每条降级后的 `tcgen05.mma` 指令本身就已经是一次协作式 MMA 启动。`Tx.wg.copy_async(...)` 是 warpgroup(线程束组)作用域的,所以该 warpgroup 的 128 个线程逐行瓜分 TMEM 的回读。
- **Layout:每个分块放在哪里?** A 和 B 使用 `tcgen05.mma` 所期望的 swizzle SMEM 布局。累加器位于 TMEM 中,采用 `TLane`/`TCol` 布局。寄存器回读视图把行映射到 `tid_in_wg`,于是每个 warpgroup 线程拥有一段行片段。
- **Dispatch:由哪条硬件路径执行?** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell 的 Tensor Core 路径。拷贝操作也有 dispatch 选择:这第一个内核使用普通的线程拷贝,后续的 GEMM 步骤会在不改变周围 scope 或 layout 的前提下,把这些拷贝换成 TMA(张量内存加速器)。

**和你的 agent 一起试试**:从第一个内核中挑三行——一个拷贝、一个 MMA、一个 TMEM 回读。让它按 scope、layout 和 dispatch 给每行打标签,然后核对答案是否与代码中的守卫、buffer 布局以及 `dispatch=` 参数一致。

## 编译是如何工作的

我们在上面为了测试已经编译过这个内核;现在我们更仔细地看看这一步做了什么。套路很短:把 `PrimFunc` 包进一个 `IRModule`,交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这会运行 TIRx 降级流水线,并返回一个你可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少在轮廓层面,值得了解一下 `tir_pipeline="tirx"` 启动了什么。流水线的核心 pass `LowerTIRx` 依据每个分块原语的 scope / layout / dispatch 契约来解析它:我们刚刚讨论的三个设计要素正是在这里被实际兑现为指令。之后,常规的 host/device 切分和一个收尾步骤会产生可启动的 module。如果你愿意,也可以在一个 `with target:` 块内编译,这样内核能拾取周围的 target 上下文。

这套流程的一个好处是对你毫无隐瞒:结果在两个层级上都可检视。你可以用 `.show()` 或 `.script()` 读 IR 本身,也可以直接从编译后的 module 上读到编译器最终吐出的 CUDA C。

```python
kernel.show()                          # 美化打印 TIRx(TVMScript)
print(kernel.script())                 # ……同样的内容,以字符串形式

# 编译后的 Executable 生成的 CUDA C 源码:
print(ex.mod.imports[0].inspect_source())
```

这只是一个轮廓。完整的降级故事——涵盖所有 pass、分块原语派发如何被解析、host/device 切分如何完成——请见 {ref}`chap_arch`。

## 接下来去哪里

一个内核就足以让我们认识 scope、layout 和 dispatch,并看到它们被编译和运行。三个设计要素中的每一个,以及这个内核本身,都通向一个把它们进一步展开的章节:

- {ref}`chap_tirx_layout_api`:张量布局模型(`TileLayout`、命名轴、swizzle),上面的操作数与累加器放置正是建立在它之上。如果三个设计要素里 layout 感觉最神秘,从这里开始。
- {ref}`chap_language_reference`:完整的语言特性集,涵盖解析器工具、数据类型、buffer 与内存、控制流以及线程同步,适合你想获得完整词汇表而非一次导览时使用。
- {ref}`chap_gemm_basics`:把这个内核作为 GEMM 优化路径的第 1 步,经由 K 循环累加、空间分块、TMA 和 warp specialization(线程束特化)逐步构建起来。如果你想看这三个设计要素如何扩展到一个真实内核,这里是自然的下一站。
