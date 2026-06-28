(chap_gemm_async)=
# 用 TMA 为 GEMM 建立流水线

:::{admonition} Overview
:class: overview

- 基础 GEMM 浪费了时间在轮替上(拷贝一个分块、计算、再拷贝下一个),而这两件事本可以同时进行。
- Step 4 切换到 TMA 异步加载,Step 5 对 SMEM 做双缓冲并预取(PIPE_DEPTH=2);完整的加载/计算重叠要到 Step 7 借助 warp specialization 才能到来,Step 6 则通过分块调度器把内核改造成持久化内核。
- 目标是在 Tensor Core 咀嚼当前分块的同时,加载下一个分块。
:::

Tensor Core 是芯片上最昂贵的单元,而上一章那版正确的分块化 GEMM 让它们在大部分时钟周期里都处于空闲。内核在轮替:线程把一个分块拷贝进共享显存,Tensor Core 咀嚼它,线程再拷贝下一个分块,Tensor Core 继续等。每一个阶段都要卡在前一个阶段上,哪怕加载下一个分块和在当前分块上做计算用的是完全独立的硬件、本可以同时跑。要弥合这个空隙并不需要新的数据通路;分块、布局和数学都已经是对的。真正需要改变的是工作发生的*时机*和由*谁*来调度。本章保持分块数据通路原封不动,直接攻击空闲问题。

我们分三个递进步骤走到那里,而动手之前先看清终点会有帮助。在 Step 4,我们把 GMEM 与 SMEM 之间的大块传输交给 TMA,由专门的拷贝硬件去搬运分块,而不是让线程来搬。在 Step 5,我们加入一个两级的软件流水线,让下一个 K 分块在当前分块还在被乘的时候有地方可落。在 Step 6,我们把启动方式改造成由分块调度器驱动的持久化内核,从而摊薄每个分块的启动开销,并让我们能挑选一个让操作数保持热的分块顺序。贯穿全程,SMEM、TMEM 和寄存器的布局都保持上一章留下的原样。唯一真正的新想法,是硬件单元之间的异步交接:让一个引擎跑在另一个引擎前面,而不是让它们齐步前进。

(chap_tma_async)=
## Step 4:TMA 异步加载

我们的第一招,是把拷贝本身挪出关键路径。回想一下 Step 1-3 里 CTA 在做什么:它的每一个线程都计算地址、派发加载指令,目的不过是为了把分块搬进 SMEM。这些指令带宽都花在了管道铺设上,而不是数学上。Step 4 用 TMA 替换掉同步的 `Tx.copy`,只需一个线程派发一条命令,TMA 引擎就会自己完成整个分块的传输。从这一步起,示例都跑在完整的 M=N=K=4096 规模上,而不再是 Step 1-3 的小规模,它们的端到端计时出现在 {ref}`chap_gemm_advanced` 末尾的 *End-to-End Result* 表里。

> **本步改变的是:Dispatch**
> - Scope:不变,仍是一个 warpgroup。
> - Layout:不变,沿用相同的 SMEM/TMEM/寄存器分块。
> - Dispatch:GMEM → SMEM 的加载从同步 `Tx.copy` 改走 TMA 引擎。

### TMA 的派发模式

Step 4 的唯一改动,就是把同步的分块拷贝换成 TMA 加载,所以值得仔细看看这一加载是如何派发的。对源码的修改只有寥寥几行,但这些行背后的执行模型却是本质上的不同。同步的 `Tx.copy` 是 CTA 线程用自己的指令自己干的活;TMA 拷贝则是由一个线程派发的一条命令,之后所有搬运都由 TMA 硬件完成。两者并排放在一起看一看是有价值的。

**之前(Step 3)**:全部 128 个线程参与拷贝,随后 `cta_sync` 让共享显存的写入对后续可见:
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # 全部 128 个线程
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**之后(Step 4)**:一个线程派发 TMA 加载,mbarrier 跟踪硬件传输何时完成:
```python
tid = warp_id * 32 + lane_id                 # warpgroup 内的 0..127
if tid == 0:  # 恰好由一个线程启动 TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # TMA 期望收到的字节数
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # 在 MMA 读取 SMEM 之前等待
```

注意,加载是以 `tid == 0` 为门控,而不是以 `elect_sync()` 为门控,这个区别比看上去更重要。`elect.sync` 是*每个 warp* 选出一个活跃 lane,而一个 warpgroup 有四个 warp,所以 `elect_sync()` 实际上会让四个线程进入加载协议。问题在于,该协议要向 mbarrier 宣告期望的字节数,而且必须恰好宣告一次;宣告四次会破坏这个计数,等待也就永远无法正确释放。按照 warpgroup 范围内的 id 精确挑出一个线程,才是干净避开这一问题的办法。

关于加速来自何处,必须诚实。Step 4 在每次 TMA 加载之后仍然要等待,所以此时我们还没有把加载和计算重叠起来;那是 Step 5 的工作。此处的收益纯粹来自数据搬移通路的改变:

- `Tx.copy` 用 CTA 线程去计算地址、派发加载/存储指令。
- TMA 用一条派发出去的命令启动一次硬件分块传输。地址生成、合并和 swizzling 都由 TMA 描述符描述,并由 TMA 引擎执行。

所以,即便 Step 4 仍在每次加载上阻塞,它最终依然更快。TMA 吸收掉了大块传输,把 CTA 线程从为搬运分块耗费指令带宽中解放出来,光是这一项节省就足以带来可观的提升。

### TMA 加载与存储的同步

我们已经看到 TMA 拷贝是如何派发的;故事的另一半,是知道它何时完成。切换到 TMA 会同时改变两件事:谁启动一次拷贝,以及代码如何知道它完成了。第一点从代码里一目了然;第二点却很容易被忽视,而且弄错了不会崩溃,只会给你一个悄无声息的正确性 bug。用 `Tx.cta.copy` 时,CTA 线程一起做拷贝,后面跟一个 `cta_sync()` 就足以知道它完成了。用 TMA 时,一个被选中的线程派发 `Tx.copy_async(..., dispatch="tma")`,引擎按自己的节奏完成传输,并通过一个 mbarrier 信号告知完成。

这正是 `cta_sync()` 不再够用的原因。`cta_sync()` 只等待 CTA 自己的线程,也只对这些线程的共享显存写入排序;它对一次在途的 TMA 传输一无所知,所以会在分块还在抵达时就欣然返回。修正办法是把完成判定显式化:对于一次 TMA 加载,被选中的线程先告诉 mbarrier 期望多少字节,然后 CTA 在任何 MMA 触碰这个 SMEM 分块之前,在*那个* mbarrier 上等待。下图端到端地描绘了这一握手。

![TMA 异步加载:同步流程](../img/tma_sync_flow.png)

上图把加载侧的握手单独剥离出来:一个被选中的线程启动 TMA,mbarrier 计数期望的字节,MMA 在读取 SMEM 之前等待其释放。文中凡说"Elected Thread"之处,都指那个启动 TMA 的被选中的线程,在我们的代码里就是 `tid == 0` 那个线程,而不是 `elect_sync()` 选出的某个 lane。

于是把加载通路拼起来:被选中的线程派发出两个 `copy_async`,再跟上 `arrive.expect_tx(total_bytes)`,其中字节数恰好就是 mbarrier 应当等候多少数据。一旦引擎搬完了那么多字节,与之匹配的 `mbarrier.try_wait(phase)` 就释放,只有到那时这个 SMEM 分块才安全地可以喂给 MMA。

存储侧走的是同一套硬件,但等待方式不同,所以值得在脑子里把这两个协议清楚地区分开:加载用 mbarrier 和字节数来追踪完成,而存储用提交组(commit group)和等待组(wait group)来追踪完成。在线程把它们的 fp16 结果写进 `Dsmem` 并同步之后,一个被选中的线程启动 `Tx.copy_async(D[...], Dsmem, dispatch="tma")`,随后 `cp_async.bulk.commit_group()` 加上 `cp_async.bulk.wait_group(0)` 会阻塞到存储排干为止。这个等待不是可选的:`Dsmem` 在前一次存储排干之前,不能被下一个分块复用。

**和你的 agent 一起试**:针对一个 K 分块,追踪 Step 4 的加载和存储同步。指出哪个线程启动每条 TMA 命令、哪个 mbarrier 或提交组追踪其完成、哪个等待保护 MMA 对 `Asmem` 和 `Bsmem` 的读取、哪个等待保护 `Dsmem` 的复用。为什么在这里,`elect_sync()` 对 TMA 加载协议而言会是错误的线程选择?

### 完整内核

完整内核把 TMA 加载和存储折叠进 Step 3 的结构里,其余部分一概不动。导入语句和之前一样:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包在 `hgemm_v4(M, N, K)` 里,这是我们贯穿全书遵循的模式:这层包装把依赖形状的常量和布局放在使用它们的内核紧旁边。

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM 分配(现在为 TMA 存储纳入了 Dsmem)---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()
    
        # --- 屏障与 TMEM 初始化 ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
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
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0
    
        # --- 内联辅助函数 ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )
    
        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        # --- 带 TMA 异步的 K 循环 ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
    
            # 单个线程派发 TMA 加载
            if tid == 0:
                tma_load(k_st)
    
            # 等待 TMA 完成;mbarrier 的释放会把 SMEM 可见性带给
            # 后续的 MMA,所以不需要额外的 fence。
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
    
            # 单个线程派发 MMA
            if tid == 0:
                mma(accum=k != 0)
    
            # 等待 MMA 完成
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1
    
        # --- TMA 存储回写 ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    
        # 读取 TMEM -> 寄存器(异步;先 wait.ld 再 cta_sync 以确保读取完成)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # 把 fp32 转为 fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # 写寄存器 -> Dsmem,刷出,然后同步
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA 存储:Dsmem -> GMEM。一个被选中的线程启动存储,
        # 并在 Dsmem 被复用之前把存储组排干。
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    
        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### 内核里的 TMA 配置

那个内核里几乎一切都沿袭自 Step 3。真正承载 TMA 语义的只有五个配置点,每一个都值得按名字认识清楚:

- **TMA 配置**:`{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 告诉 `Tx.copy_async` 使用 TMA,并通过 `tma_bar` 报告加载完成。

- **字节数**:`(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两个 fp16 操作数分块加载的字节数。`arrive.expect_tx(...)` 把这个计数交给 mbarrier。

- **mbarrier 初始化**:`init(tma_bar.ptr_to([0]), 1)` 创建出 TMA 加载所用的完成屏障。

- **`@T.inline`**:`tma_load(...)` 和 `mma(...)` 是辅助函数。它们在编译期被展开进内核体里,可以使用外层内核的变量。

- **TMA 存储同步**:收尾先把 fp16 行写进 `Dsmem`。`fence.proxy_async` 和 `warpgroup_sync` 让这些由线程写入的 SMEM 值为 TMA 存储通路做好准备。随后存储用 `commit_group()` 和 `wait_group(0)` 等待 SMEM 到 GMEM 的传输完成。

到这里,我们手里有了对的零件,却还没有对的节奏。Step 4 仍然要在开始对应的 MMA 之前先完成每一次加载,所以加载和乘法从未真正同时跑过;我们费了那么大力气分开的两个引擎,还是在轮替。下一步保持 TMA 加载和存储通路原封不动,转而重排调度,使得加载某个 K 分块能在对另一个分块做计算的同时进行。

(chap_software_pipeline)=
## Step 5:软件流水线(PIPE_DEPTH=2)

既然两个引擎明显相互独立,为什么 Step 4 不能把加载和计算重叠起来?障碍其实是存储。只有一个 SMEM 分块对时,下一次加载无处可去:它要等到当前 MMA 读完了那一对才能开始,因为提前开始会覆盖仍在使用中的数据。Step 5 通过对共享显存做双缓冲来消除这个存储冲突。单 warpgroup 的循环仍然在每次 MMA 之后才启动下一次 TMA 加载,但现在它有了不同的流水级可以预取进去、再复用。我们仍然跑在完整的 M=N=K=4096 规模上。

> **本步改变的是:Layout**
> - Scope:不变,仍是一个 warpgroup。
> - Layout:单一 SMEM 分块对变成一个 `PIPE_DEPTH` 级的环形缓冲。
> - Dispatch:不变,TMA 加载和 `tcgen05` MMA;本步加入预取和流水级复用,而完整的加载/计算重叠要到 Step 7 才到来。

### 流水线走查

设 `PIPE_DEPTH=2`,内核分配两个 SMEM 流水级,给加载通路和 MMA 通路各自独立的工作槽位。

请把下图当作两级缓冲意在支撑的流水线结构来读,而不是这个单 warpgroup 内核的精确执行轨迹。Step 5 建好了环形缓冲并预取更后面的流水级,但主循环在派发下一次 TMA 加载之前仍然要等待当前 MMA。完整的加载/计算重叠要到 Step 7 才到来,届时 warp specialization 给 TMA 和 MMA 赋予了各自独立的角色。

![*Pipeline PIPE_DEPTH=2,目标调度;本单 warpgroup 步骤只做预取,完整重叠要到 Step 7 借助 warp specialization 才到来*](../img/pipe_depth2.png)

一旦它被灌满,循环就在两个流水级之间交替。两个 TMA 加载一开始把两个流水级都填满;此后,循环等待当前流水级、在其上跑 MMA、等待该 MMA 读完这个流水级,然后再为 `k + PIPE_DEPTH` 把加载派发进刚刚变得可复用的那个流水级。这还不是并发的 TMA/MMA 调度,但它建立了环形缓冲结构,Step 7 会把这一结构在 producer 和 consumer 角色之间拆开。

具体而言,代码与 Step 4 在四处不同:

1. `Asmem` 和 `Bsmem` 多出一个前导的 `PIPE_DEPTH` 维,所以每个流水级都有自己独立的 SMEM 存储。
2. `tma_bar` 变成一个数组,每个流水级一个 mbarrier。
3. 在主 K 循环之前,内核预取头两个流水级。
4. K 循环使用 `stage = k % PIPE_DEPTH`:等待当前流水级、在其上跑 MMA,然后把这个流水级复用给 `k + PIPE_DEPTH`。

### 流水线机制

**1. 预取**:在主循环真正运行之前,我们加载头 `PIPE_DEPTH` 个流水级,这样循环在第一次迭代时就能找到等在那里的数据:
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. 主循环**:对每一个 K 分块,我们等它的流水级就绪、在其上跑 MMA,然后立刻把这个现已空闲的流水级重新派上用场——为 `PIPE_DEPTH` 之后的那个分块派发加载:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. 相位管理**:这部分最容易绊倒人,但规则比乍看上去要简单。每个屏障的相位翻转规则,直接来源于该屏障有多少个槽位,这也是为什么两个屏障以不同的节奏翻转。MMA 累加器住在一个 TMEM 槽位里,所以 `mma_bar` 是一个每次迭代都要重访的单屏障(`mma_bar.ptr_to([0])`),而一个你每次迭代都要重访的屏障,必须每次迭代都翻转相位。TMA 屏障讲的则是另一个故事:它们构成一个 `PIPE_DEPTH` 元素的数组,每个流水级一个屏障,而任一给定流水级的屏障在一圈环形里只回来一次。所以 `phase_tma` 只在流水级索引绕回 0 时才翻转:
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**和你的 agent 一起试**:设 `PIPE_DEPTH=2` 且 `K_TILES=5`,让它追踪主循环。对每个 `k`,列出 `stage`、传给各次等待的 `phase_tma` 和 `phase_mma` 值,以及是否派发了新的预取。`phase_tma` 到底在哪里翻转,为什么最后两次迭代没有预取?

### 完整内核

完整内核逐字保留 Step 4 的 TMA 加载和存储通路,再把它们包进我们刚才描述的分级缓冲和相位逻辑里。导入语句不变:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包在 `hgemm_v5(M, N, K)` 里。`PIPE_DEPTH=2` 常量设定流水级的数量(这里是两个,恰好就是双缓冲):

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # 双缓冲布局:第一维是流水级
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # 双缓冲的 TMA 屏障(每流水级一个),单个 MMA 屏障
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # 初始化屏障:TMA 用 PIPE_DEPTH 个,MMA 用 1 个
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === 预取:加载头 PIPE_DEPTH 个流水级 ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === 主循环 ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # 等待 TMA 加载完本流水级
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # 在本流水级的数据上做 MMA
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # 派发下一次预取加载(k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # 流水级绕回时 TMA 相位翻转
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA 存储回写:TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # 释放 TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## Step 6:持久化内核 + 分块调度器

到目前为止的一切,都只是在优化单个分块内部的工作。Step 6 改变问题的尺度,改为跨分块进行优化。

Step 5 为每个 128 x 128 的输出分块启动一个 CTA。对一个 4096 x 4096 的输出,这意味着 1024 个独立的 CTA,每个都要付出各自的启动开销,然后在自己的分块一做完就消失。

Step 6 则启动一个固定数量的 CTA 池,再让每个 CTA 依次处理多个分块。这给我们换来两样东西:启动工作被摊薄到多个分块上,而分块指派搬进了内核内部——在这里调度器可以挑选一个能复用操作数的顺序。我们仍然跑在完整的 M=N=K=4096 规模上。

> **本步改变的是:Scope**
> - Scope:一个固定数量的持久化 CTA 池,每个 CTA 通过调度器循环处理多个输出分块。
> - Layout:不变,沿用相同的按分块的 SMEM/TMEM/寄存器通路。
> - Dispatch:不变。

### 持久化调度

持久化内核的定义性想法,是把自己的网格尺寸对齐硬件,而不是对齐问题本身。它启动 `SM_COUNT` 个 CTA,大致每个 SM 一个,而不管恰好有多少个输出分块,目的是让每个 SM 持续被占满。我们特意说"大致":精确的 1:1 占用并不保证,因为它取决于占用率以及硬件如何选择调度 CTA。

在我们此处作为目标的 B200 上,`SM_COUNT=148`。这 148 个 CTA 中的每一个,都在循环处理由 `ClusterPersistentScheduler2D` 交给它的分块。

第一份回报来自摊薄。TMEM 分配、屏障初始化以及调度器状态,现在每个 CTA 只发生一次,并在该 CTA 处理的那大约 7 个分块上复用,而不是在 1024 个用完即弃的 CTA 上被重复 1024 次。

第二份回报来自调度器挑选的顺序。设 `l2_group_size=8` 把相邻分块成组,使得共享同一行带的分块复用相同的 A 行分块,共享同一列带的分块复用相同的 B 分块。把这些分块背靠背地跑,就让操作数在 L2 里保持热,而不是从 HBM 里重新取。这恰好就是 Step 3 留在桌上的那份复用。

```python
bx = T.cta_id([SM_COUNT])  # 一维网格,每个 SM 一个 CTA

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # 把相邻 8 个分块成组
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

跨分块循环会带来一个容易忽视的正确性后果。每个分块都跑自己全新的 K 循环,这意味着它的屏障相位必须从一个已知状态开始。在 Step 5 里,一个 CTA 恰好只处理一个分块,所以把 `phase_tma` 和 `phase_mma` 初始化一次完全没问题。在 Step 6 里,这些初始化必须移到 `while tile_scheduler.valid()` 循环*内部*,好让每个分块都以匹配自身 TMA 和 MMA 工作的相位状态开始,而不是继承上一个分块恰好留下来的状态:

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### 完整内核

在结构上,这个内核无非就是把 Step 5 的流水线包进一个分块级别的外层循环里。唯一的新依赖是调度器本身,我们把它和其余导入一起引入:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

网格维度现在直接就是 `SM_COUNT`,而不再是 `(M//BLK_M, N//BLK_N)`,并且一个 `ClusterPersistentScheduler2D` 接手了把分块派发给各 CTA 的工作:

```python
SM_COUNT = 148  # NVIDIA B200 GPU 上的 SM 数量
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 一维网格:每个 SM 一个 CTA(不再是二维网格!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM 分配(同 Step 5)---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- 屏障与 TMEM 初始化(同 Step 5)---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # 分块调度器:以对 L2 友好的顺序把分块派给各 CTA
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === 外层循环:遍历分块 ===
        while tile_scheduler.valid():
            # 从调度器取当前分块位置
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === 内层循环:同 Step 5 的流水线 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # 预取头 PIPE_DEPTH 个流水级
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # 主 K 循环
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA 存储回写:TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # 移到下一个分块

        # 释放 TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```
## 练习

1. 在 Step 4 里,`arrive.expect_tx` 用了 `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 字节。如果这个字节数太小或太大,mbarrier 会等待什么?
2. 在 Step 5 里,为什么每个 SMEM 流水级需要自己独立的 TMA 屏障,而不是两个流水级共用一个 `tma_bar`?
3. 在 Step 6 里,一个 4096 x 4096 的输出,在 `BLK_M=BLK_N=128` 下有多少个输出分块?在 `SM_COUNT=148` 时,每个持久化 CTA 平均处理多少个分块?
