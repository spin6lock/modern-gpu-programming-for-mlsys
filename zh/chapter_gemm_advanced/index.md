(chap_gemm_advanced)=
# 用线程束特化与集群扩展 GEMM

:::{admonition} Overview
:class: overview

- 流水线化的 GEMM 仍由一个 warpgroup 顺序完成加载、MMA 与回写,这正是本章要消除的瓶颈。
- 第 7 步把 warp 特化成不同角色,第 8 步加入 2-CTA 集群,第 9 步加入多个消费者。
- 每一步都移除一个串行瓶颈,最终逼近当前最优的吞吐量。
:::

上一章({ref}`chap_gemm_async`)得到的流水线 GEMM 已经很快,但它仍要求一个 warpgroup 包揽一切:发起加载、运行 MMA、再把结果写回。即便有软件流水线,这一队线程也仍是三台引擎交汇的唯一通道。

症状很明显。Tensor Cores 运行时 TMA 单元安静下来,结果排空到显存时 Tensor Cores 又安静下来,每台引擎都通过同一组线程等待彼此。突破之道就是不再让一支队伍包揽一切。

我们分三步逐步扩大协作范围来实现这一思路。第 7 步({ref}`chap_warp_specialization`)把 warp 特化为生产者、消费者与回写三种角色。第 8 步({ref}`chap_cta_cluster`)把两个 CTA 联合成一个集群,通过彼此的共享内存共享操作数。第 9 步({ref}`chap_multi_consumer`)加入第二个 MMA 消费者,让一份已加载的分块驱动两倍的运算。

把这三步看作同一模式在不同尺度上的展开会很有帮助。第 7 步把整条流水线保留在一个 CTA 内:TMA 与 MMA 共用一个 warpgroup,回写在另一个 warpgroup 中运行。第 8 步把协作扩展到 CTA 之间,产生一个横跨两者的 256×256 分块。第 9 步进一步推高计算密度:集群输出增长到 512×256,每份已加载的 B 分块被两个消费者复用,由此得到本教程中最密集的变体。

这一切中有一件事始终不变。SMEM、TMEM 与寄存器布局仍然遵循前两章建立的契约;变化的是*谁来协作*,而不是数据如何摆放。第 8 步是协作作用域首次超出单个 CTA,因此它的操作数分块被拆分到两个 CTA 的共享内存中,一份布局沿 `cbx` 集群轴横跨两个 CTA。


(chap_warp_specialization)=
## 第 7 步:线程束特化 + 流水线

单 warpgroup 内核之所以把性能留在桌上,原因很简单:每个线程走同一条路径——加载、计算、再写回,于是加载时 Tensor Cores 无事可做,计算时 TMA 引擎也无事可做。解决办法是*warp specialization(线程束特化)*。我们不再让一队线程轮流承担每项工作,而是把每项工作交给一个专职 warp,让这些 warp 同时运行,再由一条软件流水线把它们缝合起来。这是 GEMM 路径上最大的架构变化,本章余下部分都建立在它之上。这里的基准测试取 M=N=K=4096。

> **本步改变了什么:作用域**
> - 作用域:一个 warpgroup 顺序走过 加载 → MMA → 回写,变成由满/空屏障连接的三个并发角色(TMA 生产者、MMA 消费者、回写)。
> - 布局:不变,沿用第 6 步的 SMEM 流水级与 TMEM 累加器。
> - 派发:不变,TMA 加载,`tcgen05` MMA。

**主题。**

- warp specialization(线程束特化):把不同的 warp/warpgroup 专门用于不同任务

- 高层屏障抽象:`TMABar`、`TCGen05Bar`、`MBarrier`

- `PipelineState` 自动管理流水级/相位

- `warpgroup_sync` 的屏障 ID 用于按 warpgroup 同步

(多级 SMEM 流水线与持久化的 `ClusterPersistentScheduler2D` 直接复用第 5–6 步的实现;这里只有作用域拆分是新的。)

### 从串行到并发

在介绍角色与屏障之前,先孤立出 warp specialization 所移除的调度瓶颈会很有帮助。下图用第 4 步风格的串行时间线,作为第 4–6 步中前置特化内核的紧凑参照,再把它放在第 7 步 warp 特化调度的上方,使引擎利用率差异一目了然。

![线程束特化时间线](../img/warp_specialization_timeline.png)

上方是特化前的单 warpgroup 模式:同一组未特化的线程同时拥有加载路径和 MMA 路径,因此一台引擎很容易在另一台活跃时空闲。第 5、6 步用双缓冲和持久化调度改进了这一基线,但它们尚未把加载与计算拆成独立的生产者和消费者角色。下方,特化打破了这种轮替。TMA 生产者在 MMA 消费者忙于计算时预取下一个分块,回写则自行推进。生产者 warp 3 在消费者 warp 0 仍在处理当前 MMA 时就发起下一次加载,因此两台引擎都不必等待对方。加载/MMA 的交接使用两个屏障:

- **`tma2mma`**(TMA → MMA):表示已加载的 SMEM 数据已就绪,可供 MMA 消费。

- **`mma2tma`**(MMA → TMA):表示 MMA 已读完某个缓冲,该缓冲可由 TMA 用于下一次加载。

图中有一个细节乍看像是错的:`mma2tma` 的箭头会跳过一个流水级。原因在于环形缓冲。在 `PIPE_DEPTH=2` 时有两个 SMEM 缓冲,即 0 号和 1 号流水级;TMA Load k=0 填充缓冲 0,TMA Load k=1 填充缓冲 1。当 MMA Compute k=0 读完缓冲 0 后,它会发出 `mma2tma` 信号表示该缓冲已空闲,但真正想要回缓冲 0 的加载是 TMA Load k=2,而不是 k=1(k=1 用的是缓冲 1)。这就是为什么从 MMA Compute k=0 发出的 `mma2tma` 箭头会一直延伸到 TMA Load k=2。释放跳过一个流水级,纯粹是因为这个环有两个槽位。

### Warp 角色

时间线展示了*为什么*要拆分工作;接下来的问题是每部分*由谁*来做。特化把三项工作(加载、计算、回写)分配给特定的 warp,使它们能并发运行。在 `WG_NUMBER=2` 下,内核使用两个 warpgroup(角色表中简写为 WG):

| 参与方 | 位置 | 职责 |
|-------|----------|-----|
| **TMA 生产者** | Warpgroup 1,warp 3 | 持续通过 TMA 加载 A 和 B 分块 |
| **MMA 消费者** | Warpgroup 1,warp 0 | 数据一就绪就立即运行 MMA |
| **回写** | Warpgroup 0(全部 warp) | 读 TMEM 结果,写入 GMEM |

### 4 个屏障

三个并发参与方需要四个屏障,这四个屏障可以整齐地分成两个相反的方向。前向路径(TMA → MMA → 回写)发出数据*就绪*信号;它的消息是"你等的那个分块到了"。反向路径(回写 → MMA → TMA)发出缓冲*释放*信号:"你要的那个槽位又空了。"一旦掌握了命名约定,这些名字就不言自明:每个屏障都形如 `source2destination`,所以 `tma2mma` 不过是 TMA 用来通知 MMA 的那个屏障。

| 屏障 | 类型 | 方向 | 含义 |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM 数据已就绪" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM 缓冲可复用" |
| **mma2ld** | `TCGen05Bar` | MMA -> 回写 | "TMEM 结果已就绪" |
| **ld2mma** | `MBarrier` | 回写 -> MMA | "TMEM 已空闲,可用于下一个分块" |

为什么每个屏障的*类型*各是现在这样?类型取决于生产者如何宣布自己完成。**TMA Loads** 用 `TMABar`,一种带字节计数的 mbarrier:传输的字节一落地,TMA 硬件本身就到达该屏障,于是消费者无需任何线程轮询就能得知数据已就绪。**TMA Stores** 用不了它(一次存储没人可通知),所以回退到 `cp_async.bulk.commit_group()` + `wait_group(0)`,由发起线程自己等待它的写操作排空。**MMA 操作** 用 `TCGen05Bar`,其上 `tcgen05.commit()` 指令在 MMA 完成时通知屏障。

这里有个小细节会在第 8 步派上用场。各 `arrive` 调用传入 `cta_mask=0`,因为在单 CTA 内核中没有别的 CTA 可通知。等第 8 步组成集群时,正是这个参数变为非零,成为唤醒协作 CTA 的机制。

### PipelineState

四个屏障告诉各角色某个缓冲*何时*就绪;但当流水线循环时,还需要有人跟踪每个角色*正在*哪个缓冲。这种记账正是 `PipelineState` 管理的内容。环形缓冲同时带着两份记账:当前在哪个槽位,以及等待的是该槽位屏障的哪个"相位"。在流水化循环里手工同时跟踪这两者,正是滋生差一错误的地方,而这里一个差一就会让整个内核死锁。`PipelineState` 的存在就是把两者绑在一起,省去你来操心:

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # 生产者以就绪状态启动(phase=1)
# tma_ps.stage = 当前的流水级索引
# tma_ps.phase = 当前的相位(0 或 1)
tma_ps.advance()                          # 推进到下一个流水级
```

初始的 `phase` 决定了一个角色的第一次 `wait` 是放行还是阻塞,而流水线两端的正确答案正好相反,这正是容易绊倒人的地方:
- `phase=1`(生产者)-> 第一次 `wait(phase=1)` 时屏障仍处于相位 0,由于 0 != 1,它会**立即放行**。这正是我们想要的,因为缓冲一开始是空的,生产者应当能自由地立刻开始填充。

- `phase=0`(消费者)-> 第一次 `wait(phase=0)` 时屏障处于相位 0,由于 0 == 0,它会**阻塞**。同样是我们想要的,因为此时还没有数据,在生产者到达之前消费者无从读取。

给两端设相同的起始相位会得到一个死锁,更糟的话是悄无声息的数据损坏,所以这一处选择值得做对。

### `warpgroup_sync` 屏障 ID

线程束特化引入了一种容易踩中的同步隐患。一旦每个 warpgroup 跑不同的代码路径,熟悉的 `cta_sync()` 就会死锁:它使用硬件屏障 #0 并坚持*每个* CTA 线程都到达,然而在 warpgroup 分支内只有其中一部分线程在场。我们真正需要的是一个作用域限于单个 warpgroup 的屏障。GPU 提供了 16 个具名屏障(ID 0–15),因此这些内核改用 `warpgroup_sync(10)`,它只同步一个 warpgroup 内的线程。当多个 warpgroup 各自需要独立同步时(如多消费者的第 9 步),它们通过 `warpgroup_sync(wg_id + 10)` 取用不同的 ID,从而永远不会撞上同一个硬件屏障。

**实现。**

这里用 `PIPE_DEPTH=2`,这是仍能让加载与计算有任何重叠的最小深度。再深一些可以隐藏更多访存延迟,上限是 SMEM 预算;下文*当第 7 步出问题时*会详细推演这一权衡。至此所有部件(角色、四个屏障、`PipelineState` 和 warpgroup 作用域同步)都已齐备,我们可以拼出完整内核:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # NVIDIA B200 GPU 上的 SM 数量
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- 屏障初始化 ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # Warpgroup 0 的全部 128 个线程都到达
        pool.commit()

        # --- TMEM 分配 + 栅栏 ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- 分块调度器 ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1:TMA 生产者(warp 3)+ MMA 消费者(warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA 生产者 ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA 消费者 ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # 等待 TMEM 被上一分块的回写释放
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # 通知回写结果已就绪
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0:回写
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # 等待 MMA 结果
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # 读 TMEM -> 寄存器(warpgroup 作用域)
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # 通知 TMEM 已空闲(全部 128 个线程到达)
                ld2mma.arrive(0, cta_id=0, pred=True)

                # 把 fp32 转换为 fp16
                Tx.cast(reg_f16[:], reg[:])

                # 写入 Dsmem + TMA 存储
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

要运行其中任何一个内核,复用第 1 步({ref}`chap_gemm_basics`)中展示过一次的编译/运行/校验脚手架即可:把 `hgemm_v1` 换成 `hgemm_v7`、`hgemm_v8` 或 `hgemm_v9`,并选一个问题规模例如 `M=N=K=4096`。注意集群化的步骤需要 `M` 和 `N` 是其集群分块的整数倍(第 8 步为 `256×256`,第 9 步为 `512×256`),所以一个极小的 `128×128` 规模根本产生不出任何分块。每一步都要在一个全新的 Python 会话中编译,切换步骤前重启内核,因为这些内核复用了内部名字,而编译器会持有每会话状态。各步的耗时汇总在下面的*端到端结果*中。

### 收尾(回写)细节

第 7 步可以采用一种简单得令人愉快的收尾。由于只有 `BLK_N=128` 列,回写 warpgroup 只需一轮就把整块 TMEM 分块读进寄存器,然后发起一次 TMA 存储。第 8、9 步没有这种奢侈,这正是它们后来要引入分块读取的原因;就目前而言,顺序是:

1. 等待 MMA:`mma2ld.wait(phase)`。本教程第 8、9 步会在此额外加一个 `fence.after_thread_sync()` 作为保守补强;MMA 完成的 mbarrier 已经覆盖了该顺序,而大多数内核(包括 CUTLASS)都省略它,所以第 7 步同样省略。
2. 读 TMEM -> 寄存器(每线程 128 个 fp32,warpgroup 作用域,通过 `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` 再接 `T.ptx.tcgen05.wait.ld()`)。
3. 通知 MMA:`ld2mma.arrive(0, cta_id=0, pred=True)`(全部 128 个线程到达);TMEM 此刻对下一个分块空闲。两个 `arrive` 关键字参数在集群化步骤中会再次出现:`cta_id` 指明*哪个 CTA 的*屏障副本被通知(`0` = 本 CTA,即本地屏障;第 8 步中协作式的 arrive 通过 `cta_mask` 指向 CTA-0),而 `pred` 是逐线程的谓词,门控该线程是否真正参与到达(此处为 `True`,因此每个回写线程都计入到达总数)。
4. 在寄存器中把 fp32 转换为 fp16。
5. 把寄存器 -> Dsmem 写回,然后 `fence.proxy_async("shared::cta") + warpgroup_sync(10)` 刷出。
6. 通过 `cp_async.bulk.commit_group() + wait_group(0)` 把 Dsmem 用 TMA 存储 -> GMEM。

第 8 步(用 `BLK_N=256`)和第 9 步(每个消费者用 `MMA_N=256`)无法保持这种单轮形态,原因在于寄存器压力。每线程读 256 个 fp32 值意味着 256 × 4 = 1024 字节必须同时存活在每个线程的寄存器中,这有溢出到 local memory 的风险,此外还迫使 Dsmem 缓冲更大。所以这些步骤把回写拆成 `EPI_N` 列的小块(`EPI_N=64`):每次迭代只让 `EPI_N` 个 fp32 寄存器处于活跃,并发起相应更小的 TMA 存储,用多一点存储指令换取仍可舒适容纳的寄存器预算。

**实现说明。**

- **持久化内核**:`bx = T.cta_id([SM_COUNT])` --- 每个 SM 一个 CTA,循环遍历各分块

- **对 L2 友好的调度**:`ClusterPersistentScheduler2D` 按缓存局部性排序分块

- 这种模式 --- warp specialization 加软件流水线 --- 在高性能 GEMM 内核中很常见,包括 CUTLASS 风格的设计。

### 当第 7 步出问题时

第 7 步是第一个让 TMA 加载、`tcgen05` MMA 与回写同时并发的 GEMM 内核。同样的失败模式在第 8、9 步会再次出现:屏障计数不匹配、角色守卫放错位置、缺失栅栏,或暂存缓冲在 TMA 存储排空前被复用。这些情况的调试清单汇总在 {ref}`chap_warp_spec_debug`。

**流水线深度调优。** 第 7 步内核以 `PIPE_DEPTH=2`(最小值)运行。把它推到 4 或 6 能让 TMA 生产者比 MMA 消费者跑得更靠前,隐藏更多访存延迟,但代价是消耗更多 SMEM,而 SMEM 是有限的。B200 每个 SM 提供 228 KB(见 {ref}`chap_background` 的*需要记住的数字*)。在 `BLK_M=BLK_N=128, BLK_K=64, fp16` 下,每个流水级仅 A 和 B 就占 `(128*64 + 128*64) * 2 = 32 KB`,再加上 `Dsmem` 回写暂存缓冲又多 32 KB。这让 `PIPE_DEPTH=4` 大约 160 KB,`PIPE_DEPTH=6` 大约 224 KB,正好贴着预算。要再深一些,就得重新考虑回写暂存策略。

---

warp specialization 让一个 CTA 内的线程协作起来。下一步把这种协作扩大到 CTA 自身的边界之外,让两个 CTA 共同处理一个更大的分块。


(chap_cta_cluster)=
## 第 8 步:2-CTA 集群

第 7 步让各引擎重叠起来,但每个 CTA 仍在孤立地计算自己的 128×128 分块,重新加载任何邻居都借不到的操作数。第 8 步打破这种孤立。两个 CTA 联合成一个集群,并获得访问彼此共享内存的能力,于是一次协作式的 `tcgen05` MMA 就能产出一个横跨两者的 256×256 分块,而一次 B 的加载现在能驱动两倍的 MMA 工作。一如既往,M=N=K=4096。

> **本步改变了什么:作用域 + 布局 + 派发**
> - 作用域:协作作用域现在横跨集群中的两个 CTA,而不止一个。
> - 布局:操作数分块被拆分到两个 CTA 的 SMEM 上;CTA 0 拥有共享的完成屏障(`remote_view`)。
> - 派发:MMA 增加 `cta_group` / `cta_mask`,使 `tcgen05` 作为 2-CTA 协作操作运行。

**主题。**

- CTA cluster(CTA 集群):多个 CTA 协作处理一个更大的分块

- 通过 `map_shared_rank` 跨 CTA 访问 SMEM

- 用 `cta_group=2` 在 256x256 集群分块上做协作式 MMA

- 用 `cta_mask` 跨 CTA 发出屏障信号


### 集群分块形状

整个优化建立在一项硬件能力之上:在 `cta_group=2` 下,MMA 允许读取由*两个* CTA 暂存的操作数分块,而不只是它所在的那个。每个 CTA 加载一行 128 行的存储 B 切片,经转置后变成 128 个逻辑输出列,而协作式 MMA 把这两个切片重新拼合成一个操作数。下图追踪两个 CTA 的 A、B 切片如何合并为单一的 256×256 集群分块:

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*可交互:每个 CTA 拥有一片 A 行切片和一片存储-B 行切片,然后跨集群(DSMEM)读取另一个 CTA 的存储-B 切片。`B.T` 之后,两片存储-B 切片覆盖完整的输出列跨度,于是这对 CTA 产出一个 256×256 输出分块。*

**为何 A 和 B 跨集群拆分**:要看清 256×256 分块如何划分,回忆本教程把 GEMM 存储为 `D = A @ B.T`,其中存储的 B 形状为 `N x K`。集群中有两个 CTA,划分就干净地落下来:

- **A 按列拆分**:CTA-0 持有 A0(行 0-127),CTA-1 持有 A1(行 128-255)。堆叠起来:`[A0; A1]`(256 行)。

- **存储 B 按行拆分**:CTA-0 加载 B 的行 0-127,CTA-1 加载行 128-255。由于计算用的是 `B.T`,这两片存储的行切片变成逻辑右操作数的两片 128 列切片。

- 在 `cta_group=2` 下,MMA 硬件通过跨 CTA 共享内存访问从**两个** CTA 的 SMEM 读 B,于是它看到完整的逻辑输出列跨度。

- 结果:两个 CTA 在一个 256x256 输出分块上协作。每个 CTA 写出该分块的一条 128x256 行带。

值得停下来看清这是真正的收益,而不仅仅是对工作的重新洗牌。每个 CTA 仍只加载 128×K 的 A 和 128×K 的 B,所以集群整体暂存的操作数大约是单个 CTA 的 2×,然而它产出 256×256 的分块,携带约 4× 于 128×128 分块的输出 FLOPs。于是 MMA 对每字节已暂存操作数做的工作大约翻倍,因为每个 CTA 的 B 切片会通过协作式 MMA 与另一个 CTA 的 A 切片配合复用。换句话说,算术强度大约翻倍,而这正是一个仍偏访存的内核所需要的杠杆:端到端表中约 2.2× 的加速就来自把同样的字节喂给更多运算。

### 分块地址计算

既然集群成了工作单元,分块调度器也得以集群分块来计数。它返回的每个 `(m_idx, n_idx)` 命名一整块 256×256 区域,而集群内的两个 CTA 在它们之间拆分该区域。把一个集群坐标翻译成每个 CTA 实际加载的逐 CTA 切片,看起来是这样:

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTA 处理*同一个* 256×256 集群分块,而单一坐标 `cbx`(CTA 在集群内的位置,0 或 1)正是沿两条轴选出本 CTA 贡献的关键。`m_st` 选出本 CTA 拥有的输出行带,`n_st` 选出它喂给协作式 MMA 的存储-B 切片,而稍后的回写则发出 256 列输出跨度的两段 128 列。还要注意 `num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 数的是集群分块,而不是单个 CTA 分块。

乍看之下 `cbx` 同时出现在 `m_st` 和 `n_st` 中,仿佛一个行偏移不知怎么漏进了列里,但两处用法都是对的,值得理清原因。在回写路径上,`cbx` 只属于 M 轴:每个 CTA 拥有一条不同的 128 行带(`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`,于是 CTA-0 写行 `m_idx*256 .. +128`,CTA-1 写接下来的 128 行),然而两个 CTA 都写出集群分块的*完整* 256 输出列。这正是为什么存储从集群的 `n_idx` 推导列(`n_st_epi = n_idx * 256 + no * 128`,看不见 `cbx`),而不是从逐 CTA 的 `n_st` 推导。`n_st` 之所以带 `cbx`,是因为每个 CTA 把不同的存储-B 行切片加载进 MMA:在那里,`cbx` 是一个*加载*偏移,而不是该 CTA 的输出列偏移。

### 相对第 7 步的代码改动

相对第 7 步的 diff 有六处编辑,每一处编码了上面所述集群契约中的一条:

```python
# 1. 集群启动
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA 在集群内的索引(0 或 1)

# 2. 协作式 MMA(原来是 cta_group=1)
Tx.gemm_async(..., cta_group=2)

# 3. 跨 CTA 共享内存访问
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. 跨 CTA 屏障
tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. mma2tma / mma2ld 的 arrive 从 cta_mask=0(单 CTA,第 7 步)
#    变为 cta_mask=3(通知集群内的两个 CTA)
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. 结尾用 cluster_sync 取代 cta_sync
T.cuda.cluster_sync()
```


### 集群作用域的改动

这六处编辑都源于同一个转变:协作作用域现在是集群,而不是单个 CTA。下面几点具体说明这种扩大在实践中意味着什么:每个 CTA 如何找到自己的位置、集群在谁的屏障上协调、以及究竟哪个 CTA 发起协作式 MMA。

- **集群 CTA ID**:`cbx` 告诉每个 CTA 它在集群中的位置(0 或 1)。CTA-0 处理 A 的行 0-127,CTA-1 处理行 128-255。

- **远程屏障视图**:在集群中,每个 CTA 有自己的 SMEM 和自己的屏障,这引出一个显而易见的问题:如果 CTA-1 需要等待 CTA-0 产生的某物,它实际触碰的是谁的屏障?答案是提名 CTA-0 的屏障作为唯一协调点,并让集群内任何 CTA 都能访问它。`map_shared_rank(tma2mma.ptr_to([0]), 0)` 返回一个集群范围内指向 CTA-0 屏障的指针,经 TIRx 包装为 `tma2mma.remote_view(0)`,此后每个 arrive 和 wait 都指向 CTA-0 的副本。

- **仅由 CTA-0 派发 MMA**:很容易把 `cta_group=2` 读成同时触发两台引擎并行,但实际并非如此。CTA-0 恰好发起一次 `tcgen05.mma`,硬件随后驱动一次*单一协作式* MMA,横跨两个 CTA,从两个 SM 的 SMEM 读操作数,并把累加器写到两个 SM 的 TMEM 上。CTA-1 完全不发 MMA。(每个 SM 只有一个 `tcgen05` 引擎,所以 `cta_group=2` 是一次跨 SM 的 MMA,而不是两台引擎并排运行。)这就是为什么代码用 `if cbx == 0:` 守护 MMA。

- **多播 arrive**:`tcgen05.commit(..., cta_group=2, cta_mask=3)` 只由 CTA-0 发起,但通知两个 CTA 的屏障。`cta_mask=3`(二进制 `11`)意味着同时指向 CTA-0 和 CTA-1。

- **ld2mma 初始化计数**:`init(128 * CTA_GROUP)` --- 两个 CTA 的回写 warpgroup(各 128 个线程)都到达。


**实现。**

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- 屏障初始化 ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的回写线程
        pool.commit()

        # --- TMEM 分配(协作式)---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- 分块调度器(集群分块)---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- 跨 CTA 屏障视图 ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1:TMA 生产者(warp 3)+ MMA 消费者(warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0:回写(256 列分成 2 个 128 列小块)
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 个 128 列小块 = 共 256 列
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**2 个 CTA 带来的变化。**

- `CTA_GROUP = 2`,`MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- 两个 CTA 的回写 WG 都到达

- TMA arrive 的字节计数包含两个 CTA:`CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须用 `cta_group=2`

- 回写把 256 个输出列拆成两个 128 列小块 --- 一次读完所有 256 个 TMEM 列会超出寄存器容量。第 9 步把小块进一步缩小到 `EPI_N=64`

- 结尾用 `cluster_sync()` 取代 `cta_sync()`(确保 TMEM 释放前所有 CTA 都已完成)

所有这些额外的算术强度直接反映在墙上时间上:第 8 步在 4096³ 下达到 **0.104 ms**,约为同样规模下 70 ms 第 1 步算法的 676×(见端到端表)。内核现在倾向于计算受限,而这恰好为第 9 步打下基础——我们在那里加入第二个 MMA 消费者,让更多 Tensor Core 工作并发起来。

如果第 8 步比第 7 步*更慢*,罪魁几乎总是某条新的集群契约被略微搞错。值得优先检查三件事:TMA arrive 的字节计数是否为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`;针对 256×256 集群分块,调度器维度是否为 `num_m_tiles=M//256, num_n_tiles=N//256`;以及回写是否发起了两次 TMA 存储,每个 128 列小块一次,且每次都在 Dsmem 复用前排空。

---

集群提升了*跨* CTA 的复用。最后一步转向内部,通过给生产者加上第二个 MMA 消费者来喂养,在每个 CTA *内部*提升计算密度。


(chap_multi_consumer)=
## 第 9 步:多消费者的线程束特化

到第 8 步,MMA 确实已经很忙,但单个消费者 warp 消化一份已加载 B 分块的速度也就那么快,而那份 B 分块在这段时间里就一直待在 SMEM 里,任何想读它的人都可读取。最后的优化正是利用这一点:它加入第二个 MMA 消费者,把一块*不同的* A 块乘到*同一份* B 分块上。每个 CTA 的计算密度翻倍,集群输出从 256×256 增长到 512×256。一如既往,M=N=K=4096。

> **本步改变了什么:作用域 + 布局**
> - 作用域:一个 MMA 消费者变成两个,由 `warp_id` 选出。
> - 布局:一份已加载的 B 分块被两个消费者复用;A 增加一个消费者轴。
> - 派发:不变。

**主题。**

- 多个 MMA warp(消费者)以获得更高吞吐量

- 多个回写 warpgroup,各自有独立的屏障槽位

- 本教程中最优化的 GEMM 变体所采用的结构


### 多消费者结构

加入第二个消费者意味着内核现在有更多不同角色要安排:两个 MMA warp 而不是一个,外加一个与之配套的第二个回写 warpgroup 来排空额外的累加器。在 `NUM_CONSUMER=2` 和 `WG_NUMBER=3` 下,内核现在横跨三个 warpgroup(角色表中简写为 WG):

| Warpgroup | Warp | 角色 |
|-----------|------|------|
| **WG 2** | warp 0 | MMA 消费者 0:`Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| **WG 2** | warp 1 | MMA 消费者 1:`Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| **WG 2** | warp 3 | TMA 生产者:每个流水级加载 2 份 A 块 + 1 份 B 块 |
| **WG 0** | 全部 | 消费者 0 的回写:读 TMEM `[0:256]` |
| **WG 1** | 全部 | 消费者 1 的回写:读 TMEM `[256:512]` |

整套安排的关键在于一处不对称。每个消费者把各自的 A 块乘到*同一份*已加载的 B 分块上,所以一次 B 的加载现在驱动 2× 的 MMA 工作,B 的加载成本相对于有效 FLOP 实际上减半。我们共享 B 而不是 A 的原因在于,两个消费者覆盖不同的 M 行带:它们的 A 块是真正不同的数据,而 B 对两者相同。练习 3 让你确信这是唯一可行的共享方式。

### 相对第 8 步的改动

具体来说,支持第二个消费者会在几处触及内核,而每一处改动都可追溯到同一个事实:现在每个流水级要喂入并排空两个 A 块和两段 TMEM 范围,而 B 保持共享。下面的编辑暂存额外一个 A 块,给每个消费者自己的屏障槽位,并为更高的 512×256 集群分块调整分块寻址。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 每个流水级 2 个 A 块,每个消费者一个

- TMA 同时加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`,TMA arrive 字节数现在是 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`(多出的 A 块)

- MMA warp 的 `warp_id` 选出使用哪个 A 块和 TMEM 范围

- `mma2tma.init(NUM_CONSUMER)` --- 两个消费者每个流水级都通知 TMA

- `mma2ld` 和 `ld2mma` 的 `depth=NUM_CONSUMER` --- 每个消费者用自己的屏障槽位(MMA 侧用 `warp_id`,回写侧用 `wg_id`)

- 分块地址:`m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M 方向上多出一个 `NUM_CONSUMER` 因子,因为每个集群分块现在在 M 上横跨 `NUM_CONSUMER` 个消费者。分块调度器使用 `num_m_tiles = M // 256 // NUM_CONSUMER`(集群分块为 512x256)

- 回写采用分块的 `EPI_N`,使每次迭代在寄存器中存活的 TMEM 回读值更少


**实现。**

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2,每个消费者一个槽位
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2,每个消费者一个槽位
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- 屏障初始化 ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # 每个流水级期望 2 次到达
        mma2ld.init(1)              # 每个槽位获得 1 次到达
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的回写线程
        pool.commit()

        # --- TMEM 分配(协作式)---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- 分块调度器(512x256 集群分块)---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2:TMA 生产者(warp 3)+ 2 个 MMA 消费者(warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA 生产者:每个流水级加载 2 份 A 块 + 1 份 B 块 ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA 消费者:warp_id 选出 A 块和 TMEM 范围 ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1:回写(各自读其消费者的 TMEM 范围)
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # 等待本消费者
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # 按 EPI_N=64 列的小块读 TMEM(256 列需 4 次迭代)
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**实现说明。**

- 在第 9 步的这种设计中,`mma2ld` 和 `ld2mma` 各自是单个共享对象且 `depth=NUM_CONSUMER`,而不是每个消费者各自独立。槽位 0 把 MMA warp 0 连到 Warpgroup 0,槽位 1 把 MMA warp 1 连到 Warpgroup 1;MMA 侧用 `warp_id` 索引,回写侧用 `wg_id` 索引。

## 端到端结果

下表给出从朴素基线到线程束特化集群内核的实测里程碑,并附 cuBLAS 参照。NVIDIA B200 上的参照数据,M=N=K=4096,fp16,锁定频率,1000 次迭代计时基准:

| 步骤 | 技术 | 耗时 | 加速比 |
|------|-----------|------|---------|
| 1 | 同步加载 + MMA | 70 ms | 1× |
| 2 | K 循环累加 | --- | 处理 K 大于一个分块 |
| 3 | 空间分块 | 53.6 ms | ~1.3× |
| 4 | TMA 异步加载 | 0.49 ms | ~142× |
| 5 | 软件流水线 | --- | 重叠加载 + 计算 |
| 6 | 持久化内核 | --- | L2 缓存局部性 |
| 7 | 线程束特化 | 0.23 ms | ~309× |
| 8 | 2-CTA 集群 | 0.104 ms | ~676× |
| 9 | 多消费者 | 0.094 ms | ~744× |
| --- | cuBLAS(参照) | 0.094 ms | ~744× |

这张表里的每个耗时,包括 70 ms 的第 1 步基线,都是在同一个 M=N=K=4096 规模下测得的,正是这一点让整条加速链可以端到端比较。值得精确说明这 70 ms 究竟是什么,因为它很容易被误读。它*不是* {ref}`chap_gemm_basics` 中那个单分块第 1 步内核在 4096³ 下跑出来的结果;那个内核只会算一个 128×128 分块,只在较小规模下运行。这 70 ms 是一个朴素的全规模基线,采用同样的串行单分块方法,把它放大到完整的 4096³ 问题。第 1–3 步在 {ref}`chap_gemm_basics` 中以小规模(128×128 和 256³)引入,以让最初几趟走读保持简单;此处的第 1、3 步两行是它们在全规模下的基准对应。余下的横线(第 2、5、6 步)标记那些为展示结构而呈现、但并未单独计时的步骤。

请把这些数字看作受控条件下 B200 的一次参照运行,而不是榜单条目。各步骤中内嵌的 `{.python .input}` 基准测试单元是冒烟基准:它们适合发现趋势,不适合声称峰值性能。

四项技术几乎贡献了全部增益:

1. **TMA 异步数据搬运**:硬件拷贝引擎取代软件拷贝(第 1 步 → 第 4 步约 142×)。正确解读这个 142× 很重要:它反映的是从一个 128×128 单分块内核(grid 1×1)一路走到一个带 K 循环、空间分块、多 CTA 的完整分块并行内核,*再加上* TMA;它并不是 TMA 单独的贡献。要单独剥离 TMA,得比较两个仅在拷贝机制上不同的全规模内核。
2. **软件流水线 + 线程束特化**:通过让加载和计算各有专职角色来重叠二者(第 4 步 → 第 7 步约 2.2×)。
3. **CTA 集群**:一次 2-SM 协作式 MMA 改善了跨 CTA 的 B 分块复用(本基准中第 7 步 → 第 8 步约 2.2×)。
4. **多消费者**:两个 MMA warp 提高计算密度(第 8 步 → 第 9 步约 10%)。

按实测里程碑绘制,这四项贡献正好描绘出从同步分块内核向 cuBLAS 参照下探的轨迹。下图展示所选的实测点:

![GEMM 优化历程](../img/gemm_perf.png)

注意越往下增益越小,这背后有结构性原因,而不是努力减弱。前几步针对*访存*瓶颈(TMA 取代软件拷贝,集群提升算术强度),而 70 ms 大部分确实花在那里,所以这些步骤回报最大。到第 8 步,内核已经距 cuBLAS 不到 10%(0.104 vs 0.094 ms),接近*计算受限*,意味着已几乎没有可隐藏的访存停顿;第 9 步的多消费者重叠回收了所剩无几的大部分。约 10% 的最终增益正是逼近计算上限时应有的预期:它是一个近乎已解问题的边际递减,而非一次孱弱的优化。

本章构建的所有内容(TMA 加载、`tcgen05` MMA、TMEM 回读、线程束特化屏障)都直接延续到下一章。FlashAttention 复用了全部这些,随后通过在两个 MMA 阶段之间楔入一个在线 softmax 步骤,而非简单重复单一阶段,把难度又提了一档。


## 练习

1. 在第 7 步中,如果把 TMA 和 MMA 的 `PipelineState` 初始 `phase` 都设为 `0` 会怎样?画出死锁场景。
2. 在第 8 步的 `cta_group=2` 下,TMA arrive 的字节计数是 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`。每个 CTA 各自加载自己的数据,为什么还要乘以 `CTA_GROUP`?
3. 在第 9 步中,每个消费者处理不同的 M 行但同一份 B 分块。为什么共享 B(而非 A)是正确的选择?

**交给你的 agent 试试**:粘贴第 7 步内核,让它追踪一个 K 分块穿过四个屏障(`tma2mma`、`mma2tma`、`mma2ld`、`ld2mma`)的全过程。针对每个屏障,问它谁在等待、谁在到达、哪个分块变得可安全读取、之后哪个缓冲变得可复用。
