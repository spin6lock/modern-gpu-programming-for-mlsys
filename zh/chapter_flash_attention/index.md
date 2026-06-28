(chap_flash_attention)=
# Flash Attention 4

:::{admonition} Overview
:class: overview

- 注意力由两个 MMA 组成，softmax 夹在它们中间，因此不能像 GEMM 那样只重复一个 MMA。
- 这个内核把 Part I 的硬件原语(TMA、`tcgen05`、TMEM、barrier)与 Part III 的 GEMM 技术结合起来，并加入 warp 角色、在线 softmax 重缩放、因果掩码以及 GQA。
:::

注意力是决定一个 transformer 能否高效运行的核心内核，也是我们到目前为止构建的一切最终必须协同工作的地方。我们为 GEMM 组装的每一个部件在这里都用得上：TMA 分块搬运、`tcgen05` MMA、TMEM、warpgroup 寄存器分块，以及显式 barrier。

难点在于,注意力并不是一个 MMA 的简单重复。它是两个 MMA,中间夹着真正的工作:在线 softmax、因果掩码,以及让更早和更晚的分块保持同一尺度的重缩放。

那个中间阶段才是新难点的所在。一个普通的矩阵乘只往它的累加器里累加;而注意力必须在新 key 和 value 不断流入时,重新审视并重缩放它已经计算过的结果。softmax 工作本身也运行在两个 Tensor Core MMA 之间的 CUDA core 上,因此指数运算和按行规约直接落在关键路径上。

正因如此，注意力优化的很大一部分其实是 softmax 优化：改写 `exp`，并把 softmax 与 MMA 重叠起来，而不是让 MMA 在 softmax 上等待。

本章的目标不是从头重新推导 Flash Attention。我们只保留刚好够用的算法部分,让内核保持可读,然后把注意力放到真正新的部分:这个算法如何变成 TIRx。

最清晰的切入点是跟随单个分块,看它如何流过内核。`Q`、`K`、`V` 作为输入分块,从 GMEM 加载到 SMEM。score MMA 把 `Q` 和 `K` 相乘,在 TMEM 中得到 score 分块 `S`。softmax 把 `S` 转成分子分块 `P`,value MMA 把 `P` 和 `V` 组合起来更新输出累加器 `O`。

到目前为止,这看起来像是两个矩阵乘粘在一起,但有一个 GEMM 从不需要处理的转折:每当运行中的 softmax 最大值改变时,已经累加好的 `O` 突然尺度就不对了。在下一个 value MMA 能够安全地往里加之前,它必须被重缩放。下面的章节先追溯这条路径,然后才展示 TIRx 如何把每个阶段交给一个 warpgroup,并把各阶段连接起来。

## 算法形状

在把分块映射到具体存储之前,需要先明确这些分块所服务的算法。对于一个 query 块,Flash Attention 计算:

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

字面上读,这个公式说要先形成完整的 score 矩阵 `S = QKᵀ`,对它做 softmax,再乘以 `V`。这恰恰是我们不能用的一种做法,因为完整的 `S` 极其庞大。在 seq=4096 时,每个 head 大约有 16M 个元素,fp32 下约 64 MB,比 SMEM 或单个 128×512 TMEM 区域都要大好几个数量级。片上根本没有地方放得下它。Flash Attention 的答案是:根本不去物化 `S`。它以分块的形式流式处理 `K/V`,并维护三个逐行的运行状态来概括目前看到的一切:

- `row_max`:目前见过的最大 score。
- `row_sum`:softmax 的运行中分母。
- `O`:运行中的输出累加器。

流式更新就是在新分块到来时保持这些状态正确。微妙之处在于,每处理一个分块,运行中的最大值都可能上升,而一旦上升,我们在旧最大值下计算的一切现在尺度就错了。所以在加入新的贡献之前,我们先把旧状态拉回到新尺度:

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

这里单个 `scale` 因子身兼二职:它同时重缩放运行中分母和运行中输出,使得来自更早和更晚分块的贡献最终都落在同一尺度下被衡量。

上面的伪代码用自然的 `exp` 和显式的 `/sqrt(d)` 写成,因为那样最容易读,但内核走了一条更省的路线。它把 `1/sqrt(d)` 和 `log2(e)` 一起折叠进一个常量 `scale_log2 = log2(e)/sqrt(d)`,并用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)` 在原始 score 上用硬件 `exp2` 求每一个指数。动机很简单:在这块硬件上 `exp2` 比自然的 `exp` 快。

在继续之前有一点值得敲定:这里的 `P` 并*不是*最终归一化的注意力矩阵。它只是当前 K/V 分块的 softmax 分子。归一化被刻意推迟,只有在最后一个分块之后,内核才写入 `O / row_sum`。

对 TIRx 来说，知道算法计算什么只是一半。另一半是在内核运行时*每个分块位于何处*，因为这正是决定布局与 barrier 代码的因素。`S`、`P`、`O` 都是分块值，并且各自位于不同的存储区域：

- `S` 是 score 分块。score MMA 把它写到 TMEM。
- `P` 是 softmax 分子分块。softmax 把 `S` 从 TMEM 读进寄存器,计算 `P = exp((S - m_new) / sqrt(d))`,再把 `P` 写回 TMEM。
- `O` 是输出累加器分块。value MMA 从 TMEM 读 `P`、从 SMEM 读 `V`,然后累加进 TMEM 中的 `O`。

我们前面标记的重缩放也是一个分块操作,而不是一段标量的簿记:当 `row_max` 改变时,旧的 `O` 从 TMEM 读出,在寄存器中相乘,再写回 TMEM,然后下一个 value MMA 才往里累加。后面每一节都遵循同样的结构:一个分块放置,一条硬件路径,以及一道证明下一个消费者可以运行的 barrier。

## 分块原语图

有了运行状态和它们的归属,我们就能把算法铺开成一条具体的分块搬运序列。对于一个 K/V 块,内核自上而下走这条分块路径:

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

与 GEMM 的区别归结为一行。GEMM 是一条 MMA 链的重复;FA4 有两个 MMA 阶段,softmax 坐在链的中间。接下来几乎所有的不同,都是这多出来的一个阶段的后果。

如果我们把这条短路径展开成显式的生产者-消费者边,就得到完整的图:

| 阶段 | 分块搬运或计算 | TIRx 原语 | 硬件路径 |
|-------|--------------------------|----------------|---------------|
| 加载 Q/K/V | GMEM 分块 -> SMEM 分块 | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | SMEM 中的 Q 和 SMEM 中的 K -> TMEM 中的 score 分块 `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| softmax 读 | TMEM 中的 `S` -> warpgroup 寄存器分块 | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| softmax 写 | 寄存器中的分子分块 `P` -> fp16 TMEM 视图 | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store, followed by `tcgen05.wait.st()` |
| Value MMA | TMEM 中的 `P` 和 SMEM 中的 V -> TMEM 中的输出累加器 `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` with a TMEM operand |
| 修正 | TMEM 中的 `O` -> 寄存器 -> TMEM 中的 `O` | TMEM 回读、寄存器相乘、TMEM 存储 | `tcgen05.ld` / TMEM store |
| 收尾 | TMEM 中的最终 `O` -> 寄存器 -> SMEM -> GMEM | TMEM 回读、`Tx.copy`、TMA store | `tcgen05.ld` + TMA store |

新加的行是 softmax 和修正。两者都增加了 TMEM -> 寄存器 -> TMEM 的流量,也都在 score MMA 和 value MMA 之间制造了额外的交接。

**与你的 agent 一起试试**:让它只追溯上面这条短路径。对每个箭头,说出生产者阶段、消费者阶段、源分块、目的分块和硬件路径。然后问哪些箭头在 GEMM 章节里并不存在。

## Warp 角色与作用域

数据路径定下来之后,自然要问的是每个阶段到底由谁来跑。这里每个 CTA 有 4 个 warpgroup,共 512 个线程,它们的划分不是按各自碰到的数据,而是按一个 warpgroup 做*哪类工作*:

- WG3 驱动硬件引擎:TMA load、MMA、TMA store。
- WG0、WG1、WG2 做那些夹在引擎调用之间的、寄存器密集型数学:softmax、修正、收尾。

精确的角色表是:

| 拥有者 | 角色 | 做什么 |
|-------|------|--------------|
| WG3, warp 1 | TMA load | 把 Q、K、V 分块从 GMEM 加载到 SMEM |
| WG3, warp 0 | MMA | 既发 score MMA 又发 value MMA |
| WG3, warp 2 | TMA store | 把最终 O 分块从 SMEM 存到 GMEM |
| WG0 | Q 阶段 0 的 softmax | 从 TMEM 读 S,计算 P,把 P 写到 TMEM |
| WG1 | Q 阶段 1 的 softmax | 为第二个 Q 流水级做同样的工作 |
| WG2 | 修正与收尾 | 重缩放 TMEM 中的 O,归一化,暂存输出 |

很容易把"两个 Q 阶段"误读成两个注意力 head,但并非如此。它们只是 Q 流水线里的两个槽,WG0 拥有一个,WG1 拥有另一个,这样两个 Q 分块就能同时在天上飞。这正是 softmax 工作出现两次的原因,一次在 WG0,一次在 WG1。

代码用符号坐标把这些角色挑出来:

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

读这个内核时,先找到角色分支。它会告诉你哪个团队拥有嵌在它里面的每一个分块原语。

- WG3 warp 1 启动 TMA load 命令。一个被选中的 lane 发起拷贝,TMA 引擎搬运分块。
- WG3 warp 0 发出 `tcgen05.mma` 指令。
- WG0 和 WG1 在完整的 warpgroup 作用域下跑 softmax。
- WG2 在完整的 warpgroup 作用域下跑修正与收尾工作。

有一种不对称最终塑造了整个 barrier 图:*每一个* MMA,无论是 score 还是 value,都只由 WG3 warp 0 单独发出。WG0 和 WG1 根本不发任何 MMA。它们只消费 score 分块、跑 softmax,再把 `P` 写回 TMEM。

这种分离恰恰是 softmax 需要被 barrier 围起来的原因。`s_ready` 把 score 分块从 MMA warp 送到 softmax;`p_o_rescale` 携带 `P` 和一个对 value MMA 安全的 `O` 槽——要么已经重缩放,要么因为不需要重缩放而被释放。本章剩下部分会反复回到这两个名字。

## 阅读这些片段

本章的片段摘自 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py),因此不可避免地会引用我们在内核里没有复现的部分中所定义的名字。那些自描述的名字(`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、各种 `SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` 深度、`should_accumulate`,以及 `CTA_GROUP`(这里为 1))我们会在它们首次重要的地方介绍。其余的在下面这张表里给出一句注释,这样当一个片段在你面前摆出一个陌生的名字时,你有地方可以查:

| 名字 | 含义 |
|------|---------|
| `q_stage`, `i_q` | Q 流水级,0 或 1,即哪个 Q 分块槽(`SMEM_PIPE_DEPTH_Q = 2`)。在 WG0/WG1 的 softmax 内部,该 warpgroup 自己的 `wg_id`(0 或 1)就是这个同样的阶段索引,所以 `S_region[q_stage]`、`P_region[wg_id]`、`O_region[i_q]` 都选中同一个 Q 阶段 |
| `MMA_N` | TMEM 列数上的 score/输出分块宽度(128) |
| `MMA_K` | `P`/`V` 列上的 MMA 内层 K 步长(16);`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | value-MMA 调度的切分点(见*两个 MMA 阶段*);第一个 value MMA 覆盖列 `0:K_SPLIT`(`6 * MMA_K = 96`) |
| `should_rescale` | WG2 逐行标志:旧的 `O` 是否需要在下一个 value MMA 之前重缩放(用 `any_sync` 在 warpgroup 内规约) |
| `rescale_threshold` | 针对微小 row-max 变化的跳过阈值;当前内核用 `8.0`,被跳过的重缩放会把 `acc_scale` 置为恰好 `1.0` |
| `scale_log2` | 以 log2 为单位的 softmax 尺度,`log2(e)/√d`,即 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | 逐行重缩放因子,softmax 通过 SMEM mailbox 传给 WG2 |
| `chunk_start`/`chunk_end`, `p_start`/`p_end` | 正在被读 / 写的 32 宽 softmax chunk 的列范围 |

## 两个 MMA 阶段

对于每个流入的 K/V 分块,Flash Attention 跑两个 MMA 阶段,softmax 把它们桥接起来:

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

可以把它理解为一条由三个连续生产者组成的流水线。第一个 MMA 产出注意力分数 `S`，softmax 把 `S` 转成分子 `P`，第二个 MMA 消费 `P` 来更新输出累加器 `O`。除以 `row_sum` 的归一化被推迟到收尾阶段，等所有 K/V 分块的贡献都累加完之后才执行。

下面每个分块操作都沿用 GEMM 章节中的同一张 **作用域 / 布局 / 派发** 解读卡，只是多加一行 **交接**，指出把分块传给下一个角色的 barrier。

计算代码从不说原始 TMEM 列号。内核把自己单一的 TMEM 分配切成按阶段的视图(`S_region`、`P_region`、`O_region`),并按流水级索引(`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`)。这些视图在 [TMEM 布局与复用](#tmem-layout-and-reuse) 一节里用 `T.TMEMStages` 定义;就现在而言,把每个区域当作同一块物理 TMEM 的一个具名切片就够了。

### Score MMA

两个阶段里的第一个是 score MMA,即每个 K/V 迭代开头那个矩阵乘。它计算:

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并把 `128 x 128` 的 score 分块写到 TMEM:

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以继续沿用 GEMM 章节中的四个问题：谁来运行它、分块位于何处、如何派发、如何交接：

> **分块原语解读:Score MMA**
> - 作用域:WG3 warp 0 发出它;一个被选中的 lane 到达 `s_ready`。
> - 布局:SMEM 中的 Q、K → TMEM 中的 `S`(`S_region[q_stage]`)。
> - 派发:`tcgen05`。
> - 交接:`s_ready`(→ softmax)。

单个被选中的线程在 `s_ready` 上到达,就是全部的交接。它宣布这块 score 分块已经完成,softmax warpgroup 现在可以自由地读它了。

### 两个 MMA 之间的 softmax

softmax 位于两个 MMA 之间，负责把 score 分块 `S` 转成分子分块 `P`。它的解读卡是：

> **分块原语解读:softmax**
> - 作用域:WG0(Q 阶段 0)/ WG1(Q 阶段 1),完整 warpgroup。
> - 布局:TMEM 中的 `S` → 寄存器 → fp16 TMEM 中的 `P`(`P_region[wg_id]`)。
> - 派发:用 `tcgen05.ld` 读,用 TMEM store 写;两者之间是在寄存器里做按行的数学。
> - 交接:等 `s_ready`;到达 `p_o_rescale`(前 96 列)和 `p_ready_2`(最后 32 列)。

这个阶段是完全没有 GEMM 对应的那个阶段。WG0/WG1 等 score 分块在 `s_ready` 上到达,然后一次按寄存器大小的 chunk 从 TMEM 把它读出来:

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

这是一次在 warpgroup 作用域下的 TMEM-到-寄存器分块读。既然分数已经坐在寄存器里,softmax warpgroup 依次做三件事:

1. 算出行最大值和行求和,
2. 算出 softmax 分子分块 `P`,
3. 把 `P` 以 fp16 写回 TMEM。

最后一步长这样:

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

为什么刚在寄存器里算完 `P` 还要把它写回 TMEM?因为 value MMA 需要 `P` 作为*分块操作数*，而 MMA 不能把散落在每个线程标量寄存器里的值直接当成矩阵读取。在本内核里，`P` 的 MMA 可读形式是 `P_region`，也就是 fp16 TMEM 别名 `tmem_as_f16` 上的一个视图。所以这次回写不是多余动作；它正是把 `P` 转换成下一个 MMA 能消费的表示。

### Value MMA

第二个阶段,也是结束每个 K/V 迭代的那个,是 value MMA。它计算:

$$O = O + P_{\text{block}}V_{\text{block}}$$

当这个 MMA 开始运行时，`O` 已经处于当前 K/V 块所需的正确状态：第一个块上已初始化，后续块上已重缩放。因此 MMA 要做的只是累加。它与 GEMM 的区别在于操作数所在的位置：A 操作数是 TMEM 中的 `P`，B 操作数是 SMEM 中的 V，累加器 `O` 也在 TMEM 中：

```python
# 第一个子 MMA:列 0:K_SPLIT(P 的前 96 列 / V 的前 96 行)。
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# 第二个子 MMA(形式相同,accum=True,由 p_ready_2 门控)覆盖剩下的
# 列 K_SPLIT:BLK_N。
```

> **分块原语解读:Value MMA**
> - 作用域:WG3 warp 0。
> - 布局:TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O`(`O_region[i_q]`)。
> - 派发:带 TMEM 操作数的 `tcgen05`。
> - 交接:等 `p_o_rescale`、`p_ready_2`、`kv_load.full`;到达 `o_ready`(→ 收尾)。

这种操作数摆放正是两个 MMA 之间的硬件差异:

- Score MMA 从 SMEM 读两个操作数:Q 和 K。
- Value MMA 从 TMEM 读一个操作数 `P`。
- Value MMA 从 SMEM 读另一个操作数 V。
- 结果累加进 TMEM 中的 `O`。

`accum=should_accumulate` 这个标志实现了算法里"初始化或累加"的选择:在一个 query 块的第一个 K/V 分块上为假,其后每个分块上为真。

你可能也注意到,value MMA 并不是一次性跑完,而是拆成了 `96 + 32` 的调度:

1. softmax 以四个 32 列的 chunk 写出 `P`。
2. 前三个 chunk 一旦就绪,value MMA 就开始在 `P` 的前 96 列以及 V 对应行上跑。
3. 最后 32 列等 `p_ready_2`。
4. 第二个 MMA 消费最后那个 chunk 并完成整块分块。

之所以拆分，是为了让 Tensor Core 保持忙碌。如果把 value MMA 当成一次完整操作运行，整个阶段会一直停滞，直到全部四个 32 列 `P` chunk 都做完指数并写好。通过在前三个 chunk 就绪后立即启动，内核把最后一个 chunk 的 `exp` 和 TMEM 写入，与已经发出的 96 宽 MMA 重叠起来，把本来空闲的时间转化为有效工作。

(tmem-layout-and-reuse)=
## TMEM 布局与复用

`S`、`P`、`O` 全部要共享同一块 `128 x 512` 的 TMEM 分配，而它们的打包方式恰恰解释了为什么在这个内核里 barrier 与布局不可分割：

下面的图直接展示了这种打包方式：score 槽、分子槽、输出槽全部共享同一块 TMEM 分配，因此正是 barrier 协议让这种复用合法。

![TMEM 布局](../img/zh/tmem_layout_v3.png)

这张图可以读成一组分块槽:

- score 槽放 `S = QK^T`。
- 分子槽放 softmax 指数步骤之后的 `P` 分块。
- 输出槽放 fp32 的 `O` 累加器。

这些并不是独立的缓冲区。它们是*同一块*分配的不同区域,这种共享不是风格上的选择,而是被迫的。在 Q 流水深度为 2 时,两个 `S` 槽(2 × MMA_N = 256 列)和两个 `O` 槽(2 × MMA_N = 256 列)已经占满全部 512 个 fp32 列。再没有剩下给 `P` 的空间,所以 `P` 别无选择,只能通过更窄的 fp16 视图去别名同一些字节。这之所以安全,唯一的原因是每个区域都严格在它之前的消费者用完之后才被复用,而那个时序正是 barrier 保证的。所以在 FA4 里,barrier 不仅仅是调度;它们才让布局从一开始就合法。

这种别名技巧通过 `T.TMEMPool` 建立。内核先取一个 fp32 视图(`tmem`)给 score 与输出累加器使用，然后把 pool 的基地址回退到 0，在*同一批*物理字节上取第二个 fp16 视图(`tmem_as_f16`)：

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

因为 fp16 元素宽度只有一半,fp16 视图在同样这些字节上暴露出两倍多的可索引列,而这正是 `P` 住进的空间,是 fp32 布局里腾不出来的空间。两个视图在手之后,内核用 `T.TMEMStages` 把 `S`、`P`、`O` 槽切成按阶段划分的区域,让计算代码可以按流水级索引,而不是按原始列号:

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

`P_region` 步幅里的 `* 2`，是别名关系显式出现在代码中的唯一一处。`S_region` 和 `O_region` 以 fp32 `tmem` 列来度量，而 `P_region` 以 fp16 `tmem_as_f16` 列来度量，后者宽度只有一半，所以阶段到阶段的移动需要双倍步幅才能落在同一批物理字节上。不过，一旦区域定义好，计算代码就可以保持简洁：它写 `S_region[q_stage]`、读 `S_region[wg_id, ...]`、写 `P_region[wg_id, ...]`、累加进 `O_region[i_q]`，从来不碰任何原始列索引。

**与你的 agent 一起试试**:让它解释这个 FA4 内核里的 fp32(`tmem`)和 fp16(`tmem_as_f16`)视图。哪些物理 TMEM 区域放 `S`、`P`、`O`,而 `P_region` 的步幅为什么用 `MMA_N * 2`?把复用的问题留到下一节:在 barrier 表之后,检查每个区域被复用之前必须有哪些消费者用完。

## barrier 如何把角色连起来

这是内核最难的部分,所以循序渐进地切入有好处。先从那少数几个沿主计算路径搬运数据的 barrier 入手,其余的都当作可以之后再查的簿记。那些"数据就绪"的交接是:

| 交接 | 含义 |
|---------|---------|
| TMA load -> score/value MMA | Q、K 或 V 已经到达 SMEM,可以供 MMA 使用 |
| score MMA -> softmax | TMEM 里的 `S` 已就绪 |
| softmax/修正 -> value MMA | TMEM 里的 `P` 已就绪,`O` 可以安全累加 |
| value MMA -> 收尾 | TMEM 里的最终 `O` 已就绪 |
| 收尾 -> TMA store | `O_smem` 已就绪可以存储 |

不在这张表里的全是流水线簿记:那些释放某个 SMEM、TMEM 或暂存缓冲区、好让另一个角色复用的 barrier。有用的一点是,每一道 barrier,无论它携带的是数据还是仅是簿记,读起来都一样,即一次分块交接。你问谁生产了数据、谁消费它、它们都做完之后哪个缓冲区变空。

下一张图把这些交接坍缩成两个 MMA 阶段的精确就绪门:score MMA 等什么,value MMA 在它能够累加之前必须等什么。

![Flash Attention 4 MMA 输入门控](../img/zh/flash_attention_main_handoff.png)

把这张图当作一组正确性门来读，而不是一份调度。它回答“这道 MMA 可以启动之前什么必须成立”，而不描述具体时序。score MMA 等 SMEM 里的 Q 和 K，然后产出 `S`。value MMA 同时等待三样东西：SMEM 里的 V、来自 softmax 的 `P` 分块，以及一个 WG2 已经释放或重缩放过的 `O` 槽。softmax 到 value 的那道门被拆成两段，原因我们已经见过：value MMA 在 `P` 的前 96 列就位后就可以开始，`p_ready_2` 再放出最后 32 列。

有一处交接不符合"分块就绪"的模式:softmax 到修正的那条边。它不传分块,而是通过一个单槽 SMEM mailbox 把单个标量(K/V 循环里传 `acc_scale`,收尾时传最终的 `row_sum`)传给 WG2。由于这个槽每次迭代都要复用,必须有一对 `full`/`empty` barrier 来守护它:

下面的图放大了那次 mailbox 握手,这也正是为什么这对 barrier 应该被读成一条标量生产者-消费者通道,而不是一道分块就绪门。

![Flash Attention 4 softmax 尺度槽握手](../img/zh/flash_attention_softmax_correction.png)

把 `softmax_corr.full` 和 `softmax_corr.empty` 当作一对生产者-消费者:

1. softmax 在复用 scale/sum 槽之前先等 `softmax_corr.empty`。
2. softmax 把 `acc_scale` 或最终的 `row_sum` 写进那个槽。
3. softmax 在 `softmax_corr.full` 上到达。
4. WG2 等 `softmax_corr.full`,然后读那个槽。
5. WG2 在 `softmax_corr.empty` 上到达。
6. softmax warpgroup 可以在下一阶段复用这个槽。

需要小心区分 `softmax_corr.empty` 表示什么、不表示什么。它只表示 WG2 已经消费了 scale/sum 槽；它不说明 `P` 是否就绪，而且*绝对不是*让 value MMA 开始的那道门。那道门是 `p_o_rescale`，它在 `P` 的前 96 列写好、`O` 槽可以安全累加时到达。把两者搞混，是结果错误的典型来源。

主路径在手之后,完整的 barrier 列表可以作为参考:

| Barrier | 生产者 -> 消费者 | 什么变得安全 |
|---------|----------------------|-------------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM 分块可以供 MMA 使用 |
| `q_load.empty` | 这个 Q 阶段的所有 score MMA -> TMA load | Q SMEM 阶段可以复用给下一个任务 |
| `kv_load.full` | TMA load -> score/value MMA | K 或 V SMEM 分块可以供 MMA 使用 |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM 阶段可以复用 |
| `s_ready` | score MMA -> softmax | S TMEM 分块可以读 |
| `p_o_rescale` | softmax + WG2 -> value MMA | P 的前 96 列在 TMEM 里,O 槽对 value MMA 安全 |
| `p_ready_2` | softmax -> value MMA | P 的最后四分之一在 TMEM 里 |
| `o_ready` | value MMA -> 收尾 | 最终 O 累加器就绪 |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` 或最终 `row_sum` 在 SMEM mailbox 里就绪 |
| `softmax_corr.empty` | WG2 -> softmax | 同一个 SMEM mailbox 槽在 WG2 读完之后可以复用 |
| `corr_epi.full` | 收尾 -> TMA store | O_smem 可以存储 |
| `corr_epi.empty` | TMA store -> 收尾 | O_smem 阶段可以复用 |

就和 GEMM 里一样,你可以从谁产生信号来预测 barrier 的类型:

- TMA load 用 `TMABar`,因为 TMA 引擎自己按字节数计数自己的完成。
- MMA 完成用 `TCGen05Bar`,因为 `tcgen05.commit` 给完成组发信号。
- 纯线程到线程的交接用 `MBarrier`,参与的线程显式到达。

拆开的 softmax 到 value 交接值得仔细看。它用了两道门:

- `p_o_rescale` 让 value MMA 在 `P` 的前 96 列写好、`O` 分块可以安全累加时就开始。
- `p_ready_2` 放出 `P` 的最后 32 列,对应上一节那个 `96 + 32` 的 value-MMA 调度。

第一个 K/V 块是简单情形。WG2 预先在 `p_o_rescale` 上到达,因为还没有旧 `O` 分块要重缩放。

后面的块得小心些。WG2 要么跳过一次不必要的重缩放、要么把旧 `O` 重缩放完之后,才在 `p_o_rescale` 上到达。跳过测试是刻意保守的:softmax 计算以 log2 缩放的差值 `(m_old - m_new) * scale_log2`;如果它仍高于 `-rescale_threshold`,说明新最大值移动得还不够,不足以证明重缩放划算,于是内核保留旧最大值并把 `acc_scale` 置为恰好 1.0。只有更大的最大值跳跃才走 `exp2` 路径,并请求 WG2 重缩放 `O`。

然后 WG2 用 `any_sync` 在 warpgroup 内规约 `should_rescale`。如果没有一行需要这次更新，它就跳过对 `O` 的处理。这次跳过很重要，因为重缩放 `O` 是一次覆盖整个累加器的完整 TMEM -> RF -> TMEM 读-改-写；当阈值逻辑已经把 `acc_scale` 保持在 1.0 时，这一步纯属浪费。

注意所有新的 barrier 都聚集在一处。`s_ready`、`p_o_rescale`、`p_ready_2`,以及 softmax/修正那一对,全是 softmax 周围的 barrier。它们存在出于一个原因:score MMA 和 value MMA 不再相邻。寄存器数学、TMEM 重写、输出重缩放现在夹在它们之间,而其中每一步都需要自己的交接。

**与你的 agent 一起试试**:让它把一个 K/V 块追过 `s_ready`、`p_o_rescale`、`p_ready_2`、`o_ready`。对每道 barrier,问谁在等、谁到达、哪个分块变得可以读、之后哪个存储可以复用。

## 流水化结构

barrier 告诉我们在一个角色消费某个分块之前什么必须*就绪*。它们没告诉我们的是到底什么在*并发*地跑,而那正是我们现在转向的问题。这两者确实不同:一道正确性门可以在生产者碰巧运行之前很久、或之后很久被满足。

这里没有单一的流水深度,因为不同的分块流以不同的速率移动。于是内核为每条流各维护一个独立的环:

- Q 流水深度 2:一个 CTA 同时处理两个 Q 阶段。WG0 处理一个阶段,WG1 处理另一个。
- KV 流水深度 3:K 和 V 块在内层循环里流动,同时同一批 Q 阶段被复用。
- TMEM 流水深度 2:每个 Q 阶段有自己的 S/P/O TMEM 槽,这些槽在对应的 barrier 到达后被复用。

下面的图从正确性门切换到时间线视角,展示在这些独立环都飞起来之后,哪些角色大约能同时活动。

![Flash Attention 4 流水结构](../img/zh/flash_attention_pipeline_v2.png)

把这张图当作时间线,而不是 barrier 图。它展示哪些角色大约在同一时刻活动,而更早那张 barrier 流图才是你去查精确生产者-消费者等待的地方。这两张图合起来,回答了我们在本节开头提出的两个不同问题。

每一行对应代码中的一个角色分支:

- WG3 warp 1 发出 TMA load。
- WG3 warp 0 既发 score MMA 又发 value MMA。
- WG0 和 WG1 为两个 Q 阶段跑 softmax。
- WG2 释放或重缩放 `O`,之后再把最终输出归一化。
- WG3 warp 2 发出 TMA store。

从左到右顺着这张图,就追过一道有代表性的流水波。load warp 以 `Q0`、`K[n-1]`、`Q1`、`V[n-1]` 开始,然后持续流入更低索引的 K/V 块。MMA warp 发出头几个 score MMA 来产出 `S0` 和 `S1`,WG0/WG1 把它们转成 `P0` 和 `P1`。

重要的是,MMA warp 并*不是*先跑完所有 score MMA 再跑所有 value MMA。两个 Q 阶段都就绪之后,它把两类交替起来:为当前 `V` 块发一个 value MMA,再为下一个 `K` 块发一个 score MMA,以此类推:

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

正是这种交替,使得图里 score、softmax、修正、value 这几行是重叠的,而不是齐整地依次运行。

WG2 那一行标记为 `release / rescale`,两半对应我们见过的两种情形。在第一个 K/V 块上还没有旧 `O`,所以 WG2 只参与那道让 value MMA 继续的交接;在后续块上它可能在 value MMA 累加进旧 `O` 之前先重缩放它。归一化和 TMA store 恰好各做一次,在注意力任务的最后一个 K/V 块之后。

没有哪条单一的 GEMM 式流水能描述 FA4,因为 Q、K/V 和 TMEM 槽都按各自的进度推进。TIRx 把这些进度显式保留,作为独立的分块缓冲区、`PipelineState` 游标和 barrier 相位,而不是把内核藏在一个大一统的原语背后。代价是更多活动部件,但好处是复杂度保持可见、可检视。

## 重缩放与回写

重缩放是强制的,不是我们可以丢掉的优化。在线 softmax 会随着每个新 score 分块抬高逐行最大值,而每当它抬高时,从更早分块累加来的 `O` 是用*旧*最大值标度的。这让每个更早的项都偏大了一个 `exp(m_new - m_old)` 的因子。跳过修正,那些块就会被过度加权,最终输出就干脆是错的。修复是一次 TMEM → 寄存器 → TMEM 的分块操作:

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

工作被拆给两个角色。softmax 计算逐行尺度并把它写入 SMEM mailbox；WG2 等 `softmax_corr.full`，从 TMEM 把当前 `O` 读出来，乘上这个尺度，再把 `O` 写回：

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调的是，这是一次覆盖整个 `O` 累加器的完整 TMEM → 寄存器 → TMEM 分块操作，而不是一点标量簿记，它和其他阶段一样也有自己的解读卡：

> **分块原语解读:修正(重缩放)**
> - 作用域:WG2,完整 warpgroup。
> - 布局:TMEM 中的 `O` → 寄存器 → TMEM 中的 `O`(`O_region[i_q]`)。
> - 派发:用 `tcgen05.ld` 读,用 TMEM store 写;两者之间是寄存器相乘。
> - 交接:等 `softmax_corr.full`;到达 `p_o_rescale`(→ value MMA)和 `softmax_corr.empty`(→ softmax)。

从端到端追一遍同步:

1. softmax 把尺度值写到 SMEM。
2. WG2 等 `softmax_corr.full`。
3. WG2 重缩放 TMEM 里的 `O`。
4. WG2 在 `p_o_rescale` 上到达。
5. WG3 的 value MMA 现在可以消费 `P` 并累加进重缩放过的 `O` 分块。

当 WG2 读完后 `softmax_corr.empty` 释放那个 SMEM 槽,循环就合上,这让 softmax 可以在下一次迭代复用 mailbox。

K/V 循环一结束,WG2 就从修正切换到收尾。它等最终的 `row_sum` 和 `o_ready`,从 TMEM 读出最终 `O`,乘上 `1 / row_sum`(也就是我们在一开始推迟的归一化),转成 fp16,写出 `O_smem`。WG3 的 TMA store warp 再把 `O_smem` 搬回 GMEM。

对任何打算扩展这个内核的人来说，有一个局限值得标出。它只计算前向输出，而训练前向通常还会保存反向所需的 log-sum-exp(LSE)。加入 LSE 时要留意一个尺度细节：这个内核把 `row_max` 保持为*原始*、未缩放 `QK^T` 分数的最大值，而 `row_sum` 累加的是 `exp((S - row_max) / sqrt(d))`。所以形成自然对数 LSE 时，`1/\sqrt{d}` 因子必须重新施加到 `row_max` 上：

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

这个实现只产出前向输出,不写 LSE。

## 因果掩码

因果注意力加了一个约束(一个 query 只能注意到自己位置或之前的 key),内核用两种互补的方式来满足它,一种便宜,一种精确。

便宜的方式是干脆跳过工作。很多 K/V 块完全坐在对角线之上,对一个给定 Q 块毫无贡献,所以 `get_n_block_max(...)` 计算出这个块可能需要的最后一个块,循环干脆不再加载或计算剩下的那些。

精确的方式处理那些跨在对角线上的块,即有些列有效、有些列无效。那些块照样跑 score MMA,但 softmax 在做指数之前把无效列掩掉。对每一行,它从该行的 query 位置和块偏移推导出一个列上限,保留下限及以下各列,把超过上限的每一列在寄存器里置为 `-inf`,使这些列既不对行最大值、也不对 `exp2` 分子有任何贡献。

实现并不逐元素分支,而是用 `mask_r2p(...)` 应用这个上限,它把上限变成覆盖整个 32 宽 score chunk 的位掩码,一次性把 chunk 掩掉。完全在对角线之下的块保留所有列,完全不需要掩码。

从分块原语的视角看,因果模式根本不改写数据路径。它只是修剪 K/V 的循环计数,并在 score MMA 和 `P` 回写之间,往驻留在寄存器里的 softmax 插入一步掩码。

## GQA 支持

分组查询注意力(GQA)让若干个 query head 共享单个 K/V head。这能节省显存带宽，却带来一个打包问题：如何只保留一个 K/V 分块，同时仍让多个 query head 使用它?内核的答案是：针对一个被调度的 `kv_head_idx`，一次处理一整个 query head 组：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

诀窍是重新解读那 128 行 Q 分块。对 `GQA_RATIO=4`,它们不再表示 128 个序列位置;它们表示 32 个序列位置乘以 4 个 query head,打包在一起,使全部四个 head 搭同一个 K/V 分块。行的解码是:

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q load 用一个 3D 视图表达这种打包。源端是自然的 `Q[batch, seq, qo_head, dim]` 布局,目的端是同一个 score MMA 之后会当作平坦 `128 x HEAD_DIM` 操作数来读的 SMEM 分块。这个视图正是调和两者的东西,而且它不涉及任何拷贝:

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K 和 V 从不在内存中展开，这正是 GQA 的全部意义：`kv_head_idx` 对应的单个 K/V 分块，会被打包进 Q 行的全部 `GQA_RATIO` 个 query head 复用。输出端与输入端对称，收尾之后用一个匹配的 3D 视图把打包好的行存回 `O[batch, seq, qo_head, dim]`。

后果是 GQA 完全活在 Q-load 和 O-store 的边界上。计算路径内部,score MMA 看到的仍是普通的 `128 x HEAD_DIM` Q 分块,分块原语图的其余部分原封不动。

## 分块调度

调度器的工作是把每个 CTA 映射到一个 `(batch, kv_head, m_block)` 注意力任务,而正确策略取决于掩码是否让这些任务代价相等:

- 非因果模式用 `FlashAttentionLinearScheduler`。每个任务做同样多的工作,所以一个以 `num_ctas` 推进的固定 CTA 池就足以把它们均匀摊开。
- 因果模式用 `FlashAttentionLPTScheduler`,因为因果掩码让工作量极不均匀:靠近开头的 Q 块大约只注意一个 K/V 块,靠近末尾的却注意全部。朴素的切分会让某些 CTA 远远晚于其他 CTA 完成,所以最长处理时间优先(longest-processing-time)的调度器把重块前移以抹平完成时间,同时仍把相邻的 batch/head 任务放在一起以保 L2 局部性。

尽管差异巨大,两个调度器暴露出完全一致的循环接口:

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # 处理一个 Q 块,对应它的 K/V 块范围
    scheduler.next_tile()
```

唯一的行为差异在于 `next_tile()` 做什么：非因果模式下它把 CTA 推进到另一个任务，因果模式下它在当前任务之后结束循环。无论哪种，这都纯粹是调度决策：它选择 CTA 拥有*哪个*注意力分块，而不改变该分块如何计算。循环内部运行的是同样的本地原语：TMA load、score MMA、softmax、value MMA、修正、TMA store。

## 编译与验证

上面这些都是摘录。要把它们组合起来真正运行内核，我们从 `tirx-kernels` 导入完整实现，编译它，再对照 torch 参考实现检查。完整内核位于 `tirx-kernels` 仓库中的 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，本章讨论过的每个部分都在这个文件中。与 GEMM 验证 cell 有两处不同：Flash Attention 有更丰富的入口(`get_flash_attention4_kernel`)，并且它多接一个 `profiler_buf` 参数供其内置 profiler 使用。整章只需要运行这一个 cell：

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA:32 个 query head 共享 8 个 KV head
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod 直接接收 torch 张量,和每一章一样
torch.cuda.synchronize()

# torch 参考实现;enable_gqa 让 32 个 query head 共享 8 个 KV head
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**期望输出**:`... -> PASS`。内核以 fp32 累加在线 softmax，但它的结果与高精度参考之间仍有几层近似差异，包括：输入与操作数的 fp16 存储与舍入；基于 `exp2` 的 softmax 改写，即用 `scale_log2 = log2(e)/√d` 重新表达每个指数；在线 softmax 的重排与逐行重缩放，即在运行尺度下逐步求和，而不是一次性求和；最后还有写回时 `O` 的 fp16 转换。这里选用的 `rtol`/`atol` 与源内核自家测试用的容差一致，是相对 torch 参考“同时覆盖以上全部”而设定的，不是只覆盖 fp16 舍入。因此，如果这里出现明确失败，而不是接近阈值的边界情况，就应把它当作 softmax 路径的排查线索：可能漏掉了 `s_ready` / `p_o_rescale` / `p_ready_2` 等待，或者某次 `row_max` / `row_sum` 更新没有配套施加重缩放。这些正是本章花大量篇幅解释 barrier 交接的原因。

## 与 GEMM 的差异

下表沿那些发生变化的轴,把 FA4 与 GEMM 做对比:

| 维度 | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA 阶段 | 一个 MMA 的重复 | score MMA 和 value MMA |
| MMA 之间的工作 | 除了流水交接之外没有 | 在线 softmax、掩码,以及 O 重缩放 |
| 运行状态 | 仅累加器 | 行最大值、行求和、O 累加器 |
| 主要中间量 | 累加器 TMEM 分块 | S、P、O 的 TMEM 分块区域 |
| warp 角色 | TMA 生产者、MMA 消费者、回写 | TMA load、MMA、softmax、修正、TMA store |
| barrier | 主要是 load/计算/回写交接 | 额外的 score/softmax/value/修正交接 |
| 调度单位 | 输出矩阵分块 | 注意力任务:`(batch, kv_head, m_block)` |

这些差异的每一条都能追溯到我们开篇时说的那个结构性变化:第二个 MMA,softmax 楔在两者之间。而底层的 TIRx 契约则完全没有变:

- 分块原语说哪个分块被搬运或计算,
- 外围作用域说哪些线程协作,
- 布局说明分块位于何处,
- barrier 说下一个角色何时可以消费它。

所以 FA4 比 GEMM 难,不是因为它依赖不同的硬件,而是因为它分块值更多、它们之间的交接更多。

## 练习

1. 与 GEMM 相比,FA4 的两个 MMA 阶段之间出现了什么新的分块交接?说出生产者、TMEM 分块和消费者。
2. 为什么 softmax 把分子分块 `P` 写回 TMEM,而不是为 value MMA 把它只留在寄存器里?
3. 选 `p_o_rescale` 或 `p_ready_2`。这道 barrier 究竟证明了什么?如果 value MMA 跳过那次等待,会出什么问题?

**与你的 agent 一起试试**:挑一个没有注释的分块原语,比如收尾里的一个 `Tx.copy_async`、fp32 -> fp16 的 `Tx.cast`,或第二个 `gemm_pv` 子 MMA。让它给出作用域 / 布局 / 派发 / 交接卡,然后用源码里的守卫、分配和等待来核对答案。
