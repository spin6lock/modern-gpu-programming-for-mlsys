(chap_tirx_layout_api)=
# TIRx 布局 API

:::{admonition} Overview
:class: overview

- TIRx 布局 API 把 {ref}`chap_data_layout` 中的布局记法转换为编译器对象。主要对象有 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述在命名硬件轴上的仿射放置。它由分片规格 `S[...]`、副本规格 `R[...]` 以及可选偏移构成。
- 一个布局将一个逻辑坐标映射到一个或多个物理坐标。`layout.apply()` 对该映射求值。
- `SwizzleLayout` 描述基于 XOR 的共享内存 swizzle,用于避免 bank 冲突。`ComposeLayout` 在 tile 布局之上叠加一层 swizzle。
- 诸如 `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 这类现成构造器,覆盖了内核中反复出现的硬件布局。
:::

{ref}`chap_data_layout` 引入了贯穿全书的记法:一个 tile 形状、一组在命名轴上的步幅(stride),以及一个用于被复制而非被划分的值的可选复制项。本章把这种记法转换为编译器使用的 API。

目标是让页面上的记法与内核中的代码看起来几乎一致。当你写出一个诸如以下的布局时:

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你写的不仅仅是一段说明,而是在构造一个可附加到 buffer 的 `TileLayout` 对象。此后,每一个触及该 buffer 的 tile 操作都能从布局中读取自己的放置位置。放置只写一次,只检查一次,然后由编译器复用。

布局既可以在从池中分配时附加,也可以在声明 buffer 时附加:

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从此刻起,该 buffer 就携带了自己的物理放置。各 tile 操作无需重复说明每个元素位于何处。

这些布局对象都位于同一个模块中:

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

该 API 背后有一个核心思想:布局不必把一个逻辑索引映射到单一物理地址,而是把一个逻辑索引映射到一组位于命名轴上的物理坐标。在通常情况下这组坐标只有一个元素;当存在复制时,同一个逻辑元素会有多个物理放置。

这正是布局模型由三部分组成的原因:分片(shard)、副本(replica)和偏移(offset)。分片负责放置元素;副本把它复制到额外的坐标;偏移则平移整个放置。

## 通过示例看布局

下面的示例展示了 API 的基本形态。

TMEM(张量内存)中的累加器可以直接写成一个在 TMEM 轴上的放置:

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里逻辑行映射到 `TLane`,逻辑列映射到 `TCol`。在 {ref}`chap_tmem` 中,硬件坐标称作 Lane 和 Col。在 TIRx 布局记法里,这些硬件轴被写作 `TLane` 和 `TCol`。

一个块缩放(block-scaled)MMA 的缩放因子布局用到了复制:

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

分片把一个 32 行的组放置在 TMEM 中。副本以 32 条 lane 的步幅将该组重复四次,使这个 32 行的组在整个 128 lane 的 TMEM 空间中都可见。

一个 Tensor Core(张量核)寄存器片段可以分布在 lane 和 warp 之间:

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一个物理轴可以出现不止一次。在本例中,两个不同的 iter 都贡献到 `laneid`。没有显式轴的步幅使用默认内存轴 `m`。

在真实内核中,常见的硬件布局通常来自构造器:

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些构造器返回的是普通的 `TileLayout` 对象。它们只是便利封装,并非另一套机制。你可以检查返回的布局,把它与其他布局组合,或者在形状特殊时手写出底层的 `S[...]` 和 `R[...]` 形式。

## 交互式演示

在讲解机制之前,先有一个具体可玩的东西会有帮助。下面的演示允许你选择一个预设布局、编辑逻辑形状和 `S` 或 `R` 项、选择 dtype 与 swizzle 模式,然后点击某个元素,查看它被哪个或哪些物理坐标所拥有。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ Open the demo full screen ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

这个演示之所以有用,是因为 API 的大部分内容只不过是该演示所展示内容的精确版本。一个逻辑元素进入布局,布局把它展平、按 iter 切分、在命名轴上累加坐标,最后在需要时应用复制。

## TileLayout

`TileLayout` 是主要的仿射布局对象。它通常用与正文相同的记法书写:

```python
TileLayout(S[shape : strides])
```

`S` 项是分片规格。可以这样读它:取一个具有该形状的逻辑 tile,并按这些步幅在命名轴上放置它。

当一个值需要出现在多个位置时,分片规格会扩展出一个副本规格:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

还可以加上一个可选的偏移:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在底层,这些组成部分由 iter 表示。一个 iter 是一个三元组:

```text
(extent, stride, axis)
```

它描述了沿某一个命名轴的一次步进式行走。extent 表示该 iter 拥有多少个位置;stride 表示每一步移动多远;axis 表示正在改变哪一个硬件坐标。

一个布局有三部分。

### 分片

分片,即 `D`,是由 `S[...]` 构造的部分。它把逻辑索引划分到一个或多个 iter 上,并产生基础物理坐标。

例如:

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个分片 iter。它们的 extent 分别是 `8`、`2`、`4` 和 `2`。它们的步幅分别把数据放置到 `laneid`、`warpid`、再次到 `laneid`,以及默认内存轴 `m` 上。

这是对普通「形状-步幅」规则的推广。区别在于,步幅是绑定到命名硬件轴上的,而不是绑定到单一扁平地址。

### 副本

副本,即 `R`,描述同一个逻辑元素的额外物理拷贝。副本 iter 与逻辑索引相互独立。它们枚举的是硬件空间中的额外偏移。

例如:

```python
R[2 : 4@warpid]
```

在 `warpid` 轴上创建两个相隔四个 warp 的拷贝。

复制并不是为图方便而耍的小把戏,它描述的是真实的硬件行为。某些数据会被广播到多个 warp、lane 或内存区域。逻辑到物理的映射天然就能支持这一点,因为一个逻辑元素可以映射到一组物理坐标。

### 偏移

偏移,即 `O`,是加到每个结果上的一个固定坐标。

例如:

```python
5@warpid
```

把整个放置在 `warpid` 轴上平移五。

偏移可用于把一个 tile 放置到选定的基础坐标、为独占使用预留一段区域,或描述一个在同一资源中紧接另一个 tile 之后开始的 tile。

### 把各部分组合起来

一个布局按顺序应用这三部分。

首先,分片计算出基础坐标;然后,副本把该坐标扇出到零个或多个额外拷贝;最后,偏移平移每一个坐标。

对于逻辑坐标 `x`,结果是:

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有副本,`R` 只包含零偏移,所以结果是单元素集合。如果有副本,结果中每个副本位置对应一个坐标。

在 TIRx 语法里,一个完整布局看起来可以是这样的:

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左向右读:分片放置该逻辑 tile,副本在相隔四个 warp ID 处创建第二份拷贝,偏移则把整个放置平移到从 `warpid = 5` 开始。

如果各 iter 已经作为对象构造好,同一个布局也可以直接构造:

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数用户代码使用 `S[...]` 和 `R[...]` 记法,因为它更接近数学形式。

## 命名轴

布局中的轴并非匿名维度。每个轴都命名一个真实的硬件坐标或一个编译器层面的放置坐标。

例如:

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

grid 轴(如 `bx`、`by`、`bz`)把工作分配到各 CTA 之间。cluster 轴(如 `cbx`、`cby`、`cbz`)在一个 CTA 集群(CTA cluster)内部分配工作。线程轴(如 `tx`、`warpid`、`laneid`、`tid_in_wg`、`wid_in_wg`)描述一个 CTA 或 warpgroup 内部的归属。轴 `m` 是默认的线性内存轴。`P` 和 `F` 用于二维 scratchpad 式放置。`Bank` 命名共享内存的 bank。`TLane` 和 `TCol` 是 TIRx 布局中对 TMEM 的 Lane 与 Col 坐标的命名。

轴名是布局的一部分。这很重要,因为两个取相同整数值的坐标可能意味着不同的硬件含义。`1@tx` 与 `1@tid_in_wg` 不同;`1@laneid` 与 `1@TLane` 也不同。布局把这些含义保持显式。

## 正向映射

对布局求值,意味着取一个逻辑坐标并计算它落在哪个物理位置。API 方法是:

```python
layout.apply(*coord)
```

对于没有复制的布局,结果是一个坐标字典。对于带复制的布局,结果是一组坐标字典。坐标字典把轴名映射到整数位置,例如:

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

求值规则有四步。

第一步,按行优先顺序展平逻辑坐标。对于逻辑坐标:

```text
x = (x0, x1, ..., xr-1)
```

在逻辑形状:

```text
(S0, S1, ..., Sr-1)
```

内,扁平索引是:

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

第二步,按分片 extent 切分该扁平索引。如果分片 extent 是:

```text
(e0, e1, ..., en-1)
```

那么切分得到的分量是:

```text
c0, c1, ..., cn-1
```

对分片 extent 使用同样的行优先顺序。

第三步,用每个分量的步幅把它累加到对应轴上。如果分片 iter `k` 的 extent 为 `ek`、步幅为 `sk`、轴为 `ak`,那么分量 `ck` 贡献:

```text
ck * sk @ ak
```

所有对同一轴的贡献都相加在一起,随后再加上偏移。

第四步,应用副本 iter。每个副本 iter 贡献一个与逻辑索引无关的额外偏移。如果有多个副本 iter,布局会枚举所有组合。

这条规则的一个有用推论是:布局不需要硬编码输入形状。它需要的只是逻辑 tile 的元素总数等于各分片 extent 的乘积。只要这一点成立,展平和切分就定义了该映射。

## 案例:Tensor Core 寄存器 tile

考虑一个分布在两个各含 32 条 lane 的 warp 之间的逻辑 `(8, 16)` tile。每条 lane 拥有一个小的寄存器片段。寄存器槽位由默认内存轴 `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从 `(8, 16)` tile 中取一个逻辑元素 `(i, j)`。

行优先扁平索引是:

```text
flat = 16 * i + j
```

按分片 extent `(8, 2, 4, 2)` 切分得到:

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

分片贡献是:

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

加上偏移 `5@warpid` 后,变为:

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

副本项:

```python
R[2 : 4@warpid]
```

要么给 `warpid` 加 `0`,要么加 `4`。所以完整映射是:

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, where r in {0, 1}
m      = j mod 2
```

分片把 tile 放置在 warp 5 和 6 上。副本再把它复制到 warp 9 和 10。因此同一个逻辑元素出现在两个 warp 位置。

本例说明了为什么模型要使用一组物理坐标。复制无法用「从物理坐标到逻辑坐标的函数」自然地表示,但可以用「从一个逻辑坐标到多个物理坐标的函数」自然地表示。

## 案例:Blackwell 张量内存

同一个布局模型也适用于内存放置。轴不必是线程轴,也可以是内存轴。

TMEM 由硬件的 Lane 和 Col 坐标寻址。在 TIRx 布局记法中,这两个轴写作 `TLane` 和 `TCol`。

考虑如下布局:

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果逻辑 tile 形状是 `(2, 128, 112)`,那么切分分量就是逻辑坐标本身。对于元素 `(a, l, c)`,映射是:

```text
TLane = l
TCol  = 112 * a + c
```

extent 为 128、步幅为 `1@TLane` 的 iter 填满 128 行 TMEM Lane。extent 为 2、步幅为 `112@TCol` 的 iter 与 extent 为 112、步幅为 `1@TCol` 的 iter 合起来覆盖 224 列:

```text
TCol in [0, 224)
```

这个 224 列的跨度是有意为之的。TMEM 布局不必是 2 的幂。一个块缩放的 FP8 GEMM 可能会选择 224 列的累加器,因为一个完整的 256 列 tile 不会为两个累加器流水级加上缩放因子留下足够的 TMEM 容量。布局 API 能够直接表达这种形状。

## 缩放因子布局

上面的累加器布局是纯放置:每个逻辑累加器元素映射到一个 TMEM 坐标。块缩放 MMA 的缩放因子则不同,因为同一个物理组可能需要在多个 warp 窗口中都可见。这正是复制派上用场的地方。

一个紧凑的缩放因子布局可以写成:

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

分片把一个 32 行的缩放因子组放置在 TMEM 中:

```text
TLane = r
TCol  = s
```

这是对逻辑缩放坐标 `(r, s)` 而言。

副本项创建四个相隔 32 条 lane 的拷贝:

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

于是这个 32 行的组在 TMEM lane 0 到 31、32 到 63、64 到 95、96 到 127 处都可见。这就是 `warpx4` 广播模式({ref}`chap_layout_generations`)。四个 warp 大小的 TMEM lane 窗口各自都能看到同一个缩放因子组。

在完整的块缩放 MMA 布局中,这个 atom 会与 M 行和 K 个缩放因子组上的外层 iter 组合在一起。依据缩放因子的 dtype,多个缩放因子也可能被打包进同一个 32 位 `TCol` 单元。例如,fp8 缩放因子可以把四个值打包进一个 32位列单元。可选的步幅为零复用和流水级深度的 iter 随后可以描述跨多个 MMA 的缩放因子复用以及双缓冲(double buffering)。

关键之处在于,同一个 `TileLayout` 模型描述了这两种情形:累加器是 TMEM 中的单一放置,而缩放因子是同一 TMEM 地址空间中的复制放置。

## 现成布局

大多数内核并不逐一手写每一个硬件布局。TIRx 为常见布局提供了构造器。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回由 `tcgen05.mma` 写入的 TMEM 累加器布局。`datapath` 参数选择行放置模式。例如,`"D"` 对应 `M = 128` 的恒等式风格放置,而 `"F"` 对应 `M = 64` 的分散式放置。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回由一个 `tcgen05.ld` 或 `tcgen05.st` atom 搬运的寄存器 tile 布局。指令形状的例子包括 `.32x32b`、`.16x64b`、`.16x128b` 及相关形式。在 DSL 层面这是一个 warpgroup 分布式 tile。在降级(lowering)过程中它会变成四条 warp 协作的 `tcgen05.ld` 或 `tcgen05.st` 指令,每条 warp 一条,各自处理自己的 32 条 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回一个 warpgroup 本地寄存器 tile,通常在 `tid_in_wg` 上每个线程一行。

这些辅助函数是为了避免手写重复的常见硬件映射。它们并不隐藏模型:每个辅助函数返回的都是一个由上述同样的 `S` 和 `R` 构造的普通 `TileLayout`。

## SwizzleLayout 与 ComposeLayout

`TileLayout` 是仿射的,它能表达命名轴上的步幅、复制和偏移。这对许多放置已足够,包括线程片段、TMEM tile 以及紧凑的缩放因子布局。

共享内存的 swizzle 却需要别的东西。用于避免 bank 冲突的 swizzle 并不是仿射步幅模式,而是对线性共享内存地址的一次基于 XOR 的置换。

因此 TIRx 把 swizzle 保留为一个独立的布局对象:

```python
SwizzleLayout(...)
```

并把它与 tile 布局组合:

```python
ComposeLayout(swizzle, tile)
```

tile 布局先产生一个线性内存地址,swizzle 再对该地址做置换。把这两层保持分离,比把 XOR 置换硬塞进仿射布局模型更清晰。

## 为什么要 swizzle

共享内存被划分为 32 个 bank,每个 bank 字(word)占 4 字节。当一次访问的各 lane 触及同一 bank 内的不同地址时,这次访问会因 bank 冲突而被串行化。

一个普通的行优先 tile 在结构上就可能制造这种冲突。考虑一个采用行优先布局的 `(8, 64)` float16 tile:

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的线性元素地址是:

```text
m = 64 * i + j
```

每一行是 64 个 float16 值,即 128 字节,正好是一整条共享内存 bank 线。如果某个 warp 固定 `j` 沿一列向下读,每前进一行就跨过整整一条 128 字节的线。bank 索引随之重复,于是该列读会跨行塌缩到同一个 bank 上。

swizzle 通过让低位地址比特依赖于更高的行比特来改变这一点。原本会反复落在同一个 bank 上的一列,会被分散到不同 bank 上。

## Swizzle 变换

一个 `SwizzleLayout` 由三个整数参数控制:

```text
per_element = M
swizzle_len = B
atom_len    = S
```

输入是一个线性元素地址 `m`。

`m` 的低 `M` 位保持不变,以此保留一小段连续的元素组。较高位则被右移进一个临时值:

```text
x = m >> M
```

随后,`x` 中位于 `[S, S + B)` 的比特组会被异或进 `x` 的 `[0, B)` 比特组。把保持不变的低 `M` 位放回去,就得到 swizzle 后的地址。

等价地:

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

为使布局良构,`S` 至少要等于 `B`。

这个变换的要点不在于改变 tile 中包含哪些逻辑元素,而在于改变这些元素落在共享内存中的何处。MMA 仍然读同一个逻辑 tile,swizzle 只是让物理 bank 模式更优。

## 选择 swizzle 参数

在常规使用中,swizzle 参数依据 dtype 和共享内存 swizzle 模式选定。常见模式有 32 字节、64 字节和 128 字节 swizzle。

`per_element` 参数的选择要让一小段向量大小的组保持连续。对 float16 而言,一个 16 字节向量含 8 个元素,故:

```text
M = log2(8) = 3
```

采用 128 字节 swizzle 时,布局使用:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这既保持了 16 字节向量组完整,又足以置换较大的共享内存地址模式,从而打破列上的 bank 冲突。

大多数代码不应手工推导这些参数。dtype 和描述符模式通常会决定它们。对程序员而言重要的是:TIRx 布局里的 swizzle、TMA 描述符和 MMA 的期望三者必须匹配。

因此,一个 swizzle 过的共享内存分配看起来是这样的:

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

组合后的布局才是附加到共享内存 buffer 上的那个。

## 元素的 bank 与 line

要判断一个 swizzle 是否有帮助,可把 swizzle 后的元素地址换算回共享内存的 bank。

设 `addr` 为 swizzle 后的元素地址,`b` 为元素大小(字节)。字节地址是:

```text
byte = addr * b
```

bank 是:

```text
bank = floor(byte / 4) mod 32
```

128 字节的 bank 线是:

```text
line = floor(byte / 128)
```

对 float16,`b = 2`,所以 bank 公式变为:

```text
bank = floor(addr / 2) mod 32
```

这就是下面计算示例所用的公式。

## 计算示例:在 `(8, 64)` float16 tile 上的 128B swizzle

回到行优先的 float16 tile:

```text
m = 64 * i + j
```

使用:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

变换变为:

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

由于:

```text
m = 64 * i + j
```

我们可以写:

```text
q = floor(j / 8)
r = j mod 8
```

而 swizzle 后的地址是:

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在看列 `j = 0`。此时 `q = 0` 且 `r = 0`,故:

```text
addr = 72 * i
```

对 float16,bank 是:

```text
bank = floor(addr / 2) mod 32
```

所以这八行映射到:

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

该列现在触及八个不同的 bank,冲突消失了。

若不 swizzle,同一列的地址是:

```text
m = 64 * i
```

因此:

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每一行都落在 bank 0 上,于是该访问被串行化。swizzle 只改变了物理放置,但这已经足够把列访问变成无冲突访问。

这一保证依赖于按其设计意图使用 swizzle。dtype、swizzle 宽度和访问形状必须与 TMA 和 MMA 描述符模式匹配。128 字节 float16 swizzle 是围绕相关的 16 字节行块与 Tensor Core 访问模式设计的,它并不承诺任意共享内存访问都能无冲突。本章顶部的演示把这一点可视化:选择一个 dtype 和 swizzle 模式,观察一列在不加 swizzle 时塌缩到同一个 bank 上,再在施加匹配的 swizzle 后于 bank 视图中散开。

## 设计依据

布局 API 遵循三个设计取舍。

第一,它支持一般形状。硬件 tile 并不总是 2 的幂。全局张量、共享内存的各流水级、TMEM 累加器和缩放因子 buffer,常常具有源自容量限制或算法选择的形状。布局模型把这些形状当作正常情况对待。

第二,映射方向是从逻辑坐标到物理坐标。这一方向很重要,因为复制很常见:一个逻辑元素可能存在于多个物理位置。逻辑到物理的映射能直接用一组坐标表示这一点。

第三,硬件轴是显式的。布局不使用匿名维度、再依赖上下文事后解释。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 与 `TCol` 之间的区别,被直接写进布局本身。

合法性与可行性检查并不只由布局对象负责。布局能说明数据放在何处,更高层的 tile 原语决定某次操作能否合法且高效地使用该放置。这种分离使布局 API 保持精简,同时仍给编译器足够信息来派发真实的硬件操作。
