(chap_data_layout)=
# 数据布局及其记法

:::{admonition} Overview
:class: overview

- *数据布局* 将一个张量的逻辑索引映射到物理位置,它决定了访存的合并性、bank 冲突,以及某个引擎能否读取一个分块。
- 本书用一种记法来书写布局:`S[(shape) : (strides)]`,并带有具名轴(`@laneid`、`@TLane`,……)以及一个用于广播或复制数据的复制项 `R[...]`。
- Swizzle 是一种对地址的 XOR(异或)重映射,用于消除共享内存的 bank 冲突。
:::

同样的一组数字,以不同的物理排布写入内存,在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于,一个张量的逻辑索引丝毫不能说明它的字节实际存放在哪里。硬件对这种存放位置极为敏感:它决定了 32 个 lane 的访存是合并成一次事务,还是分散成 32 次;这些地址是落到不同的内存 bank 上,还是会发生冲突并被串行化;它甚至决定了一个分块的字节排布是否与 Tensor Core(张量核)能读取的格式相符。

机器学习程序通常用张量的逻辑形状来描述它们。一个**数据布局**补上了缺失的物理部分:它说明具有逻辑索引 `(i, j, …)` 的某个元素到底存放在哪里——是在内存中、在寄存器中,还是在某种其他硬件存储里。

本章介绍现代 GPU 编程中出现的几种主要布局。为了让讨论易于展开,我们发展出一套紧凑的**记法**,用以统一描述机器学习系统在各种情形下遇到的布局。最后我们以 **swizzling** 收尾,这种机制能同时让对一个分块的行向访问和列向访问都变得高效。

## 形状-步长模型

在进入 GPU 专有的布局之前,值得从最简单的布局讲起,因为本章其余所有内容都建立在它之上。布局的核心其实只有两样东西:一个**形状**和一组与之匹配的**步长**。我们把这对值写成 `S[(shape) : (strides)]`,要找到某个逻辑索引对应的位置,我们取该索引与步长的点积。例如,一个行主序的 4×4 矩阵看起来是这样的:

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这无非就是经典的形状/步长模型,只是写得紧凑了一些(是 CuTe 记法的行主序简化版),后续所有内容都由它构建而成。

事实上,你几乎肯定已经用过这个模型了。凡是写过 PyTorch 或 NumPy 的人都用过,因为在这些库里,一个张量*本身就是*一个形状加上在一个平坦存储缓冲区上的一组步长:

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

一旦你以这种方式看待张量,就会明白为什么有那么多"重塑"操作根本不碰数据。它们只是改写步长,然后返回一个指向同一份存储的**视图**,最清晰的例子就是转置,或者说 permute:

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True,同一批字节
```

这里 `t.permute(1, 0)` 是在*同一块*内存上的 `S[(4, 3) : (1, 4)]`:转置纯粹是一次步长的改变,没有搬动一个字节。对连续张量做 `reshape` 或 `view` 的情形也一样:在旧存储上换一组新的形状和新的步长。(NumPy 的行为完全一致;唯一的区别是它的 `.strides` 以字节而非元素来计数。)

GPU 上的布局也正是这样工作的,本章余下部分其实都是围绕同一个想法展开的一系列变体:一个分块的映射(无论映射到内存,还是通过我们马上引入的具名轴映射到 lane 和寄存器)都是一条作用在固定缓冲区上的步长规则,所以重排一个分块通常只是改变*布局*,而不是一次拷贝。不过,我们要小心这种推理的边界。零拷贝的故事只对一个线性地址空间之上的逻辑视图才干净成立;在 GPU 上,它只有在新的视图与既有的字节排布和归属关系相容时才适用。一旦你改变了哪个线程或哪个寄存器拥有某个元素,或者改动了 SMEM(共享内存)的 swizzle,通常就需要真正的数据搬动:load、store、shuffle、`ldmatrix`、转置。

## 分块布局

到目前为止,我们描述的都是整张张量的布局。然而 GPU 的内核很少一次性处理整张矩阵;它们处理的是更小的分块,这些分块被硬件的不同部分加载、变换、计算。好消息是,分块化并不需要任何新东西。它仍然只是一个布局,只不过现在写出来多了几个维度。把一个 8×8 矩阵切成 2×4 的分块,我们就得到一个 4 维布局,其坐标为 `(tile_row, row_in_tile, tile_col, col_in_tile)`,步长则被选成让每个分块保持连续:

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

一个逻辑坐标 `(i, j)` 先变成 `(i//2, i%2, j//4, j%4)`,再经过这些步长计算。值得注意的是,这套记法在表达分块化时根本用不到任何特殊的"分块"概念:它就是和之前一样的形状-步长模型,只是索引被拆成了外层和内层坐标。

下方的交互式可视化展示了逻辑矩阵索引是如何被分解成分块坐标,再映射到物理地址的。

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:点击某个单元格,查看它的分块化索引和地址。*

## 具名轴

到目前为止,`S[...]` 中的每一条步长都指向线性内存里的某个偏移,我们也把地址当作内存里的某个位置来看待。但在 GPU 上,数据可以存放在不止一个地方:除了内存,一个分块还可以分布在 warp(线程束)的 lane 之间、线程的寄存器之间,或者 TMEM(张量内存)的 lane 和列之间。为了统一描述所有这些情形,我们用**具名轴**来扩展这套记法。其想法是让每个步长系数带上一枚轴标签,说明它是在哪个空间里移动的:`@m` 表示普通内存,`@laneid` 表示 warp 的 lane,`@reg` 表示寄存器,`@warpid` 表示 warp,`@TLane` / `@TCol` 表示 TMEM 坐标。有了这些标签,一个布局就不仅能描述数据在内存中的位置,还能描述数据是如何分布在操作它的那些硬件资源之上的。

一旦内存标签被明确写出来,内存中一个行主序的 8×16 分块就简单地写成

```text
S[(8, 16) : (16@m, 1@m)]
```

当一个布局描述的是*分散在线程之间*而非铺在内存里的数据时,这些标签才真正派上用场。以 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` 为例:它不再指向线性内存,而是把行和列映射到 lane ID 和每个 lane 的一个寄存器上。这里的 `laneid` 指 warp 内的 lane 索引,即 `thread_index % warp_size`。这正是你将在 {ref}`chap_layout_generations` 中遇到的 Tensor Core 寄存器片段。

下方的交互式可视化展示了布局如何把张量元素分散到 warp 的 lane 和每个 lane 的寄存器上,而不是放到线性内存中。

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:建立在 `@laneid` 和 `@reg` 之上的布局;点击某个单元格,查看哪个 lane / 寄存器持有它。*

## 分布式布局

具名轴之所以如此有用,在于它让我们能统一地描述系统多个层级上的数据放置,包括*跨整台设备*的放置。我们刚刚把它用在单个 GPU 内部的 lane 和寄存器上,但同样的想法也能向外延伸:`@gpuid_x`、`@gpuid_y` 这样的轴可以说明数据位于 GPU 阵列中的何处,有了它们,这套记法就能刻画分布式训练和推理中出现的分片模式。轴尚未刻画的一件事是*复制*,即被拷贝到不止一个地方的数据,为此我们加入记法 `R[n : stride]`,其中 `R` 标记被复制的维度。例如,`R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` 轴的复制。把两者合在一起,一个表达式就能既把张量分片到一个 2×2 的 GPU 阵列上,又沿其中一根轴复制它:

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下方的演示在一个小的 GPU 阵列上展示了这种"分片加复制"的组合模式。点击任意单元格,查看它由哪个设备持有,并观察 `@gpuid_x` 复制是如何在配对设备上放置一份完全相同的副本的;按钮可以在"全分片"、"分片 + 副本"和"分片 + 偏移"三种布局之间切换。

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:分布在 2×2 GPU 阵列上的布局;点击某个单元格,查看哪些设备持有它。*

### 内核内的复制模式:TMEM 中的缩放因子

我们刚刚为 GPU 阵列引入的复制维度 `R[...]` 并不只是关于多台设备。同样的构造恰好也能描述完全发生在单个内核内部的一件事:*跨 lane 广播*的数据。Blackwell(架构代号)的块缩放 MMA(矩阵乘加)({ref}`chap_layout_generations`)就是一个好例子。它的缩放因子存放在 TMEM 里,一个 128 行的缩放向量只存在 **32 个 TMEM lane** 中:逻辑行 `r` 落到 TMEM lane `r % 32` 上,而 `r // 32` 则沿列方向延伸。这 32 个被存储的 TMEM lane 随后**沿 TMEM 的 `TLane` 轴被复制**,从 32 扩展到 128 个 TMEM lane,这样发起读取的 warpgroup(线程束组)中的四个 warp 就都能在各自那 32-lane 的 TMEM 窗口里找到一份副本。这是一次 `warpx4` 广播,我们用复制维度来书写它。这些读取本身则由这些 warp 的线程来完成:

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

于是得到四份副本,间隔为 32 个 TMEM lane:TMEM lane `l`、`l+32`、`l+64`、`l+96` 都持有同一个缩放值。如前所述,复制维度并不携带新数据;它只是说明"同一个值,出现在四个 TMEM-lane 位置上",其方式与刚才 `@gpuid_x` 在 GPU 阵列上广播一行完全相同。

下方的交互式演示把两步合在一起展示:先紧凑地打包进 32 个 TMEM lane,再 `warpx4` 广播到 128 个读取 lane。

```{raw} html
<iframe src="../demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:点击某个缩放因子 `SFA[m, sf]`;它会打包进 TMEM 中 lane 为 `m mod 32`、列号为 `(m // 32)·4 + sf` 的位置,然后沿 `TLane` 轴 `warpx4` 广播到四个 lane 副本(`l`、`l+32`、`l+64`、`l+96`),每个 warp 的 32-lane 窗口各得一份。*

每一列内部的字节打包(`scale_vec` 的 1X/2X/4X 模式)以及 `cta_group::2` 的拆分,在 {ref}`chap_layout_generations` 中介绍。

已经熟悉 CuTe 的读者,可以把本章这套记法看作 CuTe 的一个行主序变体,只是额外加上了显式的硬件具名轴和一套专门的复制结构。

## Swizzle 布局

本章的最后一种布局是为了解决一个具体的硬件问题而存在的。GPU 的共享内存被组织成多个内存 bank,当不同的 lane 落到不同的 bank 上时,访存速度最快。当若干 lane 反而落到*同一个* bank 内的不同地址时,硬件别无选择,只能把它们串行化,于是我们就要为一次 **bank 冲突**付出代价。

在张量程序中这一点很难避免,因为内存并非按纯线性的顺序访问。在处理矩阵时,我们常常需要同时读取同一分块的行切片和列切片,这造成了一种实实在在的张力:一种对行向访问高效的布局往往会在列向访问时引发 bank 冲突,而偏向列的布局又会损害行向访问。**Swizzling** 正是用来打破这种张力的技术。

swizzle 背后的想法是对地址映射做置换,通常是把列索引与行索引做 XOR(异或),使得行访问和列访问*同时*都被分散到各个 bank 上。它所提供的无冲突保证是具体的:它只对匹配的元素位宽、swizzle 模式和访问模式(也就是某个引擎的描述符所期望的那种)成立,而对任意的元素位宽或对齐方式并不成立。

下方的第一个交互式演示把它讲得很具体。点击某个列索引,观察每个元素落到哪个 bank:左侧纯行主序的分块里,一列会让全部八个元素落入同一个 bank,于是这次读取被串行化成八个周期;而右侧经过 XOR swizzle 的布局里,同一列被分散到八个不同的 bank,一个周期就能读完。

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:一个 8×8 分块,纯行主序下列方向存在 bank 冲突,经 XOR swizzle 后变为无冲突。*

这个小小的 8×8 例子抓住了核心想法,但真实 GPU 的内存 bank 数远比这个示意要多。要让 swizzle 在全规模下生效,我们并不把整个分块当作一个整体来对待,而是把内存切成一个个小段,并在每一段内部套用 swizzle 模式。实践中最常见的情况是 `SWIZZLE_128B`,它围绕 128 字节的段来组织,这样同一套行/列重映射技巧就能很自然地嵌入到 32-bank 的内存系统中。

下方的交互式演示展示了那个具体的硬件 swizzle——`SWIZZLE_128B`,在我们推广到各种格式之前,先把这种逐段重复的模式看清楚。

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:128 字节段内的 `SWIZZLE_128B` 模式;逐步走过各个读取周期,可以看到 `physical_sector = logical_sector XOR row` 把每一列都分散到不同的 bank 上。*

同样的想法可以推广到这种 128 字节情形之外。为了简化可视化,接下来我们用一个色块来表示一个段,而不再逐个画出 bank。一般而言,硬件会定义一个小的、可重复的**原子单元**,置换就在其上施加,不同的 swizzle 模式选择不同的原子大小。`SWIZZLE_128B` 使用 8 × 128 B 的原子,`SWIZZLE_64B` 使用 8 × 64 B 的原子,`SWIZZLE_32B` 使用 8 × 32 B 的原子;然后整个分块就由当时所用原子来分块化铺满。

最后一个交互式演示让你在这些格式(包括一种 16 B 交错模式)之间切换、选择数据类型,并把鼠标悬停在任意单元格上来直接查看一个原子内部的元素排布——这正是讨论某条 load/store 指令期望哪种 swizzle 时所需要的细节粒度。

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互:选择一种 swizzle 格式(和数据类型),查看它的原子形状(8 × N B);悬停某个单元格,查看其内部元素是如何被置换的。*

该选哪种模式呢?经验法则是优先选用分块所能填满的*最大*原子。一个 N 字节的原子要求分块的连续维度至少有 N 字节、并且是它的整数倍,所以 `SWIZZLE_128B` 仅在一行至少跨 128 字节、即 64 个 `float16` 元素时才适用。一旦能用,它就是默认选择,因为它的 8 × 128 B 原子正好覆盖完整的一条 128 字节 bank 线,从而能一次把一列分散到全部 32 个 bank 上,在 fp16 下同时对 8 行和 8 列提供无冲突访问。而当问题的形状迫使连续维度变小时,分块就填不满 128 B 原子,这时你就退到 `SWIZZLE_64B` 或 `SWIZZLE_32B`,即这一行仍能覆盖的最大原子。

你绝不需要手工算出这些置换后的地址,这里值得把 swizzle 与 `S[...]` 记法的关系说精确:它*不是*那条仿射映射的一部分。它是叠加在其上的一层独立的、非仿射的层。`S[...]` 布局把一个元素放到一个线性内存(`@m`)地址,swizzle 则置换这个地址,在 TIRx(Python DSL)的布局 API 中写作 `ComposeLayout(swizzle, tile)`({ref}`chap_tirx_layout_api`)。你要做的只是为涉及该分块的每一个 op 都选定一个一致的模式,剩下的交给这个复合布局即可。

这个复合布局也正是硬件要填充的东西,swizzling 和分块化也正是在这里走到一起。一个 TMA(张量内存加速器)描述符是多维的,所以一个单一的三维方框就能同时描述分块的原子分块化,以及每个原子内部的 swizzle;一次 TMA 加载就会逐个原子地把分块铺开,并在写共享内存的同时完成 swizzle({ref}`chap_tma`),不需要单独的 swizzle 步骤。至于每种引擎*需要哪种* swizzle,则因架构代际而异,这正是下一章的主题。
