(chap_layout_generations)=
# 跨 GPU 世代的 Tensor Core 操作数布局

:::{admonition} Overview
:class: overview

- 在 Ampere、Hopper 和 Blackwell 三代中,Tensor Core(张量核)仍然执行同样的高层运算:`D = A B + C`。
- 各世代之间变化的是:操作数如何抵达 Tensor Core、支持哪些分块形状与 dtype(数据类型),以及累加器存放在何处。
- Ampere 使用 warp(线程束)级的寄存器片段。共享内存(SMEM)分块通过 `ldmatrix` 加载进该片段,累加器保留在寄存器中。
- Hopper 允许 `wgmma` 通过矩阵描述符直接从共享内存读取操作数。描述符指定了 Tensor Core 所期望的共享内存 swizzle 格式。
- Blackwell 保留了共享内存操作数通路,但把累加器搬进了 TMEM(张量内存)。块缩放 MMA 也会通过 TMEM 暂存其缩放因子。
- 在所有世代中都一直存在两条内存约束:全局内存合并访问与共享内存 bank(存储体)冲突。
:::

从远处看,Tensor Core 的运算显得很稳定。它把 A 和 B 的分块相乘,加上累加器 C,得到 D。这种形式自 Volta 以来就没变过。

围绕该运算的细节却并非固定不变。在某一代上跑得很快的内核,到了下一代可能就很慢。使用了错误布局的内核还可能算出错误结果,哪怕逻辑上的数学式仍写作 `D = A B + C`。原因是 Tensor Core 并不消费抽象矩阵。它消费的是处于非常具体的硬件布局中的操作数。

本章追随这一布局契约走过三代。Ampere 通过 warp 级寄存器片段暴露 Tensor Core。Hopper 把输入操作数移到共享内存描述符。Blackwell 保留共享内存操作数,但把累加器搬进 TMEM。运算始终是矩阵乘加,但进出 Tensor Core 的路径每一次都在变。

{ref}`数据布局 <chap_data_layout>` 一章中的布局记法,正是我们用来描述这些契约的语言。Blackwell TMEM 的细节另行在 {ref}`chap_tmem` 中讨论。

## 始终未曾消失的两条约束

在 Tensor Core 介入之前,两条普通的内存约束就已经在塑造 GPU 内核的布局了。

第一条是全局内存合并访问。当 warp 的 32 条 lane(通道)发起一次全局内存加载时,内存系统希望这些地址落在少数几段连续、对齐的内存段内。如果地址是分散的,warp 加载就会变成多次内存事务。同样的逻辑数据搬运会消耗更多带宽、花费更多时间。

第二条是共享内存 bank 冲突。共享内存被划分为 32 个 bank。如果 warp 中的若干 lane 访问映射到同一个 bank 的不同地址,这些访问就无法一次性全部服务。硬件会把它们串行化。于是一个在平坦共享内存数组看来无害的布局,可能因为其 bank 模式而变得很慢。

Swizzle 是修复共享内存一侧的常用手段。逻辑分块保持不变,但物理地址映射被置换,使访问模式在各个 bank 之间铺开,而不是堆叠到同一个 bank 上。

这两条约束即便对从不使用 Tensor Core 的内核也成立。Tensor Core 内核还加上第三条约束:操作数必须按 Tensor Core 指令自身所期望的布局来排列。本章余下部分就讲这第三条约束如何在 Ampere、Hopper 和 Blackwell 之间变迁。

## Ampere:分布在 warp lane 上的寄存器片段

在 Ampere 类 GPU 上,主要的 Tensor Core 指令是 warp 级的 `mma.sync.aligned.m16n8k*` 家族。关键事实是指令在哪里读写数据:寄存器。

A、B 以及 C 或 D 累加器,都是分布在 warp 的 32 条 lane 上的逐线程寄存器片段。共享内存只是中转区。在 MMA 能运行之前,操作数分块必须从共享内存搬到指令所期望的精确寄存器片段布局中。

数据通路如下:

```text
SMEM to registers with ldmatrix
registers to registers with mma.sync
registers back to SMEM with ordinary stores
```

Ampere 的布局故事大多源自这条通路。内核必须把分块以可高效加载的形式存放在共享内存中,然后用 `ldmatrix` 产生 `mma.sync` 所需的寄存器片段。

## Ampere Tensor Core 的期望

Ampere 的 Tensor Core 读取由 8×8 子分块单元构建的寄存器片段。这些单元正是 `ldmatrix` 加载、MMA 消费的对象。

以 `mma.m16n8k16`、fp16 或 bf16 输入、fp32 累加为例。累加器分块的形状是 `16×8`。它以固定模式分布在 32 条 lane 上。

对于 C 或 D 累加器,lane `l` 持有的行是:

```text
l / 4
l / 4 + 8
```

列是:

```text
2 * (l % 4)
2 * (l % 4) + 1
```

因此每条 lane 拥有四个 fp32 累加值:来自两个 8 行半区的各两行,与两个相邻列交叉。四条连续 lane 覆盖一行的八列。

A 操作数使用相同的 M 侧行切分。K 维度散布在 `l % 4` 与该 lane 所持寄存器之间。对 fp16 或 bf16,每个 32 位寄存器打包两个 K 值。

B 操作数使用相匹配的 K 摆放,并把 N 侧散布到 lane 组与寄存器之间。

确切细节随指令形状与 dtype 而异,但原则固定不变。Tensor Core 期望某种特定的逐 lane 寄存器片段。如果数值不在这些寄存器、不在这种模式中,指令就会乘错元素。

用布局记法写,m8n8 片段正是那种以命名 lane 轴书写的模式,例如:

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@m)]
```

两个 `laneid` 迭代量合起来描述行块和列块如何在 lane 间散开,而最后的 `m` 分量描述逐 lane 的寄存器槽位。

## `ldmatrix`:从共享内存到寄存器片段

`ldmatrix` 是 Ampere 中连接共享内存与 Tensor Core 寄存器片段的指令。它是一种 warp 协作加载。一条指令把一个或多个 8×8 的 16 位矩阵从共享内存搬入 `mma.sync` 所期望的分布式寄存器布局。

指令形式为:

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

可附带可选的 `.trans` 限定符。

`.x1`、`.x2`、`.x4` 形式分别加载一、二、四个 8×8 矩阵。行基地址由 lane 提供。对矩阵 `m` 与行 `r`,基地址来自 lane `m * 8 + r`。也就是说 `.x1` 用 lane 0 到 7 提供行地址,`.x2` 用 lane 0 到 15,`.x4` 用 lane 0 到 31。

结果直接落入 MMA 片段。对于基本的 8×8 情形,lane `l` 收到 Tensor Core 所期望的那一对行与列。一段朴素的逐 lane `ld.shared` 指令循环本要手动复现那种散布。`ldmatrix` 则以一条 warp 协作指令完成共享内存到片段的重排。

`.trans` 形式在加载时对每个 8×8 矩阵做转置。当操作数的存放朝向与 MMA 指令所期望的相反时,就用得上它。

![ldmatrix 把 8x8 共享内存分块加载进 warp 寄存器片段;Ampere 上的反向通路使用普通存储,Hopper 上后来出现了专用的 stmatrix 指令](../img/zh/ldstmatrix.svg)

## 把 Ampere 片段写回

`mma.sync` 完成后,累加器仍是寄存器片段。收尾必须把这个片段搬出去。

在 Ampere 上没有 `ldmatrix` 的专用逆操作。内核使用普通的逐线程存储,有时在存储前配以 warp shuffle 或局部重排,把累加器以有用的布局写入共享内存或全局内存。

这让 Ampere 的模型保持简单,但也把大量布局工作暴露给内核。输入侧用 `ldmatrix` 生成片段。计算指令读写寄存器片段。输出侧则由从这些片段出发的普通存储来处理。

## Ampere 上的 Swizzle

Ampere 内核已经需要共享内存 swizzle。原因是共享内存分块通常以一种访问模式写入、又以另一种模式读出。

假设某个分块沿行从全局内存填入。行主序布局使那次写入既合并又 bank 友好。但 `ldmatrix` 之后可能以一种实际上沿列下行或横跨 8×8 子分块的模式读取该分块。在朴素的行主序布局下,这些读取可能堆叠到同一个共享内存 bank 上。

对一个简单的 `(8, 64)` float16 分块,一行是:

```text
64 * 2 bytes = 128 bytes
```

恰好是一整条共享内存 bank 线。沿固定列每往下一行就前进 128 字节,于是 bank 索引不断重复。八行可能塌缩到同一个 bank 上,造成 8 路冲突。

改成朴素的列主序布局并不能解决全部问题。它通常只是把冲突挪到另一次访问。行写入变差,而列式读取变好。

XOR swizzle 通过让物理列依赖于行来修复这一点。一个简单版本是:

```text
physical_col = logical_col xor row
```

逻辑分块不变。共享内存中的物理摆放被置换,使行式写入与 Tensor Core 读取模式都能避开 bank 冲突。

在 Ampere 上,这种 swizzle 通常通过手写的共享内存索引算术来表达。后续世代把它纳入了硬件引擎所用的描述符格式。

![在朴素的行主序分块上,行写入在各 bank 间铺开,而列读取撞在同一个 bank 上;XOR swizzle 在不放弃合并的行写入的前提下,把列读取散布到各 bank](../img/zh/swizzle_conflict.svg)

## Hopper:`wgmma`、共享内存描述符与 swizzle 格式

Hopper 改变了 Tensor Core 通路的输入侧。Hopper 的 `wgmma` 不再要求每个操作数都通过 `ldmatrix` 加载进寄存器,而能直接从共享内存读取操作数。

B 操作数从共享内存矩阵描述符读取。A 操作数既可从共享内存描述符读取,也可从寄存器读取,这便有了 `.ss` 与 `.rs` 两种形式。

这去掉了源在 SMEM 的操作数那段显式 `ldmatrix` 步骤,但并未去掉布局要求。Tensor Core 仍期望操作数以精确的共享内存格式存放。区别在于,该格式现在通过矩阵描述符告知硬件。

## Hopper Tensor Core 的期望

Hopper 的共享内存矩阵描述符是对共享内存中一个矩阵分块的紧凑描述。它告诉 `wgmma` 如何把逻辑操作数坐标转换成共享内存地址。

描述符包含如下字段:

```text
start address
leading dimension offset
stride dimension offset
swizzle mode
base offset
```

确切含义取决于操作数的主维模式。对一个 K 主序分块,一条步长沿 K 推进,另一条沿 M 推进。对一个 MN 主序分块,两者角色互换。

swizzle 模式是若干共享内存描述符格式之一,例如:

```text
SWIZZLE_NONE
SWIZZLE_32B
SWIZZLE_64B
SWIZZLE_128B
```

swizzle 模式决定两件事。它决定描述符所用的原子形状,也决定在该原子内部施加的 XOR 置换。例如,128 字节 swizzle 模式把操作数视为 8 行×128 字节原子的网格,swizzle 在每个原子内部施加。

内核仍须把字节摆对。TMA(张量内存加速器)通常负责填充共享内存分块,而 TMA 描述符必须使用与 `wgmma` 描述符后续所指定的相同的 swizzle 格式。如果 TMA 写出的是 128 字节 swizzled 分块,`wgmma` 描述符就必须按 128 字节 swizzled 分块来读。描述符与数据若不一致,Tensor Core 就会读到乱序的操作数。

这是相对 Ampere 的主要转变。swizzle 不再仅隐藏在手写的共享内存索引里。Hopper 把它升格为一等描述符格式。写分块的 TMA 加载与读分块的 `wgmma` 指令,都能指名同一种格式。

![Hopper 的共享内存矩阵描述符把操作数坐标映射到 swizzled 共享内存原子:描述符的步长选出原子,而 swizzle 选定原子内的字节位置](../img/zh/smem_descriptor.svg)

## Hopper 的输出仍用寄存器

Hopper 改变了输入通路,但累加器仍住在寄存器里。

一条 `wgmma` 指令把累加器写入逐线程寄存器片段。片段的确切大小与寄存器数量取决于指令形状,例如 `m64nNk16`,其中 N 改变累加器寄存器的数量。但基本思路与 Ampere 相同:收尾消费的是一个寄存器片段。

因此 Hopper 拥有混合的布局模型。输入操作数可直接来自共享内存描述符,swizzle 由硬件描述。输出累加器则仍是一个寄存器布局问题。

Blackwell 改变了这一输出侧。

## Blackwell:`tcgen05` 与 TMEM

Blackwell 为数据操作数保留了共享内存描述符的思路。A 和 B 仍以 Tensor Core 所期望的布局在共享内存中备好。某些模式还能从 TMEM 读取 A 操作数。

主要变化在累加器。`tcgen05.mma` 把累加器写入 TMEM(张量内存),而不是让它作为一个长寿命的寄存器片段留存。在计算阶段,累加器留在 TMEM 中。收尾随后用 `tcgen05.ld` 把它装回寄存器。

这把输出布局问题从寄存器挪到了 TMEM。内核必须分配 TMEM、选择正确的 TMEM 布局、等待 MMA 完成,然后用匹配的 `tcgen05.ld` 通路为收尾恢复累加器片段。

`cta_group::1` 与 `cta_group::2` 如何把累加器在一个或两个 CTA(协作线程阵列)之间切分,其细节在 {ref}`chap_tensor_cores` 中讨论。与早期世代差异最大的布局,是块缩放的缩放因子布局。

## TMEM 中的缩放因子布局

块缩放 MMA 模式,例如 `mxfp8` 与 `nvfp4`,增加了缩放因子操作数。除 A 和 B 之外,MMA 还读取:

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK` 是 K 缩放块的数量。

数据操作数 A 和 B 住在共享内存中。缩放因子住在 TMEM 中。于是它们拥有不同的搬运通路。

TMA 从全局内存加载到共享内存。它不直接加载进 TMEM。因此缩放因子通常分两步搬运:

```text
global memory to shared memory with TMA
shared memory to TMEM with tcgen05.cp
```

只有在这次拷贝之后,缩放因子才进入 `tcgen05.mma` 所期望读取的内存空间。

TMEM 缩放因子布局使用 TMEM 硬件坐标 Lane 与 Col。在 TIRx 布局记法中,这两根轴写作 `TLane` 与 `TCol`。

一个 128 行的缩放向量被压缩进一个 32 lane 组,然后在 TMEM 的四个 32 lane 窗口之间复制。在布局记法中,核心模式是:

```text
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
```

分片摆放基础的 32 行组:

```text
TLane = r
TCol  = s
```

复制项在 lane 偏移 0、32、64、96 处添加副本:

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

这就是 `warpx4` 广播模式。同一个压缩缩放因子组在整个 128 lane 的 TMEM 空间中都变得可见。

在 32 位 `TCol` 单元内部还有字节打包。打包取决于 `scale_vec` 模式:

```text
1X: one scale value is broadcast across the 32-bit cell
2X: two scale values are packed, each duplicated
4X: four K-block scale values are packed
```

![scale_vec 字节打包:1X 把一个缩放值在整个 4 字节单元中广播;2X 打包两个缩放值,各复制一份;4X 打包四个 K 块缩放值](../img/zh/sf_scale_vec.svg)

这种打包在 Ampere 或 Hopper 上没有直接的对应物,因为那两代没有针对 `tcgen05` 块缩放 MMA 的 TMEM 缩放因子操作数。

在 `cta_group::2` 中,缩放因子跟随它们所缩放的数据。SFA 缩放 A,因此它沿 M 在两个 CTA 之间切分,与每个 CTA 所拥有的 A 行相匹配。SFB 缩放 B,而 B 为计算的两个 CTA 半体所共享,因此 SFB 被多播到两个 CTA({ref}`chap_tensor_cores`)。

## 一个反复出现的片段

尽管周围的内存通路在变,有一种结构不断归来:m8n8 式的寄存器片段。

在 Ampere 上,`ldmatrix` 构建该片段供 `mma.sync` 读取。

在 Hopper 上,`wgmma` 把累加器作为寄存器片段写出来供收尾使用。

在 Blackwell 上,累加器在计算期间住在 TMEM 中,但 `tcgen05.ld` 会在收尾处理并存储它之前把它装回成寄存器片段({ref}`chap_tmem`)。

因此片段并未消失。它的角色在变。早期世代在整个计算阶段都把累加器放在那里。Blackwell 主要在 TMEM 与收尾的边界处使用它。

## 主线

在 Ampere 上,内核显式构建 Tensor Core 寄存器片段。共享内存 swizzle 主要是内核通过索引算术来承担的责任。

在 Hopper 上,Tensor Core 可通过矩阵描述符直接从共享内存读取操作数。swizzle 变成了 TMA 与 `wgmma` 共享的具名描述符格式。

在 Blackwell 上,输入侧仍使用共享内存操作数,但累加器搬到了 TMEM。块缩放 MMA 还增加了必须暂存进 TMEM 的缩放因子操作数。

描述符并不消除布局工作。它把契约显式化。内核仍须确保数据搬运通路、内存布局与 Tensor Core 指令三者一致。写 swizzled SMEM 分块的 TMA 描述符、读该分块的 MMA 描述符,以及附着于缓冲区的布局,三者必须描述同一种物理摆放。

这些部件中只要有任何一处不一致,硬件照常运行。它只是会读错字节,或者读得慢。正因如此,布局不是 Tensor Core 内核之外的装饰。它是指令接口的一部分。
