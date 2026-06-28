(chap_tma)=
# 异步数据搬运:TMA

:::{admonition} Overview
:class: overview

- TMA(张量内存加速器)是一种硬件引擎,用于在全局显存(GMEM)与共享内存(SMEM)之间进行异步分块拷贝。由一个线程发起拷贝,引擎负责搬运字节。
- 一次 TMA 拷贝由一个 tensor-map 描述符(descriptor)描述。该描述符告知引擎全局张量的形状、步长、分块坐标,以及共享内存的 swizzle(混洗)模式。
- 在加载路径上,TMA 可以在写入共享内存时对分块进行 swizzle,使分块直接落到 Tensor Core(张量核)所期望的布局上。
- TMA 加载通过带字节计数追踪的 `mbarrier`(内存屏障)完成;TMA 存储则使用提交组(commit group)与等待组(wait group)。
:::

只有当数据准备就绪可供消费时,Tensor Core 才能发挥作用。在 GEMM(通用矩阵乘)或注意力 kernel(内核)中,一旦流水线填满,计算可能成为计算受限的部分({ref}`chap_performance`),但流水线只有当下一个操作数分块及时到达时才能保持填满。

搬运一个分块的旧做法是让线程自行拷贝。每个线程计算地址、从全局显存发起加载,并把数值写入共享内存。这可行,但它把 warp 指令花在了地址计算与拷贝簿记上,而不是用在计算上。它还使得拷贝路径暴露在那些本应喂给 Tensor Core 的同一批 warp 的指令流中。

Tensor Memory Accelerator,即 TMA,把这部分工作搬进了硬件拷贝引擎。一个线程发起一次分块拷贝,拷贝引擎随后在全局显存与共享内存之间异步搬运一个矩形分块。当引擎在搬运字节时,CTA(协作线程阵列)的其余部分可以继续做其他工作。

TMA 还处理了布局问题的一部分。Tensor Core 不仅需要在共享内存中放正确的数值,还需要它们处于正确的共享内存布局上。在加载路径上,TMA 可以在写入分块时应用共享内存 swizzle。这使得分块可以直接落到后续 MMA(矩阵乘加)所期望的布局上。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示:TMA 将一个分块从全局显存拷贝到共享内存。切换 swizzle 模式,并将鼠标悬停在某个源单元上,即可看到它落在共享内存中的位置。*

## 一个线程发起,硬件搬运分块

一次 TMA 拷贝从一个发起线程开始。该线程并不会遍历分块中的所有元素。它向硬件提供一份拷贝描述,然后由 TMA 引擎执行传输。

主要输入是一个 tensor-map 描述符。该描述符描述了全局张量以及应当如何从中读取一个分块。它记录了诸如张量形状、步长、元素大小、分块形状与 swizzle 模式等信息。发起线程还需提供分块应当落入的共享内存地址。

指令发起后,拷贝异步执行。发起线程可以继续执行。CTA 中的其他线程也可以继续执行。此时传输由 TMA 引擎负责,而不再由一组普通的加载/存储指令循环承担。

这给 kernel 提供了两种不同的方式来表达同一个逻辑操作——「拷贝这个分块」。

一种路径是线程拷贝。线程协作从全局显存加载并写入共享内存。这让 kernel 能对每一次访问进行直接控制,但会消耗线程指令与寄存器来计算地址。

另一种路径是 TMA 拷贝。由一个线程发起传输,硬件拷贝引擎执行矩形拷贝。这是大型规整分块(尤其是 Tensor Core kernel 所使用的操作数分块)的自然路径。

这两条路径有不同的同步规则和不同的性能表现。在二者之间作出选择是一个派发(dispatch)决策。布局告诉 kernel 它想要何种内存排布;作用域(scope)告诉它有哪些线程或 CTA 参与;派发则决定这次拷贝是由普通线程代码实现,还是由 TMA 实现。

## Swizzled 布局

仅搬运分块是不够的。分块还必须以一种 Tensor Core 能够高效读取的布局放入共享内存。

这正是 TMA swizzling 派上用场的地方。当 TMA 把分块写入共享内存时,它可以对共享内存的地址模式进行置换。全局内存中的分块仍是一个逻辑矩形,但共享内存中的目标布局可以被 swizzle。

swizzle 模式是 TMA 描述符的一部分。一旦描述符设置好,发起线程就无需手动应用 swizzle。引擎会在字节落入共享内存时应用它。

关键的要求是一致性。TMA 描述符、共享内存分块布局以及后续的 MMA 指令,三者必须描述同一个布局({ref}`chap_data_layout`)。如果 TMA 用一种 swizzle 写入分块,而 MMA 却按另一种来读取它,硬件仍会一丝不苟地执行被要求做的事——只不过对计算而言,字节的排布将是错误的。

正是在这一点上,布局记号不再只是簿记工具。DSL(领域特定语言)所使用的布局必须与 TMA 描述符和 Tensor Core 指令所使用的硬件布局相匹配。例如,如果 kernel 声明某个操作数分块以 128 字节 swizzled 布局存储,那么 TMA 描述符就必须使用匹配的 swizzle 模式,MMA 派发也必须期待同样的共享内存排布。上面的演示允许你在「无 swizzle」与「128 字节 swizzle」之间切换;将鼠标悬停在某个源元素上,即可看到应用 swizzle 后它落在哪里。

理解 swizzle 的一个有用方式是:TMA 并没有改变逻辑分块,它改变的是逻辑元素在物理上落在共享内存的何处。后续 MMA 仍消费同一个逻辑 A 或 B 分块。swizzle 只决定该分块如何分布在共享内存的各 bank 之间。

## 用于分块化与 Swizzling 的 3D TMA

一次普通的 TMA 拷贝搬运的是扁平的 2D 分块,但 Tensor Core 所期望的共享内存布局通常已被*分块化(tiled)*成 swizzle 原子(即 {ref}`chap_data_layout` 中 8 × 128 字节的原子)。TMA 通过一个额外的描述符维度来处理这一点。一个 **3D TMA** 把共享内存盒子描述为 `(group, row, col)`,其中 group 维度跨原子推进,内部两个维度在单个原子内寻址。于是,单次 3D 拷贝既逐原子地铺排分块(分块化),又在每个原子内部应用 swizzle,使数据抵达时便已处于 MMA 所期望的布局,无需单独的分块化或 swizzling 步骤。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示:一次 3D TMA 拷贝,以 (group, row, col) 寻址,分块化地写入 swizzled 共享内存。*

选择 swizzle *格式*与这种分块化紧密相关。更宽的 swizzle 会把一列打散到更多 bank 上,因此在能够容纳时 128 字节 swizzle 是默认选择;但一个 N 字节原子要求分块的连续维度能够填满它。因此,因形状约束而偏小的分块无法使用 128 字节 swizzle,必须退而使用 64 字节或 32 字节:经验法则是选取分块所能填满的最大 swizzle({ref}`chap_data_layout`)。下面的演示直接展示了这一约束:对 16 × 16 分块应用 128 字节 swizzle,只有在把分块拆分成与原子匹配的 16 × 8 组之后,才能做到无冲突。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*交互演示:对 16 × 16 分块应用 128 字节 swizzle,只有分块化为 16 × 8 组后才能做到无冲突。*

## 完成:加载

拷贝是异步的,因此仅仅发起它是不够的。消费者不能仅仅因为 TMA 指令已发起就去读共享内存中的分块。只有当引擎写完字节之后,分块才是可安全读取的。

对 TMA 加载而言,完成信号是一个 `mbarrier`({ref}`chap_async_barriers`)。

通常的流程是:

1. 为该流水级初始化或复用一个 `mbarrier`;
2. 告知屏障 TMA 传输预期将写入多少字节;
3. 发起 TMA 加载;
4. 让 TMA 引擎随着字节到达而更新屏障;
5. 让消费者在读共享内存分块之前,先在屏障相位(phase)上等待。

字节数通过如下操作来设置:

```text
mbarrier.arrive.expect_tx(bytes)
```

它做两件事:记录预期传输大小,同时执行发起线程对屏障的到达(arrival)。屏障并不会仅仅因为这个调用发生过就完成。它仍要等待 TMA 引擎报告预期的字节已经到达。

随着传输推进,引擎对屏障执行 complete-tx 更新。屏障相位只有在两个条件都满足时才会翻转:到达计数满足,且待处理字节数归零。

随后消费者在该屏障上等待。一旦对预期相位的等待完成,共享内存分块就准备就绪。此时 MMA 路径就可以安全地读取它了。

![TMA 加载同步流程](../img/zh/tma_sync_flow.png)

这与其他异步生产者-消费者交接所使用的屏障模型是同一个。生产者是 TMA 引擎,消费者是 MMA 路径或任何读取该共享内存分块的代码。屏障则是它们之间显式的交接点。

## 完成:存储

TMA 存储按相反方向搬运数据,从共享内存到全局显存。它们同样是异步的,但完成机制不同。

TMA 加载通常喂给同一个 kernel 内部的消费者。MMA 路径需要知道共享内存分块何时就绪,这正是加载路径使用 `mbarrier` 的原因。

TMA 存储通常把最终数据写出至全局显存。往往没有即时的、kernel 内部的消费者在等待被存储的结果。kernel 主要需要知道的是:何时可以安全地复用共享内存缓冲区,或者何时可以结束这串存储。

为此,TMA 存储使用提交组与等待组。kernel 发起一个或多个存储,提交该组,随后等待该组排空。等待完成后,从 kernel 视角看该组内的存储已经完成,被存储所占用的共享内存区域即可安全复用。

所以规则很简单:

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

这两种机制在不同的交接点上服务于同一目的。加载需要让一个共享内存分块对后续消费者可见;存储则需要在 kernel 复用源存储或依赖该存储已排空之前,确保这次外出传输已经完成。

## TMA 对流水化的意义

TMA 在作为流水线一部分时最为有用。一个 kernel 可以在 Tensor Core 计算当前分块的同时,发起对未来分块的加载。加载在后台运行,计算在前台运行。当未来的分块变成当前分块时,屏障把二者衔接起来。

一个典型的 GEMM 循环会反复使用这种结构。共享内存的一段持有当前由 MMA 消费的分块,另一段正由 TMA 填充。随着循环推进,这些角色轮换。在 MMA 读取某一段之前,它先在该段的加载屏障上等待。在 TMA 覆盖某一段之前,kernel 要确保之前的消费者已经用完它。

这就是为什么 TMA 与 `mbarrier` 通常一起出现在 Blackwell 和 Hopper 风格的 kernel 中。TMA 给 kernel 提供了一个异步拷贝引擎,屏障则给 kernel 提供了一种精确获知被拷贝字节何时就绪的方式。
