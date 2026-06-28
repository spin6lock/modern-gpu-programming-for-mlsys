..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Buffer 与内存
=============

参数 buffer 用 ``T.match_buffer`` 绑定;临时 buffer 用下面两种声明 API 之一在函数体内创建。用 ``A[i, j]`` 索引 buffer,用 ``A[m0:m0+BM, 0:BK]`` 对其切片(得到一个 ``BufferRegion``),用 ``A.ptr_to([i, j])`` 取指针,或用原始数据指针 ``A.data``。

声明 buffer
-----------

两种基本 API 用于创建 buffer:

- ``T.alloc_buffer(shape, dtype, scope=..., ...)`` —— **分配新存储** (生成一个 ``AllocBuffer`` 节点)并返回该 ``Buffer``。``T.alloc_shared`` / ``T.alloc_local`` 只是分别带 ``scope="shared"`` / ``scope="local"`` 的 ``alloc_buffer``。
- ``T.decl_buffer(shape, dtype, data=..., ...)`` —— 在一个已有指针 ``data`` 上**声明一个视图** (不分配存储);用于给存储起别名或重新解释——可以是池的一个子区域,或一个张量内存(tensor memory)地址。当 ``data=None`` 时它会像 ``alloc_buffer`` 一样分配存储。

buffer 的 ``data`` 指针是一个不可变 ``Var`` (``alloc_buffer`` 定义它;``decl_buffer`` 接收一个)。若要用一个指针*表达式*来支撑一个 buffer,需先绑定它——见 :doc:`data_types`。

二者共享同一个描述符;最关键的参数如下:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - 含义
   * - ``dtype``
     - 元素类型 —— ``"float32"``、``"float16"``、``"float4_e2m1fn"``、……
   * - ``shape``
     - 逻辑形状(由各维 extent 组成的元组)
   * - ``layout``
     - 物理映射(:ref:`TileLayout <chap_tirx_layout_api>`);``"default"`` = 稠密行主序
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset`` (或 ``byte_offset``)把一个*视图*放在 ``data`` 内的某个偏移处;``allocated_addr`` 携带一个预分配地址(张量内存)
   * - ``align``
     - 数据指针对齐,以字节为单位

``scope`` 参数选择内存空间:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - 简写
     - 内存
   * - ``"global"``
     - (默认)
     - 设备全局内存
   * - ``"shared"``
     - ``T.alloc_shared``
     - 静态共享内存(``__shared__``)
   * - ``"shared.dyn"``
     - (池)
     - 动态共享内存(池化——见下文)
   * - ``"local"``
     - ``T.alloc_local``
     - 每线程寄存器
   * - ``"tmem"``
     - (TMEM 池)
     - Blackwell 张量内存(见下文)

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # 参数 buffer
    As = T.alloc_shared((BM, BK), "float16")                 # 新的共享分块
    acc = T.alloc_local((4,), "float32")                     # 寄存器累加器
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # As 上的一个视图

**基于指针的 buffer 只是指针之上的元数据。** 对任何非 tmem 的 buffer,其声明就是「一个指针 + 一个布局」,而索引解析为一个地址::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

(``layout.apply`` 返回逐轴映射;其中的 ``"m"`` 分量是元素偏移。)因此,*同一个*逻辑访问会纯粹依据 buffer 的元数据被编译成不同的地址算术。在一个 4×8 区域上写 ``B[i, j] = A[i, j] + 1``,把 ``B`` 以四种方式声明:

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # 行主序
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # 列主序
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # 平移视图
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # 行步长 16

每一种都让 ``B[i, j]`` 在生成的 CUDA 中降级(lowering)为不同的索引(``A[i, j]`` 的加载保持 ``i*8 + j`` 不变——只有 ``B`` 的元数据变了):

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // 行主序:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // 列主序:        j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64: i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // 行步长 16:      i*16 + j

共享内存
--------

共享内存有两种形式——**静态** (编译期固定)与**动态** (启动时确定大小)——外加一个管理动态情形的池辅助工具。

静态
~~~~

最简单的共享 buffer 是**静态** 的——``T.alloc_shared`` (即 ``scope="shared"``),在编译期确定大小。把数据搬进来,``cta_sync`` 让整个 block 都看到这些写入,然后再读回:

.. code-block:: python

    @T.prim_func
    def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        B = T.match_buffer(B_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128])
        sm = T.alloc_shared((128,), "float32")   # 静态共享内存
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2.0)

它降级为一个普通的 ``__shared__`` 数组(生成的 CUDA,样板已省略):

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

动态
~~~~

**动态** 共享内存(``scope="shared.dyn"``)按启动确定大小(即 ``sharedMemBytes`` 启动参数),而非编译期。一个内核**只能有唯一一个** 动态共享分配——即那个 *arena*。所以你只需分配它一次,然后把每个 buffer 作为一个视图 ``decl`` 进去:用 ``T.decl_buffer``,``data=`` 指向 arena 指针,并带上一个 ``elem_offset``:

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # 唯一的那个 arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # 偏移 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # 偏移 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个视图共享同一个 ``extern __shared__`` arena(生成的 CUDA,样板已省略;为清晰起见把 arena 命名为 ``smem``):

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // 唯一的那个动态共享 arena
    smem[tx]      = A_ptr[tx];                       // As —— 偏移 0 处的视图
    smem[tx + 64] = B_ptr[tx];                       // Bs —— 偏移 64 处的视图
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

(两次单独的 ``alloc_buffer(scope="shared.dyn")`` 调用是错误的——*只允许一次动态共享内存分配*。)所以静态共享内存在编译期确定大小(``__shared__ T x[N];``);动态共享内存则是这个唯一的、按启动确定大小的 arena,视图在它内部的各偏移处被声明。

.. note::

   **TVM 如何标注动态共享大小。** arena 的大小在编译期已知(此处是 ``128`` 个 float = ``512`` 字节)。在降级过程中,TVM 会给设备内核的 ``tirx.kernel_launch_params`` 追加一个 ``"tirx.use_dyn_shared_memory"`` 标签,而 host 端启动器会算出总字节数,把它作为最后一个启动参数传入:

   .. code-block:: python

       # 设备内核属性:
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host 端启动调用  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   运行时,那个 ``512`` 会成为 ``cuLaunchKernelEx`` 调用中的 ``config.sharedMemBytes``。你永远不需要手动设置它——它是从 ``shared.dyn`` 分配的大小推导出来的。

池语法糖
~~~~~~~~~~

``T.SMEMPool`` 自动完成这套 arena 簿记——它会用游标分配各偏移,这样你就不必手动 ``decl`` 视图。除了 ``alloc`` / ``commit`` 之外,它还提供逐 buffer 的 ``align=``、一个能为你构造与 MMA 兼容的 swizzle 布局的 ``alloc_mma`` 辅助工具,以及一个把游标倒回以复用空间的 ``move_base_to``:

.. code-block:: python

    pool = T.SMEMPool()                          # shared.dyn 上的游标分配器
    As = pool.alloc((BM, BK), "float16", align=128)   # 切出一个分块
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # 与 MMA 兼容,swizzle 自动推断
    pool.commit()                                 # 定稿池的大小
    # pool.move_base_to(offset) 把游标倒回以复用空间

下文的 TMEM 池(tensor memory)就建立在 ``SMEMPool`` 之上。

寄存器
------

每线程的临时数据存放在寄存器里。用 ``T.alloc_local(shape, dtype)`` (即 ``scope="local"``)分配:它对每个线程私有,降级为一个保存在寄存器中的本地数组。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # 每线程寄存器数组
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... 在 r[0..3] 上计算 ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // 每线程,驻留寄存器
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   这个 ``alignas(64)`` 是*默认*的 buffer 对齐——一个 buffer 的 ``data_alignment`` 默认为 ``runtime::kAllocAlignment`` (64 字节),CUDA codegen 会把它应用到每个分配上,包括那些对齐毫无意义的逐线程 ``local`` 数组。对这类驻留寄存器的数组,它**没有任何性能影响**:一个索引可在编译期解析的线程本地数组,会被 nvcc/ptxas 提升为寄存器(聚合体的标量替换,SROA),因此它从不落入可寻址的本地内存,这个对齐就是个空操作。(只有被动态索引、溢出到本地内存的数组,才会真正受这个过度对齐影响,但那是少见情形。)寄存器本地变量的这种过度对齐是一个已知的粗糙之处,我们计划修复(对 ``local`` 作用域改用 dtype 的自然对齐)。

标量
~~~~

标量就是**单元素** 的寄存器数组——严格说,你并不需要一个单独的概念。你可以分配一个大小为 1 的 ``local`` buffer 并索引 ``[0]``:

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 单元素寄存器数组
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

但到处写 ``phase[0]`` 很笨拙,所以**标量** 正是为此而生的语法糖——一个单元素寄存器 buffer,你**按名字** 读写它:

.. code-block:: python

    phase: T.int32 = 0                 # 可变标量(上面写法的语法糖)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # 显式形式;按名字赋值(s = ...,而非 s[0])
    acc: T.float32 = 0.0               # 一个带类型标注的赋值也会生成一个标量

二者不只是相似——它们被解析为**结构上完全相同的 TIRx**。语法糖完全在解析器中消解:``phase: T.int32`` *就是*那个单元素 ``local`` buffer,而 ``phase`` / ``phase += 1`` *就是* ``phase[0]`` / ``phase[0] += 1``。对两个内核跑 ``tvm.ir.assert_structural_equal`` 会通过,而且 printer 甚至会把显式的 ``alloc_local`` + ``[0]`` 形式**反过来** 渲染成标量形式——所以一旦解析完成,二者毫无差别。因此二者都降级为同一个 ``alignas(64) int phase_ptr[1];``;标量只是让你省掉 ``[0]``。(``T.local_scalar`` / ``T.shared_scalar`` / ``T.alloc_scalar`` 显式选择作用域。)

.. note::

   **为什么不是** ``Var`` \ **?** TIRx 的 ``Var`` 是*不可变*的——一个静态绑定(它正是下文 ``T.let`` 所产生的)。而标量需要是*可变*的——你会在循环和累加器中重新赋值给它——所以它必须由一个可反复写入的单元素 buffer 支撑,而不是一个 ``Var``。

``let``
~~~~~~~

一个 ``T.let`` 绑定是**不可变** 的——单个 ``LetStmt`` (一个具名值,而非 buffer)。用它来表达派生常量:

.. code-block:: python

    n: T.let = M * K               # 不可变绑定(LetStmt)
    half: T.let[T.int32] = N // 2  # ……带一个显式类型

它降级为一个**普通的标量 C 变量**——不是 buffer(没有数组,没有 ``[0]``)。对 ``half: T.let = m * 2`` (其中 ``m`` 是运行时值):

.. code-block:: c++

    int half = m * 2;     // 这个 `let` -> 一个类 const 的局部变量

因为该值不可变,简化器可以自由地传播它并对其做 CSE,所以在使用点你常常直接看到 ``m * 2`` 被替换进去(或通过一个公共子表达式临时量共享),而不是对 ``half`` 的引用。

.. note::

   **为什么要有不可变绑定?** 因为该值不会改变,算术分析器在简化一个 ``LetStmt`` 时会把它绑定到该 var(``analyzer.Bind(var, value)``),于是关于该值所证明的事实——常量边界、模集合(整除性 / 对齐)、范围——**会传播到每一处使用**。这会提供给索引简化、边界检查消除,以及对齐 / 向量化决策。而*可变*标量是一次内存加载(``buf[0]``):分析器不能假设它保持恒定,所以那些性质都无法传递。``let`` 也是一个纯值——不分配存储,可自由内联 / 替换 / CSE——而标量是一个带加载 / 存储语义的单元素 buffer。

张量内存
--------

Blackwell 的*张量内存*不是一个普通的临时作用域:它必须用 warp 一致的 ``T.ptx.tcgen05.alloc`` / ``tcgen05.dealloc`` 内显式预约和释放,每个张量都是其中用 ``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)`` 声明的一个视图。``allocated_addr`` (一个列偏移)是必填的——tensor-core 派发会断言它——所以 ``T.alloc_buffer(scope="tmem")`` (它**不** 设置该地址)行不通。与共享内存不同,张量内存不可直接寻址:它只能通过 ``tcgen05`` 的 ``mma`` / ``ld`` / ``st`` / ``cp`` 来读写。

手动做法是:一个 warp 把分配发到一个共享槽位,你把每个张量在某列偏移处 ``decl`` 为视图,最后一个 warp 在结尾释放它:

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # 存放已分配基址的槽位
    if warp_id == alloc_warp:                         # tcgen05.alloc 是 warp 一致的
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # 列 0 处的视图
    # ... 把 acc 用作 gemm_async / copy_async 的操作数 ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)

列偏移和 ``tmem_layout`` (一个数据通路 D/F 布局)由你自己管理。这正是下面那个池所生成的序列。

池
~~

``T.TMEMPool`` 把以上全部封装起来——warp 一致的 alloc/dealloc、列的 bump 分配,以及数据通路布局:

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = 内核的 smem 池
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # 为你设好 allocated_addr
    tmem_pool.commit()                               # 发出 tcgen05.alloc(由一个 warp)
    # ... 使用 acc ...
    tmem_pool.dealloc()                              # 发出 tcgen05.dealloc(由一个 warp)

完整示例见第三部分的 GEMM 内核。

Buffer API
----------

一个 ``Buffer`` 是指针之上的元数据(见上文*声明 buffer*),所以它的大多数方法都是*编译期*的 reshape / 重解释,改的是索引算术或给你一个指针——它们本身不发出任何运行时操作。常用方法:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - 是什么
   * - ``B.data``
     - 原始数据指针(一个 ``Var``);打印为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 指向某个元素的有类型指针(``address_of``);打印为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - 向量化加载 / 存储;打印为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 在新形状 / 布局下重新解释同一份存储(不拷贝)
   * - ``B.local(*shape, layout=…)``
     - 调用线程对 ``local`` buffer 拥有的私有寄存器切片
   * - ``B.permute(*dims)``
     - 轴被置换后的视图(一个转置后的布局)
   * - ``B.access_ptr(mask, …)``
     - 一个带掩码的访问指针(``tvm_access_ptr`` 内建),用于把一个区域传给某个 intrinsic

**指针 —— ``ptr_to`` / ``data``。** ``ptr_to`` 是你把一个元素地址交给某个 intrinsic 或内联函数的方式;``data`` 是基地址指针:

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

**向量化访问 —— ``vload`` / ``vstore``。** 把多个元素作为一次宽传输移动(另见 :doc:`data_types`):

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**reshape / 重解释 —— ``view`` / ``permute``。** 二者都是纯元数据;数据指针不变,只是索引算术不同。``A.view(64, 4)`` 把这个 256 元素的 buffer 看作 ``64×4``;``A.permute(1, 0)`` 转置各轴:

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j] -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view:行主序 64x4 索引
    At_ptr[(j * 4) + i]                       // permute:步长交换

**寄存器 —— ``local``。** 把一个线程轴的 ``local`` 布局分解为调用线程的平坦寄存器束(被分块原语广泛使用):

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # 当前 lane 的 8 个寄存器

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             // 该 lane 的私有寄存器
