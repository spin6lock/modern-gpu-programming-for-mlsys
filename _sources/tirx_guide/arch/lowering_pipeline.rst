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

TIRx 降级流水线
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 会将一个手写的 TIRx 模块送入
**tirx pipeline**——一条有序的 TIR pass 序列，它把你编写的高层构造（tile 原语、
``TileLayout`` 类型的 buffer、执行作用域 id）转换成拆分后的 **host** + **device** 函数，再由
CUDA 后端渲染成源代码。该流水线定义于
``python/tvm/tirx/compilation_pipeline.py`` (``tirx_pipeline`` ）；本页按顺序逐个介绍这些 pass。

它的位置
-------------

``tvm.compile`` 先绑定 target，运行 **tirx pipeline** (下面的模块级 pass），然后分别对 host 和
device 函数应用 **finalization** (收尾）pass，最后把每个 device 函数交给 CUDA 代码生成器：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

各 pass
----------

``tirx_pipeline`` 模块 pass 按如下确切顺序执行（少数几个受 ``PassContext`` 配置控制）：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - 核心降级——见下文 Inside LowerTIRx
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的线程轴绑定，使得每个 ``threadIdx`` / ``blockIdx``
       轴只声明一次
   * - 3
     - ``StmtSimplify``
     - 语句级算术化简（arith analyzer）
   * - 4
     - ``LowerTIRxOpaque``
     - 把剩余的不透明 TIRx 构造降级为普通 TIR
   * - 5
     - ``FlattenBuffer``
     - 把多维 ``BufferLoad`` / ``BufferStore`` 拍平为一维
   * - 6
     - ``BF16ComputeLegalize``
     - 把 ``bfloat16`` 计算改写为合法形式（上转为 f32）
   * - 7
     - ``NarrowDataType(32)``
     - 在可证明安全的前提下，把 index/loop ``PrimExpr`` 的 dtype 收窄为 32 位
   * - 8
     - ``VectorizeLoop``
     - 把 ``T.vectorized`` 循环转成向量操作（若
       ``tir.disable_vectorize`` 则跳过）
   * - 9
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的循环（以及小的常量循环）
   * - 10
     - ``StmtSimplify``
     - 再次化简，因为 vectorize/unroll 暴露出了常量
   * - 11
     - ``CommonSubexprElim``
     - 把重复出现的子表达式提取为临时变量（若
       ``tir.disable_cse_tir`` 则跳过）
   * - 12
     - ``FP8ComputeLegalize``
     - 把 ``float8`` 计算改写为合法形式
   * - 13
     - ``VerifyMemory``
     - 检查 host 侧代码不会直接解引用 device 内存（一道安全闸门）
   * - 14
     - ``AnnotateEntryFunc``
     - 把唯一的 PrimFunc 标记为模块入口
   * - 15
     - ``SplitHostDevice``
     - 在 ``launch_thread`` 边界处把每个内核拆成一个 **host** 函数和一个
       **device** 函数
   * - 16
     - ``MakePackedAPI``
     - 把 host 函数改写为 packed-func ABI（TVM 调用的启动器）
   * - 17
     - ``FP8StorageLegalize``
     - 合法化 ``float8`` 存储（打包进受支持的容器类型）
   * - 18
     - ``BF16StorageLegalize``
     - 合法化 ``bfloat16`` 存储

随后 **Finalization** (收尾）按函数类型分别运行：

- **host**：``LowerTVMBuiltin`` (降级 ``tvm_*`` builtins）、``LowerIntrin``
  （target 相关的 intrinsics）
- **device**：``LowerWarpMemory`` (warp 作用域 buffer → shuffles）、``StmtSimplify``、
  ``LowerIntrin``

LowerTIRx 内部
----------------

``LowerTIRx`` 本身是一小段序列（``src/tirx/transform/lower_tirx.cc`` ）：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 把每个 ``TilePrimitiveCall`` (``copy``、
  ``gemm``、``reduction``、……）替换成由其选中的后端派发所生成的函数体——即它的变体选择与代码生成。
- **``LowerTIRxCleanup``** 运行 ``LayoutApplier``：它把每个
  ``TileLayout`` 类型的 buffer 访问解析为具体的物理地址算术
  （``addr = data + elem_offset + layout.apply(coord)`` ），拍平 buffer，并
  降级执行作用域 id（``T.cta_id`` / ``T.thread_id`` / …… 经由
  ``launch_thread`` 变成 ``blockIdx`` / ``threadIdx`` ）。

因此在 ``LowerTIRx`` 之后，模块就是普通 TIR：没有 tile 原语，没有
``TileLayout`` 间接寻址，作用域 id 已解析为线程轴。

一个完整示例
----------------

取一个一行的 scale 内核：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**``LowerTIRx`` 之后**，作用域 id 已是真实的线程轴，layout 也已应用
（``A_1`` / ``B_1`` 是拍平后的一维视图）：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**``SplitHostDevice`` + ``MakePackedAPI`` 之后** ，原来这一个函数变成了两个——
一个 host 启动器和一个 device 内核：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host:packed-API 启动器(计算 grid/block 并启动)
            ...
        def scale_kernel(...):  # device:__global__ 函数体,在 GPU 上运行

随后 CUDA 后端把 ``scale_kernel`` 渲染成 ``__global__`` 函数
（``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f`` ）。

自己复现一遍
---------------------

你可以手工运行流水线的任意前缀来检视某个阶段——本文档各处的
IR 片段就是这样得到的：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile 原语已派发，layout 已应用
    print(mod.script())               # 检视降级后的 TIRx IR

或者编译整个模块，再读出生成的 CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
