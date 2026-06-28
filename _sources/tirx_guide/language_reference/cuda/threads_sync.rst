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

CUDA C++/PTX intrinsics
=======================

当没有分块原语能覆盖你的需求时,有两条逃生路径可直接触达硬件:**调用后端 intrinsic**(来自 ``tvm.backend.cuda`` 的 ``T.cuda.*`` / ``T.ptx.*`` 命名空间),或**内联原始 CUDA** 源码。

调用后端 intrinsic
------------------

``T.cuda.*`` 和 ``T.ptx.*`` 直接暴露 CUDA 后端的设备 intrinsic——同步、mbarrier、归约,以及 PTX 的数据搬移 / MMA 家族:

.. code-block:: python

    T.cuda.cta_sync()                    # block 屏障(__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup 屏障
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block 级归约

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # 用于异步完成的 mbarrier
    T.ptx.mbarrier.try_wait(bar.data, phase)

一个完整、可运行的示例——通过 ``T.tvm_warp_shuffle_xor`` 做一次 warp all-reduce:

.. code-block:: python

    @T.prim_func
    def warp_reduce(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32", align=16)
        T.device_entry()
        cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

该 shuffle 直接降级为 ``__shfl_xor_sync``:

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 下的其他家族:``cp_async``(LDGSTS)、``cp_async.bulk.tensor``(TMA)、``ldmatrix`` / ``stmatrix``、``tcgen05.*``(Blackwell MMA)、``atomic_add``、``fence`` ……完整的 ``tvm.backend.cuda`` 参考见后端 API 文档。

同步语义
--------

在 GEMM 和 Flash Attention 内核中,有四种同步机制会不断出现。因为它们控制的是异步引擎和并行线程组,误用其中任何一个通常都会导致静默的数据损坏或死锁。

**Mbarrier 的相位。** mbarrier 用单个内部相位位来跟踪到达情况。``T.ptx.mbarrier.try_wait(bar, phase)`` intrinsic 会阻塞,直到 barrier 的内部相位与调用者传入的 ``phase`` 参数*不同*。因此,在循环迭代间复用一个 barrier 时,调用者必须在每次等待后翻转自己的本地相位跟踪量(``phase ^= 1``)。否则会让随后的等待立即返回,使引擎读到写了一半的内存。:ref:`chap_gemm_basics` 给出了完整的相位跟踪表。

**选举。** ``T.ptx.elect_sync()`` 选出的是*warp 内单个活跃的 lane*,不是 lane 0,也不是每个 CTA 一个线程。要把发令者收窄到恰好一个线程,你必须把它与一个 warp 级守卫配对使用。``if warp_id == 0:`` 后接 ``if T.ptx.elect_sync():`` 这个模式,在 :ref:`chap_gemm_basics` 中用于发出 ``Tx.gemm_async`` 和 ``tcgen05.commit``。

**具名 warpgroup 屏障。** ``T.cuda.cta_sync()`` 映射为 ``__syncthreads()``,要求*每个* CTA 线程都到达。一旦各 warpgroup 特化到不同代码路径上,把 ``cta_sync()`` 放在一个 warpgroup 分支内就会让内核死锁,因为其他 warpgroup 永远到不了那里。硬件提供 16 个具名屏障(ID 0 到 15);``T.cuda.warpgroup_sync(10)`` 只同步某一个 warpgroup 的线程。不同的 warpgroup 取不同的 ID(例如 ``warpgroup_sync(wg_id + 10)``),这样它们永远不会在同一个硬件屏障上相撞。见 :ref:`chap_gemm_advanced`。

**栅栏(fence)。** fence 用来排序:让生产者的写在消费者(常常是某个异步引擎)读取之前先完成:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - 排序的内容
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - 线程写入的共享内存,先于某个异步代理(TMA store / MMA)读取它
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier 的初始化,先于后续的到达或等待使用该 barrier
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - ``tcgen05`` 写回边上的一道保守排序 fence(第 8、9 步加上了它;在 TMA 到 MMA 的路径上不需要)

内联原始 CUDA
-------------

对那些根本没有 intrinsic 的需求,用 ``T.cuda.func_call(name, *args, source_code=..., return_type=...)`` 从一个源码字符串注入一个 ``__device__`` 函数:

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

源码被原样输出,调用被接上:

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
