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

控制流
======

控制流就是 ``if``、循环族,以及 ``while``——每一种都映射到显而易见的 CUDA。

if
--

Python 的 ``if`` / ``else`` 变成 CUDA 的 ``if`` / ``else``。用一个线程 / lane 比较来守卫工作,或者用 ``T.ptx.elect_sync()`` 选出一个单独的发令线程:

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # 单个被选中的 lane(例如用于发出 TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

若要做表达式级的选择(无分支),用 ``T.if_then_else(cond, a, b)``。它降级为一个三元表达式,因此不引入任何控制流发散:

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

一致 vs 发散的控制流
----------------------

像 ``if tx < 128`` 这种逐线程守卫,对普通工作没问题,但**集体** 操作必须被它所要同步的每一个线程*一致地*到达。

例如,``T.cuda.cta_sync()`` 映射为 ``__syncthreads()``,后者要求 thread block 中的所有线程到达。它绝不能位于一个线程或 warpgroup 发散的分支内:若放在 ``if wg_id == 0:`` 之内,其他 warpgroup 永远不会到达,内核将死锁。当只需要一个 warpgroup 同步时,使用 warpgroup 作用域的 ``T.cuda.warpgroup_sync(id)`` (见 :ref:`chap_gemm_advanced` 与 :doc:`threads_sync`)。

同样的谨慎也适用于 barrier 的初始化。一个 ``mbarrier`` 的 ``.init()`` 降级为单线程守卫(``if (threadIdx.x < 1)``)。把它嵌套在另一个发散分支内,可能让 barrier 处于未初始化状态,导致未指明的启动失败。

loop
----

循环有四种形式;普通的 Python ``range`` 变成 ``T.serial``:

- ``T.serial(n)`` —— 顺序循环(ptxas 仍可能展开它)。
- ``T.unroll(n)`` —— 完全展开(扩展为直线语句)。
- ``T.vectorized(n)`` —— 向量化循环。
- ``T.grid(*extents)`` —— 嵌套循环族。

``break`` / ``continue`` 在循环内可用。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 则展开为四条没有循环的直线语句。

while
-----

``while`` 循环一直运行到其条件为假为止。使用一个可变标量计数器(见 :doc:`buffers`):

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它降级为带提前退出 ``break`` 的 ``while (1)`` (计数器是一个单元素寄存器 buffer):

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
