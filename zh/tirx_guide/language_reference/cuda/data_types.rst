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

数据类型与表达式
================

每个 TIRx 表达式都携带一个底层 **dtype** 和一个高层 **type**。

表达式 dtype
------------

一个 ``PrimExpr`` 的 ``.dtype`` 是它的标量(或向量)元素类型——``float32``、``float16``、``bfloat16``、``int32``、``uint8``、``bool``、低精度的 ``float8_e4m3fn`` / ``float4_e2m1fn`` ……、``handle``(指针),以及诸如 ``float32x4`` 的向量形式。每种都打印为对应的 CUDA 类型。跨若干 dtype 分配 local 和 shared buffer,外加一次向量化的 ``float32x4`` 加载 / 存储:

.. code-block:: python

    @T.prim_func
    def dtypes(A_ptr: T.handle, O_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        O = T.match_buffer(O_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
        f16  = T.alloc_local((1,), "float16")        # 寄存器标量 ......
        bf16 = T.alloc_local((1,), "bfloat16")
        i32  = T.alloc_local((1,), "int32")
        u8   = T.alloc_local((1,), "uint8")
        b1   = T.alloc_local((1,), "bool")
        sm   = T.alloc_shared((64,), "float16")      # ......以及一个 shared 分块
        v    = T.alloc_local((1,), "float32x4")      # 一个向量 dtype 寄存器(float4)
        v[0] = A.vload([tx * 4], dtype="float32x4")  # 向量化加载
        O.vstore([tx * 4], v[0])                     # 向量化存储
        # ... (使用 f16/bf16/i32/u8/b1/sm) ...

降级为(生成的 CUDA,已省略):

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    signed char   b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (向量)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // 向量化加载
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // 向量化存储

一个 buffer 的 dtype 本身就可以是**向量类型**:``T.alloc_local((1,), "float32x4")`` 直接声明一个 ``float4`` 寄存器(你以 ``v[0]`` 索引它),而一次 ``float32x4`` 的 ``vload`` / ``vstore`` 随后把它作为一次 16 字节访问来移动。向量 dtype 并不与 ``vload`` 绑定——任何 buffer 或标量都可以携带它。

所以 dtype → CUDA 的映射是:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - dtype → CUDA
     - dtype → CUDA
     - dtype → CUDA
   * - ``float32`` → ``float``
     - ``float16`` → ``half``
     - ``bfloat16`` → ``nv_bfloat16``
   * - ``int32`` → ``int``
     - ``uint8`` → ``uchar``
     - ``bool`` → ``signed char``
   * - ``float32x4`` → ``float4``
     - ``handle`` → ``T*`` (指针)
     - (向量 dtype → CUDA 向量类型)

dtype vs type
-------------

``dtype`` 是*底层*的——它说的是「什么比特」。另外,一个值还有一个高层 **type**:标量是 ``PrimType(dtype)``,指针是 ``PointerType(PrimType(dtype), scope)``。大多数表达式都是标量(``PrimType``);类型系统主要对**指针**有意义。

指针(``handle``)
-----------------

一个 buffer 的 ``data``——即它的指针——是一个指针类型的 ``Var``,且它是**不可变**的(指针不会被重新赋值)。这决定了你如何获得一个指针:

- ``T.alloc_buffer(...)`` 分配存储**并**定义它的 ``data`` 指针。
- ``T.decl_buffer(..., data=ptr)`` 在一个已有指针 ``Var`` ``ptr`` 之上声明一个 buffer。
- 若要用一个指针**表达式**来支撑一个 buffer——例如 ``T.ptx.map_shared_rank``(PTX ``mapa``)给出另一个集群 CTA 的共享地址——你必须先把该表达式绑定到一个指针 ``Var``(``data`` 必须是一个 ``Var``,不能是表达式),使用一个 ``PointerType`` 的 ``T.let``:

  .. code-block:: python

      from tvm.ir.type import PointerType, PrimType

      ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
          T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
