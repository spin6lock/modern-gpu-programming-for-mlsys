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

解析器工具
==========

少数几个辅助工具作用于**解析期**(即 TVMScript 被转换为 TIRx 的时刻),让你可以内联 Python 计算出的值、抽离可复用片段,并把解析器侧的状态打包起来。

``T.meta_var`` —— 内联一个 Python 值
-------------------------------------

``T.meta_var(x)`` 告诉解析器把 ``x``——一个在 **Python** 中算出的值——当作一个编译期 *meta* 值,直接内联进 IR,而不是把它当作脚本变量来解析。它避免了那种用完即弃的临时局部变量,并驱动元编程:一个普通的 Python ``for`` 遍历某个 meta 值时会在解析器中展开。

.. code-block:: python

    n = T.meta_var(4)              # n 是一个 Python int,被内联
    for j in range(n):            # 解析期展开
        acc[0] = acc[0] + A[tx, j]

``@T.inline`` —— 内联函数
-------------------------

``@T.inline`` 定义一个函数,其函数体在解析期间**在每个调用点被内联**——生成的代码中不会出现调用。它遵循 Python 的词法(LEGB)作用域与延迟绑定,因此参数会遮蔽外层变量:

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # 内联 -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` —— 解析器侧状态对象
--------------------------------------

``@T.meta_class`` 标记一个普通 Python 类,其**实例就是解析器的 meta 值**:它们的字段可以持有 buffer 和标量,所以你可以把相关的分配与状态打包进一个对象,并在内核体中使用它。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # 像普通 buffer 一样使用它的字段
    # ... s.buf[i] ...

这很适合用来把一个内核的流水线状态(barrier、累加器、临时视图)打包成一组,而不是把许多零散的局部变量穿过函数体。

``T.constexpr``
---------------

``T.constexpr`` 标记一个编译期内核参数,由 ``@T.jit`` 的 ``.specialize(...)`` 烘焙进来。细节见 :ref:`chap_tirx_primer`。
