(chap_warp_spec_debug)=
# 调试线程束特化内核

{ref}`chap_gemm_advanced` 中的 GEMM 步骤 7-9 让 TMA 加载、`tcgen05` MMA 以及 TMEM/SMEM 写回相互重叠。同样的调试方法也适用于 Flash Attention 的交接:先识别角色,再识别每个角色拥有的存储,然后用该模型去核对生成的 CUDA。

不要一上来就重写内核。首先确认这次运行本身是有效的,然后再去检查生成的 CUDA。在排除了环境与编译期问题之后,这些内核的运行时失败通常可以归结为一次失败的交接:一个未初始化的屏障、错误的到达计数、被藏在角色守卫里的集合操作、陈旧的屏障相位,或者存储在生产者让其写入可见之前就被复用了。

## 调试内核之前

先排除运行时环境问题:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些内核面向 Blackwell(`sm_100a`)。如果 Python 导入的是一份陈旧的 TVM 检出版本,或者 GPU 不是 Blackwell 系列,请在改动内核之前先解决这些问题。然后运行该内核最小的正确性检查(例如 `run_correctness()`),再去看性能。

## 调试工作流

1. 在仍然失败的最小形状上复现这次失败。如果失败是一次非法内存访问,在下次运行前请重启 Python。
2. 如果编译失败,在阅读运行时同步代码之前,先检查已安装的 API、target、`dispatch=` 以及 buffer 作用域。
3. 保存 `inspect_source("cuda")` 的输出。在重新阅读 Python 之前,先在其中搜索角色守卫、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的那条内核路径写出角色 / 存储 / 交接 / 生命周期表。
5. 用该表核对生成的 CUDA:屏障初始化是否在角色分支之前、预期的 TMA 生产者、MMA 派发者、写回组,以及是否在仅 warpgroup 的分支内出现了 CTA 级别的集合操作。
6. 把这次运行归类为死锁、崩溃、错误结果或正确但缓慢,然后使用下方对应的章节。
7. 每次只改一处交接:初始化计数、arrive/wait 相位、角色守卫、fence、TMA 存储排空、TMEM 分配/释放,或分块调度器推进。
8. 在测量性能之前先重跑正确性检查。

## 要记录的内容

对任何异步内核,在改动代码之前都先做一张小的工作表:

| 条目 | 要写下的内容 |
|---|---|
| 角色 | 派发每个异步操作的确切线程、warp、warpgroup 或 CTA。 |
| 存储 | 每个分块在每一步的存活位置:GMEM、SMEM、TMEM 或寄存器。 |
| 交接 | 生产者、消费者、信号对象、到达计数、相位,以及让数据可见所用的 fence 或排空。 |
| 生命周期 | 每个存储槽最早可以复用、读回或释放的时刻。 |

然后用这张工作表去核对生成的 CUDA:

- 角色守卫与角色表一致。
- 屏障初始化出现在带守卫的角色分支之前。
- 集合操作没有意外地被 lane、warp 或 warpgroup 守卫收窄。
- arrive/wait 相位与交接表一致。
- TMA 存储排空、TMEM 释放以及 SMEM 复用只发生在生命周期表允许之后。

对 TMA->MMA->写回的 GEMM 流水线,以及 Flash Attention 中 score/softmax/value/correction 的交接,都用同一张工作表。

## 如果编译失败

在调试运行时同步之前,先解决编译期失败:

| 症状 | 可能的方面 | 首先检查 |
|---|---|---|
| 未知的 TIRx API 或属性错误 | 已安装的 wheel 与教程代码不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`;将 API 名与 {ref}`chap_language_reference` 对照。 |
| 不支持的 `dispatch=` | 选定的 target 或原语不支持该路径 | 检查 `dispatch` 参数与 target 能力;本教程中的 `tcgen05` 路径需要 Blackwell。 |
| Buffer 作用域不匹配 | 某个 buffer 被通过错误的硬件路径使用 | 检查工作表的存储行:TMEM 必须通过 `tcgen05` 访问,TMA 操作数必须使用兼容的 GMEM/SMEM 布局。 |
| 编译通过但生成的 CUDA 缺少预期路径 | 派发没有按预期降级 | 在改动算法之前,先在生成的 CUDA 中检查 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成的代码

对任何已编译的内核,都把 CUDA 保存下来,以便搜索和 diff:

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成的代码按如下方式把 TIRx 构造映射到 CUDA:

| TIRx | 生成的 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部守卫 | `((int)threadIdx.x) < 1`(仅 CTA 线程 0) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

在通读整个内核之前,先搜索这些字符串:

| 生成的 CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单 CTA 线程守卫,通常是屏障初始化 |
| `mbarrier_init` | 屏障初始化存在,且出现在角色分支之前 |
| `tcgen05` | 生成了 Tensor Core 路径 |
| `cp.async.bulk.tensor` | 拷贝降级到了 TMA |
| `cta_sync();` | CTA 级屏障;它绝不能出现在 `wg_id` 分支内 |

## 步骤 7 参考骨架

一个正确编译的步骤 7 内核具有如下的顶层形状。下方的守卫为了可读性写成角色名;在生成的 CUDA 中,请搜索上表中对应的表达式。

```c
// (1) 屏障初始化:顶层,仅 CTA 线程 0
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // 由 WG0 的全部 128 个线程 arrive
}

// (2) TMEM 分配:WG0 的 warp 0,派发 warp 的全部 lane
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) fence + cta_sync,然后相位初始化:生产者=1,消费者=0

// (4) 线程束特化循环
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) 清理:派发 warp,无 lane 守卫
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

在改动算法之前,先检查这些项:

- 屏障初始化位于顶层,不在 `wg_id` 守卫内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp 守卫但没有 lane 守卫;派发 warp 的全部 lane 都参与。
- TMA 和 MMA 循环都迭代 `K_TILES` 次。
- 相位初始化为生产者=`1`,消费者=`0`。

## 症状对照表

从症状出发,但把它当成线索而不是最终诊断:

| 线索 | 可能的方面 | 首先检查 |
|---|---|---|
| 内核挂起,随后运行时报未指定的启动失败 | 死锁 | 屏障初始化位置、到达计数、`cta_sync()` 位置以及 `next_tile()` 参与情况 |
| 非法内存访问、XID,或后续无关的 CUDA 调用也失败 | 崩溃 / 上下文被污染 | 重启 Python,然后检查指针范围、存储生命周期和集合操作的参与情况 |
| 在 128 行或分块大小的条纹中出现错误的行 | 同步竞争或分块索引不匹配 | 生产者/消费者相位、调度器推进,以及每个行条纹属于哪个 warpgroup |
| 出现 `NaN` 或明显非法的值 | 描述符、操作数设置,或未初始化的累加 | SMEM/TMEM 描述符设置、swizzle/布局,以及累加器初始化 |
| 有限但呈规律性错误的数据 | 陈旧或仅部分可见的数据 | 缺失 fence、缺失 TMA 存储排空,或存储在生命周期表允许之前就被复用 |
| 输出正确但没有预期的加速 | 派发或资源问题 | 生成的 CUDA 路径、流水线深度、占用率以及寄存器溢出 |

## 何时重启 Python

一次 CUDA 错误并不总会自己清理干净。在非法内存访问、XID 或「CUDA 上下文被污染」错误之后,后续无关的调用(例如 `torch.randn`)可能持续失败。在测试下一个修复之前请重启 Python 进程,否则你可能是在调试上一次崩溃,而不是当前的代码。

## 死锁

按以下顺序检查:

- **到达计数与初始化计数不匹配。** 常见情况:`MBarrier.init(128)`,但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` 守卫,于是只有 1 个线程到达,wait 永不返回。

  | 屏障 | init(count) | 谁到达 | 到达数 |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA 引擎,通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp,通过 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | WG0 的全部线程,通过 `arrive` | 128 |

- **屏障初始化嵌套在 `wg_id` 守卫内。** `.init()` 降级为 `if threadIdx.x < 1:`,即 CTA 线程 0。CTA 线程 0 属于 WG0,所以 `if wg_id == 1:` 会让所有线程都跑不到 init。初始化必须在顶层;在 `inspect_source()` 中 `grep mbarrier_init` 来核实。

- **`cta_sync()` 出现在 warpgroup 分支内。** `cta_sync` 即 `__syncthreads()`,它要求全部 CTA 线程参与。在 `if wg_id == 0:` 内部,WG1 永远到不了它。对单个 warpgroup 的屏障,请用 `T.cuda.warpgroup_sync(10)`。

- **`tile_scheduler.next_tile()` 被某些消费端 warpgroup 的线程跳过。** 调度器维护每线程状态;跳过它的线程可能永远循环下去。

- **TMA 与 MMA 在 K 分块计数上不一致。** 如果 MMA 执行的是 `K_TILES - 1` 而不是 `K_TILES`,屏障相位就会漂移,第二个外层分块就会死锁。

- **`PipelineState` 初始相位错误。** 生产者从 `phase=1` 起步,这样第一次 wait 就能通过;消费者从 `phase=0` 起步,这样第一次 wait 会阻塞。如果两者从同一相位起步,第一次交接可能立刻死锁。

## 崩溃与上下文污染

常见原因:

- **在 `pool.commit()` 之后调用 `pool.alloc()`。** 屏障封装在内部调用了 `alloc`。正确顺序:`tmem_addr -> 屏障封装 -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **`tcgen05.alloc` 或 `tcgen05.dealloc` 带 lane 守卫。** 派发 warp 必须以全部 lane 参与。`if lane_id == 0:` 只让一个线程运行,这是未定义行为。
- **在 `tcgen05.dealloc` 之前缺失 `cta_sync()`。** 在写回仍在读取时 TMEM 就被释放了。
- **GMEM 或 SMEM 越界访问。** 缩小到一个分块,检查调度器的 `m_idx` / `n_idx`,并确认当前形状是内核分块或集群分块的整数倍。

## 错误结果

在猜测之前先按规律对错误输出分类。整行的条纹往往指向生产者/消费者相位、分块索引或角色归属不匹配。`NaN` 输出往往指向描述符设置、操作数设置或未初始化的累加。有限但呈规律性错误的数据往往意味着消费者读到的是旧分块、只写了一部分的分块,或存储尚未排空的数据。

- **`tcgen05.commit` 不在 `elect_sync` 内。** 全部 32 个线程都创建了 commit 组;那 31 个空组会立刻向 mbarrier 发信号。于是 TMA 可能在 MMA 读取 SMEM 之前就覆盖了它。
- **在 TMA 存储之前缺失 `fence.proxy_async("shared::cta")`。** TMA 引擎可能看不到来自各线程的 SMEM 写入。
- **在 TMA 存储之后缺失 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一个分块可能在存储排空之前就复用了 Dsmem。
- **持久化内核在小尺寸(例如 1024x1024)下间歇性失败。** 更大的尺寸可能用更长的 K 循环掩盖了竞争。重新检查分块之间的相位复位,以及 TMA 存储的 commit/wait。
- **`fence.after_thread_sync()` 通常不是解药。** MMA 完成的 mbarrier 已经带有 release-acquire 语义。步骤 8 和 9 出于保守在写回边缘、即 `mma2ld.wait` 之后和第一个 `tcgen05.ld` 之前添加了它;不要在 TMA 到 MMA 的边缘常规性地添加它。

## 正确但缓慢

如果输出正确但性能远低于预期,使用同样的检查循环:

| 线索 | 可能的方面 | 首先检查 |
|---|---|---|
| 生成的 CUDA 没有 `cp.async.bulk.tensor` | 拷贝没有降级到 TMA | 检查 `dispatch="tma"`、target 能力以及操作数布局 |
| 生成的 CUDA 没有 `tcgen05` 路径 | MMA 没有降级到 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、target 能力以及操作数布局 |
| TMA 与 MMA 没有重叠 | 流水线太浅,或相位让生产者/消费者串行化 | 检查生成的 CUDA 中 wait/arrive/advance 的顺序 |
| 小形状正确性良好但大形状速度差 | 寄存器溢出、占用率或暂存缓冲压力 | 检查编译器资源报告;减小分块大小、分块写回或降低流水线深度 |

## 如何提交一个好的 issue

如果这次失败在上述检查之后依然存在,在向 [Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues)提交 issue 之前先把它缩小。请附上:

- `tvm.__file__` / `tvm.__version__` 的输出以及 GPU 能力;
- 复现失败的最小形状;
- 这次失败是编译期、死锁、崩溃、错误结果还是正确但缓慢;
- 最小的内核或 notebook 单元,以及它的正确性检查;
- 保存的 `inspect_source("cuda")` 输出,或能体现可疑守卫、屏障或派发路径的最小摘录。
