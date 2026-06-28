"""ZH version of gen_gemm_perf.py — Simplified-Chinese labels -> img/zh/gemm_perf.png.

Original (English) generator stays untouched in gen_gemm_perf.py.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _zhfont import setup_zh, ZH_OUT
setup_zh()

steps = ['Step 3\n分块同步', 'Step 4\nTMA', 'Step 7\nWarp 特化', 'Step 8\n2-CTA', 'Step 9\n多消费者', 'cuBLAS']
times = [53.642159, 0.493814, 0.226613, 0.103529, 0.094139, 0.094139]
colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff', '#5352ed', '#a0a0a0']

fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

# 纵向对数刻度耗时图。正文表格承载加速比数值;本图聚焦运行时下降。
xpos = list(range(len(steps)))
ax.bar(xpos, times, color=colors, width=0.68)
ax.set_yscale('log')
ax.set_ylim(0.06, 120)
ax.set_xticks(xpos)
ax.set_xticklabels(steps)
ax.set_ylabel('时间 (ms, 对数刻度)')
ax.set_title('GEMM 优化之旅 (M=N=K=4096, fp16, B200)')
ax.grid(axis='y', which='major', linestyle='--', alpha=0.35)

for x, t in enumerate(times):
    time_label = f'{t:.3f} ms' if t < 0.2 else (f'{t:.2f} ms' if t < 10 else f'{t:.1f} ms')
    ax.text(x, t * 1.20, time_label, ha='center', va='bottom', fontsize=9, clip_on=False)

plt.savefig(os.path.join(ZH_OUT, 'gemm_perf.png'), dpi=150, bbox_inches='tight')
print(f'Saved {os.path.join(ZH_OUT, "gemm_perf.png")}')
plt.close()
