"""ZH version of gen_roofline.py — Simplified-Chinese labels -> img/zh/roofline.png.

Original (English) generator stays untouched in gen_roofline.py.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _zhfont import setup_zh, ZH_OUT
setup_zh()

PEAK_TFLOPS = 2000.0      # ~2 PFLOP/s dense fp16 tensor core (order of magnitude)
BW_TB_S = 8.0             # HBM3e, TB/s  (==> attainable = 8 * AI  TFLOP/s)
RIDGE = PEAK_TFLOPS / BW_TB_S   # ~281 FLOP/byte

ai = np.logspace(-1, 4.3, 500)            # arithmetic intensity, FLOP/byte
roof = np.minimum(PEAK_TFLOPS, BW_TB_S * ai)

fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
ax.plot(ai, roof, color='#222', lw=2.2, zorder=3)
ax.axhline(PEAK_TFLOPS, color='#888', ls='--', lw=1, alpha=0.6)
ax.axvline(RIDGE, color='#888', ls=':', lw=1, alpha=0.6)
ax.text(RIDGE * 1.1, 3.5, f'脊点 ≈ {RIDGE:.0f} FLOP/byte', color='#555', fontsize=8.5, rotation=90, va='bottom')
ax.text(0.12, PEAK_TFLOPS * 1.07, f'算力屋顶 ≈ {PEAK_TFLOPS/1000:.0f} PFLOP/s (fp16)', color='#555', fontsize=8.5)
ax.text(0.13, 8 * 0.13 * 1.15, f'带宽屋顶: {BW_TB_S:.0f} TB/s', color='#555', fontsize=8.5, rotation=34)

# 示例工作负载: (标签, 算术强度, 实测 TFLOP/s, 颜色, 偏移, 对齐)
pts = [
    ('逐元素 / RMSNorm\n(访存受限)', 0.4, 8 * 0.4 * 0.7, '#ff6b6b', (14, 10), 'left'),
    ('GEMM 4096³ — 朴素\n(SM 空闲)', 1365, 2.9, '#ffa502', (12, -2), 'left'),
    ('GEMM 4096³ — SOTA\n(约峰值的 2/3)', 1365, 1320, '#2ed573', (-8, 10), 'right'),
]
for label, x, y, c, xytext, ha in pts:
    ax.scatter([x], [y], s=70, color=c, zorder=5, edgecolor='white', linewidth=0.8)
    ax.annotate(label, (x, y), textcoords='offset points',
                xytext=xytext, ha=ha, fontsize=8.5, color='#333')
# GEMM 的优化差距箭头
ax.annotate('', xy=(1365, 1320), xytext=(1365, 2.9),
            arrowprops=dict(arrowstyle='->', color='#2ed573', lw=1.6, alpha=0.8))
ax.text(1365 * 0.62, 70, '优化\n在此爬升', color='#2ed573', fontsize=8.5, ha='right')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.1, 2e4); ax.set_ylim(2, 4000)
ax.set_xlabel('算术强度 (FLOP / byte)')
ax.set_ylabel('可达性能 (TFLOP/s)')
ax.set_title('Roofline (约 B200)——工作负载所在，以及优化能换来什么')
ax.grid(which='both', ls='--', alpha=0.25)
plt.savefig(os.path.join(ZH_OUT, 'roofline.png'), dpi=150, bbox_inches='tight')
print(f'Saved {os.path.join(ZH_OUT, "roofline.png")}')
plt.close()
