"""ZH version of gen_tma_sync_flow.py — Simplified-Chinese labels.

Output: img/zh/tma_sync_flow.png
Original (English) generator stays untouched in gen_tma_sync_flow.py.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _zhfont import setup_zh, ZH_OUT
setup_zh()

COLORS = {
    "thread": "#f8fafc",
    "tma": "#bfdbfe",
    "tma_edge": "#2563eb",
    "barrier": "#fde68a",
    "barrier_edge": "#d97706",
    "mma": "#bbf7d0",
    "mma_edge": "#059669",
    "neutral": "#475569",
}

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

lanes = {
    '当选\n线程': (2.5, COLORS["thread"]),
    'TMA\n硬件': (6.0, COLORS["tma"]),
    'mbarrier': (9.5, COLORS["barrier"]),
    'tcgen05\nMMA': (13.0, COLORS["mma"]),
}

for label, (x, color) in lanes.items():
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.9, -0.1), 1.8, 0.7, boxstyle='round,pad=0.05',
                 fc=color, ec='black', lw=2))
    ax.text(x, 0.25, label, ha='center', va='center', fontsize=11, fontweight='bold')

for label, (x, color) in lanes.items():
    ax.plot([x, x], [0.8, 11.5], color='#cccccc', lw=1.5, linestyle=':')

def action_arrow(y, x_from, x_to, label, color='#333333', lw=2):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    mx = (x_from + x_to) / 2
    ax.text(mx, y - 0.25, label, ha='center', va='center', fontsize=9, fontweight='bold',
            color=color, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

def wait_bar(y, x, label, color='#e74c3c'):
    ax.plot([x - 0.3, x + 0.3], [y, y], color=color, lw=4, solid_capstyle='round')
    ax.text(x, y + 0.2, label, ha='center', va='bottom', fontsize=8, color=color, style='italic')

def step_label(y, num, text):
    ax.text(-1.2, y, '%d' % num, ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', bbox=dict(boxstyle='circle,pad=0.3', fc='#4a90d9', ec='black', lw=1.5))
    ax.text(-0.5, y, text, ha='left', va='center', fontsize=8, color='#555')

et_x = 2.5
tma_x = 6.0
bar_x = 9.5
mma_x = 13.0

y = 1.5
step_label(y, 1, '发射 TMA')
action_arrow(y, et_x, tma_x, 'copy_async(A)', COLORS["tma_edge"])

y = 2.3
action_arrow(y, et_x, tma_x, 'copy_async(B)', COLORS["tma_edge"])

y = 3.2
step_label(y, 2, '设置字节数')
action_arrow(y, et_x, bar_x, 'arrive.expect_tx(bytes)', COLORS["barrier_edge"])

y = 4.3
step_label(y, 3, '硬件搬运')
ax.add_patch(mpatches.FancyBboxPatch((tma_x - 0.55, 3.6), 1.1, 1.0, boxstyle='round,pad=0.05',
             fc=COLORS["tma"], ec=COLORS["tma_edge"], lw=1.5, alpha=0.75))
ax.text(tma_x, 4.1, 'TMA 搬运\n(GMEM->SMEM)', ha='center', va='center', fontsize=7, color=COLORS["tma_edge"], style='italic')

y = 4.8
action_arrow(y, tma_x, bar_x, 'arrive (自动)', COLORS["barrier_edge"])

y = 5.8
step_label(y, 4, '等待数据')
action_arrow(y, et_x, bar_x, 'try_wait(phase)', COLORS["barrier_edge"])
wait_bar(5.8, et_x, '阻塞', COLORS["barrier_edge"])

y = 6.6
action_arrow(y, bar_x, et_x, 'phase 完成!', COLORS["barrier_edge"])

y = 7.8
step_label(y, 5, '发射 MMA')
action_arrow(y, et_x, mma_x, 'gemm_async + commit', COLORS["mma_edge"])

ax.add_patch(mpatches.FancyBboxPatch((mma_x - 0.55, 8.2), 1.1, 1.0, boxstyle='round,pad=0.05',
             fc=COLORS["mma"], ec=COLORS["mma_edge"], lw=1.5, alpha=0.75))
ax.text(mma_x, 8.7, 'MMA 计算\n(SMEM->TMEM)', ha='center', va='center', fontsize=7, color=COLORS["mma_edge"], style='italic')

ax.text(et_x, 9.6,
        'mbarrier 的 try_wait\n已携带 release->acquire 边:\n在此 TMA->MMA 路径上\n无需额外 fence。',
        ha='center', va='center', fontsize=7, color='#555', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#cccccc', lw=1))

ax.set_xlim(-2.0, 15.0)
ax.set_ylim(-0.5, 11.0)
ax.invert_yaxis()
ax.set_title('TMA 异步加载:同步流程', fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(os.path.join(ZH_OUT, 'tma_sync_flow.png'), dpi=150, bbox_inches='tight')
print(f'Saved {os.path.join(ZH_OUT, "tma_sync_flow.png")}')
plt.close()
