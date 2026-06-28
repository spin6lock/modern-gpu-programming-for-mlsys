"""ZH reconstruction of img/pipe_depth2.png — Simplified-Chinese labels -> img/zh/pipe_depth2.png.

The checked-in ``img/pipe_depth2.png`` has no original generator script; this
reproduces it from the figure's content (PIPE_DEPTH=2 software-pipeline sketch:
one TMA lane of loads leading one MMA lane of computes by 2 stages).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _zhfont import setup_zh, ZH_OUT
setup_zh()

BOX_COLOR = "#6c5ce7"   # medium purple, matching the original
EDGE = "#3b2fa3"
TMA_Y = 1.5
MMA_Y = 0.5
BOX_H = 0.72


def box(ax, x0, x1, y, text):
    ax.add_patch(FancyBboxPatch(
        (x0, y - BOX_H / 2), x1 - x0, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, facecolor=BOX_COLOR, edgecolor=EDGE, zorder=3))
    ax.text((x0 + x1) / 2, y, text, ha="center", va="center",
            fontsize=12, weight="bold", color="white", zorder=4)


fig, ax = plt.subplots(figsize=(10, 2.2), dpi=200)
ax.set_xlim(-1.3, 8.4)
ax.set_ylim(0, 2.4)
ax.axis("off")

ax.text(3.5, 2.15, "PIPE_DEPTH=2", ha="center", va="center",
        fontsize=15, weight="bold")

# 泳道标签
ax.text(-0.4, TMA_Y, "TMA", ha="right", va="center", fontsize=12, weight="bold")
ax.text(-0.4, MMA_Y, "MMA", ha="right", va="center", fontsize=12, weight="bold")

# 时间轴网格 0..8
for t in range(0, 9):
    ax.plot([t, t], [0.1, 2.05], color="#cccccc", lw=0.8, zorder=0)
    ax.text(t, 0.05, str(t), ha="center", va="top", fontsize=9, color="#888")

# TMA:Load k0/k1/k2(领先 2 级)
box(ax, 0, 2, TMA_Y, "加载 k0")
box(ax, 2, 4, TMA_Y, "加载 k1")
box(ax, 4, 6, TMA_Y, "加载 k2")

# MMA:Compute k0/k1/k2(滞后 2 级)
box(ax, 2, 4, MMA_Y, "计算 k0")
box(ax, 4, 6, MMA_Y, "计算 k1")
box(ax, 6, 8, MMA_Y, "计算 k2")

out_path = os.path.join(ZH_OUT, "pipe_depth2.png")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(f"Saved {out_path}")
