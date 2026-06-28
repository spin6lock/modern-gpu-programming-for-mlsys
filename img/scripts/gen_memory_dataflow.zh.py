"""ZH version of gen_memory_dataflow.py — Simplified-Chinese labels -> img/zh/memory_dataflow.png.

Original (English) generator stays untouched in gen_memory_dataflow.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _zhfont import setup_zh, ZH_OUT
setup_zh()


BOXES = [
    ("全局\n内存\n(GMEM)", "gmem"),
    ("共享\n内存\n(SMEM)", "smem"),
    ("张量\n内存\n(TMEM)", "tmem"),
    ("寄存器\n堆\n(RF)", "rf"),
    ("共享\n内存\n(SMEM)", "smem"),
    ("全局\n内存\n(GMEM)", "gmem"),
]

# 英文标签用作颜色查找 key，中文仅为显示文本。
EDGE_LABELS = [
    "TMA load",
    "tcgen05 MMA",
    "tcgen05.ld",
    "thread write",
    "TMA store",
]

EDGE_ZH = {
    "TMA load": "TMA 加载",
    "tcgen05 MMA": "tcgen05 MMA",
    "tcgen05.ld": "tcgen05.ld",
    "thread write": "线程写回",
    "TMA store": "TMA 存储",
}

# 配色与硬件约定一致:GMEM 中性、SMEM 紫、TMEM 橙、RF/寄存器 红。
COLOURS = {
    "gmem": ("#f8fafc", "#94a3b8"),  # (fill, edge)
    "smem": ("#ede9fe", "#8b5cf6"),
    "tmem": ("#fffbeb", "#f59e0b"),
    "rf":   ("#fee2e2", "#dc2626"),
}

EDGE_COLOURS = {
    "TMA load": "#3b82f6",
    "tcgen05 MMA": "#059669",
    "tcgen05.ld": "#f59e0b",
    "thread write": "#dc2626",
    "TMA store": "#3b82f6",
}

BOX_W = 1.6
BOX_H = 1.3
GAP = 1.6
Y = 0.0


def draw_box(ax, cx, cy, w, h, text, fill, edge):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        facecolor=fill,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11)


def draw_arrow(ax, x_from, x_to, y, key):
    color = EDGE_COLOURS[key]
    arrow = FancyArrowPatch(
        (x_from, y),
        (x_to, y),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5,
        color=color,
    )
    ax.add_patch(arrow)
    ax.text(
        (x_from + x_to) / 2,
        y + 0.25,
        EDGE_ZH[key],
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
    )


def main():
    n_boxes = len(BOXES)
    total_w = n_boxes * BOX_W + (n_boxes - 1) * GAP
    fig, ax = plt.subplots(figsize=(total_w * 0.95, 1.9), dpi=200)

    centres = []
    x = BOX_W / 2
    for text, kind in BOXES:
        centres.append(x)
        fill, edge = COLOURS[kind]
        draw_box(ax, x, Y, BOX_W, BOX_H, text, fill, edge)
        x += BOX_W + GAP

    for i, key in enumerate(EDGE_LABELS):
        x_from = centres[i] + BOX_W / 2
        x_to = centres[i + 1] - BOX_W / 2
        draw_arrow(ax, x_from, x_to, Y, key)

    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)

    out_path = os.path.join(ZH_OUT, "memory_dataflow.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
