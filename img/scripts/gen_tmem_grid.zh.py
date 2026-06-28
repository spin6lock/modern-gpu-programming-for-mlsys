"""ZH version of gen_tmem_grid.py — Simplified-Chinese labels -> img/zh/tmem_grid.png.

Original (English) generator stays untouched in gen_tmem_grid.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _zhfont import setup_zh, ZH_OUT
setup_zh()

NCOL = 512   # TCol: up to 512 32-bit columns
NROW = 128   # TLane: 128 rows
ACC_N = 256  # example accumulator width (columns)


def main():
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=180)

    # 完整地址空间。
    ax.add_patch(Rectangle((0, 0), NCOL, NROW, facecolor="#eef3fb",
                           edgecolor="#222222", linewidth=1.6, zorder=1))

    # 浅色网格线:列每 64、行每 32(分配粒度)。
    for c in range(0, NCOL + 1, 64):
        ax.plot([c, c], [0, NROW], color="#c3d2e8", linewidth=0.7, zorder=2)
    for r in range(0, NROW + 1, 32):
        ax.plot([0, NCOL], [r, r], color="#c3d2e8", linewidth=0.7, zorder=2)

    # 示例累加器:占据全部 128 行的前 ACC_N 列。
    ax.add_patch(Rectangle((0, 0), ACC_N, NROW, facecolor="#cfe8d4",
                           edgecolor="#2f8f4e", linewidth=1.6, zorder=3))
    ax.text(ACC_N / 2, NROW / 2,
            "一个累加器\nS[(128, 256) : (1@TLane, 1@TCol)]",
            ha="center", va="center", fontsize=11.5, weight="bold",
            color="#1f5e36", zorder=5)

    # 高亮一个元素,展示 (行, 列) 寻址。
    cx, cy, cw = 352, 44, 10
    ax.add_patch(Rectangle((cx, cy), cw, 4, facecolor="#ffd166",
                           edgecolor="#b8860b", linewidth=1.2, zorder=4))
    ax.annotate("单个元素位于\n行 TLane = l, 列 TCol = c",
                xy=(cx + cw, cy + 2), xytext=(cx + 30, cy + 46),
                fontsize=9.5, color="#7a5a00", ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.3))

    # 第 0 行在顶部。
    ax.set_xlim(-10, NCOL + 12)
    ax.set_ylim(0, NROW)
    ax.invert_yaxis()
    ax.axis("off")

    # 底部列轴 (TCol)。
    for c in [0, 128, 256, 384, 512]:
        ax.plot([c, c], [NROW, NROW + 3], color="#222222", linewidth=1.0,
                clip_on=False, zorder=4)
        ax.text(c, NROW + 7, str(c), ha="center", va="top", fontsize=9,
                color="#333333")
    ax.text(NCOL / 2, NROW + 20,
            "TCol——最多 512 个 32 位列(按 32 对齐分配)",
            ha="center", va="top", fontsize=10.5, color="#333333")

    # 左侧行轴 (TLane)。
    for r in [0, 64, 127]:
        ax.text(-6, r, str(r), ha="right", va="center", fontsize=9,
                color="#333333")
    ax.text(-34, NROW / 2, "TLane——128 行", ha="center", va="center",
            fontsize=10.5, color="#333333", rotation=90)

    ax.set_title("TMEM 是二维地址空间:每个 CTA 拥有 128 个 TLane 行 × 最多 512 个 "
                 "TCol 列", fontsize=13, weight="bold", pad=14)

    fig.tight_layout(pad=0.5)
    out_path = os.path.join(ZH_OUT, "tmem_grid.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
