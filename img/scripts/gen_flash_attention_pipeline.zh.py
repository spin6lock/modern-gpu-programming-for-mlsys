#!/usr/bin/env python3
"""ZH version of gen_flash_attention_pipeline.py — Simplified-Chinese labels -> img/zh/flash_attention_pipeline_v2.png.

Original (English) generator stays untouched in gen_flash_attention_pipeline.py.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _zhfont import setup_zh, ZH_OUT
setup_zh()


COLORS = {
    "tma": "#bfdbfe",
    "mma": "#bbf7d0",
    "softmax": "#ddd6fe",
    "corr": "#ccfbf1",
    "label": "#f8fafc",
}


def block(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None, color="#4b5563", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=0.78,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, fontsize=7, color=color, ha="center", zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(-0.25, 6.65)
    ax.axis("off")

    ax.text(6.65, 6.45, "Flash Attention 4 流水线结构", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.65,
        6.18,
        "代表性的发射顺序;MMA warp 将当前 V 的 value MMA 与下一个 K 的 score MMA 交替发射",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    arrow(ax, 1.75, 5.95, 12.75, 5.95, color="#9ca3af")
    ax.text(7.25, 6.02, "时间", ha="center", va="bottom", fontsize=8, color="#6b7280", style="italic", zorder=2)

    rows = [
        ("WG3 warp 1", "TMA 加载", 5.0),
        ("WG3 warp 0", "MMA 发射", 4.0),
        ("WG0", "softmax Q 阶段 0", 3.0),
        ("WG1", "softmax Q 阶段 1", 2.0),
        ("WG2", "校正 / epilogue", 1.0),
        ("WG3 warp 2", "TMA 存储", 0.0),
    ]
    for name, role, y in rows:
        block(ax, 0.15, y + 0.12, 1.35, 0.62, f"{name}\n{role}", COLORS["label"], fs=8)
        ax.plot([1.75, 13.0], [y + 0.43, y + 0.43], color="#e5e7eb", lw=1, zorder=0)

    # TMA 加载顺序:Q0、K_last、Q1、V_last,然后 K/V 流式加载。
    for x, text in [
        (2.0, "加载 Q0"),
        (3.1, "加载 K[n-1]"),
        (4.2, "加载 Q1"),
        (5.3, "加载 V[n-1]"),
        (6.7, "加载 K[n-2]"),
        (7.8, "加载 V[n-2]"),
        (9.2, "加载 K[n-3]"),
        (10.3, "加载 V[n-3]"),
    ]:
        block(ax, x, 5.12, 0.88, 0.62, text, COLORS["tma"], fs=8)
    ax.text(11.45, 5.43, "...", fontsize=13, color="#6b7280")

    # MMA 发射顺序:先启动 score,然后把当前 V 的 PV 与下一个 K 的 QK 交替。
    mma_blocks = [
        (4.0, "score\nQ0*K[n-1]", COLORS["mma"]),
        (5.1, "score\nQ1*K[n-1]", COLORS["mma"]),
        (6.35, "value\nP0*V[n-1]", COLORS["mma"]),
        (7.45, "score\nQ0*K[n-2]", COLORS["mma"]),
        (8.55, "value\nP1*V[n-1]", COLORS["mma"]),
        (9.65, "score\nQ1*K[n-2]", COLORS["mma"]),
        (10.75, "value\nP0*V[n-2]", COLORS["mma"]),
    ]
    for x, text, color in mma_blocks:
        block(ax, x, 4.12, 0.98, 0.66, text, color, fs=8)
    ax.text(11.95, 4.43, "... 最后的 K/V 之后", fontsize=8, color="#6b7280")

    # softmax 与校正事件。每个 Q 阶段保留一个可读的依赖环。
    block(ax, 4.75, 3.12, 1.05, 0.66, "softmax S0\n写 P0", COLORS["softmax"], fs=8)
    block(ax, 5.85, 2.12, 1.05, 0.66, "softmax S1\n写 P1", COLORS["softmax"], fs=8)
    block(ax, 6.05, 1.12, 1.02, 0.66, "释放 /\n重缩放 O0", COLORS["corr"], fs=8)
    block(ax, 8.25, 1.12, 1.02, 0.66, "释放 /\n重缩放 O1", COLORS["corr"], fs=8)
    block(ax, 8.35, 3.12, 1.05, 0.66, "softmax S0\n写 P0", COLORS["softmax"], fs=8)
    block(ax, 10.25, 2.12, 1.05, 0.66, "softmax S1\n写 P1", COLORS["softmax"], fs=8)
    block(ax, 11.25, 1.12, 1.08, 0.66, "归一化\nO0/O1", COLORS["corr"], fs=8)
    block(ax, 12.0, 0.12, 0.9, 0.62, "存储 O", COLORS["tma"], fs=8)

    # 图例。
    legend = [
        ("TMA 加载/存储", COLORS["tma"]),
        ("Tensor Core MMA", COLORS["mma"]),
        ("softmax", COLORS["softmax"]),
        ("校正/epilogue", COLORS["corr"]),
    ]
    lx = 2.0
    for name, color in legend:
        block(ax, lx, -0.12, 0.22, 0.16, "", color, fs=1)
        ax.text(lx + 0.3, -0.04, name, fontsize=8, va="center", color="#4b5563")
        lx += 1.55

    out_path = os.path.join(ZH_OUT, "flash_attention_pipeline_v2.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
