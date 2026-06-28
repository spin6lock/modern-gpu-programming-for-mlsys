#!/usr/bin/env python3
"""ZH version of gen_flash_attention_barrier_flow.py — Simplified-Chinese labels.

Outputs img/zh/flash_attention_main_handoff.png and img/zh/flash_attention_softmax_correction.png.
Original (English) generator stays untouched in gen_flash_attention_barrier_flow.py.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _zhfont import setup_zh, ZH_OUT
setup_zh()


COLORS = {
    "tma": "#bfdbfe",
    "smem": "#e9d5ff",
    "tmem": "#fed7aa",
    "mma": "#bbf7d0",
    "softmax": "#ddd6fe",
    "wg2": "#ccfbf1",
    "bar": "#fde68a",
    "merge": "#eee7fb",
}


def box(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.15,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, color="#4b5563", rad=0.0, lw=1.25):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def label(ax, x, y, text, fs=8.5, color="#374151", facecolor=None):
    if facecolor is None:
        facecolor = COLORS["bar"]
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=facecolor, edgecolor="#d97706"),
    )


def gen_main_handoff():
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(6.5, 6.62, "Flash Attention 4:MMA 输入门控", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.5,
        6.26,
        "每路 MMA 触发前必须就绪的输入",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # ---- Score MMA 门控(上):Q 和 K 必须在 SMEM 中。----
    ax.text(0.5, 5.95, "Score MMA 门控", fontsize=11.5, weight="bold", color="#1f2937")
    box(ax, 0.6, 5.12, 1.95, 0.62, "Q 分块\n在 SMEM", COLORS["smem"], fs=8.8)
    box(ax, 0.6, 4.30, 1.95, 0.62, "K 分块\n在 SMEM", COLORS["smem"], fs=8.8)
    box(ax, 6.45, 4.55, 2.3, 0.95, "score MMA\nQ,K -> S", COLORS["mma"], fs=10.5)
    arrow(ax, 2.55, 5.43, 6.45, 5.18, rad=-0.04)
    arrow(ax, 2.55, 4.61, 6.45, 4.88, rad=0.04)
    label(ax, 4.45, 5.50, "q_load.full", fs=8.3)
    label(ax, 4.45, 4.54, "kv_load.full", fs=8.3)
    ax.text(7.6, 4.34, "所有输入就绪后触发", ha="center", fontsize=7.8,
            color="#6b7280", style="italic")

    # ---- Value MMA 门控(下):V 在 SMEM、P 在 TMEM(分段)、O 槽位安全。----
    ax.text(0.5, 3.35, "Value MMA 门控", fontsize=11.5, weight="bold", color="#1f2937")
    box(ax, 0.6, 2.55, 1.95, 0.62, "V 分块\n在 SMEM", COLORS["smem"], fs=8.8)
    box(ax, 0.6, 1.55, 3.15, 0.7, "TMEM 中 P 列 0:96\n+ O 槽位安全 (WG2)", COLORS["tmem"], fs=8.2)
    box(ax, 0.6, 0.55, 3.15, 0.62, "TMEM 中 P 列 96:128", COLORS["tmem"], fs=8.4)
    box(ax, 6.45, 1.35, 2.3, 0.95, "value MMA\nP,V -> O", COLORS["mma"], fs=10.5)
    arrow(ax, 2.55, 2.86, 6.45, 2.05, rad=-0.05)
    arrow(ax, 3.75, 1.90, 6.45, 1.83, rad=0.0)
    arrow(ax, 3.75, 0.86, 6.45, 1.52, rad=0.06)
    label(ax, 4.95, 2.42, "kv_load.full", fs=8.3)
    label(ax, 4.95, 1.93, "p_o_rescale.full", fs=8.3)
    label(ax, 4.95, 1.08, "p_ready_2.full", fs=8.3)
    ax.text(7.6, 1.14, "分两段的 MMA:先在列 0:96 启动,再到 96:128", ha="center", fontsize=7.8,
            color="#6b7280", style="italic")

    # ---- 图例(右侧空隙)。----
    lx = 9.55
    ax.text(lx, 5.95, "图例", fontsize=11.5, weight="bold", color="#1f2937")

    def swatch(y, color, text):
        box(ax, lx, y, 0.42, 0.34, "", color)
        ax.text(lx + 0.6, y + 0.17, text, ha="left", va="center", fontsize=8.7, color="#374151")

    swatch(5.40, COLORS["smem"], "SMEM 分块 (TMA 加载)")
    swatch(4.86, COLORS["tmem"], "TMEM 分块 / O 槽位")
    swatch(4.32, COLORS["mma"], "MMA 操作")
    label(ax, lx + 0.5, 3.66, "barrier", fs=8.0)
    ax.text(lx + 1.05, 3.66, "MMA 触发前必须\n发出信号的门控",
            ha="left", va="center", fontsize=8.7, color="#374151")
    ax.text(lx, 2.55, "kv_load.full 同时门控 K 与 V 的加载,\n因此出现在两个门控中。",
            ha="left", va="center", fontsize=8.5, color="#6b7280", style="italic")

    out_path = os.path.join(ZH_OUT, "flash_attention_main_handoff.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


def gen_softmax_correction():
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    ax.text(6.0, 5.0, "Softmax / WG2 缩放槽位握手", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.0,
        4.66,
        "softmax_corr.full 与 softmax_corr.empty 保护的是一个 SMEM 槽位,而非 P/O 计算路径",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # full/empty 主生命周期。
    box(ax, 0.55, 3.35, 1.75, 0.72, "槽位空闲\nsoftmax 可写入", COLORS["bar"], fs=8.8)
    box(ax, 2.8, 3.35, 1.95, 0.72, "softmax 写入\nacc_scale / row_sum", COLORS["softmax"], fs=8.8)
    box(ax, 5.25, 3.35, 1.7, 0.72, "bar_softmax\n_corr_full", COLORS["bar"], fs=8.4)
    box(ax, 7.45, 3.35, 1.9, 0.72, "WG2 读取\n该 SMEM 槽位", COLORS["wg2"], fs=8.8)
    box(ax, 9.85, 3.35, 1.7, 0.72, "bar_softmax\n_corr_empty", COLORS["bar"], fs=8.4)

    arrow(ax, 2.3, 3.71, 2.8, 3.71)
    arrow(ax, 4.75, 3.71, 5.25, 3.71)
    arrow(ax, 6.95, 3.71, 7.45, 3.71)
    arrow(ax, 9.35, 3.71, 9.85, 3.71)
    arrow(ax, 10.7, 3.35, 1.42, 3.35, color="#7c3aed", rad=-0.18, lw=1.35)
    label(ax, 5.95, 2.7, "empty 回到 softmax:\n该槽位下次可被覆写", fs=8.4)

    ax.text(0.7, 4.25, "生产者", fontsize=9, weight="bold", color="#92400e")
    ax.text(7.95, 4.25, "消费者", fontsize=9, weight="bold", color="#166534")
    ax.text(0.6, 1.95, "full/empty 这一对证明了什么", fontsize=11, weight="bold")
    ax.text(
        0.6,
        1.58,
        "full:WG2 可从 SMEM 读取缩放系数或最终 row_sum\n"
        "empty:softmax 可复用同一 SMEM 槽位\n"
        "范围:每个 Q 阶段一个槽位,由 128 个 warpgroup 线程送达",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    ax.text(7.05, 1.95, "它不能证明什么", fontsize=11, weight="bold")
    ax.text(
        7.05,
        1.58,
        "否:P 已写入 TMEM\n"
        "否:O 已重缩放\n"
        "否:value MMA 可以启动\n"
        "这些由 p_o_rescale.full 与 p_ready_2.full 覆盖",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    out_path = os.path.join(ZH_OUT, "flash_attention_softmax_correction.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


def main():
    gen_main_handoff()
    gen_softmax_correction()


if __name__ == "__main__":
    main()
