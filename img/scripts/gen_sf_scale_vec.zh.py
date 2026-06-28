"""ZH version of gen_sf_scale_vec.py — Simplified-Chinese labels -> img/zh/sf_scale_vec.svg.

Original (English) generator stays untouched in gen_sf_scale_vec.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _zhfont import setup_zh, ZH_OUT as OUT
setup_zh()

TXT = "#1f2937"
SFC = {"SF0": "#ef4444", "SF1": "#3b82f6", "SF2": "#10b981", "SF3": "#f59e0b"}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "scale_vec 模式:一个 TMEM u32 如何打包 K-block 缩放系数", ha="center", va="top",
        fontsize=12, fontweight="bold", color=TXT)

SX, SW = 30, 44          # 条带 x 起点、宽度
bw = SW / 4
BYTES = ["[0:7]", "[8:15]", "[16:23]", "[24:31]"]

# 字节位置表头
for b in range(4):
    ax.text(SX + (b + 0.5) * bw, 86, BYTES[b], ha="center", va="center", fontsize=7.5, color=TXT)
ax.text(SX + SW / 2, 90, "32 位字的字节", ha="center", fontsize=8, color=TXT)

ROWS = [
    ("1X", "fp8 / mxfp8", ["SF0", "SF0", "SF0", "SF0"], "一个缩放系数,广播 ×4   (SF_VEC = 32)"),
    ("2X", "mxfp4", ["SF0", "SF1", "SF0", "SF1"], "两个缩放系数,各 ×2   (SF_VEC = 32)"),
    ("4X", "nvfp4", ["SF0", "SF1", "SF2", "SF3"], "四个缩放系数 = 四个 K-block   (SF_VEC = 16)"),
]
ys = [66, 46, 26]
sh = 13
for (mode, fmt, bytes_, cap), y in zip(ROWS, ys):
    ax.text(SX - 3, y + sh / 2, f"{mode}", ha="right", va="center", fontsize=11, fontweight="bold", color=TXT)
    ax.text(SX - 3, y - 2.5, fmt, ha="right", va="center", fontsize=7.5, color=TXT)
    for b, lab in enumerate(bytes_):
        ax.add_patch(Rectangle((SX + b * bw, y), bw, sh, facecolor=SFC[lab], edgecolor="white",
                               linewidth=1.6, alpha=0.92))
        ax.text(SX + (b + 0.5) * bw, y + sh / 2, lab, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.text(SX + SW + 3, y + sh / 2, cap, ha="left", va="center", fontsize=8, color=TXT)

ax.text(50, 9, "SFk = 某个 M 行第 k 个 K-block 的缩放系数。sf_per_mma(即 ×N)= mma_k / SF_VEC:"
        "fp8 为 1、mxfp4 为 2、nvfp4 为 4。", ha="center", fontsize=7.6, color=TXT, style="italic")
ax.text(50, 4, "三种模式的 lane/列放置(TLane = m%32,m//32 → 列)都相同,"
        "只是字节打包方式不同。", ha="center", fontsize=7.6, color=TXT, style="italic")

fig.savefig(f"{OUT}/sf_scale_vec.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/sf_scale_vec.svg")
