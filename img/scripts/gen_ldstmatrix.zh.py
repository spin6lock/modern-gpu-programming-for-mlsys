"""ZH version of gen_ldstmatrix.py — Simplified-Chinese labels -> img/zh/ldstmatrix.svg.

Original (English) generator stays untouched in gen_ldstmatrix.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

from _zhfont import setup_zh, ZH_OUT as OUT
setup_zh()

ROWC = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#65a30d"]
TXT = "#1f2937"

fig, ax = plt.subplots(figsize=(9.2, 4.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97, "ldmatrix / stmatrix:8×8 b16 SMEM 分块  ↔  warp 寄存器片段",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TXT)

# --- 几何 ---
TOP, BOT = 82, 14
H = (TOP - BOT) / 8.0          # 行高
SX0, SW = 9, 32                # SMEM 分块 x 起点、宽度
FX0, FW = 59, 32              # 寄存器片段 x 起点、宽度
cw_s = SW / 8.0
cw_f = FW / 8.0


def row_y(r):
    return TOP - (r + 1) * H


# 表头
ax.text(SX0 + SW / 2, TOP + 3.5, "SMEM:8×8 fp16(行主序)", ha="center", fontsize=8.5,
        fontweight="bold", color=TXT)
ax.text(FX0 + FW / 2, TOP + 3.5, "32 个 lane 上的寄存器", ha="center", fontsize=8.5,
        fontweight="bold", color=TXT)

# SMEM 分块:8 行 × 8 个 fp16,按行着色;地址由 lane T{r} 提供
for r in range(8):
    y = row_y(r)
    for c in range(8):
        ax.add_patch(Rectangle((SX0 + c * cw_s, y), cw_s, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=0.8, alpha=0.5))
    ax.text(SX0 - 1.5, y + H / 2, f"T{r}", ha="right", va="center", fontsize=7.5,
            fontweight="bold", color=ROWC[r])
ax.text(SX0 + SW / 2, BOT - 3.2, "行 r 地址 ← lane T{r}", ha="center", fontsize=7, color=TXT)

# 寄存器片段:同样 8×8,按行着色;每对列 = 一个 lane 的 b32 寄存器
for r in range(8):
    y = row_y(r)
    for j in range(4):  # 每行 4 个 lane,各占列 2j、2j+1
        lane = 4 * r + j
        x = FX0 + 2 * j * cw_f
        ax.add_patch(Rectangle((x, y), 2 * cw_f, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=1.6, alpha=0.9))
        ax.text(x + cw_f, y + H / 2, f"L{lane}", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white")
ax.text(FX0 + FW / 2, BOT - 3.2, "lane l → 行 l/4,列 2·(l%4)、+1  (1 b32 = 2 fp16)",
        ha="center", fontsize=7, color=TXT)

# 中间箭头
ax.annotate("", xy=(FX0 - 2, 56), xytext=(SX0 + SW + 2, 56),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 60, "ldmatrix  (SMEM → 寄存器)", ha="center", fontsize=7.8, color="#475569")
ax.annotate("", xy=(SX0 + SW + 2, 44), xytext=(FX0 - 2, 44),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 40, "stmatrix  (寄存器 → SMEM)", ha="center", fontsize=7.8, color="#475569")

ax.text(50, 6, ".x1 加载一个 8×8(地址来自 T0–T7);.x2 / .x4 加载 2 / 4 个矩阵 "
        "(T0–T15 / T0–T31)。.trans 按列主序读取每个 8×8。",
        ha="center", fontsize=6.8, color=TXT, style="italic")

fig.savefig(f"{OUT}/ldstmatrix.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/ldstmatrix.svg")
