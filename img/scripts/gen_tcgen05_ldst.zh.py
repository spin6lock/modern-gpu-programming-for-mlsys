"""ZH version of gen_tcgen05_ldst.py — Simplified-Chinese labels -> img/zh/tcgen05_ldst.svg.

Original (English) generator stays untouched in gen_tcgen05_ldst.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _zhfont import setup_zh, ZH_OUT as OUT
setup_zh()

ROWC = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9", "#ec4899", "#65a30d"]
TXT = "#1f2937"

fig, ax = plt.subplots(figsize=(9.2, 4.7))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "tcgen05.ld / st:TMEM 累加器  ↔  m8n8 寄存器片段",
        ha="center", va="top", fontsize=11, fontweight="bold", color=TXT)

TOP, BOT = 82, 16
H = (TOP - BOT) / 8.0
SX0, SW = 9, 32          # TMEM 分块
FX0, FW = 59, 32         # 寄存器片段
cw_s = SW / 8.0
cw_f = FW / 8.0


def row_y(r):
    return TOP - (r + 1) * H


ax.text(SX0 + SW / 2, TOP + 3.5, "TMEM 累加器", ha="center", fontsize=8.5, fontweight="bold", color=TXT)
ax.text(FX0 + FW / 2, TOP + 3.5, "寄存器(m8n8 片段)", ha="center", fontsize=8.5, fontweight="bold", color=TXT)

# TMEM 分块:8 个 TLane × 8 个 TCol,按行(TLane)着色
for r in range(8):
    y = row_y(r)
    for c in range(8):
        ax.add_patch(Rectangle((SX0 + c * cw_s, y), cw_s, H, facecolor=ROWC[r],
                               edgecolor="white", linewidth=0.8, alpha=0.5))
    ax.text(SX0 - 1.5, y + H / 2, f"TLane {r}", ha="right", va="center", fontsize=6.8,
            fontweight="bold", color=ROWC[r])
ax.text(SX0 + SW / 2, BOT - 3.2, "行 m → TLane m  (TCol → N)", ha="center", fontsize=7, color=TXT)

# 寄存器片段:每对列 = 一个 lane 的 b32 寄存器;lane l → 行 l/4,列 2(l%4)、+1
for r in range(8):
    y = row_y(r)
    for j in range(4):
        lane = 4 * r + j
        x = FX0 + 2 * j * cw_f
        ax.add_patch(Rectangle((x, y), 2 * cw_f, H, facecolor=ROWC[r], edgecolor="white",
                               linewidth=1.6, alpha=0.9))
        ax.text(x + cw_f, y + H / 2, f"L{lane}", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white")
ax.text(FX0 + FW / 2, BOT - 3.2, "lane l → 行 l/4,列 2·(l%4)、+1  (1 b32 = 2 元素)",
        ha="center", fontsize=7, color=TXT)

# 箭头
ax.annotate("", xy=(FX0 - 2, 56), xytext=(SX0 + SW + 2, 56),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 60, "tcgen05.ld  (TMEM → 寄存器)", ha="center", fontsize=7.8, color="#475569")
ax.annotate("", xy=(SX0 + SW + 2, 44), xytext=(FX0 - 2, 44),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))
ax.text(50, 40, "tcgen05.st  (寄存器 → TMEM)", ha="center", fontsize=7.8, color="#475569")

ax.text(50, 6.5, "由 warpgroup 协作、异步执行(由 tcgen05.wait 门控)。该片段与 ldmatrix 构建的 "
        "(Ampere)以及 wgmma 输出的(Hopper)m8n8 布局相同。",
        ha="center", fontsize=7.2, color=TXT, style="italic")

fig.savefig(f"{OUT}/tcgen05_ldst.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/tcgen05_ldst.svg")
