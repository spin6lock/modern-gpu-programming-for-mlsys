"""ZH version of gen_smem_descriptor.py — Simplified-Chinese labels -> img/zh/smem_descriptor.svg.

Original (English) generator stays untouched in gen_smem_descriptor.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _zhfont import setup_zh, ZH_OUT as OUT
setup_zh()

TXT = "#1f2937"
C_A = "#3b82f6"
C_B = "#60a5fa"

fig, ax = plt.subplots(figsize=(9.8, 5.8))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 98, "SMEM 矩阵描述符 → A(M×K) 在共享内存中的摆放方式",
        ha="center", va="top", fontsize=12.5, fontweight="bold", color=TXT)

NR, NC = 2, 3
X0, X1, YB, YT = 30, 88, 24, 72
bw = (X1 - X0) / NC
bh = (YT - YB) / NR

for br in range(NR):
    for bc in range(NC):
        x = X0 + bc * bw
        y = YT - (br + 1) * bh
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=(C_A if (br + bc) % 2 == 0 else C_B),
                               edgecolor="white", linewidth=2.0, alpha=0.9))
        for k in range(1, 8):  # 8 行,背靠背绘制(连续)
            ax.plot([x, x + bw], [y + k * bh / 8, y + k * bh / 8], color="white", lw=0.4, alpha=0.5)
        # 第一个 atom:标签右移,给连续性刻度留位置
        lx = x + bw * 0.62 if (br == 0 and bc == 0) else x + bw / 2
        ax.text(lx, y + bh / 2, "atom\n8 × 128 B", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    ax.text(X0 - 2, YT - (br + 0.5) * bh, ["行 0–7", "行 8–15"][br], ha="right", va="center",
            fontsize=8.5, fontweight="bold", color=TXT)

# 第一个(左上)atom 内部的连续性细节:8 行是连续的 128 B
fx, fy = X0, YT - bh
ax.add_patch(Rectangle((fx, fy), bw, bh, facecolor="none", edgecolor="#111827", linewidth=2.4))
ax.annotate("", xy=(fx + 3.0, fy + 2.5), xytext=(fx + 3.0, fy + bh - 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.5))
ax.text(fx + 4.2, fy + bh - 3.5, "字节 0", fontsize=7, color="#111827", va="center")
ax.text(fx + 4.2, fy + 3.5, "+896 B", fontsize=7, color="#111827", va="center")
ax.text(fx + bw / 2, fy + bh + 1.8, "连续 1 KB", ha="center", fontsize=7,
        color="#111827", fontweight="bold")

# 坐标轴
ax.text((X0 + X1) / 2, YB - 8.5, "K  (字节) →", ha="center", fontsize=9.5, color=TXT)
ax.annotate("", xy=(X1 + 1.5, YT), xytext=(X1 + 1.5, YB), arrowprops=dict(arrowstyle="-|>", color=TXT, lw=1.3))
ax.text(X1 + 4, (YB + YT) / 2, "M ↓", ha="left", va="center", fontsize=9.5, color=TXT)

# start_address 标记
ax.plot(X0, YT, marker="o", color="#111827", markersize=6)
ax.text(X0 - 2.5, YT + 6, "start_address (addr ≫ 4)", ha="left", va="bottom", fontsize=9, color="#111827")

# ldo:沿主维(此处为 K)的 atom 间步幅
ax.annotate("", xy=(X0 + 2.5 * bw, YT - 3.2), xytext=(X0 + 1.5 * bw, YT - 3.2),
            arrowprops=dict(arrowstyle="<|-|>", color="#111827", lw=1.6))
ax.text(X0 + 2 * bw, YT - 7.5, "ldo  (主维,此处为 K)", ha="center", fontsize=8.5,
        fontweight="bold", color="#111827")

# sdo:沿另一维(此处为 M)的 atom 间步幅
ax.annotate("", xy=(X0 - 9.5, YT - 1.5 * bh), xytext=(X0 - 9.5, YT - 0.5 * bh),
            arrowprops=dict(arrowstyle="<|-|>", color="#111827", lw=1.6))
ax.text(X0 - 12, YT - bh, "sdo\n(另一\n维, M)", ha="right", va="center", fontsize=8,
        fontweight="bold", color="#111827")

# swizzle 格式说明
ax.text((X0 + X1) / 2, YB - 4.0, "swizzle 格式决定了 atom 形状(此处为 8 × 128 B;其余为 64 / 32 / 16 B)"
        "及其内部的 XOR 模式", ha="center", fontsize=8, color="#7c3aed")

ax.text(50, 7, "每个 atom 是一个连续的 8 × 128 B 块——其 8 行在内存中背靠背排列;"
        "ldo / sdo 在 atom 之间跳步。", ha="center", fontsize=8.2, color=TXT, style="italic")
ax.text(50, 3, "ldo = 沿主维的步幅,sdo = 沿另一维。图中为 K-major"
        "(主维 = K);MN-major 操作数则交换 ldo 与 sdo。", ha="center", fontsize=8.2,
        color=TXT, style="italic")

fig.savefig(f"{OUT}/smem_descriptor.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/smem_descriptor.svg")
