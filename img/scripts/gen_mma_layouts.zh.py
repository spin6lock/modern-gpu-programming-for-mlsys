"""ZH version of gen_mma_layouts.py — Simplified-Chinese labels.

Outputs 5 SVGs into img/zh/: mma_cg1_m128, mma_cg1_m64, mma_cg2_m256,
mma_cg2_m128, mma_block_scaled. Original (English) generator stays untouched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from _zhfont import setup_zh, ZH_OUT as OUT
setup_zh()

SMEM_BG = "#ede9fe"
TMEM_BG = "#fffbeb"
SMEM_EDGE = "#8b5cf6"
TMEM_EDGE = "#f59e0b"
MMA_C = "#059669"
EDGE = "#94a3b8"
TXT = "#334155"
A_C = "#8b5cf6"
B_C = "#6d28d9"
C0 = "#f59e0b"
C1 = "#f59e0b"
C0B = "#fbbf24"
C1B = "#fbbf24"
SFA_C = "#fb923c"
SFB_C = "#f97316"
GRAY = "#cbd5e1"


def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def group(ax, x, y, w, h, label, bg):
    edge = SMEM_EDGE if bg == SMEM_BG else TMEM_EDGE if bg == TMEM_BG else EDGE
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=bg, edgecolor=edge, linewidth=1.2))
    ax.text(x + 1.5, y + h - 1.2, label, fontsize=8, fontweight="bold", color=TXT,
            ha="left", va="top")


def tile(ax, x, y, w, h, color, title, sub=None, fs=8.5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=1.2))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + h * 0.14, title, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, cy - h * 0.16, sub, ha="center", va="center", color="white",
                fontsize=fs - 1.8)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold")


def arrow(ax, x0, x1, y, text="tcgen05\nMMA"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=MMA_C, lw=1.8))
    ax.text((x0 + x1) / 2, y + 6, text, ha="center", va="center", fontsize=7, color=MMA_C)


def title(ax, t):
    ax.text(50, 99, t, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#1f2937")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def smem_tmem_strip(ax, ylo, yhi, a_lab, a_sub, b_lab, b_sub, c_color, c_lab, c_sub,
                    smem_title, tmem_title):
    """单个 CTA:SMEM(A,B) -> TMEM(C)。"""
    h = yhi - ylo
    group(ax, 3, ylo, 38, h, smem_title, SMEM_BG)
    tile(ax, 6, ylo + h * 0.50, 32, h * 0.40, A_C, a_lab, a_sub)
    tile(ax, 6, ylo + h * 0.06, 32, h * 0.36, B_C, b_lab, b_sub)
    arrow(ax, 42, 55, ylo + h * 0.5)
    group(ax, 59, ylo, 38, h, tmem_title, TMEM_BG)
    tile(ax, 62, ylo + h * 0.06, 32, h * 0.78, c_color, c_lab, c_sub)


# ---------- cta_group::1, M=128(identity) ----------
fig, ax = setup(8.5, 3.1)
title(ax, "cta_group::1, M=128  —  1 CTA")
smem_tmem_strip(ax, 8, 88,
                "A   (M, K)", "128 × K", "B   (N, K)", None,
                C0, "C   (M, N)", "行 m -> lane m  (lane 0-127)",
                "SMEM (1 CTA)", "TMEM (1 CTA)")
save(fig, "mma_cg1_m128.svg")

# ---------- cta_group::1, M=64(四次运行,步幅 32)----------
fig, ax = setup(8.5, 3.3)
title(ax, "cta_group::1, M=64  —  1 CTA")
group(ax, 3, 8, 38, 80, "SMEM (1 CTA)", SMEM_BG)
tile(ax, 6, 48, 32, 32, A_C, "A   (M, K)", "64 × K")
tile(ax, 6, 12, 32, 30, B_C, "B   (N, K)", None)
arrow(ax, 42, 55, 50)
group(ax, 59, 8, 38, 80, "TMEM (1 CTA)", TMEM_BG)
# C:在 lane 步幅 32 处的四个 16 行运行;交错的 Lane 带保持未用。
runs = ["行 0-15", "行 16-31", "行 32-47", "行 48-63"]
band_h = 70 / 8.0
for j in range(8):
    yb = 80 - (j + 1) * band_h
    if j % 2 == 0:
        tile(ax, 62, yb, 32, band_h - 0.6, C0, runs[j // 2], None, fs=7.5)
    else:
        ax.add_patch(Rectangle((62, yb), 32, band_h - 0.6, facecolor=GRAY,
                               edgecolor="white", linewidth=1.0, alpha=0.6, hatch="//"))
ax.text(78, 10.5, "C (M, N):未使用的 Lane 带可容纳另一个对齐的 M=64 分块",
        ha="center", fontsize=6.6, color=TXT)
save(fig, "mma_cg1_m64.svg")


def two_cta(ax, t, a0, b0, c0lab, c0sub, a1, b1, c1lab, c1sub, mid_note,
            split_c=False):
    title(ax, t)
    # CTA 0(上)
    group(ax, 3, 58, 38, 32, "CTA 0 — SMEM", SMEM_BG)
    tile(ax, 6, 74, 32, 13, A_C, a0[0], a0[1], fs=8)
    tile(ax, 6, 60, 32, 12, B_C, b0[0], b0[1], fs=8)
    arrow(ax, 42, 55, 73, "MMA")
    group(ax, 59, 58, 38, 32, "CTA 0 — TMEM", TMEM_BG)
    if split_c:
        tile(ax, 62, 74, 32, 13, C0, c0lab[0], c0lab[1], fs=7.5)
        tile(ax, 62, 60, 32, 12, C0B, c0sub[0], c0sub[1], fs=7.5)
    else:
        tile(ax, 62, 60, 32, 27, C0, c0lab, c0sub, fs=8)
    # CTA 1(下)
    group(ax, 3, 12, 38, 32, "CTA 1 — SMEM", SMEM_BG)
    tile(ax, 6, 28, 32, 13, A_C, a1[0], a1[1], fs=8)
    tile(ax, 6, 14, 32, 12, B_C, b1[0], b1[1], fs=8)
    arrow(ax, 42, 55, 27, "MMA")
    group(ax, 59, 12, 38, 32, "CTA 1 — TMEM", TMEM_BG)
    if split_c:
        tile(ax, 62, 28, 32, 13, C1, c1lab[0], c1lab[1], fs=7.5)
        tile(ax, 62, 14, 32, 12, C1B, c1sub[0], c1sub[1], fs=7.5)
    else:
        tile(ax, 62, 14, 32, 27, C1, c1lab, c1sub, fs=8)
    ax.text(50, 50, mid_note, ha="center", va="center", fontsize=6.8, color=TXT,
            style="italic")


# ---------- cta_group::2, M=256(Layout A)----------
fig, ax = setup(8.8, 4.8)
two_cta(ax, "cta_group::2, M=256  —  2 个 CTA(M 划分)",
        ("A  行 0-127", "128 × K"), ("B  列 0..N/2", "N/2 × K"),
        "C  行 0-127", "× 全 N (lane 0-127)",
        ("A  行 128-255", "128 × K"), ("B  列 N/2..N", "N/2 × K"),
        "C  行 128-255", "× 全 N (lane 0-127)",
        "B 的两半(各 N/2)跨这对 CTA 合并 -> 完整 B(N,K);\n"
        "每个 CTA 的 A 半 × 完整 B -> 它的 128 行 C")
save(fig, "mma_cg2_m256.svg")

# ---------- cta_group::2, M=128(Layout B)----------
fig, ax = setup(8.8, 4.8)
two_cta(ax, "cta_group::2, M=128  —  2 个 CTA(M 划分,N 折叠进 lane)",
        ("A  行 0-63", "64 × K"), ("B  列 0..N/2", "N/2 × K"),
        ("C 行 0-63", "N 低位 -> lane 0-63"), ("行 0-63", "N 高位 -> lane 64-127"),
        ("A  行 64-127", "64 × K"), ("B  列 N/2..N", "N/2 × K"),
        ("C 行 64-127", "N 低位 -> lane 0-63"), ("行 64-127", "N 高位 -> lane 64-127"),
        "每个 CTA:64 行,N 划分 -> 低位半在 lane 0-63,\n高位半在 lane 64-127",
        split_c=True)
save(fig, "mma_cg2_m128.svg")

# ---------- block-scaled(cta_group::2)----------
fig, ax = setup(9.6, 5.0)
title(ax, "Block-scaled MMA (cta_group::2)  —  缩放系数位于 TMEM")
for row, (ylo, cc, arows, sfarows) in enumerate(
        [(56, C0, "行 0-127", "行 0-127"), (10, C1, "行 128-255", "行 128-255")]):
    cta = row
    group(ax, 2, ylo, 30, 32, f"CTA {cta} — SMEM", SMEM_BG)
    tile(ax, 4.5, ylo + 16, 25, 13, A_C, f"A  {arows}", "fp8/fp4 打包", fs=7.5)
    tile(ax, 4.5, ylo + 2, 25, 12, B_C, "B  列一半", "fp8/fp4 打包", fs=7.5)
    arrow(ax, 33, 43, ylo + 16, "MMA")
    group(ax, 45, ylo, 53, 32, f"CTA {cta} — TMEM", TMEM_BG)
    tile(ax, 47, ylo + 16, 15, 13, SFA_C, "SFA", f"(M, SFK)\n{sfarows}", fs=7)
    tile(ax, 64, ylo + 16, 15, 13, SFB_C, "SFB", "(N, SFK)\n全 N (multicast)", fs=7)
    tile(ax, 81, ylo + 2, 15, 27, cc, "C", f"(M, N)\n{arows}", fs=7.5)
    tile(ax, 47, ylo + 2, 32, 12, GRAY, "", None)
    ax.text(63, ylo + 8, "SFA 按 M 划分(每个 CTA)· SFB 全 N 发给两者",
            ha="center", va="center", fontsize=6.4, color=TXT)
ax.text(50, 47, "A、B 在 SMEM 中打包 · SFA/SFB 经 tcgen05.cp 从 SMEM 加载到 TMEM · "
        "SFK = K / block(nvfp4 为 16,mxfp8 为 32)",
        ha="center", va="center", fontsize=6.8, color=TXT, style="italic")
save(fig, "mma_block_scaled.svg")

print(f"wrote 5 figures into {OUT}: mma_cg1_m128, mma_cg1_m64, mma_cg2_m256, mma_cg2_m128, mma_block_scaled")
