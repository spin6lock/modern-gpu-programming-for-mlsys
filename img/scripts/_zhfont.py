"""CJK font setup shared by the dedicated zh figure scripts.

These scripts live alongside the originals as ``gen_<name>.zh.py`` and always
render Simplified Chinese into ``img/zh/<name>.png``. The original
``gen_<name>.py`` scripts stay untouched and keep producing the English figures.

matplotlib 3.6 has no per-glyph font fallback, so a single font must cover both
Latin/digits and CJK. ``NotoSansCJK-Regular.ttc`` is a TrueType collection whose
first face is the JP region; ``addfont`` exposes it as "Noto Sans CJK JP". That
face covers Latin, digits, and the full CJK Unified Ideographs set, so it renders
Simplified Chinese well (a handful of glyphs may lean JP-style).
``Droid Sans Fallback`` lacks Latin/digits, and no standalone Noto Sans SC file
is installed here, so the JP face is the best single-font option. Avoid the
vulgar fraction ⅔ (U+2154, missing) — write "2/3" instead.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ZH_OUT = os.path.join(HERE, "..", "zh")  # -> img/zh/

_NOTO_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_NOTO_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def setup_zh():
    """Register the CJK font and point matplotlib at it."""
    from matplotlib import rcParams, font_manager

    for p in (_NOTO_REG, _NOTO_BOLD):
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
