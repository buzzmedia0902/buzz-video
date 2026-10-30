"""Output canvas of the version being rendered: size per aspect ratio, frame rate, text-safe area.

render.py / finish.py set CV_ASPECT and CV_FPS before importing anything that draws, so every module can use
the module-level W, H, FPS (same pattern as sns-ad-video).
"""
from __future__ import annotations

import os

SIZES = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080), "16:9": (1920, 1080)}
# (left, top, right, bottom) fractions kept free of titles / subtitles / cards (organic posts, not ads):
# 9:16: Shorts / Reels / TikTok put the like column on the right and the caption + account at the bottom.
SAFE = {"9:16": (0.06, 0.10, 0.13, 0.25), "4:5": (0.05, 0.05, 0.05, 0.08), "1:1": (0.05, 0.05, 0.05, 0.07),
        "16:9": (0.04, 0.05, 0.04, 0.08)}

ASPECT = os.environ.get("CV_ASPECT", "16:9")
if ASPECT not in SIZES:
    raise SystemExit(f"CV_ASPECT must be one of {', '.join(SIZES)} (got {ASPECT})")
W, H = SIZES[ASPECT]
FPS = float(os.environ.get("CV_FPS", "30"))  # 29.97 sources render at 30000/1001
TALL = H > W                                 # 9:16 and 4:5 lay things out vertically


def fps_arg() -> str:
    """FPS for ffmpeg (-r / fps=): 30000/1001 for 29.97, else the number."""
    return "30000/1001" if abs(FPS - 29.97) < 0.01 else f"{FPS:g}"


def safe_rect() -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) in px of the text-safe area."""
    l, t, r, b = SAFE[ASPECT]
    return W * l, H * t, W * (1 - r), H * (1 - b)


def tag(aspect: str | None = None) -> str:
    return (aspect or ASPECT).replace(":", "x")
