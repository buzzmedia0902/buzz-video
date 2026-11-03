"""Output canvas: size per aspect ratio, frame rate and text safe zones.

render.py sets SNS_ASPECT / SNS_FPS / SNS_PLATFORM in the environment before importing anything from core,
so every module sees the same W, H, FPS at import time.
"""
from __future__ import annotations

import os

SIZES = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080), "16:9": (1920, 1080)}

# Text/logo safe zone as fractions of the frame: (left, top, right, bottom) margins.
#  9:16 meta   : Reels/Stories ads (top 14 %, bottom 35 %, sides 6 %)
#  9:16 tiktok : In-Feed (top ~13 %, bottom ~35 %, left ~11 %, right ~17 % for the button column)
#  feed formats: comfortable title-safe margins; 16:9 keeps the bottom clear for player UI.
SAFE = {
    ("9:16", "meta"): (0.06, 0.14, 0.06, 0.35),
    ("9:16", "tiktok"): (0.11, 0.13, 0.17, 0.35),
    ("4:5", "meta"): (0.06, 0.06, 0.06, 0.10),
    ("1:1", "meta"): (0.06, 0.06, 0.06, 0.08),
    ("16:9", "meta"): (0.05, 0.07, 0.05, 0.15),
}

ASPECT = os.environ.get("SNS_ASPECT", "9:16")
if ASPECT not in SIZES:
    raise SystemExit(f"SNS_ASPECT={ASPECT!r}: choose one of {', '.join(SIZES)}")
PLATFORM = os.environ.get("SNS_PLATFORM", "meta")
W, H = SIZES[ASPECT]
FPS = int(os.environ.get("SNS_FPS", "30"))


def safe_margins(aspect: str = ASPECT, platform: str = PLATFORM) -> tuple[float, float, float, float]:
    return SAFE.get((aspect, platform), SAFE[(aspect, "meta")])


def tag(aspect: str = ASPECT) -> str:
    """File-name friendly aspect tag: 9:16 -> 9x16."""
    return aspect.replace(":", "x")
