"""Layout helpers: the text safe zone as a rectangle, anchors as fractions of it, and review guides.

Templates place text with L.y(f) / L.x(f) (0 = top/left edge of the safe zone, 1 = bottom/right edge)
instead of absolute pixels, so the same template works in 9:16, 4:5, 1:1 and 16:9.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.canvas import ASPECT, H, PLATFORM, SAFE, W, safe_margins


class Layout:
    def __init__(self, aspect: str = ASPECT, platform: str = PLATFORM):
        left, top, right, bottom = safe_margins(aspect, platform)
        self.W, self.H, self.aspect, self.platform = W, H, aspect, platform
        self.x0, self.y0 = W * left, H * top
        self.x1, self.y1 = W * (1 - right), H * (1 - bottom)
        self.sw, self.sh = self.x1 - self.x0, self.y1 - self.y0
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        ratio = H / W
        self.kind = "tall" if ratio > 1.5 else "portrait" if ratio > 1.1 else "square" if ratio > 0.9 else "wide"
        self.text_w = min(self.sw, 900.0)  # widest a single caption line should get

    def y(self, f: float) -> float:
        return self.y0 + f * self.sh

    def x(self, f: float) -> float:
        return self.x0 + f * self.sw

    @property
    def tall(self) -> bool:
        return self.kind == "tall"

    @property
    def wide(self) -> bool:
        return self.kind == "wide"

    def fit(self, needed: float, avail: float | None = None) -> float:
        """Scale (<= 1) that makes a stack of height `needed` fit into `avail` (default: safe height)."""
        avail = self.sh if avail is None else avail
        return min(1.0, avail / max(needed, 1.0))

    def pick(self, **by_kind):
        """Per-format value: L.pick(tall=1130, wide=880, default=...)."""
        return by_kind.get(self.kind, by_kind.get("default"))


def draw_guides(frame_u8: np.ndarray, aspect: str = ASPECT) -> np.ndarray:
    """Safe-zone outlines for review stills: yellow = Meta (all formats), red = TikTok (9:16)."""
    img = Image.fromarray(frame_u8).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    for (asp, plat), (l, t, r, b) in SAFE.items():
        if asp != aspect:
            continue
        col = (255, 220, 0, 230) if plat == "meta" else (255, 60, 60, 230)
        box = (W * l, H * t, W * (1 - r), H * (1 - b))
        d.rectangle(box, outline=col, width=3)
        d.text((box[0] + 6, box[1] + 4 if plat == "meta" else box[3] - 28), f"safe: {plat}", fill=col, font=f)
    img.alpha_composite(over)
    return np.asarray(img.convert("RGB"))
