"""Easing curves, colours and premultiplied-RGBA sprite helpers (canvas-independent).

Re-exported by core.engine, so templates keep using E.ease_out_cubic, E.blit, ...
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------- easing
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease_out_cubic(p: float) -> float:
    return 1 - (1 - clamp01(p)) ** 3


def ease_in_cubic(p: float) -> float:
    return clamp01(p) ** 3


def ease_in_out_sine(p: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(p))


def ease_in_out_cubic(p: float) -> float:
    p = clamp01(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out_back(p: float, s: float = 1.7) -> float:
    p = clamp01(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def smoothstep(p: float) -> float:
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def window(t: float, t_in: float, t_out: float, fade_in: float, fade_out: float) -> float:
    """0..1 envelope: ease in after t_in, ease out before t_out."""
    return ease_out_cubic((t - t_in) / max(fade_in, 1e-6)) * ease_out_cubic((t_out - t) / max(fade_out, 1e-6))


def leave(t: float, t_end: float, dur: float) -> float:
    """1 -> 0 over the `dur` seconds that END at t_end (smoothstep). Use for exits so elements are gone
    by a section boundary instead of popping off on the cut (ease-out fades drop most alpha too late)."""
    return 1.0 - smoothstep((t - (t_end - dur)) / max(dur, 1e-6))


# ---------------------------------------------------------------- colours
def rgb(c) -> tuple[int, int, int]:
    """'#D2BF5B' or (210, 191, 91) -> (210, 191, 91)."""
    if isinstance(c, str):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return tuple(int(v) for v in c[:3])


def mix(a, b, k: float) -> tuple[int, int, int]:
    a, b = rgb(a), rgb(b)
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


# ---------------------------------------------------------------- sprites
def premult(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def blit(frame: np.ndarray, spr: np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
    """Alpha-composite a premultiplied RGBA float sprite onto frame (in place), top-left at (x, y)."""
    if opacity <= 0.003:
        return
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    region = frame[y0:y1, x0:x1]
    region *= 1 - s[..., 3:4] * opacity
    region += s[..., :3] * opacity


def blit_img(frame: np.ndarray, img: Image.Image, cx: float, cy: float, scale: float = 1.0,
             opacity: float = 1.0, angle: float = 0.0) -> None:
    """Composite a PIL RGBA image centred at (cx, cy) with scale / rotation (degrees, CCW)."""
    if opacity <= 0.003 or scale <= 0.01:
        return
    im = img
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if abs(angle) > 0.05:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    blit(frame, premult(im), cx - im.width / 2, cy - im.height / 2, opacity)


def solid_sprite(alpha: np.ndarray, color, strength: float = 1.0) -> np.ndarray:
    a = np.clip(alpha * strength, 0, 1).astype(np.float32)
    c = np.array(rgb(color), np.float32) / 255.0
    return np.dstack([a[..., None] * c, a[..., None]])


def shadow_for(img: Image.Image, radius: float, strength: float, color=(10, 8, 6), pad: int = 40) -> Image.Image:
    """Blurred alpha of img (padded by pad on every side) as a coloured RGBA shadow/glow."""
    a = np.asarray(img.convert("RGBA"))[..., 3].astype(np.float32) / 255
    a = np.pad(a, pad)
    blur = np.clip(gaussian_filter(a, radius) * strength, 0, 1)
    out = np.dstack([np.full(blur.shape + (3,), rgb(color), np.float32), blur[..., None] * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def with_shadow(img: Image.Image, radius: float = 10, strength: float = 1.1, color=(20, 14, 10),
                pad: int = 30, offset: tuple[int, int] = (0, 4)) -> Image.Image:
    """img composited over its own soft shadow (same centre)."""
    sh = shadow_for(img, radius, strength, color, pad)
    out = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (offset[0], offset[1]) if offset != (0, 0) else (0, 0))
    out.alpha_composite(img, (pad, pad))
    return out


def tight(img: Image.Image) -> Image.Image:
    box = img.getbbox()
    return img.crop(box) if box else img
