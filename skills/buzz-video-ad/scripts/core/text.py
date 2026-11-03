"""Fonts (macOS system + downloadable Japanese fonts) and text sprites: tracked lines, vertical text,
pills, sparkles, per-character animated lines.

Re-exported by core.engine, so templates keep using E.font, E.text_img, E.TextLine, ...
"""
from __future__ import annotations

import glob
import math
import os
import re
import unicodedata
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.draw import ease_out_cubic, rgb, premult, blit, shadow_for, tight


# ---------------------------------------------------------------- fonts
_SYS = "/System/Library/Fonts"
_SUP = "/System/Library/Fonts/Supplemental"
# fonts downloaded in Font Book; the folder's number changes with the macOS version (Font7, Font8, ...)
_ASSETS = sorted(glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*"),
                 key=lambda p: int(re.sub(r"\D", "", p.rsplit("Font", 1)[-1]) or 0), reverse=True)


@lru_cache(maxsize=None)
def _sysfont(part: str) -> str | None:
    for f in os.listdir(_SYS):
        if part in unicodedata.normalize("NFC", f):
            return os.path.join(_SYS, f)
    return None


@lru_cache(maxsize=None)
def _asset_font(name: str) -> str | None:
    for root in _ASSETS:
        for d in os.listdir(root):
            p = os.path.join(root, d, "AssetData", name)
            if os.path.exists(p):
                return p
    return None


# key -> list of (path-resolver, ttc index) tried in order
_FONT_SPECS = {
    "mincho": [(lambda: _asset_font("ToppanBunkyuMinchoPr6N-Regular.otf"), 0), (lambda: _sysfont("ヒラギノ明朝 ProN"), 0)],
    "midashi": [(lambda: _asset_font("ToppanBunkyuMidashiMinchoStdN-ExtraBold.otf"), 0),
                (lambda: _sysfont("ヒラギノ明朝 ProN"), 2)],
    "hmin3": [(lambda: _sysfont("ヒラギノ明朝 ProN"), 0)],
    "hmin6": [(lambda: _sysfont("ヒラギノ明朝 ProN"), 2)],
    "yumin": [(lambda: _asset_font("YuMincho.ttc"), 0), (lambda: _sysfont("ヒラギノ明朝 ProN"), 0)],
    "yumin_b": [(lambda: _asset_font("YuMincho.ttc"), 1), (lambda: _sysfont("ヒラギノ明朝 ProN"), 2)],
    "goth3": [(lambda: _sysfont("ヒラギノ角ゴシック W3"), 0)],
    "goth4": [(lambda: _sysfont("ヒラギノ角ゴシック W4"), 0)],
    "goth6": [(lambda: _sysfont("ヒラギノ角ゴシック W6"), 0)],
    "goth8": [(lambda: _sysfont("ヒラギノ角ゴシック W8"), 0)],
    "tgoth": [(lambda: _asset_font("ToppanBunkyuGothicPr6N.ttc"), 0), (lambda: _sysfont("ヒラギノ角ゴシック W4"), 0)],
    "tgoth_x": [(lambda: _asset_font("ToppanBunkyuMidashiGothicStdN-ExtraBold.otf"), 0),
                (lambda: _sysfont("ヒラギノ角ゴシック W8"), 0)],
    "maru": [(lambda: _sysfont("ヒラギノ丸ゴ ProN W4"), 1)],
    "tmaru": [(lambda: _asset_font("TsukushiAMaruGothic.ttc"), 0), (lambda: _sysfont("ヒラギノ丸ゴ ProN W4"), 1)],
    "klee": [(lambda: _asset_font("Klee.ttc"), 1), (lambda: _sysfont("ヒラギノ丸ゴ ProN W4"), 1)],
    "hand": [(lambda: _asset_font("Hannotate.ttc"), 0), (lambda: _asset_font("Klee.ttc"), 1)],
    "optima": [(lambda: f"{_SYS}/Optima.ttc", 0)],
    "optima_b": [(lambda: f"{_SYS}/Optima.ttc", 1)],
    "didot": [(lambda: f"{_SUP}/Didot.ttc", 0)],
    "didot_i": [(lambda: f"{_SUP}/Didot.ttc", 1)],
    "avenir": [(lambda: f"{_SYS}/Avenir Next.ttc", 7)],
    "avenir_m": [(lambda: f"{_SYS}/Avenir Next.ttc", 5)],
    "avenir_b": [(lambda: f"{_SYS}/Avenir Next.ttc", 2)],
    "avenir_x": [(lambda: f"{_SYS}/Avenir Next.ttc", 8)],
    "avenir_ul": [(lambda: f"{_SYS}/Avenir Next.ttc", 10)],
    "futura": [(lambda: f"{_SUP}/Futura.ttc", 0)],
    "helv": [(lambda: f"{_SYS}/HelveticaNeue.ttc", 0)],
    "helv_b": [(lambda: f"{_SYS}/HelveticaNeue.ttc", 1)],
    "helv_m": [(lambda: f"{_SYS}/HelveticaNeue.ttc", 10)],
}
FONT_KEYS = tuple(_FONT_SPECS)
_font_cache: dict = {}


def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    """Font by key (see FONT_KEYS); falls back to Hiragino Sans W6 with a warning if a file is missing."""
    k = (key, int(size))
    if k in _font_cache:
        return _font_cache[k]
    for resolve, idx in _FONT_SPECS.get(key, []):
        path = resolve()
        if path and os.path.exists(path):
            _font_cache[k] = ImageFont.truetype(path, int(size), index=idx)
            return _font_cache[k]
    print(f"[engine] font {key!r} not found, using Hiragino Sans W6", flush=True)
    _font_cache[k] = ImageFont.truetype(_sysfont("ヒラギノ角ゴシック W6"), int(size), index=0)
    return _font_cache[k]


# ---------------------------------------------------------------- text sprites
SQUEEZE = "、。，．"
OPEN = "「『（"


def _advance(f: ImageFont.FreeTypeFont, ch: str) -> tuple[float, float]:
    """(draw offset, advance) with Japanese punctuation squeezed."""
    adv = f.getlength(ch)
    if ch in SQUEEZE:
        return 0.0, adv * 0.55
    if ch in OPEN:
        return -adv * 0.45, adv * 0.55
    return 0.0, adv


def text_img(text: str, key: str, size: int, color, tracking: float = 0.0, stroke: int = 0,
             stroke_color=(0, 0, 0), pad: int = 12) -> Image.Image:
    """Horizontal single-line text as an RGBA image (manual tracking + punctuation squeeze)."""
    f = font(key, size)
    color = rgb(color)
    xs, x = [], 0.0
    for ch in text:
        dx, eff = _advance(f, ch)
        xs.append(x + dx)
        x += eff + tracking * size
    width = int(x - tracking * size) + 2 * pad + 2 * stroke
    height = int(size * 1.3) + 2 * pad + 2 * stroke
    img = Image.new("RGBA", (max(1, width), height), (*color, 0))
    d = ImageDraw.Draw(img)
    for ch, cx in zip(text, xs):
        d.text((pad + stroke + cx, pad + stroke + int(size * 1.0)), ch, font=f, fill=(*color, 255),
               anchor="ls", stroke_width=stroke, stroke_fill=(*rgb(stroke_color), 255))
    return img


def text_width(text: str, key: str, size: int, tracking: float = 0.0) -> float:
    f = font(key, size)
    return sum(_advance(f, ch)[1] + tracking * size for ch in text) - tracking * size


def fit_size(lines: list[str], key: str, size: int, max_w: float, tracking: float = 0.0, min_size: int = 24) -> int:
    """Largest font size <= size at which every line fits within max_w px."""
    s = int(size)
    while s > min_size and max(text_width(t, key, s, tracking) for t in lines) > max_w:
        s -= 2
    return s


def vtext_img(text: str, key: str, size: int, color, tracking: float = 0.12, pad: int = 12) -> Image.Image:
    """Vertical (tategaki) text column; 、。 moved to the upper-right of their cell."""
    f = font(key, size)
    color = rgb(color)
    step = size * (1 + tracking)
    img = Image.new("RGBA", (int(size * 1.3) + 2 * pad, int(step * len(text) + 2 * pad)), (*color, 0))
    d = ImageDraw.Draw(img)
    y = pad
    for ch in text:
        x, yy = pad + size * 0.15, y
        if ch in SQUEEZE:
            x, yy = x + size * 0.6, y - size * 0.58
        d.text((x, yy + size * 0.88), ch, font=f, fill=(*color, 255), anchor="ls")
        y += step * (0.6 if ch in SQUEEZE else 1.0)
    return img.crop((0, 0, img.width, int(y + pad)))


def pill_img(text: str, key: str, size: int, fg, bg, pad_x: int = 34, pad_y: int = 18,
             tracking: float = 0.04, bg_alpha: int = 235, radius: int | None = None) -> Image.Image:
    """Text on a rounded pill (radius defaults to fully round)."""
    t = tight(text_img(text, key, size, fg, tracking, pad=0))
    w, h = t.width + 2 * pad_x, t.height + 2 * pad_y
    img = Image.new("RGBA", (w + 4, h + 4), (*rgb(bg), 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, w + 1, h + 1), radius=h // 2 if radius is None else radius, fill=(*rgb(bg), bg_alpha))
    img.alpha_composite(t, (2 + pad_x, 2 + pad_y))
    return img


def sparkle_img(r: int, color=(255, 246, 214)) -> Image.Image:
    """4-point star glint."""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    ax, ay = np.abs(xx) / r, np.abs(yy) / r
    star = np.exp(-ax * 9) * np.exp(-ay * 1.6) + np.exp(-ay * 9) * np.exp(-ax * 1.6)
    glow = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.18) ** 2))
    a = np.clip(star * 0.9 + glow * 0.8, 0, 1)
    out = np.dstack([np.full(a.shape + (3,), rgb(color), np.float32), a[..., None] * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


class TextLine:
    """Horizontal line with per-character reveal (fade + rise + blur-to-sharp)."""

    BLUR = (0, 2, 5, 9)

    def __init__(self, text: str, key: str, size: int, color, tracking: float = 0.08, stroke: int = 0,
                 stroke_color=(0, 0, 0)):
        f = font(key, size)
        color = rgb(color)
        self.size, self.box_h = size, int(size * 1.22)
        pad = 26 + stroke
        self.chars, x = [], 0.0
        for ch in text:
            dx, eff = _advance(f, ch)
            adv = f.getlength(ch)
            img = Image.new("RGBA", (int(math.ceil(adv)) + 2 * pad, self.box_h + 2 * pad), (*color, 0))
            ImageDraw.Draw(img).text((pad, pad + int(size * 0.98)), ch, font=f, fill=(*color, 255), anchor="ls",
                                     stroke_width=stroke, stroke_fill=(*rgb(stroke_color), 255))
            levels = [premult(img.filter(ImageFilter.GaussianBlur(r)) if r else img) for r in self.BLUR]
            self.chars.append({"x": x + dx - pad, "y": -pad, "levels": levels})
            x += eff + tracking * size
        self.width = x - tracking * size

    def draw(self, frame, ox, oy, t, start, stagger, dur, fade=1.0, drift=0.0, rise=28) -> None:
        """Top-left of the text box at (ox, oy); character i starts at start + i*stagger."""
        for i, c in enumerate(self.chars):
            e = ease_out_cubic((t - start - i * stagger) / dur)
            if e <= 0:
                continue
            lvl = min(3, int(round((1 - e) * 3)))
            blit(frame, c["levels"][lvl], ox + c["x"], oy + c["y"] + (1 - e) * rise - drift, e * fade)

    def draw_centered(self, frame, cx, cy, t, start, stagger=0.04, dur=0.5, fade=1.0, rise=28) -> None:
        self.draw(frame, cx - self.width / 2, cy - self.box_h / 2, t, start, stagger, dur, fade, rise=rise)


class VText:
    """Vertical text column with per-character reveal (fade + slight drop + blur-to-sharp) and a soft glow."""

    def __init__(self, text: str, key: str, size: int, color, tracking: float = 0.28, glow=None):
        f = font(key, size)
        color = rgb(color)
        self.chars, y = [], 0.0
        step = size * (1 + tracking)
        pad = 24
        for ch in text:
            img = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (*color, 0))
            dx, dy = (size * 0.6, -size * 0.58) if ch in SQUEEZE else (0.0, 0.0)
            ImageDraw.Draw(img).text((pad + dx, pad + dy + size * 0.88), ch, font=f, fill=(*color, 255), anchor="ls")
            levels = [premult(img.filter(ImageFilter.GaussianBlur(r)) if r else img) for r in (0, 2, 5, 9)]
            self.chars.append({"y": y - pad, "x": -pad - size / 2, "levels": levels, "img": img})
            y += step * (0.6 if ch in SQUEEZE else 1.0)
        self.height = y
        col = Image.new("RGBA", (size + 2 * pad, int(y) + 2 * pad), (0, 0, 0, 0))
        for c in self.chars:
            col.alpha_composite(c["img"], (0, int(c["y"] + pad)))
        dark = sum(color) < 384
        gcol = glow or ((252, 248, 240) if dark else (8, 6, 4))
        self.glow = premult(shadow_for(col, 12 if dark else 10, 2.6, gcol, pad=40))
        self.glow_off = (-pad - size / 2 - 40, -pad - 40)

    def draw(self, frame, cx, top, t, t_in, t_out, stagger=0.09) -> None:
        fade = ease_out_cubic((t_out - t) / 0.5)
        if t < t_in or fade <= 0:
            return
        blit(frame, self.glow, cx + self.glow_off[0], top + self.glow_off[1],
             0.95 * ease_out_cubic((t - t_in) / 0.8) * fade)
        for i, c in enumerate(self.chars):
            e = ease_out_cubic((t - t_in - i * stagger) / 0.6)
            if e <= 0:
                continue
            lvl = min(3, int(round((1 - e) * 3)))
            blit(frame, c["levels"][lvl], cx + c["x"], top + c["y"] - (1 - e) * 18, e * fade)
