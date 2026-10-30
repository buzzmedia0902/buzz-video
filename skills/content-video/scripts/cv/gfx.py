"""Skia drawing kit shared by every tone: colours, easing, paints, fonts, text, outlines, images, surfaces.

The canvas size comes from cv.canvas.
Fonts are looked up by family name through skia's FontMgr (macOS system + downloaded asset fonts), or by file.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
import skia
from PIL import Image

from cv.canvas import FPS, H, W  # noqa: F401  (re-exported for tones)

BLACK, WHITE, INK = (13, 13, 13), (255, 255, 255), (24, 24, 27)


# ---------------------------------------------------------------- easing / maths
def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1 else float(x)


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out(t: float, p: float = 3) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** p


def ease_in(t: float, p: float = 3) -> float:
    return clamp01(t) ** p


def ease_io(t: float) -> float:
    t = clamp01(t)
    return t * t * (3 - 2 * t) if t < 1 else 1.0


def ease_io5(t: float) -> float:
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def back(t: float, s: float = 1.6) -> float:
    """0 -> 1 overshooting a little (pop-in)."""
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def spring(t: float, freq: float = 2.2, damp: float = 5.0) -> float:
    """0 -> 1 with a small decaying wobble."""
    t = max(0.0, t)
    return 1 - math.exp(-damp * t) * math.cos(2 * math.pi * freq * t)


def win(t: float, t0: float, t1: float, fin: float = 0.3, fout: float = 0.3) -> float:
    """1 inside [t0, t1] with eased ramps; 0 outside."""
    if t < t0 or t > t1:
        return 0.0
    a = ease_io((t - t0) / fin) if fin > 0 else 1.0
    b = ease_io((t1 - t) / fout) if fout > 0 else 1.0
    return min(a, b)


def hexrgb(c) -> tuple[int, int, int]:
    """'#FEDB2E' or (254, 219, 46) -> (254, 219, 46)."""
    if isinstance(c, str):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return tuple(int(v) for v in c[:3])


def mix_rgb(a, b, t):
    return tuple(int(round(lerp(x, y, t))) for x, y in zip(hexrgb(a), hexrgb(b)))


# ---------------------------------------------------------------- paints
def col(rgb, a: float = 1.0) -> int:
    r, g, b = hexrgb(rgb)
    return skia.ColorSetARGB(int(round(255 * clamp01(a))), r, g, b)


def paint(rgb=INK, a: float = 1.0, stroke: float | None = None, blur: float = 0.0, cap_round: bool = False) -> skia.Paint:
    p = skia.Paint(AntiAlias=True, Color=col(rgb, a))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        if cap_round:
            p.setStrokeCap(skia.Paint.kRound_Cap)
            p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def lin_grad(p0, p1, colors, alphas=None, pos=None) -> skia.Shader:
    alphas = alphas or [1.0] * len(colors)
    return skia.GradientShader.MakeLinear([skia.Point(*p0), skia.Point(*p1)], [col(c, a) for c, a in zip(colors, alphas)],
                                          pos)


def rad_grad(center, r, colors, alphas=None, pos=None) -> skia.Shader:
    alphas = alphas or [1.0] * len(colors)
    return skia.GradientShader.MakeRadial(skia.Point(*center), r, [col(c, a) for c, a in zip(colors, alphas)], pos)


def rrect(x, y, w, h, r) -> skia.RRect:
    return skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)


def shadow(c: skia.Canvas, x, y, w, h, r, blur=30, dy=16, a=0.18, spread=0):
    c.drawRRect(rrect(x - spread, y + dy - spread, w + 2 * spread, h + 2 * spread, r + spread), paint(BLACK, a, blur=blur))


def group(c: skia.Canvas, alpha: float):
    """Begin a layer with alpha (caller must c.restore())."""
    if alpha >= 0.999:
        c.save()
    else:
        c.saveLayerAlpha(None, int(255 * clamp01(alpha)))


# ---------------------------------------------------------------- fonts / text
_FM = skia.FontMgr()
# family names by role key; tones pick keys, a version may override them with "look": {"fonts": {...}}
FAMILIES = {
    "gothic": "Hiragino Sans",                  # 300 W3 ... 600 W6 ... 800 W8 ... 900 W9
    "mincho": "Hiragino Mincho ProN",
    "maru": "Hiragino Maru Gothic ProN",
    "round": "Tsukushi A Round Gothic",         # soft rounded (friendly)
    "round_b": "Tsukushi B Round Gothic",
    "mplus_round": "Rounded Mplus 1c",          # M PLUS Rounded 1c (variety subtitles; weight 950 = Black,
                                                #   900 matches ExtraBold)
    "midashi_go": "Toppan Bunkyu Midashi Gothic",
    "midashi_min": "Toppan Bunkyu Midashi Mincho",
    "kaiti": "Kaiti SC",                         # heavy 楷書 (wa subtitles)
    "xingkai": "Xingkai SC",                     # 行書 (wa titles)
    "yugothic": "YuGothic",
    "noto": "Noto Sans JP",
    "avenir": "Avenir Next",
    "futura": "Futura",
    "helvetica": "Helvetica Neue",
    "optima": "Optima",
    "didot": "Didot",
}
FALLBACK = "Hiragino Sans"


@functools.lru_cache(maxsize=None)
def typeface(family: str, weight: int) -> skia.Typeface:
    if os.path.isfile(os.path.expanduser(family)):
        return skia.Typeface.MakeFromFile(os.path.expanduser(family))
    name = FAMILIES.get(family, family)
    tf = _FM.matchFamilyStyle(name, skia.FontStyle(weight, skia.FontStyle.kNormal_Width,
                                                   skia.FontStyle.kUpright_Slant))
    if tf is None:
        print(f"[gfx] font {name!r} not found, using {FALLBACK}", flush=True)
        tf = _FM.matchFamilyStyle(FALLBACK, skia.FontStyle(weight, skia.FontStyle.kNormal_Width,
                                                           skia.FontStyle.kUpright_Slant))
    return tf


@functools.lru_cache(maxsize=None)
def font(size: float, weight: int = 600, family: str = "gothic") -> skia.Font:
    f = skia.Font(typeface(family, weight), size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setHinting(skia.FontHinting.kNone)
    return f


def jp(size, weight=600):
    """Hiragino Sans."""
    return font(size, weight, "gothic")


def en(size, weight=600):
    return font(size, weight, "avenir")


def text_width(s: str, f: skia.Font, tracking: float = 0.0) -> float:
    if not s:
        return 0.0
    return f.measureText(s) + tracking * f.getSize() * (len(s) - 1)


def draw_text(c: skia.Canvas, s: str, x: float, y: float, f: skia.Font, p: skia.Paint, align: str = "left",
              tracking: float = 0.0) -> float:
    """Draw on baseline y. align: left/center/right. tracking in em. Returns width."""
    w = text_width(s, f, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if tracking == 0.0:
        c.drawString(s, x, y, f, p)
    else:
        for ch in s:
            c.drawString(ch, x, y, f, p)
            x += f.measureText(ch) + tracking * f.getSize()
    return w


def cap_height(f: skia.Font) -> float:
    m = f.getMetrics()
    return -m.fAscent * 0.72


def draw_text_mid(c: skia.Canvas, s: str, x: float, cy: float, f: skia.Font, p: skia.Paint, align: str = "left",
                  tracking: float = 0.0) -> float:
    """Draw with the visual middle of CJK glyphs at cy."""
    return draw_text(c, s, x, cy + cap_height(f) / 2, f, p, align, tracking)


def outlined_text(c: skia.Canvas, s: str, x: float, y: float, f: skia.Font, fill, stroke_rgb=BLACK, stroke_w: float = 8,
                  align: str = "left", tracking: float = 0.0, alpha: float = 1.0, shadow_a: float = 0.0,
                  outer=None, outer_w: float = 0.0):
    """Telop text: optional soft shadow, optional coloured outer edge, thick round-join outline, solid fill."""
    if shadow_a > 0:
        draw_text(c, s, x + 3, y + 5, f, paint(BLACK, shadow_a * alpha, stroke=stroke_w + outer_w + 2, blur=6,
                                              cap_round=True), align, tracking)
    if outer is not None and outer_w > 0:
        draw_text(c, s, x, y, f, paint(outer, alpha, stroke=stroke_w + outer_w, cap_round=True), align, tracking)
    if stroke_w > 0:
        draw_text(c, s, x, y, f, paint(stroke_rgb, alpha, stroke=stroke_w, cap_round=True), align, tracking)
    return draw_text(c, s, x, y, f, paint(fill, alpha), align, tracking)


# ---------------------------------------------------------------- images
@functools.lru_cache(maxsize=None)
def image(path: str) -> skia.Image:
    arr = np.array(Image.open(os.path.expanduser(path)).convert("RGBA"))
    return skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType).withDefaultMipmaps()


def image_from_array(arr: np.ndarray) -> skia.Image:
    """(h, w, 4) uint8 RGBA (straight alpha, opaque for video frames) -> skia image."""
    return skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType)


SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)
SAMPLING_SHARP = skia.SamplingOptions(skia.CubicResampler.Mitchell())


def draw_image(c: skia.Canvas, img: skia.Image, x, y, w=None, h=None, alpha: float = 1.0, src=None,
               sharp: bool = False):
    w = img.width() if w is None else w
    h = img.height() if h is None else h
    p = skia.Paint(AntiAlias=True)
    if alpha < 1:
        p.setAlphaf(clamp01(alpha))
    sr = skia.Rect.MakeXYWH(*src) if src else skia.Rect.MakeWH(img.width(), img.height())
    c.drawImageRect(img, sr, skia.Rect.MakeXYWH(x, y, w, h), SAMPLING_SHARP if sharp else SAMPLING, p)


# ---------------------------------------------------------------- surfaces
def new_surface(w: int = W, h: int = H) -> tuple[skia.Surface, skia.Canvas]:
    s = skia.Surface(w, h)
    return s, s.getCanvas()


def surface_rgba(s: skia.Surface) -> np.ndarray:
    """RGBA uint8 (H, W, 4) of a surface."""
    return s.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
