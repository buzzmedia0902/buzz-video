"""和・重厚 (wa): the 筆文字 look of a pitch / investment show. Gold 行書 titles (first line in heavy 楷書) on a crimson dry-brush
stroke with a 落款 seal and a glint every 6 s, heavy 楷書 subtitles (white, thickened, black edge; {..} words in gold with
a crimson edge), crimson 墨しぶき behind big words, gold frames and drifting gold dust on a dark lacquer / washi
background, taiko and hyoshigi on titles. Pauses are kept (0.5 s), the camera moves rarely, warm grade, 和楽器 BGM.

Use for: 投資番組・審査番組, 勝負・決断の場面, 対談の山場, 重みのある一言を見せたい切り抜き.
Brand keys: "seal" (落款 text, up to 4 characters; default: the first 2 characters of brand "name").
Overlay keys: name "speaker" (colour from edit.json "speakers").
"""
from __future__ import annotations

import contextlib
import functools
import math
import zlib
from dataclasses import dataclass, replace

import numpy as np
import skia
from PIL import Image
from scipy.ndimage import gaussian_filter

from cv import gfx as G
from cv.canvas import H, TALL, W
from cv.cues import JOIN
from cv.tones.base import Look, _merge, spans

WHITE, BLACK, INK = (255, 255, 255), (0, 0, 0), (14, 11, 10)
SHU, KURENAI, BROWN, GOLD = (200, 28, 34), (150, 14, 22), (70, 30, 4), (222, 160, 34)
GOLD_STOPS = ((0.0, (255, 251, 222)), (0.36, (252, 222, 112)), (0.5, GOLD), (0.6, (236, 182, 52)), (1.0, (255, 236, 150)))
SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")   # brush fonts draw them full size: 80 % instead


@contextlib.contextmanager
def grouped(c: skia.Canvas, alpha: float):
    """Everything drawn inside fades together (no outline showing through a translucent fill)."""
    G.group(c, alpha)
    try:
        yield
    finally:
        c.restore()


@contextlib.contextmanager
def placed(c: skia.Canvas, x: float, y: float, k: float = 1.0, rot: float = 0.0):
    """Canvas origin moved to (x, y), rotated and scaled by k around it."""
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(k, k)
    try:
        yield
    finally:
        c.restore()


# ---------------------------------------------------------------- textures (numpy, cached; seeds are crc32, never hash())
def seed_of(text: str) -> int:
    return zlib.crc32(text.encode()) % 9973


def _noise(rng, n: int, scale: float) -> np.ndarray:
    k = max(2, int(n / scale))
    return np.interp(np.linspace(0, k, n), np.arange(k + 3), rng.normal(0, 1, k + 3)).astype(np.float32)


def rgba(alpha: np.ndarray, rgb) -> skia.Image:
    out = np.zeros(alpha.shape + (4,), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.clip(alpha * 255, 0, 255)
    return G.image_from_array(out)


@functools.lru_cache(maxsize=96)
def brush(w: int, h: int, seed: int, rgb: tuple, dry: float = 0.5, taper: float = 0.2) -> skia.Image:
    """A horizontal 墨 brush stroke with dry-brush streaks (かすれ), tapering to the right."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx[0] / max(1, w - 1)
    cy = 0.5 * h + _noise(rng, w, 220) * h * 0.035
    half = h * 0.42 * (1 + 0.07 * _noise(rng, w, 140)) * np.clip(u / 0.03, 0, 1) ** 0.5 \
        * (0.18 + 0.82 * np.clip((1 - u) / taper, 0, 1) ** 0.9)
    top, bot = cy - half * (1 + 0.05 * _noise(rng, w, 30)), cy + half * (1 + 0.05 * _noise(rng, w, 30))
    inside = np.clip(np.minimum(yy - top[None, :], bot[None, :] - yy) / 2.5, 0, 1)
    tex = gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), sigma=(0.9, 34))
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    v = np.abs(yy - cy[None, :]) / np.maximum(half[None, :], 1)
    dryness = dry * (0.15 + 0.85 * np.clip((u - 0.3) / 0.7, 0, 1))[None, :] + 0.45 * np.clip(v - 0.72, 0, 1)
    a = inside * np.clip(1.6 * (0.55 * tex + 1.0 - 2.1 * dryness), 0, 1)
    dens = gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), 9)
    return rgba(np.clip(a * np.clip(0.9 + 0.1 * dens / (dens.std() + 1e-6), 0.7, 1.0), 0, 1), rgb)


@functools.lru_cache(maxsize=16)
def splash(seed: int, rgb: tuple) -> skia.Image:
    """An elliptical 墨しぶき blob with droplets and streaks flung outward (square, scaled when drawn)."""
    rng = np.random.default_rng(seed)
    s = 600
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    ang = np.arctan2(yy, xx)
    rad = 0.58 + sum(rng.uniform(0.01, 0.05) * np.sin(k * ang + rng.uniform(0, 6.28)) for k in range(3, 12))
    nz = gaussian_filter(rng.normal(0, 1, (s, s)).astype(np.float32), 3)
    a = np.clip((rad + 0.025 * nz / (nz.std() + 1e-6) - np.sqrt(xx ** 2 + yy ** 2) / (s / 2)) / 0.012, 0, 1)
    for _ in range(26):
        th, r0, rr, ln = rng.uniform(0, 6.28), rng.uniform(0.62, 0.95), rng.uniform(0.012, 0.04), rng.uniform(1.0, 3.5)
        dx, dy = xx / (s / 2) - math.cos(th) * r0, yy / (s / 2) - math.sin(th) * r0
        ex, ey = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        a = np.maximum(a, np.clip((rr - np.sqrt((ex / ln) ** 2 + ey ** 2)) / 0.006, 0, 1))
    return rgba(a * 0.9, rgb)


@functools.lru_cache(maxsize=2)
def lacquer(w: int, h: int) -> skia.Image:
    """Dark red-black lacquer, vignetted, with washi fibres and faint ink clouds."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    k = (np.clip(np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h * 0.42) / h) ** 2) / 0.75, 0, 1) ** 1.2)[..., None]
    base = np.array([46, 18, 14], np.float32) * (1 - k) + np.array([7, 5, 5], np.float32) * k
    f = [gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), s) for s in ((0.7, 7), (7, 0.7), 1.2)]
    paper = 1 + 0.035 * (f[0] / f[0].std() * 0.5 + f[1] / f[1].std() * 0.5 + f[2] / f[2].std() * 0.4)
    cl = gaussian_filter(rng.normal(0, 1, (h // 4 + 8, w // 4)).astype(np.float32), 18)
    cl = np.clip((cl / cl.std() - 0.2) * 0.8, 0, 1).astype(np.float32)
    cl = np.asarray(Image.fromarray(cl, "F").resize((w, h), Image.BILINEAR), np.float32)
    rgb = np.clip(base * paper[..., None] * (1 - 0.45 * cl[..., None]), 0, 255)
    return G.image_from_array(np.dstack([rgb, np.full((h, w), 255, np.float32)]).astype(np.uint8))


@functools.lru_cache(maxsize=8)
def seal_image(chars: str, size: int) -> skia.Image:
    """落款: a vermilion stamp with the characters knocked out (白文) and a worn edge."""
    surf, c = G.new_surface(size, size)
    c.drawRRect(G.rrect(4, 4, size - 8, size - 8, size * 0.12), G.paint(SHU, 1.0))
    n = len(chars)
    cells = [(0.5, 0.5)] if n == 1 else [(0.5, 0.30), (0.5, 0.70)] if n == 2 else \
        [(0.70, 0.30), (0.70, 0.70), (0.30, 0.30), (0.30, 0.70)]
    f = G.font(size * (0.62 if n == 1 else 0.40 if n == 2 else 0.34), 900, "kaiti")
    knock = skia.Paint(AntiAlias=True)
    knock.setBlendMode(skia.BlendMode.kClear)
    for ch, (px, py) in zip(chars, cells):
        G.draw_text_mid(c, ch, size * px, size * py, f, knock, "center")
    a = G.surface_rgba(surf)[..., 3].astype(np.float32) / 255
    grit = gaussian_filter(np.random.default_rng(4).normal(0, 1, (size, size)).astype(np.float32), 1.3)
    return rgba(gaussian_filter(a * ((grit > -1.9) * 0.9 + 0.1), 0.6), SHU)


@functools.lru_cache(maxsize=1)
def sparkle() -> skia.Image:
    surf, c = G.new_surface(19, 19)
    p = skia.Paint(AntiAlias=True)
    p.setShader(G.rad_grad((9.5, 9.5), 9.5, [(255, 205, 110)] * 3, [1.0, 0.35, 0.0], [0.0, 0.4, 1.0]))
    c.drawCircle(9.5, 9.5, 9.5, p)
    return surf.makeImageSnapshot()


@functools.lru_cache(maxsize=4)
def _dust(n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(18, 60, n), rng.uniform(0, 6.28, n),
            rng.uniform(0.25, 0.8, n))


def dust(c: skia.Canvas, t: float, alpha: float = 1.0, n: int = 46, seed: int = 20) -> None:
    """Gold dust drifting up and glittering."""
    xs, ys, vs, ph, al = _dust(n, seed)
    for i in range(n):
        y = (ys[i] - vs[i] * t) % (H + 40) - 20
        x = xs[i] + 18 * math.sin(t * 0.8 + ph[i])
        G.draw_image(c, sparkle(), x - 9, y - 9, alpha=alpha * al[i] * (0.55 + 0.45 * math.sin(t * 3.1 + ph[i] * 4)) * 0.7)


def gold_paint(top: float, bottom: float, alpha: float = 1.0) -> skia.Paint:
    p = skia.Paint(AntiAlias=True)
    p.setShader(G.lin_grad((0, top), (0, bottom), [rgb for _, rgb in GOLD_STOPS], pos=[q for q, _ in GOLD_STOPS]))
    p.setAlphaf(alpha)
    return p


def stroked(p: skia.Paint, width: float) -> skia.Paint:
    p.setStyle(skia.Paint.kStroke_Style)
    p.setStrokeWidth(width)
    p.setStrokeCap(skia.Paint.kRound_Cap)
    return p


# ---------------------------------------------------------------- brush lettering
@dataclass(frozen=True)
class Ink:
    """Paint of a brush-lettered row: fill (rgb or "gold"), outlines (outermost first, paint width / size), the fill
    grown by `thick` (em), shadow (dx, dy, blur in em, alpha), a gold bevel, and the look of {..} runs."""
    fill: object = WHITE
    strokes: tuple = ((INK, 0.4),)
    thick: float = 0.03
    shadow: tuple = (0.06, 0.09, 0.125, 0.65)
    bevel: bool = False
    emph: "Ink | None" = None


SUB = Ink(emph=Ink(fill="gold", strokes=((INK, 0.47), ((90, 12, 10), 0.25)), bevel=True))
TITLE_KAI = Ink(strokes=(((120, 8, 14), 0.354), (INK, 0.229)), thick=0.042, shadow=(0.04, 0.08, 0.1, 0.65),
                emph=Ink(fill="gold", strokes=(((120, 8, 14), 0.354), (INK, 0.229)), thick=0.042, bevel=True))
TITLE_GOLD = Ink(fill="gold", strokes=((INK, 0.26), (BROWN, 0.116)), thick=0.015, shadow=(0.04, 0.06, 0.07, 0.65),
                 bevel=True, emph=Ink(fill=(255, 248, 228), strokes=((INK, 0.26), (BROWN, 0.116)), thick=0.015, bevel=True))
LABEL = Ink(strokes=(), thick=0.02, shadow=(0.0, 0.0, 0.0, 0.0), emph=Ink(fill=(255, 226, 140), strokes=(), thick=0.02))
BODY = Ink(strokes=(), thick=0.025, shadow=(0.0, 0.05, 0.08, 0.5), emph=Ink(fill="gold", strokes=(), thick=0.025))


def rows_of(text: str) -> list[tuple[str, list[bool]]]:
    """[(plain, emphasis flags)] per line of a text with {..} markup and ｜ / newline breaks."""
    plain, flags = spans(text.replace(JOIN, "\n"))
    out, k = [], 0
    for s in plain.split("\n"):
        out.append((s, flags[k:k + len(s)]))
        k += len(s) + 1
    return out


@functools.lru_cache(maxsize=None)
def covers(face: tuple, ch: str) -> bool:
    return G.typeface(*face).unicharToGlyph(ord(ch)) != 0


def glyphs(s: str, flags, size: float, face: tuple, track: float = 0.02) -> tuple[list, float]:
    """[(x, char, font, emphasised)] and the width: small kana at 80 %, 、。 tightened, a little tracking; characters
    the brush font lacks (・ 〜) come from Mincho."""
    big, small = G.font(size, face[1], face[0]), G.font(size * 0.8, face[1], face[0])
    out, x = [], 0.0
    for ch, fl in zip(s, flags):
        sm = ch in SMALL_KANA
        f = small if sm else big if covers(face, ch) else G.font(size, 700, "mincho")
        adv = f.measureText(ch) * (0.6 if ch in "、。" else 1.0) * (1.08 if sm else 1.0)
        out.append((x + (adv * 0.06 if sm else 0.0), ch, f, fl))
        x += adv + track * size
    return out, max(0.0, x - track * size)


def row_width(s: str, flags, size: float, face: tuple) -> float:
    return glyphs(s, flags, size, face)[1]


def draw_row(c: skia.Canvas, s: str, flags, x: float, y: float, size: float, face: tuple, ink: Ink, alpha: float = 1.0,
             align: str = "center", glint: float | None = None) -> None:
    """One row on baseline y: shadow, outlines, thickened fill (flat or gold with bevel), glint sweep (0..1)."""
    gl, w = glyphs(s, flags, size, face)
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    inks = [ink.emph if fl and ink.emph else ink for *_, fl in gl]
    top, bot = y - 0.92 * size, y + 0.22 * size
    area = skia.Rect.MakeLTRB(x0 - size * 0.7, top - size * 0.3, x0 + w + size * 0.7, bot + size * 0.4)

    def lay(make, dy: float = 0.0, dx: float = 0.0) -> None:
        for (gx, ch, f, _), ik in zip(gl, inks):
            p = make(ik)
            if p is not None:
                c.drawString(ch, x0 + gx + dx, y + dy, f, p)

    def grown(p: skia.Paint, ik: Ink) -> skia.Paint:
        if ik.thick > 0:
            p.setStyle(skia.Paint.kStrokeAndFill_Style)
            p.setStrokeWidth(ik.thick * size)
            p.setStrokeJoin(skia.Paint.kRound_Join)
        return p

    def fill(ik: Ink) -> skia.Paint:
        return grown(gold_paint(top, bot, alpha) if ik.fill == "gold" else G.paint(ik.fill, alpha), ik)

    def cut(ik: Ink) -> skia.Paint:
        p = grown(G.paint(BLACK, 1.0), ik)
        p.setBlendMode(skia.BlendMode.kDstOut)
        return p

    sdx, sdy, sblur, sa = ink.shadow
    if sa > 0:
        lay(lambda ik: G.paint(BLACK, sa * alpha, stroke=max([sw for _, sw in ik.strokes] + [ik.thick]) * size,
                               blur=sblur * size, cap_round=True), sdy * size, sdx * size)
    for layer in range(2):
        lay(lambda ik: G.paint(ik.strokes[layer][0], alpha, stroke=ik.strokes[layer][1] * size, cap_round=True)
            if layer < len(ik.strokes) else None)
    bevel = any(ik.bevel for ik in inks)
    if bevel or glint is not None:
        c.saveLayer(area, None)
    lay(fill)
    if bevel:                                   # light rim on the top edges: white glyphs minus themselves shifted down
        c.saveLayer(area, None)
        lay(lambda ik: grown(G.paint(WHITE, 0.55 * alpha), ik) if ik.bevel else None)
        lay(lambda ik: cut(ik) if ik.bevel else None, dy=size * 0.03)
        c.restore()
    if glint is not None:
        gx = x0 - 160 + glint * (w + 320)
        gp = skia.Paint(AntiAlias=True)
        gp.setBlendMode(skia.BlendMode.kSrcATop)
        gp.setShader(G.lin_grad((gx - 60, y), (gx + 55, y - 25), [WHITE] * 3, [0.0, 0.85 * alpha, 0.0], [0.0, 0.5, 1.0]))
        c.drawRect(area, gp)
    if bevel or glint is not None:
        c.restore()


def draw_rows(c: skia.Canvas, rows, x: float, cy: float, size: float, face: tuple, ink: Ink, step: float = 1.25,
              alpha: float = 1.0, align: str = "center", glint: float | None = None) -> None:
    """Rows stacked around the middle line cy (the visual middle of the glyphs)."""
    for j, (s, fl) in enumerate(rows):
        draw_row(c, s, fl, x, cy + (j - (len(rows) - 1) / 2) * step * size + 0.36 * size, size, face, ink, alpha, align, glint)


def frame(c: skia.Canvas, x: float, y: float, w: float, h: float, u: float = 1.0, alpha: float = 1.0) -> None:
    """Gold rounded frame with L-shaped corner brackets ending in dots, a crimson hairline inside (u scales the lines)."""
    gold = stroked(gold_paint(y - 20 * u, y + h + 20 * u, alpha), 7 * u)
    dots = gold_paint(y - 20 * u, y + h + 20 * u, alpha)
    c.drawRRect(G.rrect(x - 0.5 * u, y - 0.5 * u, w + u, h + u, 26 * u), gold)
    c.drawRRect(G.rrect(x + 7 * u, y + 7 * u, w - 14 * u, h - 14 * u, 18 * u), G.paint(SHU, 0.85 * alpha, stroke=2 * u))
    ln = 70 * u
    for cx, cy, sx, sy in ((x - 16 * u, y - 16 * u, 1, 1), (x + w + 16 * u, y - 16 * u, -1, 1),
                           (x - 16 * u, y + h + 16 * u, 1, -1), (x + w + 16 * u, y + h + 16 * u, -1, -1)):
        c.drawLine(cx, cy, cx + sx * ln, cy, gold)
        c.drawLine(cx, cy, cx, cy + sy * ln, gold)
        c.drawCircle(cx + sx * ln, cy, 7 * u, dots)
        c.drawCircle(cx, cy + sy * ln, 7 * u, dots)


def thumb_path(s: float) -> skia.Path:
    pts = [(0.36, 0.46), (0.52, 0.20), (0.55, 0.04), (0.65, 0.0), (0.72, 0.09), (0.69, 0.36), (0.93, 0.36), (1.0, 0.45),
           (0.96, 0.56), (0.99, 0.66), (0.93, 0.76), (0.94, 0.86), (0.85, 0.98), (0.36, 0.98)]
    p = skia.Path()
    p.moveTo(pts[0][0] * s, pts[0][1] * s)
    for x, y in pts[1:]:
        p.lineTo(x * s, y * s)
    p.close()
    p.addRoundRect(skia.Rect.MakeLTRB(0.0, 0.44 * s, 0.27 * s, 0.98 * s), 0.05 * s, 0.05 * s)
    return p


def bell_path(s: float) -> skia.Path:
    p = skia.Path()
    p.moveTo(0.5 * s, 0.06 * s)
    p.cubicTo(0.79 * s, 0.06 * s, 0.82 * s, 0.34 * s, 0.82 * s, 0.55 * s)
    p.lineTo(0.94 * s, 0.76 * s)
    p.lineTo(0.06 * s, 0.76 * s)
    p.lineTo(0.18 * s, 0.55 * s)
    p.cubicTo(0.18 * s, 0.34 * s, 0.21 * s, 0.06 * s, 0.5 * s, 0.06 * s)
    p.close()
    p.addCircle(0.5 * s, 0.87 * s, 0.09 * s)
    return p


# ---------------------------------------------------------------- the tone
class Wa(Look):
    key = "wa"
    CUT = dict(pause=0.5, tail=0.2, lead=0.15, speed=1.0)
    CAMERA = dict(hold=12.0, med=1.12, punch_every=45.0, punch_zoom=1.08, punch_dur=0.5)
    AUDIO = dict(bgm="wa", music_db=-15.0, sfx="wa", sfx_db=-2.0)
    GRADE = "eq=contrast=1.07:saturation=1.03,colorbalance=rs=0.03:bs=-0.04:rm=0.03:bm=-0.04"
    BUBBLE = (0.80, 0.335, 0.115)
    STYLE = _merge(Look.STYLE, {
        "accent": GOLD, "ink": INK, "paper": (245, 238, 224),
        "sub": {"font": "kaiti", "weight": 900, "size": {"wide": 58, "tall": 70}, "y": {"wide": 0.93, "tall": 0.72},
                "gap": 1.25, "max_w": 0.92},
        "title": {"font": "xingkai", "weight": 700, "size": {"wide": 66, "tall": 100}, "y": {"wide": 0.17, "tall": 0.19}},
        "chapter": {"font": "kaiti", "weight": 900, "size": {"wide": 44, "tall": 54}, "y": {"wide": 0.10, "tall": 0.29}},
        "card": {"font": "kaiti", "weight": 900, "size": {"wide": 44, "tall": 54}, "w": {"wide": 0.40, "tall": 0.80}},
        "emphasis": {"font": "xingkai", "weight": 700, "size": {"wide": 150, "tall": 140},
                     "y": {"wide": 0.42, "tall": 0.42}},
        "name": {"font": "kaiti", "weight": 900, "size": {"wide": 46, "tall": 50}, "y": {"wide": 0.74, "tall": 0.56}},
        "caption": {"font": "kaiti", "weight": 900, "size": {"wide": 42, "tall": 48}, "y": {"wide": 0.11, "tall": 0.14}},
        "credit": {"font": "mincho", "weight": 700, "size": {"wide": 24, "tall": 28}, "fill": (225, 220, 210)},
    })

    # -------------------------------------------------------------- helpers
    def face(self, el: str) -> tuple:
        return self.v(el, "font"), self.v(el, "weight")

    def wrapped(self, text: str, size: float, face: tuple, max_w: float) -> list[tuple[str, list[bool]]]:
        """Rows of a text; a single row wider than max_w is broken at the best Japanese line-break spot."""
        rows = rows_of(text)
        if len(rows) == 1 and len(rows[0][0]) >= 6 and row_width(*rows[0], size, face) > max_w:
            (a, fa), (b, fb) = self._break(rows[0][0], rows[0][1], G.font(size, face[1], face[0]), max_w, size)
            rows = [(a, list(fa)), (b, list(fb))]
        return rows

    @staticmethod
    def fit(rows, size: float, face: tuple, max_w: float) -> float:
        """Scale (<= 1) that makes the widest row fit max_w."""
        widest = max(row_width(s, fl, size, face) for s, fl in rows)
        return min(1.0, max_w / widest) if widest > 0 else 1.0

    def tag(self, c, rows, rgb, x: float, y: float, size: float, face: tuple, slant: float = 2.0) -> None:
        """札: white brush lettering on a dry brush stroke in `rgb`, left edge at x, middle line y."""
        w = max(row_width(s, fl, size, face) for s, fl in rows) + size * 1.2
        h = size * (0.5 + 1.2 * len(rows))
        with placed(c, x, y, 1.0, -slant):
            G.draw_image(c, brush(int(w) + 8, int(h) + 8, seed_of(rows[0][0]), tuple(rgb), 0.25, 0.08), -4, -h / 2 - 4, w + 8, h + 8)
            draw_rows(c, rows, size * 0.6, 0, size, face, LABEL, 1.2, 1.0, "left")

    # -------------------------------------------------------------- subtitles
    def draw_sub(self, c, sub, t) -> None:
        size, lines = self.sub_layout(sub.text)
        face, tt = self.face("sub"), t - sub.t0
        p = G.clamp01(tt / 0.14)
        a = G.clamp01(p * 1.6) * G.clamp01((sub.t1 - t) / 0.06 + 0.2)
        whole = all(all(fl) for _, fl in lines)             # a line that is all {..}: crimson splash, bigger pop
        gap = size * self.v("sub", "gap")
        y_last = H * self.v("sub", "y")
        y_first = y_last - (len(lines) - 1) * gap
        cy = (y_first + y_last) / 2 - 0.3 * size
        if whole:
            e = G.ease_out(tt / 0.18)
            bw = (max(row_width(s, fl, size, face) for s, fl in lines) * 1.2 + 80) * (0.6 + 0.4 * e)
            bh = size * 1.45 * (len(lines) + 0.4) * (0.6 + 0.4 * e)
            G.draw_image(c, splash(seed_of(sub.text), KURENAI), W / 2 - bw / 2, cy - bh / 2, bw, bh, alpha=G.clamp01(e * 1.5) * a)
        with grouped(c, a), placed(c, W / 2, cy, 1 + 0.32 * (1 - G.back(p, 2.4)) if whole else 0.94 + 0.06 * G.back(p, 2.0)):
            for j, (s, fl) in enumerate(lines):
                draw_row(c, s, fl, 0, y_first + j * gap - cy, size, face, SUB)
        if sub.speaker:
            with grouped(c, a):
                self.tag(c, [(sub.speaker, [False] * len(sub.speaker))], self.speaker_rgb(sub.speaker) or (60, 60, 64), self.x0,
                         y_first - size * 1.5, size * 0.62, face)
        if whole and tt < 0.1:
            c.drawRect(skia.Rect.MakeWH(W, H), G.paint(WHITE, 0.18 * (1 - tt / 0.1)))

    # -------------------------------------------------------------- title block (title_band and title card)
    def title_layout(self, text: str, max_w: float, scale: float = 1.0) -> list[tuple]:
        """[(plain, flags, size, face, ink)] of a title: with several rows the first is heavy 楷書 (white), the rest
        gold 行書; a long single row is broken in two; each row shrunk to max_w."""
        base, gold_face = self.v("title", "size") * scale, self.face("title")
        rows = self.wrapped(text, base, gold_face, max_w)
        out = []
        for j, (s, fl) in enumerate(rows):
            gold = len(rows) == 1 or j > 0
            face, size = (gold_face, base) if gold else (self.face("sub"), base * 0.7)
            out.append((s, fl, size * min(1.0, max_w / max(1.0, row_width(s, fl, size, face))), face,
                        TITLE_GOLD if gold else TITLE_KAI))
        return out

    def title_box(self, rows: list[tuple]) -> tuple[float, float, float]:
        """(block width, block height, padding) of title rows."""
        return max(row_width(s, fl, sz, face) for s, fl, sz, face, _ in rows), sum(r[2] * 1.28 for r in rows), rows[-1][2] * 0.6

    def title_block(self, c, text: str, cx: float, cy: float, tt: float, alpha: float = 1.0, scale: float = 1.0) -> None:
        """Crimson brush stroke wipes in, the lines stamp in, the 落款 lands; gold lines glint every 6 s."""
        rows = self.title_layout(text, (self.x1 - self.x0) * 0.9, scale)
        bw, bh, pad = self.title_box(rows)
        top, ss = cy - bh / 2, int(W * (0.11 if TALL else 0.065))
        seal = (self.brand.get("seal") or self.brand.get("name", "")[:2])[:4]
        with grouped(c, alpha):
            p = G.clamp01(tt / 0.32)
            if p > 0:
                bwid, bhgt = int(min(W * 1.02, bw + pad * 3) // 10 * 10), int((bh + pad * 1.6) // 10 * 10)
                with placed(c, cx, cy, 1.0, 2.2):
                    c.clipRect(skia.Rect.MakeLTRB(-bwid / 2, -bhgt, -bwid / 2 + p * bwid, bhgt))
                    G.draw_image(c, brush(bwid, bhgt, 3, KURENAI, 0.55, 0.22), -bwid / 2, -bhgt / 2, bwid, bhgt, alpha=0.92)
            y, mids = top, []
            for k, (s, fl, sz, face, ink) in enumerate(rows):
                q, step = (tt - 0.12 - 0.18 * k) / 0.28, sz * 1.28
                mids.append((y + step / 2, row_width(s, fl, sz, face)))
                if q > 0:
                    g = ((tt - 0.95) % 6.0) / 0.7 if tt > 0.95 and ink is TITLE_GOLD else 1.0
                    with placed(c, cx, y + step / 2, 1 + 0.45 * (1 - G.ease_out(q)) if q < 1 else 1.0):
                        draw_row(c, s, fl, 0, 0.36 * sz, sz, face, ink, G.clamp01(q * 2.5), "center", g if g < 1 else None)
                y += step
            q = (tt - 0.55) / 0.22
            if seal and q > 0:
                spot = next(((cx + rw / 2 + ss * 0.62, ry) for ry, rw in reversed(mids) if cx + rw / 2 + ss * 1.15 <= self.x1),
                            (min(cx + bw / 2, self.x1 - ss * 0.6), top + bh + ss * 0.1))
                with placed(c, *spot, 1 + 0.8 * (1 - G.ease_out(q)) if q < 1 else 1.0, -8):
                    G.draw_image(c, seal_image(seal, ss), -ss / 2, -ss / 2, ss, ss, alpha=G.clamp01(q * 3) * 0.95)

    def draw_title_band(self, c, ov, t) -> None:
        tt = t - ov.t0
        dust(c, tt, G.clamp01(tt / 0.6) * G.clamp01((ov.t1 - t) / 0.3), 24, 21)
        self.title_block(c, ov.text, W / 2, H * self.v("title", "y"), tt, G.clamp01((ov.t1 - t) / 0.25))

    def impact(self, ov, scale: float = 1.0) -> tuple[float, list]:
        """(size, rows) of an impact word at `scale`: one row shrunk to the safe width, broken in two when that would
        shrink it below 60 %."""
        size, face, avail = self.v("emphasis", "size") * scale, self.face("emphasis"), self.x1 - self.x0
        rows = rows_of(ov.text)
        if len(rows) == 1 and self.fit(rows, size, face, avail) < 0.6:
            rows = self.wrapped(ov.text, size, face, avail * 0.9)
        return size * self.fit(rows, size, face, avail), rows

    def spots(self, ov) -> list[tuple[float, float]]:
        """In 9:16 an impact word goes first under the chin (between the face and the subtitles), then above the head."""
        if ov.type == "emphasis" and TALL:
            return [(0.5, 0.57), (0.5, 0.31), (0.5, self.v("emphasis", "y"))]
        return super().spots(ov)

    def box(self, ov, center=None, scale: float = 1.0):
        if ov.type == "title_band":
            bw, bh, pad = self.title_box(self.title_layout(ov.text, (self.x1 - self.x0) * 0.9))
            return W / 2 - bw / 2 - pad, H * self.v("title", "y") - bh / 2 - pad, bw + 2 * pad, bh + 2 * pad
        if ov.type == "emphasis":
            cx, cy = center or self.spots(ov)[0]
            size, rows = self.impact(ov, scale)
            w, h = max(row_width(s, fl, size, self.face("emphasis")) for s, fl in rows), size * (1.1 * (len(rows) - 1) + 1.2)
            return W * cx - w / 2, H * cy - h / 2, w, h
        return super().box(ov, center, scale)

    def draw_credit(self, c, ov, t) -> None:
        size, face, avail = self.v("credit", "size"), self.face("credit"), self.x1 - self.x0
        rows = self.wrapped(ov.text, size, face, avail)
        ink = replace(LABEL, fill=self.v("credit", "fill"), strokes=((BLACK, 0.22),))
        draw_rows(c, rows, self.x1, self.y1 + size * 1.24 + (len(rows) - 1) * 0.55 * size, size * self.fit(rows, size, face, avail),
                  face, ink, 1.1, 0.9, "right")

    # -------------------------------------------------------------- chapter: crimson brush banner sliding in
    def draw_chapter(self, c, ov, t) -> None:
        k_in = G.ease_out((t - ov.t0) / 0.35)
        a = min(G.clamp01(k_in * 1.4), G.ease_io((ov.t1 - t) / 0.3))
        if a <= 0:
            return
        size, face, rows = self.v("chapter", "size"), self.face("chapter"), rows_of(ov.text)
        w = max(row_width(s, fl, size, face) for s, fl in rows) + size * 2.2
        h = size * (0.5 + 1.5 * len(rows))
        x, y = self.x0 - (1 - k_in) * (w + 40), H * self.v("chapter", "y")
        with grouped(c, a):
            G.draw_image(c, brush(int(w) // 10 * 10, int(h) // 10 * 10, seed_of(ov.text), KURENAI, 0.35, 0.14), x, y, w, h)
            c.drawRect(skia.Rect.MakeXYWH(x + size * 0.5, y + h * 0.92, w * 0.72, 3), gold_paint(y, y + h))
            draw_rows(c, rows, x + size * 1.0, y + h / 2, size, face, replace(SUB, shadow=(0.04, 0.07, 0.09, 0.6)), 1.3, 1.0, "left")

    # -------------------------------------------------------------- card: lacquer panel in a gold frame
    def draw_card(self, c, ov, t) -> None:
        tt = t - ov.t0
        a = G.clamp01(tt / 0.15) * G.ease_io((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        x, y, w, h = self.card_rect(ov)
        _, scale, see = self.placed(ov)
        size, face, gold_face = self.v("card", "size") * scale, self.face("card"), self.face("title")
        with grouped(c, a * see), placed(c, x + w / 2, y + h / 2, 0.85 + 0.15 * G.back(tt / 0.35)):
            x, y = -w / 2, -h / 2
            G.shadow(c, x, y, w, h, size * 0.2, blur=24, dy=14, a=0.5)
            c.drawRRect(G.rrect(x, y, w, h, size * 0.2), G.paint((20, 13, 12), 0.95))
            hh = size * 1.5
            c.save()
            c.clipRRect(G.rrect(x, y, w, h, size * 0.2), doAntiAlias=True)
            G.draw_image(c, brush(int(w * 1.1) // 10 * 10, int(hh) // 10 * 10, 5, KURENAI, 0.3, 0.1), x - w * 0.05, y, w * 1.1, hh)
            c.restore()
            frame(c, x + size * 0.08, y + size * 0.08, w - size * 0.16, h - size * 0.16, size / 110)
            if ov.d.get("title"):
                rows = rows_of(str(ov.d["title"]))
                sz = size * 0.95 * self.fit(rows, size * 0.95, gold_face, w - size * 1.6)
                draw_rows(c, rows, 0, y + hh / 2, sz, gold_face, TITLE_GOLD, 1.1)
            yy = y + hh + size * 0.4
            if ov.d.get("big"):
                draw_rows(c, rows_of(str(ov.d["big"])), 0, yy + size * 0.65, size * 2.0, gold_face, TITLE_GOLD)
                yy += size * 1.6
            for j, item in enumerate(ov.d.get("items", [])):
                ka, cy = G.clamp01((tt - 0.25 - 0.18 * j) / 0.2), yy + size * 0.6 + j * size * 1.55
                rows = rows_of(str(item))
                sz = size * (0.82 if len(rows) == 1 else 0.62)
                sz *= self.fit(rows, sz, face, w - size * 2.3)
                with placed(c, x + size * 0.85, cy, 1.0, 45):
                    c.drawRect(skia.Rect.MakeXYWH(-size * 0.14, -size * 0.14, size * 0.28, size * 0.28), gold_paint(-size * 0.3, size * 0.3, ka))
                draw_rows(c, rows, x + size * 1.4 + (1 - ka) * 24, cy, sz, face, BODY, 1.0, ka, "left")

    # -------------------------------------------------------------- emphasis: gold brush word on a crimson splash
    def draw_emphasis(self, c, ov, t) -> None:
        tt = t - ov.t0
        po = G.clamp01((ov.t1 - t) / 0.2)
        a = min(G.clamp01(tt / 0.08), po)
        if a <= 0:
            return
        (fx, fy), scale, see = self.placed(ov)
        size, rows = self.impact(ov, scale)
        face, shake = self.face("emphasis"), max(0.0, 1 - tt / 0.18)
        cx, cy = W * fx + 8 * shake * math.sin(tt * 160), H * fy + 6 * shake * math.cos(tt * 130)
        e = G.ease_out(tt / 0.25)
        sw = (max(row_width(s, fl, size, face) for s, fl in rows) * 1.25 + size) * (0.6 + 0.4 * e)
        sh = (size * (1.1 * (len(rows) - 1) + 1.2) * 1.5 + size * 0.4) * (0.6 + 0.4 * e)
        G.draw_image(c, splash(seed_of(ov.text), KURENAI), cx - sw / 2, cy - sh / 2, sw, sh, alpha=a * see * G.clamp01(e * 1.5))
        with grouped(c, a * see), placed(c, cx, cy, (1 + 0.6 * (1 - G.ease_out(tt / 0.2))) * (1 + 0.04 * (1 - po))):
            draw_rows(c, rows, 0, 0, size, face, TITLE_GOLD, 1.1, 1.0, "center", G.clamp01((tt - 0.3) / 0.7) if 0.3 < tt < 1.0 else None)
        if tt < 0.1:
            c.drawRect(skia.Rect.MakeWH(W, H), G.paint(WHITE, 0.16 * (1 - tt / 0.1) * see))

    # -------------------------------------------------------------- name plate / caption / note
    def draw_name(self, c, ov, t) -> None:
        tt = t - ov.t0
        k = G.ease_out(tt / 0.4)
        a = G.clamp01(k * 1.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        (_, fy), scale, see = self.placed(ov)
        size, face = self.v("name", "size") * scale, self.face("name")
        x = self.x0 - (1 - k) * 60
        with grouped(c, a * see):
            self.tag(c, rows_of(ov.text), self.speaker_rgb(ov.d.get("speaker") or ov.text) or SHU, x, H * fy, size, face)
            if ov.d.get("sub"):
                self.tag(c, rows_of(str(ov.d["sub"])), INK, x + size * 0.5, H * fy + size * 1.55, size * 0.62, face, 1.0)

    def draw_caption(self, c, ov, t) -> None:
        a = G.clamp01((t - ov.t0) / 0.35) * G.ease_io((ov.t1 - t) / 0.35)
        if a <= 0:
            return
        size, face, y = self.v("caption", "size"), self.face("caption"), H * self.v("caption", "y")
        rows = rows_of(ov.text)
        sub = rows_of(str(ov.d["sub"])) if ov.d.get("sub") else []
        y2 = y + (len(rows) - 1) * 1.25 * size + size * 0.95
        with grouped(c, a):
            c.drawRect(skia.Rect.MakeXYWH(self.x0, y - size * 0.6, 5, y2 - y + size * (0.55 * len(sub) + 0.1 if sub else 1.1)),
                       gold_paint(y, y2))
            draw_rows(c, rows, self.x0 + size * 0.45, y + (len(rows) - 1) * 0.625 * size, size, face, SUB, 1.25, 1.0, "left")
            draw_rows(c, sub, self.x0 + size * 0.45, y2 + (len(sub) - 1) * 0.375 * size, size * 0.6, face, SUB, 1.25, 1.0, "left")

    def draw_note(self, c, ov, t) -> None:
        a = G.clamp01((t - ov.t0) / 0.2) * G.ease_io((ov.t1 - t) / 0.2)
        if a <= 0:
            return
        size = self.v("sub", "size")
        rows = rows_of("※" + ov.text.lstrip("※"))
        draw_rows(c, rows, self.x0, H * self.v("sub", "y") - size * 2.9 - 0.2 * size - (len(rows) - 1) * 0.35 * size, size * 0.55,
                  self.face("sub"), SUB, 1.25, a, "left")

    # -------------------------------------------------------------- 9:16 card mode / band background
    def draw_card_mode(self, c, frame_img, box, k, face) -> None:
        super().draw_card_mode(c, frame_img, box, k, face)
        if k <= 0:
            return
        bx, by = W * self.BUBBLE[0], H * self.BUBBLE[1]
        rr = W * self.BUBBLE[2] * G.back(k, 1.6)
        c.drawCircle(bx, by, rr, stroked(gold_paint(by - rr, by + rr), 14))       # covers the base's accent ring
        c.drawCircle(bx, by, rr - 11, G.paint(SHU, 0.9, stroke=3))

    def warm_blur(self, c, img, dest, k: float) -> None:
        """A faint warm blur of the picture added onto the lacquer."""
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(28, 28, skia.TileMode.kClamp))
        p.setColorFilter(skia.ColorFilters.Matrix([k, 0, 0, 0, 0, 0, k * 0.66, 0, 0, 0, 0, 0, k * 0.55, 0, 0, 0, 0, 0, 1, 0]))
        p.setBlendMode(skia.BlendMode.kPlus)
        c.drawImageRect(img, skia.Rect.MakeXYWH(*dest), G.SAMPLING, p)

    def draw_band_bg(self, c, frame_img, box) -> None:
        """Behind 16:9 footage in 9:16: lacquer, a faint warm blur of the picture, gold dust, gold frame around the picture."""
        ox, oy, ow, oh = box
        c.drawImage(lacquer(W, H), 0, 0)
        z = max(W / ow, H / oh) * 1.1
        self.warm_blur(c, frame_img, (W / 2 - ow * z / 2, H / 2 - oh * z / 2, ow * z, oh * z), 0.30)
        dust(c, 0.0)
        frame(c, ox, oy, ow, oh)

    # -------------------------------------------------------------- full-screen cards
    def stage(self, c, bg, tt: float, inset: float = 0.045) -> None:
        """Lacquer with a faint warm blur of the picture, drifting gold dust and a gold frame inset in the canvas."""
        c.drawImage(lacquer(W, H), 0, 0)
        if bg is not None:
            self.warm_blur(c, bg, (0, 0, W, H), 0.34)
        dust(c, tt, G.clamp01(tt / 0.5))
        m = W * inset
        frame(c, m, m, W - 2 * m, H - 2 * m, 0.7, G.clamp01(tt / 0.4))

    def card_title(self, c, part, tt, dur, bg) -> None:
        self.stage(c, bg, tt)
        text = part.get("text") or self.brand.get("channel") or self.brand.get("name", "")
        cy = H * 0.5
        if self.brand.get("logo"):
            img = G.image(self.brand["logo"])
            lw = min(W * 0.34, H * 0.2 * img.width() / img.height())
            lh, k = lw * img.height() / img.width(), G.back(tt / 0.4)
            G.draw_image(c, img, W / 2 - lw / 2 * k, H * 0.30 - lh / 2 * k, lw * k, lh * k, alpha=G.clamp01(tt / 0.2))
            cy = H * 0.58
        self.title_block(c, text, W / 2, cy, tt, G.clamp01((dur - tt) / 0.3), 1.2 if TALL else 1.9)

    def card_text(self, c, part, tt, dur, bg) -> None:
        self.stage(c, bg, tt)
        size, face = 104 if not TALL else 96, self.face("sub")
        rows = self.wrapped(part.get("text", ""), size, face, W - 260)
        size *= self.fit(rows, size, face, W - 220)
        bw = max(row_width(s, fl, size, face) for s, fl in rows)
        bwid, bhgt = int(min(W * 1.02, bw + size * 3) // 10 * 10), int(size * (1.5 * len(rows) + 1.2) // 10 * 10)
        with grouped(c, G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.3)):
            with placed(c, W / 2, H / 2, 1.0, 2.0):
                G.draw_image(c, brush(bwid, bhgt, 7, KURENAI, 0.5, 0.2), -bwid / 2, -bhgt / 2, bwid, bhgt, alpha=0.9)
            draw_rows(c, rows, W / 2, H / 2, size, face, replace(SUB, shadow=(0.04, 0.08, 0.1, 0.7)), 1.35)

    def card_eyecatch(self, c, part, tt, dur, bg) -> None:
        """A black ink stroke sweeps across, the logo / text stamps on it, the stroke sweeps off."""
        c.drawImage(lacquer(W, H), 0, 0)
        bh = int(H * 0.30) // 10 * 10
        x = G.lerp(-W * 1.1, W * 1.1, G.ease_io5(tt / max(dur, 0.1)))
        G.draw_image(c, brush(int(W * 1.1) // 10 * 10, bh, 11, (6, 4, 4), 0.35, 0.18), x, H / 2 - bh / 2, W * 1.1, bh)
        a = G.win(tt, dur * 0.3, dur * 0.7, 0.15, 0.15)
        if self.brand.get("logo"):
            img = G.image(self.brand["logo"])
            lw = W * (0.5 if TALL else 0.3)
            lh = lw * img.height() / img.width()
            G.draw_image(c, img, W / 2 - lw / 2, H / 2 - lh / 2, lw, lh, alpha=a)
        elif part.get("text"):
            draw_rows(c, rows_of(part["text"]), W / 2, H / 2, 90, self.face("title"), TITLE_GOLD, 1.3, a)

    # -------------------------------------------------------------- end card: like / subscribe (or follow)
    LIKE_IN, SUB_IN, CLICKS, BELL = 0.2, 0.35, (0.9, 1.7), 2.35      # CLICKS match card_sfx

    def card_end(self, c, part, tt, dur, bg) -> None:
        self.stage(c, bg, tt)
        yt = self.fmt == "youtube"
        face, size = self.face("title"), 76 if TALL else 84
        rows = self.wrapped(part.get("text") or ("ご視聴ありがとうございました" if yt else "フォローして続きをチェック"), size, face, W - 260)
        with grouped(c, G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.25)):
            draw_rows(c, rows, W / 2, H * (0.25 if TALL else 0.20), size * self.fit(rows, size, face, W - 220), face, TITLE_GOLD, 1.3)
            self.buttons(c, tt, yt)

    def buttons(self, c, tt: float, yt: bool) -> None:
        """Two lacquer pills with gold rims pop in; a cursor clicks like (icon fills gold) then subscribe (登録済み, bell)."""
        s = 1.1 if TALL else 1.0
        labels = ("高評価", "チャンネル登録", "登録済み") if yt else ("いいね", "フォロー", "フォロー中")
        face, fs, bh, cy = self.face("sub"), 40 * s, 96 * s, H * 0.5
        width = lambda text: row_width(text, [False] * len(text), fs, face)
        lw, sw = width(labels[0]) + 150 * s, max(width(labels[1]), width(labels[2]) + 70 * s) + 100 * s
        x0 = W / 2 - (lw + 46 * s + sw) / 2
        like_c, sub_c = (x0 + lw / 2, cy), (x0 + lw + 46 * s + sw / 2, cy)
        ring = tt - self.BELL

        def label(text: str, x: float, alpha: float = 1.0, align: str = "left") -> None:
            draw_rows(c, [(text, [False] * len(text))], x, 0, fs, face, LABEL, 1.0, alpha, align)

        for i, (center, w, t_in, t_click) in enumerate(((like_c, lw, self.LIKE_IN, self.CLICKS[0]),
                                                       (sub_c, sw, self.SUB_IN, self.CLICKS[1]))):
            done, d = tt >= t_click, tt - t_click
            k = G.back(G.clamp01((tt - t_in) / 0.32), 2.4) if tt >= t_in else 0.0
            press = 1 - 0.08 * math.sin(math.pi * min(1.0, d / 0.12)) if 0 <= d < 0.12 else \
                1 + 0.06 * math.sin(math.pi * (d - 0.12) / 0.23) if 0.12 <= d < 0.35 else 1.0
            if k <= 0:
                continue
            with placed(c, *center, k * press):
                G.shadow(c, -w / 2, -bh / 2, w, bh, bh / 2, blur=12, dy=10, a=0.5)
                c.drawRRect(G.rrect(-w / 2, -bh / 2, w, bh, bh / 2), G.paint((90, 14, 20) if i == 0 and done else (30, 20, 18) if i == 0
                                                                           else (70, 64, 66) if done else KURENAI, 0.96))
                c.drawRRect(G.rrect(-w / 2, -bh / 2, w, bh, bh / 2), stroked(gold_paint(-bh / 2, bh / 2), 5 * s))
                if i == 0:
                    ts = 50 * s * (1 + 0.35 * math.sin(math.pi * G.clamp01(d / 0.3)) if done else 1.0)
                    with placed(c, -w / 2 + 36 * s, -ts / 2 - 2):
                        c.drawPath(thumb_path(ts), gold_paint(0, ts) if done else stroked(gold_paint(0, ts), 4.5 * s))
                    label(labels[0], -w / 2 + 110 * s)
                elif not done:
                    label(labels[1], 0, 1.0, "center")
                else:
                    label(labels[2], -w / 2 + 50 * s, 0.9)
                    if yt:                      # bell: grey, then gold and swinging
                        bs = 52 * s
                        with placed(c, w / 2 - 50 * s, 2, 1.0, 22 * math.sin(math.tau * 6.5 * ring) * math.exp(-ring * 2.8) if ring >= 0 else 0.0):
                            c.translate(-bs / 2, -bs * 0.44)
                            c.drawPath(bell_path(bs), gold_paint(0, bs) if ring >= 0 else G.paint((235, 235, 240), 1.0))
            if i == 0 and done and d < 0.7:     # gold sparks off the icon
                for j in range(8):
                    ang, r = j * math.tau / 8 + 0.3, 40 + 90 * G.ease_out(d / 0.7)
                    G.draw_image(c, sparkle(), center[0] - w / 2 + 61 * s + r * math.cos(ang) - 9, cy + r * math.sin(ang) - 9,
                                 alpha=1 - d / 0.7)
        if yt and tt >= self.BELL + 0.35:       # 「よろしくお願いします」 stamps in under the buttons
            with placed(c, W / 2, cy + 112 * s, G.back((tt - self.BELL - 0.35) / 0.32, 2.4)):
                draw_rows(c, [("よろしくお願いします", [False] * 10)], 0, 0, 46 * s, face, SUB)
        self.cursor(c, tt, like_c, sub_c, lw, sw, cy, s, yt)

    def cursor(self, c, tt: float, like_c, sub_c, lw: float, sw: float, cy: float, s: float, yt: bool) -> None:
        """Cursor path: in from the bottom right -> like -> subscribe -> bell -> away; click ripples stay put."""
        clicks = self.CLICKS + ((self.BELL,) if yt else ())
        keys = [(0.35, (W * 0.85, H * (0.72 if TALL else 0.9))), (clicks[0] - 0.05, (like_c[0] - lw / 2 + 70 * s, cy + 22 * s)),
                (clicks[1] - 0.05, (sub_c[0] - 40 * s, cy + 22 * s))]
        if yt:
            keys.append((self.BELL - 0.05, (sub_c[0] + sw / 2 - 70 * s, cy + 18 * s)))
        keys.append((clicks[-1] + 0.8, (W * 0.88, H * (0.75 if TALL else 0.93))))

        def at(tq: float) -> tuple[float, float]:
            for (ta, pa), (tb, pb) in zip(keys, keys[1:]):
                if ta <= tq <= tb:
                    u = G.ease_io(G.clamp01((tq - ta) / max(0.01, (tb - ta) * 0.8)))
                    return G.lerp(pa[0], pb[0], u), G.lerp(pa[1], pb[1], u)
            return keys[-1][1]

        if not keys[0][0] <= tt <= keys[-1][0]:
            return
        for tc in clicks:
            d = (tt - tc) / 0.4
            if 0 <= d < 1:
                c.drawCircle(*at(tc), 14 + 40 * G.ease_out(d), G.paint(GOLD, 0.8 * (1 - d), stroke=4))
        self._cursor(c, *at(tt), G.clamp01((tt - keys[0][0]) / 0.2) * G.clamp01((keys[-1][0] - tt) / 0.3), tt, clicks)

    # -------------------------------------------------------------- sound
    TITLE_HITS = [(0.0, "whoosh", -6.0), (0.10, "taiko", -4.0), (0.28, "taiko", -2.0), (0.55, "hyoshigi", -6.0)]

    def sfx(self, ov) -> list[tuple[float, str, float]]:
        """taiko on chapters and emphasis, hyoshigi on titles and cards; the title chimes with each glint."""
        if ov.type == "title_band":
            glints = [(0.95 + 6.0 * k + 0.1, "chime", -20.0) for k in range(int((ov.t1 - ov.t0 - 2.0) // 6.0) + 1)]
            return [(ov.t0 + dt, n, g) for dt, n, g in self.TITLE_HITS + glints]
        m = {"chapter": [(0.0, "taiko", -4.0)], "emphasis": [(0.0, "taiko", -1.0)], "card": [(0.0, "hyoshigi", -6.0)],
             "name": [(0.0, "hyoshigi", -9.0)], "caption": [(0.0, "tick", -10.0)], "note": [(0.0, "tick", -12.0)]}
        return [(ov.t0 + dt, name, g) for dt, name, g in m.get(ov.type, [])]

    def card_sfx(self, part: dict, t0: float, dur: float) -> list[tuple[float, str, float]]:
        card = part["card"]
        if card == "title":
            return [(t0 + dt, n, g) for dt, n, g in self.TITLE_HITS]
        if card == "end":
            fx = [(t0 + self.CLICKS[0], "click", -4.0), (t0 + self.CLICKS[1], "click", -4.0)]
            return fx + ([(t0 + self.CLICKS[1] + 0.05, "chime", -6.0)] if self.fmt == "youtube" else [])
        if card == "eyecatch":
            return [(t0, "whoosh", -4.0), (t0 + dur * 0.4, "taiko", -4.0)]
        return [(t0, "taiko", -4.0)]


LOOK = Wa
