"""The three YouTube directions (1280x720) and the 9:16 cover (1080x1920), styled by a Theme.

  reaction      his own line in a speech bubble + an expressive face (+ the thing he reacts to)
  before_after  the SAME thing wrong (✕) -> right (○), big title, a pleased face
  authority     credibility badge + huge copy + a confident smile
  cover         vertical cover of a Short / clip: title huge in the upper half, the speaker below
YouTube "Test & compare" takes up to 3 thumbnails; the first uploaded is the fallback, so reaction goes first.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import skia

from cv import gfx as G
from cv.thumbs import cutout, draw
from cv.thumbs.theme import BLACK, WHITE, Theme

TW, TH = 1280, 720
CW, CH = 1080, 1920
CHIP = (TW - 170, TH - 60, 170, 60)     # the duration chip: keep text and badges out of it


def chip_clear(what: str, x: float, y: float, w: float, h: float) -> None:
    """Warn when something readable sits under YouTube's duration chip (bottom-right)."""
    cx, cy, cw, ch = CHIP
    if x < cx + cw and x + w > cx and y < cy + ch and y + h > cy:
        print(f"[thumb] {what} reaches the bottom-right corner (the duration chip): shorten it", file=sys.stderr)


@dataclass
class Copy:
    """What the user wants on the thumbnails (all optional but `copy`)."""
    copy: str
    sub: str = ""
    badge: str = ""
    thing: object = None      # reaction: the thing he reacts to (text | skia.Image)
    before: object = None     # before / after: text | skia.Image
    after: object = None
    title: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Placed:
    face: tuple     # Vision face box on the canvas (x, y, w, h)
    head: tuple     # (centre x, top y, width, height) of the head on the canvas


def surface(w: int = TW, h: int = TH) -> tuple[skia.Surface, skia.Canvas]:
    s = skia.Surface(w, h)
    return s, s.getCanvas()


def to_rgb(surf: skia.Surface) -> np.ndarray:
    return surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]


def put_speaker(c: skia.Canvas, theme: Theme, cut: cutout.Cutout, cx: float, top: float, head_h: float,
                size: tuple = (TW, TH)) -> Placed:
    """Draw the sticker so the head centre is at x = cx and the hair top at y = top, the head `head_h` px tall."""
    s = head_h / cut.head_h
    ax, ay = cut.face_cx, cut.top
    view = (ax - cx / s, ay - top / s, ax + (size[0] - cx) / s, ay + (size[1] - top) / s)
    st = cutout.sticker(cut, s, view, **{"stroke": 10, "edge": 3, **theme.sticker})
    st.draw(c, (ax, ay), (cx, top))
    fx, fy, fw, fh = cut.face
    return Placed((cx + (fx - ax) * s, top + (fy - ay) * s, fw * s, fh * s), (cx, top, cut.head_w * s, head_h))


def _card_content(theme: Theme, spec, role: str):
    """Draw function (design size) of a before / after / thing: a picture or `label｜value` text."""
    if isinstance(spec, skia.Image):
        return lambda c: draw.image_card(c, spec)
    return lambda c: draw.text_card(c, theme, str(spec), role)


# ---------------------------------------------------------------- reaction
def reaction(theme: Theme, cp: Copy, cut: cutout.Cutout) -> np.ndarray:
    surf, c = surface()
    draw.ground(c, theme, "gradient")
    if theme.key != "trust":
        draw.deco(c, theme)
    draw.glow(c, 380, 430, 720, cp.extra.get("glow") or theme.glow, 0.45 if theme.dark else 0.30)
    draw.glow(c, 1020, 340, 340, theme.accent, 0.30)
    if theme.shape == "burst":
        draw.rays(c, 1010, 330, n=36, rgb=theme.hot, a=0.5)
    draw.speed_lines(c, 1010, 340, WHITE if theme.dark else theme.ink, 0.07 if theme.dark else 0.06, n=60)
    lines = draw.split_long(draw.parse_copy(cp.copy, "last"))
    # the bubble sizes itself around its text, the card sits under it
    box_w, box_h = (690, 250) if cp.thing is not None else (700, 360)
    bx, by = 34, 30 if cp.thing is not None else 78
    fam, wt = theme.font
    sizes = draw.fit_sizes(lines, theme, box_w - 60, box_h - 40, cap=150 if cp.thing is None else 128, small=0.82)
    th_h = sum(s * 1.04 for s in sizes)
    bw = min(box_w + 10, max(draw.line_width(ln, theme, s) for ln, s in zip(lines, sizes)) + 90)
    bh = th_h + 56
    if cp.thing is not None:
        draw.card(c, _card_content(theme, cp.thing, "after"), 40, by + bh - 40, 640, -5,
                  frame=(60, 52, 110) if theme.dark else WHITE, frame_w=6)
    placed = put_speaker(c, theme, cut, 1022, 150, 400)
    fx, fy, fw, fh = placed.face
    draw.bubble(c, theme, bx, by, bw, bh, (fx - fw * 0.02, fy + fh * 0.58))
    yy = by + 30
    for ln, sz in zip(lines, sizes):
        hot = any(s.hot for s in ln)
        fill = theme.hot if hot else theme.ink if not theme.dark or theme.key == "friendly" else BLACK
        draw.runs(c, [(s.text, sz, fill) for s in ln], bx + bw / 2, yy + sz * 0.86, fam, wt,
                  layers=((BLACK, 0.12),) if hot else (), skew=-0.08, shadow=0, align="center")
        yy += sz * 1.04
    if cp.sub:
        sy = TH - 26 - 77 if cp.thing is not None else min(by + bh + 70, TH - 26 - 77)
        sw, sh = draw.strip(c, theme, cp.sub, 40, sy, 54, 640)
        chip_clear("the support line", 40, sy, sw, sh)
    return to_rgb(surf)


# ---------------------------------------------------------------- before / after
def before_after(theme: Theme, cp: Copy, cut: cutout.Cutout) -> np.ndarray:
    surf, c = surface()
    c.clear(G.col(theme.bg))
    draw.deco(c, theme)
    split = skia.Path()
    split.moveTo(760, 0)
    split.lineTo(TW, 0)
    split.lineTo(TW, TH)
    split.lineTo(640, TH)
    split.close()
    c.save()
    c.clipPath(split, doAntiAlias=True)
    c.drawRect(skia.Rect.MakeWH(TW, TH), G.paint({"accent": theme.accent, "bg2": theme.bg2, "hot": theme.hot}[theme.split]))
    draw.rays(c, 1010, 250, n=40, rgb=theme.metal if theme.key == "wa" else WHITE, a=0.25 if theme.key == "wa" else 0.35)
    c.restore()
    put_speaker(c, theme, cut, 1012, 132, 372)
    lines = draw.split_long(draw.parse_copy(cp.copy, "first"), 6)
    x, y, w, h = draw.block(c, theme, lines, (30, 22, 700, 380), cap=150, small=1.0)
    chip_clear("the title", x, y, w, h)
    draw.card(c, _card_content(theme, cp.before, "before"), 24, 424, 330, -4)
    draw.card(c, _card_content(theme, cp.after, "after"), 414, 424, 330, 3)
    draw.arrow(c, 352, 534, 410, 60, theme.accent)
    draw.mark_x(c, theme, 330, 430, 38)
    draw.mark_ok(c, theme, 726, 428, 38)
    return to_rgb(surf)


# ---------------------------------------------------------------- authority
def authority(theme: Theme, cp: Copy, cut: cutout.Cutout) -> np.ndarray:
    surf, c = surface()
    draw.ground(c, theme, "gradient" if theme.key != "trust" else "plain")
    draw.deco(c, theme)
    draw.backdrop(c, theme, 1010, 360, 330)
    put_speaker(c, theme, cut, 1010, 150, 382)
    top = 28
    if cp.badge:
        bw, bh = draw.badge(c, theme, draw.parse_badge(cp.badge), 26, 22, max_w=760)
        top = 22 + bh + 22
    lines = draw.split_long(draw.parse_copy(cp.copy, "tail"), 7)
    bottom = TH - 26 - (84 if cp.sub else 0)
    x, y, w, h = draw.block(c, theme, lines, (30, top, 640, bottom - top), cap=200, valign="middle")
    chip_clear("the copy", x, y, w, h)
    if cp.sub:
        sw, sh = draw.strip(c, theme, cp.sub, 34, TH - 26 - 62, 44, 560, rot=-1.5)
        chip_clear("the support line", 34, TH - 26 - 62, sw, sh)
    return to_rgb(surf)


# ---------------------------------------------------------------- covers of Shorts / Reels / clips (1080x1920)
# Platforms crop covers differently: the profile grid shows the middle 3:4 (y 240..1680), the feed all of 9:16.
# Title and badge stay inside the 3:4 band; the speaker's chin stays above y = 1650.
def cover_poster(theme: Theme, cp: Copy, cut: cutout.Cutout) -> np.ndarray:
    """Poster cover: tone ground, the speaker big in the lower half, the title huge above his head."""
    surf, c = surface(CW, CH)
    draw.ground(c, theme, "gradient")
    draw.deco(c, theme)
    draw.glow(c, 300, 1200, 900, theme.glow, 0.40 if theme.dark else 0.25)
    draw.backdrop(c, theme, 540, 1290, 470)
    put_speaker(c, theme, cut, 540, 900, 600, (CW, CH))
    top = 270
    if cp.badge:
        _, bh = draw.badge(c, theme, draw.parse_badge(cp.badge), 50, 252, max_w=980)
        top = 252 + bh + 28
    lines = draw.split_long(draw.parse_copy(cp.copy, "tail"), 6)
    bottom = 860 - (96 if cp.sub else 0)
    draw.block(c, theme, lines, (60, top, 960, bottom - top), cap=230, small=0.8, valign="middle")
    if cp.sub:
        draw.strip(c, theme, cp.sub, 60, 860 - 84, 54, 900, rot=-1.5)
    return to_rgb(surf)


def cover_photo(theme: Theme, cp: Copy, photo: skia.Image, face: tuple) -> np.ndarray:
    """Photo cover: the frame itself (cover-fit around the face), a dark gradient below, the title over it."""
    surf, c = surface(CW, CH)
    iw, ih = photo.width(), photo.height()
    k = max(CW / iw, CH / ih)
    sw, sh = CW / k, CH / k
    fx = face[0] + face[2] / 2
    sx = min(max(fx - sw / 2, 0), iw - sw)
    G.draw_image(c, photo, 0, 0, CW, CH, src=(sx, (ih - sh) / 2, sw, sh), sharp=True)
    shade = skia.Paint(AntiAlias=True)
    shade.setShader(G.lin_grad((0, 980), (0, CH), [BLACK, BLACK], [0.0, 0.88]))
    c.drawRect(skia.Rect.MakeXYWH(0, 980, CW, CH - 980), shade)
    lines = draw.split_long(draw.parse_copy(cp.copy, "tail"), 6)
    top = 1130
    if cp.badge:
        _, bh = draw.badge(c, theme, draw.parse_badge(cp.badge), 50, 1080, max_w=980)
        top = 1080 + bh + 20
    draw.block(c, theme, lines, (60, top, 960, 1590 - top - (90 if cp.sub else 0)), cap=220, small=0.8)
    if cp.sub:
        draw.strip(c, theme, cp.sub, 60, 1590 - 76, 50, 900, rot=-1.5)
    return to_rgb(surf)


DIRECTIONS = {"reaction": reaction, "before_after": before_after, "authority": authority}
LABELS = {"reaction": "リアクション", "before_after": "ビフォーアフター", "authority": "権威"}
