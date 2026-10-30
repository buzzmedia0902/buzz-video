"""Phone-feed preview: the thumbnails at the sizes viewers really see them, next to the title.

A thumbnail has to win at 168 px wide (the sidebar / search list), not at 1280: this sheet shows each option as a
home-feed card (dark mode, 2x of a 390 pt phone is 780 px; shown at 640), as a grid card (360) and as a list row (168)
with the planned title and channel name. 9:16 covers show the feed card (270 x 480), the 3:4 profile-grid tile
(the platforms crop the middle of a cover) and a tiny tile.
"""
from __future__ import annotations

import numpy as np
import skia
from PIL import Image

from cv import gfx as G

BG, FG, DIM = (15, 15, 15), (241, 241, 241), (170, 170, 170)


def _wrap(s: str, f: skia.Font, width: float, lines: int = 2) -> list[str]:
    """Greedy wrap at spaces when the title has them (a number is not split), else at any character."""
    toks = [t + " " for t in s.split(" ")[:-1]] + s.split(" ")[-1:] if " " in s else list(s)
    out, cur = [], ""
    for i, tok in enumerate(toks):
        if cur and G.text_width((cur + tok).rstrip(), f) > width:
            out.append(cur.rstrip())
            cur = ""
            if len(out) == lines:
                out[-1] = out[-1][:-1] + "…"
                return out
        cur += tok
    return out + [cur.rstrip()] if cur.strip() else out


def _chip(c: skia.Canvas, text: str, right: float, bottom: float, size: float) -> None:
    f = G.jp(size, 600)
    w = G.text_width(text, f) + size * 0.8
    c.drawRRect(G.rrect(right - w - 6, bottom - size * 1.6 - 6, w, size * 1.6, size * 0.3), G.paint(G.BLACK, 0.85))
    G.draw_text_mid(c, text, right - w / 2 - 6, bottom - size * 0.8 - 6, f, G.paint(G.WHITE), "center")


def _avatar(c: skia.Canvas, cx: float, cy: float, r: float, icon: skia.Image | None, letter: str, rgb) -> None:
    c.save()
    clip = skia.Path()
    clip.addCircle(cx, cy, r)
    c.clipPath(clip, doAntiAlias=True)
    if icon is not None:
        G.draw_image(c, icon, cx - r, cy - r, 2 * r, 2 * r)
    else:
        c.drawRect(skia.Rect.MakeXYWH(cx - r, cy - r, 2 * r, 2 * r), G.paint(rgb))
        G.draw_text_mid(c, letter, cx, cy, G.jp(r * 1.1, 900), G.paint(G.BLACK), "center")
    c.restore()


def _meta(c: skia.Canvas, x: float, y: float, width: float, title: str, channel: str, size: float) -> None:
    ft, fm = G.jp(size, 600), G.jp(size * 0.78, 400)
    for line in _wrap(title, ft, width):
        G.draw_text(c, line, x, y, ft, G.paint(FG))
        y += size * 1.38
    G.draw_text(c, channel, x, y + size * 0.1, fm, G.paint(DIM))


def render(items: list[tuple[str, str]], title: str, channel: str, out: str, duration: str = "9:21",
           icon: str | None = None, accent=(254, 219, 46), vertical: bool = False) -> str:
    """items = [(label, image path)]; writes a JPEG to `out`."""
    letter = (channel or "?")[0]
    ico = G.image(icon) if icon else None
    w, rows = 1120, len(items)
    row_h = 600 if vertical else 540
    surf = skia.Surface(w, row_h * rows + 20)
    c = surf.getCanvas()
    c.clear(G.col(BG))
    for i, (label, path) in enumerate(items):
        y = 20 + i * row_h
        G.draw_text(c, label, 24, y + 30, G.jp(26, 800), G.paint(accent))
        img = G.image(path)
        if vertical:
            _vertical_row(c, img, y + 56, title, channel, duration, ico, letter, accent)
        else:
            _wide_row(c, img, y + 56, title, channel, duration, ico, letter, accent)
    arr = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]
    Image.fromarray(np.ascontiguousarray(arr)).save(out, quality=90)
    return out


def _wide_row(c, img, y, title, channel, duration, ico, letter, accent) -> None:
    G.draw_image(c, img, 24, y, 640, 360, sharp=True)
    _chip(c, duration, 24 + 640, y + 360, 20)
    _avatar(c, 24 + 28, y + 360 + 44, 24, ico, letter, accent)
    _meta(c, 24 + 70, y + 360 + 40, 640 - 90, title, channel, 25)
    x2 = 700
    G.draw_image(c, img, x2, y, 360, 202, sharp=True)           # grid card
    _chip(c, duration, x2 + 360, y + 202, 14)
    _meta(c, x2, y + 202 + 32, 360, title, channel, 19)
    ys = y + 202 + 130
    G.draw_image(c, img, x2, ys, 168, 94, sharp=True)           # list row: the size that decides
    _chip(c, duration, x2 + 168, ys + 94, 11)
    _meta(c, x2 + 184, ys + 16, 176, title, channel, 16)


def _vertical_row(c, img, y, title, channel, duration, ico, letter, accent) -> None:
    G.draw_image(c, img, 24, y, 270, 480, sharp=True)           # 9:16 feed card
    x2 = 330
    iw, ih = img.width(), img.height()
    crop = (0, ih * 0.125, iw, ih * 0.75)                       # the middle 3:4 (profile grid)
    G.draw_image(c, img, x2, y, 270, 360, src=crop, sharp=True)
    _avatar(c, x2 + 16, y + 360 + 34, 16, ico, letter, accent)
    _meta(c, x2 + 42, y + 360 + 30, 230, title, channel, 18)
    x3 = 650
    for k in range(3):                                          # a grid of the tiny tiles
        G.draw_image(c, img, x3 + k * 128, y, 120, 160, src=crop, sharp=True)
    G.draw_text(c, "プロフィールの一覧（3列）での見え方", x3, y + 188, G.jp(17, 400), G.paint(DIM))
