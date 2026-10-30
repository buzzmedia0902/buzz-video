"""Thumbnail drawing kit (skia): big outlined copy that fits a box, badges, speech bubbles, marks, backgrounds.

Tone-aware through cv.thumbs.theme.Theme.
Rules: copy and badge text must survive a 168 px wide feed thumbnail (key words >= ~70 px), the
bottom-right corner belongs to the duration chip, no half-hidden elements, a copy line must not repeat the title.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import skia

from cv import gfx as G
from cv.thumbs.theme import BLACK, WHITE, Theme


def canvas_size(c: skia.Canvas) -> tuple[int, int]:
    s = c.getBaseLayerSize()
    return s.width(), s.height()


# ---------------------------------------------------------------- fonts with a Latin fallback
BRUSH = ("xingkai", "kaiti", "Xingkai SC", "Kaiti SC")     # their Latin letters read badly (AI looks like A7)


def chunks(text: str, fam: str, wt: int) -> list[tuple[str, str, int]]:
    """[(text, family, weight)]: letters and digits of a brush font are set in the heavy gothic instead."""
    if fam not in BRUSH:
        return [(text, fam, wt)]
    out: list[tuple[str, str, int]] = []
    for ch in text:
        f = ("gothic", 900) if ch.isascii() and ch.isalnum() else (fam, wt)
        if out and out[-1][1:] == f:
            out[-1] = (out[-1][0] + ch, *f)
        else:
            out.append((ch, *f))
    return out


def text_w(text: str, size: float, fam: str, wt: int) -> float:
    return sum(G.text_width(t, G.font(size, w, f)) for t, f, w in chunks(text, fam, wt))


def put(c: skia.Canvas, text: str, x: float, y: float, size: float, fam: str, wt: int, paint: skia.Paint,
        align: str = "left", mid: bool = False) -> float:
    """Draw text with the fallback; y is the baseline (or the visual middle with mid=True)."""
    total = text_w(text, size, fam, wt)
    x -= total / 2 if align == "center" else total if align == "right" else 0
    for t, f, w in chunks(text, fam, wt):
        font = G.font(size, w, f)
        (G.draw_text_mid if mid else G.draw_text)(c, t, x, y, font, paint)
        x += G.text_width(t, font)
    return total


# ---------------------------------------------------------------- copy
@dataclass
class Seg:
    text: str
    hot: bool = False


Line = list[Seg]


def parse_copy(copy: str, policy: str = "last") -> list[Line]:
    """`｜` / newline split lines, {..} marks the hot words. Without markup the policy decides which lines are hot:
    last (only the last), tail (all but the first), first, all, none."""
    lines: list[Line] = []
    marked = "{" in copy
    for raw in copy.replace("\n", "｜").split("｜"):
        segs, cur, hot = [], "", False
        for ch in raw:
            if ch in "{}":
                if cur:
                    segs.append(Seg(cur, hot))
                cur, hot = "", ch == "{"
            else:
                cur += ch
        if cur:
            segs.append(Seg(cur, hot))
        if segs:
            lines.append(segs)
    n = len(lines)
    if not marked:
        for i, ln in enumerate(lines):
            hot = {"last": i == n - 1, "tail": i > 0 or n == 1, "first": i == 0, "all": True, "none": False}[policy]
            for s in ln:
                s.hot = hot
    return lines


def plain(line: Line) -> str:
    return "".join(s.text for s in line)


def split_long(lines: list[Line], limit: int = 7) -> list[Line]:
    """A single long line is broken once at the best Japanese break (cv.linebreak) so it can get big."""
    if len(lines) != 1 or len(plain(lines[0])) <= limit:
        return lines
    from cv import linebreak
    text = plain(lines[0])
    costs = linebreak.break_costs(text)
    if not costs:
        return lines
    i = min(costs, key=lambda k: costs[k] + abs(k - len(text) / 2) * 6)
    out, pos = [[], []], 0
    for s in lines[0]:
        a, b = s.text[: max(0, i - pos)], s.text[max(0, i - pos):]
        if a:
            out[0].append(Seg(a, s.hot))
        if b:
            out[1].append(Seg(b, s.hot))
        pos += len(s.text)
    return out


def _width(line: Line, fam: str, wt: int) -> float:
    """Width of a line at font size 1."""
    return sum(text_w(s.text, 100, fam, wt) for s in line) / 100


def line_width(line: Line, theme: Theme, size: float) -> float:
    return _width(line, *theme.font) * size


def fit_sizes(lines: list[Line], theme: Theme, box_w: float, box_h: float, cap: float, small: float = 0.52,
              leading: float = 1.04) -> list[float]:
    """Font size per line: hot lines fill the width (up to `cap`), the others are `small` x the biggest hot size;
    then everything is scaled to the box height. The outline (0.32 em) counts as width."""
    fam, wt = theme.font
    fits = [min(cap, box_w / (_width(ln, fam, wt) + 0.32)) for ln in lines]
    hot = [any(s.hot for s in ln) for ln in lines]
    top = max((f for f, h in zip(fits, hot) if h), default=cap)
    sizes = [f if h else min(f, top * small) for f, h in zip(fits, hot)]
    total = sum(s * leading for s in sizes)
    return [s * box_h / total for s in sizes] if total > box_h else sizes


def runs(c: skia.Canvas, items, x: float, y: float, fam: str, wt: int, layers=((BLACK, 0.26),), skew: float = -0.10,
         shadow: float = 0.55, align: str = "left") -> float:
    """Text runs [(text, size, fill)] on one baseline: every outline pass under every fill. fill: rgb or a
    (top, bottom) gradient. layers: (rgb, stroke width in em), outermost first - one tuple for all runs or a list
    with one tuple per run."""
    per_in = layers if layers and isinstance(layers[0], list) else [layers] * len(items)
    subs = []       # (text, size, fill, font, layers)
    for (s, sz, fill), ls in zip(items, per_in):
        for t, f, w in chunks(s, fam, wt):
            subs.append((t, sz, fill, G.font(sz, w, f), ls))
    widths = [G.text_width(t, f) for t, _, _, f, _ in subs]
    total = sum(widths)
    x0 = x - (total if align == "right" else total / 2 if align == "center" else 0)
    xs = list(np.cumsum([x0] + widths[:-1]))
    c.save()
    c.translate(0, y)
    c.skew(skew, 0)
    c.translate(0, -y)
    outer = max((ls[0][1] for *_, ls in subs if ls), default=0.0)
    if shadow > 0 and outer > 0:
        for (t, sz, _, f, _), xx in zip(subs, xs):
            G.draw_text(c, t, xx + sz * 0.03, y + sz * 0.06, f,
                        G.paint(BLACK, shadow, stroke=outer * sz + 2, blur=sz * 0.06, cap_round=True))
    for k in range(max((len(ls) for *_, ls in subs), default=0)):
        for (t, sz, _, f, ls), xx in zip(subs, xs):
            if k < len(ls):
                G.draw_text(c, t, xx, y, f, G.paint(ls[k][0], stroke=ls[k][1] * sz, cap_round=True))
    for (t, sz, fill, f, _), xx in zip(subs, xs):
        if isinstance(fill[0], tuple):
            p = skia.Paint(AntiAlias=True)
            p.setShader(G.lin_grad((0, y - sz * 0.82), (0, y + sz * 0.05), list(fill)))
        else:
            p = G.paint(fill)
        G.draw_text(c, t, xx, y, f, p)
    c.restore()
    return total


def block(c: skia.Canvas, theme: Theme, lines: list[Line], box: tuple, cap: float = 210, small: float = 0.52,
          align: str = "left", valign: str = "top") -> tuple[float, float, float, float]:
    """Draw copy lines inside box (x, y, w, h); returns the area really used (x, y, w, h)."""
    x, y, w, h = box
    fam, wt = theme.font
    sizes = fit_sizes(lines, theme, w, h, cap, small)
    heights = [s * 1.04 for s in sizes]
    yy = y + (h - sum(heights)) / 2 if valign == "middle" else y
    top, used_w = yy, 0.0
    for ln, sz, hh in zip(lines, sizes, heights):
        base = yy + sz * 0.86
        items = [(s.text, sz, theme.hero_fill if s.hot else theme.plain_fill) for s in ln]
        per = [list(theme.hero_layers if s.hot else theme.plain_layers) for s in ln]
        lw = line_width(ln, theme, sz)
        px = x + (w - lw) / 2 if align == "center" else x + (w - lw if align == "right" else 0)
        if theme.marker and any(s.hot for s in ln):
            c.save()
            c.translate(0, base)
            c.skew(theme.skew, 0)
            c.translate(0, -base)
            c.drawRRect(G.rrect(px - sz * 0.12, base - sz * 0.74, lw + sz * 0.24, sz * 0.62, sz * 0.18),
                        G.paint(theme.accent, 0.92))
            c.restore()
        if theme.underline and any(s.hot for s in ln):
            c.drawRect(skia.Rect.MakeXYWH(px, base + sz * 0.10, lw, sz * 0.07), G.paint(theme.accent))
        runs(c, items, px, base, fam, wt, layers=per, skew=theme.skew,
             shadow=0.5 if theme.dark or theme.key == "variety" else 0.18)
        used_w = max(used_w, lw)
        yy += hh
    return x, top, used_w, yy - top


# ---------------------------------------------------------------- badge, bubble, strip
def badge(c: skia.Canvas, theme: Theme, parts: list[tuple[str, bool]], x: float, y: float, h: float = 106,
          max_w: float = 1200, rot: float = 0.0) -> tuple[float, float]:
    """Credibility plate: [tag] HERO rest. parts = [(text, role)] with role 'tag' | 'hero' | 'rest'. Hero text is
    ~72 px so it still reads at 168 px wide. Returns (w, h)."""
    fam, wt = theme.font2
    hfam, hwt = theme.font
    spec = {"tag": (30, wt, fam), "hero": (72, hwt, hfam), "rest": (46, wt, fam)}
    k = 1.0
    for _ in range(2):
        widths = [text_w(t, spec[r][0] * k, spec[r][2], spec[r][1]) + (26 * k if r == "tag" else 0)
                  for t, r in parts]
        w = 18 * k + sum(widths) + 16 * k * (len(parts) - 1) + 26 * k
        if w <= max_w:
            break
        k = max_w / w
    hh = h * k
    cy = y + hh / 2
    c.save()
    c.translate(x + w / 2, cy)
    c.rotate(rot)
    c.translate(-(x + w / 2), -cy)
    r_out = 18 * k if theme.badge == "plate" else hh / 2 if theme.badge == "pill" else 6
    face = (20, 16, 8) if theme.badge == "plate" else theme.accent if theme.badge == "pill" else theme.hot
    c.drawRRect(G.rrect(x + 5, y + 9, w, hh, r_out), G.paint(BLACK, 0.55, blur=8))
    c.drawRRect(G.rrect(x, y, w, hh, r_out), G.paint(face))
    c.drawRRect(G.rrect(x, y, w, hh, r_out), G.paint(theme.metal if theme.badge != "pill" else
                                                     (theme.ink if theme.key == "friendly" else BLACK), stroke=5))
    tx = x + 18 * k
    for (t, role), wd in zip(parts, widths):
        size, weight, family = spec[role][0] * k, spec[role][1], spec[role][2]
        if role == "tag":
            pill = theme.badge == "pill"
            c.drawRRect(G.rrect(tx, cy - 25 * k, wd, 50 * k, 10 * k), G.paint(BLACK if pill else theme.metal))
            put(c, t, tx + wd / 2, cy, size, family, weight, G.paint(WHITE if pill else BLACK), "center", True)
        elif role == "hero":
            p = skia.Paint(AntiAlias=True)
            if theme.badge == "plate":
                p.setShader(G.lin_grad((0, cy - 34 * k), (0, cy + 30 * k),
                                       [G.mix_rgb(theme.metal, WHITE, 0.55), G.mix_rgb(theme.metal, BLACK, 0.12)]))
            else:
                p.setColor(G.col(BLACK if theme.badge == "pill" else WHITE))
            put(c, t, tx, cy, size, family, weight, p, mid=True)
        else:
            put(c, t, tx, cy + 6 * k, size, family, weight, G.paint(WHITE if theme.badge != "pill" else BLACK), mid=True)
        tx += wd + 16 * k
    c.restore()
    return w, hh


def parse_badge(text: str) -> list[tuple[str, str]]:
    """`経営者｜AI導入｜の現場から` -> tag / hero / rest. {..} marks the hero; else 1 part = hero,
    2 = tag + hero, 3+ = tag + hero + rest."""
    segs = [s for s in text.split("｜") if s]
    for i, s in enumerate(segs):
        if "{" in s:
            segs = [x.replace("{", "").replace("}", "") for x in segs]
            return ([(t, "tag") for t in segs[:i]] + [(segs[i], "hero")] + [(t, "rest") for t in segs[i + 1:]])
    if len(segs) == 1:
        return [(segs[0], "hero")]
    if len(segs) == 2:
        return [(segs[0], "tag"), (segs[1], "hero")]
    return [(segs[0], "tag"), (segs[1], "hero")] + [(t, "rest") for t in segs[2:]]


def strip(c: skia.Canvas, theme: Theme, text: str, x: float, y: float, size: float, max_w: float,
          rot: float = -2.0) -> tuple[float, float]:
    """Accent label with a black border (the small support line); text shrinks to max_w."""
    fam, wt = theme.font2
    wt = 900 if wt >= 800 else wt
    pad = size * 0.42
    w = text_w(text, size, fam, wt) + 2 * pad
    if w > max_w:
        size *= max_w / w
        w = max_w
    h = size * 1.42
    c.save()
    c.translate(x + w / 2, y + h / 2)
    c.rotate(rot)
    c.drawRRect(G.rrect(-w / 2 + 4, -h / 2 + 7, w, h, size * 0.22), G.paint(BLACK, 0.5, blur=6))
    c.drawRRect(G.rrect(-w / 2, -h / 2, w, h, size * 0.22), G.paint(theme.accent))
    c.drawRRect(G.rrect(-w / 2, -h / 2, w, h, size * 0.22), G.paint(BLACK if theme.key != "friendly" else theme.ink,
                                                                     stroke=size * 0.08))
    put(c, text, 0, 0, size, fam, wt, G.paint(BLACK if theme.key != "friendly" else theme.ink), "center", True)
    c.restore()
    return w, h


def bubble(c: skia.Canvas, theme: Theme, x: float, y: float, w: float, h: float, tail: tuple[float, float]) -> None:
    """Speech bubble with a pointed tail to `tail` (a point on the speaker's face)."""
    body = skia.Path()
    body.addRRect(G.rrect(x, y, w, h, h * 0.32))
    t = skia.Path()
    bx = x + w * 0.80
    t.moveTo(bx - 40, y + h - 10)
    t.lineTo(tail[0], tail[1])
    t.lineTo(bx + 30, y + h - 30)
    t.close()
    both = skia.Op(body, t, skia.PathOp.kUnion_PathOp)
    c.save()
    c.translate(6, 12)
    c.drawPath(both, G.paint(BLACK, 0.55, blur=10))
    c.restore()
    c.drawPath(both, G.paint(theme.bubble_line, stroke=20, cap_round=True))
    c.drawPath(both, G.paint(WHITE if theme.key != "friendly" else theme.paper))


# ---------------------------------------------------------------- marks
def mark_x(c: skia.Canvas, theme: Theme, cx: float, cy: float, r: float) -> None:
    c.drawCircle(cx + 3, cy + 6, r, G.paint(BLACK, 0.5, blur=6))
    c.drawCircle(cx, cy, r, G.paint(theme.hot))
    c.drawCircle(cx, cy, r, G.paint(WHITE, stroke=r * 0.14))
    p = G.paint(WHITE, stroke=r * 0.26, cap_round=True)
    k = r * 0.38
    c.drawLine(cx - k, cy - k, cx + k, cy + k, p)
    c.drawLine(cx - k, cy + k, cx + k, cy - k, p)


def mark_ok(c: skia.Canvas, theme: Theme, cx: float, cy: float, r: float) -> None:
    c.drawCircle(cx + 3, cy + 6, r, G.paint(BLACK, 0.5, blur=6))
    c.drawCircle(cx, cy, r, G.paint(theme.ok))
    c.drawCircle(cx, cy, r, G.paint(WHITE, stroke=r * 0.14))
    c.drawCircle(cx, cy, r * 0.46, G.paint(WHITE, stroke=r * 0.2))


def arrow(c: skia.Canvas, x0: float, y: float, x1: float, h: float, rgb) -> None:
    p = skia.Path()
    hw, head = h * 0.3, h * 0.62
    for i, (px, py) in enumerate(((x0, y - hw), (x1 - head, y - hw), (x1 - head, y - h / 2), (x1, y),
                                  (x1 - head, y + h / 2), (x1 - head, y + hw), (x0, y + hw))):
        p.moveTo(px, py) if i == 0 else p.lineTo(px, py)
    p.close()
    c.drawPath(p, G.paint(BLACK, stroke=10, cap_round=True))
    c.drawPath(p, G.paint(rgb))


# ---------------------------------------------------------------- cards (content boxes drawn at design size)
MW, MH = 640, 420


def card(c: skia.Canvas, draw, x: float, y: float, w: float, rot: float, frame=WHITE, frame_w: float = 8,
         shadow: float = 0.55) -> None:
    """A design-size (MW x MH) drawing at width w, top-left near (x, y), rotated around its centre, framed."""
    s = w / MW
    h = MH * s
    c.save()
    c.translate(x + w / 2, y + h / 2)
    c.rotate(rot)
    c.drawRRect(G.rrect(-w / 2 + 6, -h / 2 + 12, w, h, 24 * s), G.paint(BLACK, shadow, blur=12))
    c.drawRRect(G.rrect(-w / 2 - frame_w, -h / 2 - frame_w, w + 2 * frame_w, h + 2 * frame_w, 24 * s + frame_w),
                G.paint(frame))
    c.translate(-w / 2, -h / 2)
    c.scale(s, s)
    c.save()
    c.clipRRect(G.rrect(0, 0, MW, MH, 24), doAntiAlias=True)
    draw(c)
    c.restore()
    c.restore()


def text_card(c: skia.Canvas, theme: Theme, text: str, role: str) -> None:
    """The same sheet in two states: `label｜value` (value big). role 'before' is dull and cold, 'after' bright."""
    label, _, value = text.partition("｜") if "｜" in text else ("", "", text)
    before = role == "before"
    ground = (226, 227, 231) if before else WHITE
    ink = (112, 114, 124) if before else theme.ink if not theme.dark else (24, 24, 27)
    c.drawRect(skia.Rect.MakeWH(MW, MH), G.paint(ground))
    c.drawRect(skia.Rect.MakeXYWH(0, 0, MW, 14), G.paint(theme.hot if before else theme.ok))
    if label:
        put(c, label, MW / 2, 92, 50, theme.font2[0], 800, G.paint(ink), "center", True)
    fam, wt = theme.font
    wt = 900 if wt >= 800 else wt
    size = 210 if len(value) <= 3 else min(170, 560 / max(1, len(value)) * 1.7)
    while text_w(value, size, fam, wt) > MW - 60 and size > 40:
        size -= 6
    put(c, value, MW / 2, MH * (0.64 if label else 0.52), size, fam, wt, G.paint(ink), "center", True)
    if before:   # struck through: this is what you leave behind
        c.drawLine(40, MH * 0.80, MW - 40, MH * 0.34, G.paint(theme.hot, 0.75, stroke=9, cap_round=True))


def image_card(c: skia.Canvas, img: skia.Image) -> None:
    """Cover-fit picture in the design box."""
    iw, ih = img.width(), img.height()
    k = max(MW / iw, MH / ih)
    sw, sh = MW / k, MH / k
    G.draw_image(c, img, 0, 0, MW, MH, src=((iw - sw) / 2, (ih - sh) / 2, sw, sh), sharp=True)


# ---------------------------------------------------------------- backgrounds
def hexagon(c: skia.Canvas, cx: float, cy: float, r: float, rgb, a: float = 1.0) -> None:
    path = skia.Path()
    for k in range(6):
        ang = math.radians(60 * k - 90)
        pt = (cx + r * math.cos(ang), cy + r * math.sin(ang))
        path.moveTo(*pt) if k == 0 else path.lineTo(*pt)
    path.close()
    c.drawPath(path, G.paint(rgb, a))


def hex_grid(c: skia.Canvas, r: float = 44, rgb=WHITE, a: float = 0.05, width: float = 2.0) -> None:
    w, h = canvas_size(c)
    p = G.paint(rgb, a, stroke=width)
    dx, dy = r * math.sqrt(3), r * 1.5
    for row in range(-1, int(h / dy) + 2):
        for col in range(-1, int(w / dx) + 2):
            cx = col * dx + (dx / 2 if row % 2 else 0)
            cy = row * dy
            path = skia.Path()
            for k in range(6):
                ang = math.radians(60 * k - 90)
                pt = (cx + r * math.cos(ang), cy + r * math.sin(ang))
                path.moveTo(*pt) if k == 0 else path.lineTo(*pt)
            path.close()
            c.drawPath(path, p)


def rays(c: skia.Canvas, cx: float, cy: float, n: int = 44, rgb=WHITE, a: float = 0.25, r: float = 2400) -> None:
    p = G.paint(rgb, a)
    for i in range(0, n, 2):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        path = skia.Path()
        path.moveTo(cx, cy)
        path.lineTo(cx + r * math.cos(a0), cy + r * math.sin(a0))
        path.lineTo(cx + r * math.cos(a1), cy + r * math.sin(a1))
        path.close()
        c.drawPath(path, p)


def glow(c: skia.Canvas, cx: float, cy: float, r: float, rgb, a: float) -> None:
    w, h = canvas_size(c)
    p = skia.Paint(AntiAlias=True)
    p.setShader(G.rad_grad((cx, cy), r, [rgb, rgb], [a, 0.0]))
    c.drawRect(skia.Rect.MakeWH(w, h), p)


def dots(c: skia.Canvas, rgb, a: float = 0.16, pitch: float = 34, r: float = 7, box=None) -> None:
    """Halftone dot grid (offset rows), fading from the corners."""
    w, h = canvas_size(c)
    x0, y0, x1, y1 = box or (0, 0, w, h)
    p = G.paint(rgb, a)
    row = 0
    y = y0
    while y < y1:
        x = x0 + (pitch / 2 if row % 2 else 0)
        while x < x1:
            c.drawCircle(x, y, r, p)
            x += pitch
        y += pitch * 0.86
        row += 1


def seigaiha(c: skia.Canvas, rgb, a: float = 0.12, r: float = 54, box=None) -> None:
    """Overlapping concentric waves (青海波), drawn bottom row last so the lower arcs cover the upper ones."""
    w, h = canvas_size(c)
    x0, y0, x1, y1 = box or (0, 0, w, h)
    p = G.paint(rgb, a, stroke=2.5)
    row = 0
    y = y0 - r
    while y < y1 + r:
        x = x0 - r + (r if row % 2 else 0)
        while x < x1 + r:
            for k in (1.0, 0.72, 0.44):
                c.drawCircle(x, y + r, r * k, p)
            x += r * 2
        y += r * 0.5
        row += 1


def lines(c: skia.Canvas, rgb, a: float = 0.07, pitch: float = 40) -> None:
    w, h = canvas_size(c)
    p = G.paint(rgb, a, stroke=1.5)
    for x in np.arange(0, w, pitch):
        c.drawLine(float(x), 0, float(x), h, p)
    for y in np.arange(0, h, pitch):
        c.drawLine(0, float(y), w, float(y), p)


def deco(c: skia.Canvas, theme: Theme, box=None, strength: float = 1.0) -> None:
    """Pattern of the tone over the ground."""
    if theme.deco == "hex":
        hex_grid(c, rgb=WHITE if theme.dark else theme.ink, a=0.05 * strength)
    elif theme.deco == "dots":
        dots(c, theme.hot if theme.key == "variety" else theme.accent, a=0.22 * strength, box=box)
    elif theme.deco == "seigaiha":
        seigaiha(c, theme.metal, a=0.13 * strength, box=box)
    elif theme.deco == "lines":
        lines(c, theme.ink, a=0.05 * strength)


def backdrop(c: skia.Canvas, theme: Theme, cx: float, cy: float, r: float) -> None:
    """The shape behind the speaker (his head and shoulders break its edge)."""
    if theme.shape == "hexagon":
        hexagon(c, cx, cy, r, theme.accent)
    elif theme.shape == "circle":
        c.drawCircle(cx, cy, r, G.paint(theme.accent if theme.key == "friendly" else theme.bg2, 1.0))
        c.drawCircle(cx, cy, r, G.paint(WHITE, stroke=10))
    elif theme.shape == "sun":
        c.drawCircle(cx, cy, r, G.paint(theme.hot))
        c.drawCircle(cx, cy, r * 1.06, G.paint(theme.metal, stroke=5))
    elif theme.shape == "burst":
        path = skia.Path()
        n = 18
        for i in range(n * 2):
            ang = math.pi * i / n
            rr = r * (1.12 if i % 2 == 0 else 0.86)
            pt = (cx + rr * math.cos(ang), cy + rr * math.sin(ang))
            path.moveTo(*pt) if i == 0 else path.lineTo(*pt)
        path.close()
        c.drawPath(path, G.paint(BLACK, stroke=14, cap_round=True))
        c.drawPath(path, G.paint(theme.accent))


def ground(c: skia.Canvas, theme: Theme, kind: str = "plain") -> None:
    """Fill the canvas: tone ground, optional two-colour gradient."""
    w, h = canvas_size(c)
    p = skia.Paint(AntiAlias=True)
    p.setShader(G.lin_grad((0, 0), (w, h), [theme.bg, theme.bg2 if kind == "gradient" else theme.bg]))
    c.drawRect(skia.Rect.MakeWH(w, h), p)


def speed_lines(c: skia.Canvas, cx: float, cy: float, rgb, a: float = 0.18, n: int = 90, r_in: float = 260,
                r_out: float = 1400) -> None:
    """Manga focus lines around the speaker (a reaction)."""
    rng = np.random.default_rng(7)
    p = G.paint(rgb, a)
    for i in range(n):
        ang = 2 * math.pi * i / n + rng.uniform(-0.02, 0.02)
        half = rng.uniform(0.004, 0.016)
        r0 = r_in * rng.uniform(1.0, 1.5)
        path = skia.Path()
        path.moveTo(cx + r0 * math.cos(ang - half), cy + r0 * math.sin(ang - half))
        path.lineTo(cx + r_out * math.cos(ang), cy + r_out * math.sin(ang))
        path.lineTo(cx + r0 * math.cos(ang + half), cy + r0 * math.sin(ang + half))
        path.close()
        c.drawPath(path, p)
