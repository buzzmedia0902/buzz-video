"""バラエティ・勢い (variety): a busy-telop variety look (games, reactions). Rounded black subtitles (white, black
outline, a coloured outer edge per speaker; （inner voice） smaller and tilted), gradient impact words in Midashi Gothic
that slam in with a shake, burst-ray chapter splashes, sticker panels with a rainbow rim, a small corner title, and a
pop / whoosh / impact / tick on nearly every element. Quick jet cut (0.25 s), camera that moves every ~6 s with a
punch-in every 20 s, punchy grade, pop BGM 12 dB under the voice.

Use for: 企画もの, 飲み会, リアクション, 友人との掛け合い, テンポで見せたい切り抜き.
`{..}` turns a word yellow (subtitles) or colours it (impact words). Optional overlay keys: emphasis / chapter
"style": yellow | pink | blue | red | green | white; name "speaker" (colour from edit.json "speakers").
"""
from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, replace

import skia

from cv import gfx as G
from cv.canvas import H, TALL, W
from cv.cues import JOIN
from cv.tones.base import Look, _merge, spans

WHITE, BLACK, PLUM = (255, 255, 255), (0, 0, 0), (45, 22, 80)
PINK, ORANGE, BLUE, GREEN = (255, 60, 170), (255, 150, 0), (40, 150, 255), (40, 190, 60)
NARROW = set("、。・：；！？」』）ー〜…")      # full-width marks: the advance is trimmed so lines do not look gappy
CENTERED = set("・：；ー〜…！？")             # their glyph sits mid-box, so the trim is taken from both sides
SQUEEZE = 0.72


# ---------------------------------------------------------------- telop text
@dataclass(frozen=True)
class Ink:
    """How a telop is painted: shadow, outlines (outermost first, paint width as a fraction of the size), fill."""
    fill: tuple = WHITE                       # rgb, or (top_rgb, bottom_rgb) for a vertical gradient
    emph: tuple | None = None                 # fill of {..} runs (None: same as fill)
    strokes: tuple = ((BLACK, 0.2),)
    shadow: tuple = (0.0, 0.06, 0.07, 0.5)    # dx, dy (em), blur (em), alpha
    gloss: float = 0.0                        # white sheen on the upper half of the glyphs


def _sticker(top, bottom, edge, **kw) -> Ink:
    return Ink(fill=(top, bottom), strokes=((edge, 0.213), (WHITE, 0.12)), gloss=0.35, **kw)


YELLOW, PINK_GRAD = ((255, 251, 130), (255, 196, 0)), ((255, 176, 230), (255, 44, 152))
IMPACT = {
    "yellow": Ink(fill=YELLOW, shadow=(0.0, 0.06, 0.067, 0.55), gloss=0.35),
    "pink": _sticker(*PINK_GRAD, (90, 16, 110)), "blue": _sticker((140, 230, 255), (20, 120, 255), (10, 30, 90)),
    "red": _sticker((255, 150, 110), (235, 20, 30), (60, 0, 0)), "green": _sticker((210, 255, 120), (40, 190, 60), (10, 60, 20)),
    "white": Ink(strokes=((BLACK, 0.21),), shadow=(0.0, 0.06, 0.067, 0.55)),
}
CHAPTER = Ink(fill=((255, 200, 240), (255, 60, 170)), emph=YELLOW, strokes=(((60, 10, 80), 0.326), (WHITE, 0.196)),
              shadow=(0.0, 0.087, 0.11, 0.55), gloss=0.4)
TITLE_PINK = Ink(fill=((255, 200, 240), (255, 40, 160)), emph=YELLOW, strokes=(((70, 10, 90), 0.34), (WHITE, 0.19)),
                 shadow=(0.0, 0.1, 0.12, 0.6), gloss=0.45)
TITLE_YELLOW = Ink(fill=((255, 245, 120), (255, 150, 0)), emph=PINK_GRAD, strokes=(((120, 0, 70), 0.34), (WHITE, 0.19)),
                   shadow=(0.0, 0.1, 0.12, 0.6), gloss=0.45)
HEAD = Ink(fill=((255, 245, 120), (255, 160, 0)), emph=PINK_GRAD, strokes=(((90, 20, 110), 0.3), (WHITE, 0.16)),
           shadow=(0.0, 0.08, 0.09, 0.35), gloss=0.35)
BODY = Ink(fill=PLUM, emph=(255, 40, 150), strokes=(), shadow=(0.0, 0.0, 0.0, 0.0))
PLAIN = Ink(strokes=((BLACK, 0.2),), shadow=(0.0, 0.07, 0.08, 0.45))


def gradient(rgb, light: float = 0.6) -> tuple:
    """(top, bottom) fill of a brand / speaker colour: a lighter top, a slightly warmer bottom."""
    return G.mix_rgb(rgb, WHITE, light), G.mix_rgb(rgb, (255, 170, 0), 0.12)


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


def rows_of(text: str) -> list[tuple[str, list[bool]]]:
    """[(plain, emphasis flags)] per line of a text with {..} markup and ｜ / newline breaks."""
    plain, flags = spans(text.replace(JOIN, "\n"))
    out, k = [], 0
    for s in plain.split("\n"):
        out.append((s, flags[k:k + len(s)]))
        k += len(s) + 1
    return out


def pieces(s: str, flags, f: skia.Font) -> tuple[list[tuple[float, str, bool]], float]:
    """[(x, text, emphasised)] runs of a row and its width; marks in NARROW are drawn singly with a trimmed advance."""
    out, x, i = [], 0.0, 0
    while i < len(s):
        if s[i] in NARROW:
            adv = f.measureText(s[i])
            trim = adv * (1 - SQUEEZE)
            out.append((x - (trim / 2 if s[i] in CENTERED else 0.0), s[i], flags[i]))
            x += adv - trim
            i += 1
            continue
        j = i + 1
        while j < len(s) and s[j] not in NARROW and flags[j] == flags[i]:
            j += 1
        out.append((x, s[i:j], flags[i]))
        x += f.measureText(s[i:j])
        i = j
    return out, x


def row_width(s: str, flags, f: skia.Font) -> float:
    return pieces(s, flags, f)[1]


def draw_row(c: skia.Canvas, s: str, flags, x: float, y: float, f: skia.Font, ink: Ink, alpha: float = 1.0,
             align: str = "center") -> None:
    """One row on baseline y (align: centred / left / right of x): shadow, outlines, then the (gradient) fill run by
    run, then the gloss."""
    size = f.getSize()
    ps, w = pieces(s, flags, f)
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    top, bot = y - 0.86 * size, y + 0.10 * size

    def lay(p: skia.Paint) -> None:
        for px, run, _ in ps:
            c.drawString(run, x0 + px, y, f, p)

    sdx, sdy, sblur, sa = ink.shadow
    if sa > 0:
        with placed(c, sdx * size, sdy * size):
            lay(G.paint(BLACK, sa * alpha, stroke=max((sw for _, sw in ink.strokes), default=0.0) * size + 2,
                        blur=sblur * size, cap_round=True))
    for rgb, sw in ink.strokes:
        lay(G.paint(rgb, alpha, stroke=sw * size, cap_round=True))
    if ink.gloss > 0:
        c.saveLayer(skia.Rect.MakeLTRB(x0 - 8, top - 8, x0 + w + 8, bot + 8), None)
    for px, run, em in ps:
        col = (ink.emph or ink.fill) if em else ink.fill
        p = skia.Paint(AntiAlias=True)
        if isinstance(col[0], (tuple, list)):
            p.setShader(G.lin_grad((0, top), (0, bot), list(col)))
            p.setAlphaf(alpha)
        else:
            p = G.paint(col, alpha)
        c.drawString(run, x0 + px, y, f, p)
    if ink.gloss > 0:
        mid = y - 0.42 * size
        gp = skia.Paint(AntiAlias=True)
        gp.setBlendMode(skia.BlendMode.kSrcATop)
        gp.setShader(G.lin_grad((0, top), (0, mid), [WHITE, WHITE], [ink.gloss * alpha, ink.gloss * alpha * 0.25]))
        c.drawRect(skia.Rect.MakeLTRB(x0 - 8, top, x0 + w + 8, mid), gp)
        c.restore()


def draw_rows(c: skia.Canvas, rows, x: float, cy: float, f: skia.Font, ink: Ink, step: float = 1.15, alpha: float = 1.0,
              align: str = "center") -> None:
    """Rows stacked around the middle line cy (the visual middle of the glyphs)."""
    size = f.getSize()
    for j, (s, fl) in enumerate(rows):
        draw_row(c, s, fl, x, cy + (j - (len(rows) - 1) / 2) * step * size + 0.36 * size, f, ink, alpha, align)


def star(c: skia.Canvas, x: float, y: float, r: float, rgb, a: float, rot: float = 0.0) -> None:
    path = skia.Path()
    for k in range(8):
        ang = rot + k * math.pi / 4
        rr = r if k % 2 == 0 else r * 0.38
        (path.moveTo if k == 0 else path.lineTo)(x + rr * math.cos(ang), y + rr * math.sin(ang))
    path.close()
    c.drawPath(path, G.paint(rgb, a))


def rays(c: skia.Canvas, cx: float, cy: float, a: float, rot: float, n: int = 36, alpha: float = 0.10,
         spread: float = 0.035, rgb=WHITE) -> None:
    """Every other one of n thin wedges around (cx, cy): the burst lines behind a splash."""
    path, r = skia.Path(), math.hypot(W, H)
    for k in range(0, n, 2):
        ang = k * math.tau / n + rot
        path.moveTo(cx, cy)
        path.lineTo(cx + r * math.cos(ang - spread), cy + r * math.sin(ang - spread))
        path.lineTo(cx + r * math.cos(ang + spread), cy + r * math.sin(ang + spread))
        path.close()
    c.drawPath(path, G.paint(rgb, alpha * a))


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


def saturate(s: float, k: float) -> list[float]:
    """Colour matrix: saturation s, brightness k."""
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    return [(lr * (1 - s) + s) * k, lg * (1 - s) * k, lb * (1 - s) * k, 0, 0,
            lr * (1 - s) * k, (lg * (1 - s) + s) * k, lb * (1 - s) * k, 0, 0,
            lr * (1 - s) * k, lg * (1 - s) * k, (lb * (1 - s) + s) * k, 0, 0, 0, 0, 0, 1, 0]


# ---------------------------------------------------------------- the tone
class Variety(Look):
    key = "variety"
    CUT = dict(pause=0.25, tail=0.08, lead=0.06, speed=1.0)
    CAMERA = dict(hold=6.0, med=1.18, punch_every=20.0, punch_zoom=1.10, punch_dur=0.3)
    AUDIO = dict(bgm="pop", music_db=-12.0, sfx="variety", sfx_db=-1.0)
    GRADE = "eq=contrast=1.08:saturation=1.12"
    STYLE = _merge(Look.STYLE, {
        "accent": (255, 214, 0), "ink": (24, 18, 40), "paper": (255, 250, 238),
        "sub": {"font": "mplus_round", "weight": 950, "size": {"wide": 66, "tall": 80}, "fill": WHITE,
                "stroke": BLACK, "stroke_w": 0.21, "outer": (40, 40, 48), "outer_w": 0.18, "shadow": 0.5,
                "y": {"wide": 0.935, "tall": 0.71}, "gap": 1.22, "max_w": 0.92},
        "title": {"font": "mplus_round", "weight": 950, "size": {"wide": 40, "tall": 54},
                  "y": {"wide": 0.075, "tall": 0.145}},
        "chapter": {"font": "midashi_go", "weight": 800, "size": {"wide": 92, "tall": 96},
                    "y": {"wide": 0.5, "tall": 0.53}},
        "card": {"font": "mplus_round", "weight": 950, "size": {"wide": 46, "tall": 54}, "fill": PLUM,
                 "w": {"wide": 0.40, "tall": 0.80}},
        "emphasis": {"font": "midashi_go", "weight": 800, "size": {"wide": 150, "tall": 150},
                     "y": {"wide": 0.42, "tall": 0.40}},
        "name": {"font": "midashi_go", "weight": 800, "size": {"wide": 84, "tall": 76},
                 "y": {"wide": 0.74, "tall": 0.56}},
        "caption": {"font": "mplus_round", "weight": 950, "size": {"wide": 44, "tall": 52},
                    "y": {"wide": 0.11, "tall": 0.14}},
        "credit": {"font": "mplus_round", "weight": 800, "size": {"wide": 26, "tall": 30}, "fill": (235, 235, 240)},
    })

    # -------------------------------------------------------------- text helpers
    def wrapped(self, text: str, f: skia.Font, max_w: float) -> list[tuple[str, list[bool]]]:
        """Rows of a text; a single row wider than max_w is broken at the best Japanese line-break spot."""
        rows = rows_of(text)
        if len(rows) == 1 and len(rows[0][0]) >= 6 and row_width(*rows[0], f) > max_w:
            (a, fa), (b, fb) = self._break(rows[0][0], rows[0][1], f, max_w, f.getSize())
            rows = [(a, list(fa)), (b, list(fb))]
        return rows

    @staticmethod
    def fit(rows, f: skia.Font, max_w: float, pad: float = 0.0) -> float:
        """Scale (<= 1) that makes the widest row, plus pad, fit max_w."""
        widest = max(row_width(s, fl, f) for s, fl in rows) + pad
        return min(1.0, max_w / widest) if widest > 0 else 1.0

    def accent_ink(self) -> Ink:
        """The yellow impact style in the brand accent colour."""
        return replace(IMPACT["yellow"], fill=gradient(self.S["accent"]))

    # -------------------------------------------------------------- subtitles
    def sub_line(self, c, s, flags, cx, y, size, alpha, speaker_rgb=None, inner: bool = False) -> None:
        """Speech: white fill, black inner edge, speaker-coloured (else dark) outer edge; {..} in yellow gradient."""
        sw, ow = self.v("sub", "stroke_w"), self.v("sub", "outer_w")
        edge = (90, 50, 150) if inner else (speaker_rgb if speaker_rgb is not None else self.v("sub", "outer"))
        ink = Ink(fill=self.v("sub", "fill"), emph=gradient(self.v("sub", "emph")), strokes=((edge, sw + ow), (BLACK, sw)),
                  shadow=(0.0, 0.09, 0.1, self.v("sub", "shadow")))
        draw_row(c, s, flags, cx, y, self.font("sub", size), ink, alpha)

    def draw_sub(self, c, sub, t) -> None:
        size, lines = self.sub_layout(sub.text)
        inner = spans(sub.text)[0].startswith("（")      # 心の声: smaller, leaning, swaying
        gap = size * self.v("sub", "gap")
        y_last = H * self.v("sub", "y")
        y_first = y_last - (len(lines) - 1) * gap
        cy = (y_first + y_last) / 2 - 0.3 * size
        a = G.clamp01((t - sub.t0) / 0.05) * G.clamp01((sub.t1 - t) / 0.07)
        sc = (0.86 + 0.14 * G.back((t - sub.t0) / 0.10, 2.2)) * (0.82 if inner else 1.0)
        spk = self.speaker_rgb(sub.speaker)
        with grouped(c, a), placed(c, W / 2, cy, sc, -1.6 * math.sin(t * 2.2) if inner else 0.0):
            if inner:
                c.skew(-0.12, 0)
            for j, (s, fl) in enumerate(lines):
                self.sub_line(c, s, fl, 0, y_first + j * gap - cy, size, 1.0, spk, inner)

    # -------------------------------------------------------------- persistent elements
    def title_layout(self, ov) -> tuple[float, list, float, float, float, float]:
        """(size, rows, left, top, width, height) of the corner title: right corner in 16:9, left in 9:16."""
        # shorts and clips leave the top free and the title is the hook: make it big
        size = self.v("title", "size") * (1.45 if self.fmt in ("clip", "short") else 1.0)
        rows = self.wrapped(ov.text, self.font("title", size), (self.x1 - self.x0) * 0.78)
        size *= self.fit(rows, self.font("title", size), self.x1 - self.x0, size * 0.4)
        w = max(row_width(s, fl, self.font("title", size)) for s, fl in rows) + size * 0.4
        return size, rows, (self.x0 if TALL else self.x1 - w), H * self.v("title", "y") - size * 0.5, w, \
            size * (1.15 * (len(rows) - 1) + 1.0)

    def draw_title_band(self, c, ov, t) -> None:
        """Small corner title: white rounded letters, black edge, {..} in yellow gradient."""
        a = G.clamp01((t - ov.t0) / 0.2) * G.clamp01((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        size, rows, left, top, _, h = self.title_layout(ov)
        ink = replace(PLAIN, emph=gradient(self.S["accent"]), strokes=((BLACK, 0.21),), shadow=(0.0, 0.08, 0.1, 0.45))
        with grouped(c, a), placed(c, (1 - G.ease_out((t - ov.t0) / 0.3)) * (-60 if TALL else 60), 0):
            draw_rows(c, rows, left + size * 0.2, top + h / 2, self.font("title", size), ink, 1.15, 1.0, "left")

    def draw_credit(self, c, ov, t) -> None:
        size, avail = self.v("credit", "size"), self.x1 - self.x0
        rows = self.wrapped(ov.text, self.font("credit"), avail)
        f = self.font("credit", size * self.fit(rows, self.font("credit"), avail))
        ink = Ink(fill=self.v("credit", "fill"), strokes=((BLACK, 0.22),), shadow=(0.0, 0.08, 0.1, 0.35))
        draw_rows(c, rows, self.x1, self.y1 + size * 1.24 + (len(rows) - 1) * 0.55 * size, f, ink, 1.1, 0.95, "right")

    # -------------------------------------------------------------- chapter: dimmed splash with burst lines
    def draw_chapter(self, c, ov, t) -> None:
        tt = t - ov.t0
        pi, po = G.clamp01(tt / 0.12), G.clamp01((ov.t1 - t) / 0.2)
        a = min(pi, po)
        if a <= 0:
            return
        cy, size = H * self.v("chapter", "y"), self.v("chapter", "size")
        f = self.font("chapter", size)
        rows = self.wrapped(ov.text, f, W - 200)
        sc = (0.4 + 0.6 * G.back(pi, 2.4)) * self.fit(rows, f, W - 160, size * 0.5) * (1 + 0.08 * (1 - po))
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint((10, 0, 30), 0.38 * a))
        rays(c, W / 2, cy, a, tt * 0.25)
        with grouped(c, a), placed(c, W / 2, cy, sc):
            draw_rows(c, rows, 0, 0, f, IMPACT.get(ov.d.get("style"), CHAPTER), 1.15)
        rng = random.Random(7)
        for k in range(6):
            ang, rr = rng.random() * math.tau, (0.20 + rng.random() * 0.14) * W
            star(c, W / 2 + rr * math.cos(ang) * 1.4, cy + rr * math.sin(ang) * (0.9 if TALL else 0.55),
                 (16 + rng.random() * 14) * W / 1370, (255, 230, 120), a * (0.6 + 0.4 * math.sin(tt * 10 + k)), tt * 2)

    # -------------------------------------------------------------- card: sticker panel with a rainbow rim
    def draw_card(self, c, ov, t) -> None:
        tt = t - ov.t0
        a = G.clamp01(tt / 0.12) * G.ease_io((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        x, y, w, h = self.card_rect(ov)
        _, scale, see = self.placed(ov)
        size = self.v("card", "size") * scale
        with grouped(c, a * see), placed(c, x + w / 2, y + h / 2, 0.72 + 0.28 * G.back(tt / 0.35, 2.0)):
            x, y = -w / 2, -h / 2
            G.shadow(c, x + 4, y, w, h, size * 0.5, blur=14, dy=10, a=0.35)
            c.drawRRect(G.rrect(x, y, w, h, size * 0.5), G.paint(self.S["paper"], 0.97))
            rim = skia.Paint(AntiAlias=True)
            rim.setStyle(skia.Paint.kStroke_Style)
            rim.setStrokeWidth(7)
            rim.setShader(G.lin_grad((x, y), (x + w, y + h), [(255, 120, 200), (170, 60, 220), (90, 120, 255)]))
            c.drawRRect(G.rrect(x - 3.5, y - 3.5, w + 7, h + 7, size * 0.5 + 3.5), rim)
            hh = size * 1.6
            if ov.d.get("title"):
                rows = rows_of(str(ov.d["title"]))
                fh = self.font("card", size * 0.95 * self.fit(rows, self.font("card", size * 0.95), w - size * 2.4, size * 0.4))
                draw_rows(c, rows, 0, y + hh / 2, fh, HEAD, 1.0)
                for sx in (x + size * 0.55, x + w - size * 0.55):
                    star(c, sx, y + hh / 2, size * 0.3, ORANGE, 1.0, tt * 2)
            yy = y + hh + size * 0.55
            if ov.d.get("big"):
                draw_rows(c, rows_of(str(ov.d["big"])), 0, yy + size * 0.7, self.font("card", size * 2.0), IMPACT["pink"], 1.0)
                yy += size * 1.6
            for j, item in enumerate(ov.d.get("items", [])):
                k, cy, bx = G.back((tt - 0.25 - 0.15 * j) / 0.28, 2.0), yy + size * 0.6 + j * size * 1.55, x + size * 0.95
                with placed(c, bx, cy, k):
                    c.drawCircle(0, 0, size * 0.44, G.paint(WHITE, 1.0))
                    c.drawCircle(0, 0, size * 0.38, G.paint((PINK, ORANGE, BLUE, GREEN)[j % 4], 1.0))
                    draw_rows(c, [(str(j + 1), [False])], 0, 0, self.font("card", size * 0.78),
                              Ink(strokes=(), shadow=(0.0, 0.0, 0.0, 0.0)))
                rows = rows_of(str(item))
                fs = size * (0.82 if len(rows) == 1 else 0.62)
                fi = self.font("card", fs * self.fit(rows, self.font("card", fs), w - size * 2.3))
                draw_rows(c, rows, x + size * 1.65, cy, fi, BODY, 0.95, G.clamp01(k), "left")

    # -------------------------------------------------------------- emphasis: gradient impact word that slams in
    def impact(self, ov, scale: float = 1.0) -> tuple[float, list]:
        """(size, rows) of an impact word at `scale`: one row shrunk to the safe width, broken in two when that would
        shrink it below 60 %."""
        size = self.v("emphasis", "size") * scale
        f, avail = self.font("emphasis", size), self.x1 - self.x0
        rows = rows_of(ov.text)
        if len(rows) == 1 and self.fit(rows, f, avail, size * 0.4) < 0.6:
            rows = self.wrapped(ov.text, f, avail * 0.9)
        return size * self.fit(rows, f, avail, size * 0.4), rows

    def draw_emphasis(self, c, ov, t) -> None:
        tt = t - ov.t0
        po = G.clamp01((ov.t1 - t) / 0.14)
        a = min(G.clamp01(tt / 0.06), po)
        if a <= 0:
            return
        (fx, fy), scale, see = self.placed(ov)
        size, rows = self.impact(ov, scale)
        ink = IMPACT.get(ov.d.get("style")) or self.accent_ink()
        if any(any(fl) for _, fl in rows):          # only the {..} words take the colour
            ink = replace(ink, fill=WHITE, emph=ink.fill)
        sc = (1 + 0.85 * (1 - G.ease_out(tt / 0.14))) * (1 + 0.025 * math.sin(tt * 9) * G.clamp01((tt - 0.2) / 0.2)) \
            * (1 + 0.12 * (1 - po))
        shake = max(0.0, 1 - tt / 0.18)
        cx, cy = W * fx + 9 * shake * math.sin(tt * 160), H * fy + 7 * shake * math.cos(tt * 130)
        rays(c, cx, cy, a * see * G.clamp01(1 - tt / 0.5), tt * 0.5, n=28, alpha=0.22, spread=0.05)
        with grouped(c, a * see), placed(c, cx, cy, sc, float(ov.d.get("rot", -3.0))):
            draw_rows(c, rows, 0, 0, self.font("emphasis", size), ink, 1.1)

    # -------------------------------------------------------------- name plate: ribbon + big coloured name
    def name_layout(self, ov, scale: float = 1.0) -> tuple[float, list, list, float, float]:
        """(size, name rows, ribbon rows, plate width, ribbon width)."""
        size = self.v("name", "size") * scale
        rows = rows_of(ov.text)
        sub = rows_of(str(ov.d["sub"])) if ov.d.get("sub") else []
        rw = max((row_width(s, fl, self.font("caption", size * 0.46)) for s, fl in sub), default=0.0) + size * 0.7
        w = max(row_width(s, fl, self.font("name", size)) for s, fl in rows) + size * 0.4
        return size, rows, sub, max(w, rw if sub else 0.0), rw

    def spots(self, ov) -> list[tuple[float, float]]:
        """In 9:16 an impact word goes first under the chin (between the face and the subtitles), then above the head."""
        if ov.type == "emphasis" and TALL:
            return [(0.5, 0.57), (0.5, 0.27), (0.5, self.v("emphasis", "y"))]
        return super().spots(ov)

    def box(self, ov, center=None, scale: float = 1.0):
        if ov.type == "title_band":
            _, _, left, top, w, h = self.title_layout(ov)
            return left, top, w, h
        if ov.type == "emphasis":
            cx, cy = center or self.spots(ov)[0]
            size, rows = self.impact(ov, scale)
            w = max(row_width(s, fl, self.font("emphasis", size)) for s, fl in rows)
            h = size * (1.1 * (len(rows) - 1) + 1.2)
            return W * cx - w / 2, H * cy - h / 2, w, h
        if ov.type == "name":
            _, cy = center or self.spots(ov)[0]
            size, _, sub, w, _ = self.name_layout(ov, scale)
            return self.x0, H * cy - size * (1.1 + 0.46 * max(0, len(sub) - 1)), w, size * (1.8 + 0.46 * max(0, len(sub) - 1))
        return super().box(ov, center, scale)

    def draw_name(self, c, ov, t) -> None:
        tt = t - ov.t0
        k = G.ease_out(tt / 0.4)
        a = G.clamp01(tt / 0.15) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        (_, fy), scale, see = self.placed(ov)
        size, rows, sub, w, rw = self.name_layout(ov, scale)
        spk = self.speaker_rgb(ov.d.get("speaker") or ov.text)
        edge = G.mix_rgb(spk, BLACK, 0.7) if spk else (90, 16, 110)
        ink = Ink(fill=gradient(spk, 0.35) if spk else PINK_GRAD, emph=YELLOW, strokes=((edge, 0.2), (WHITE, 0.11)),
                  shadow=(0.0, 0.07, 0.08, 0.5), gloss=0.35)
        cy = H * fy
        with grouped(c, a * see), placed(c, self.x0 - (1 - k) * 120, 0):
            if sub:
                ry, rh = cy - size * (0.78 + 0.23 * (len(sub) - 1)), size * (0.32 + 0.32 * len(sub))
                with placed(c, size * 0.1, ry):
                    c.skew(-0.25, 0)
                    c.drawRect(skia.Rect.MakeXYWH(0, -rh / 2, rw, rh), G.paint(spk or PINK, 0.95))
                draw_rows(c, sub, size * 0.45, ry, self.font("caption", size * 0.46),
                          Ink(strokes=((BLACK, 0.2),), shadow=(0.0, 0.0, 0.0, 0.0)), 1.1, 1.0, "left")
            draw_rows(c, rows, w / 2, cy + size * 0.12, self.font("name", size), ink, 1.0)

    # -------------------------------------------------------------- caption / note
    def draw_caption(self, c, ov, t) -> None:
        a = G.clamp01((t - ov.t0) / 0.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        size, y = self.v("caption", "size"), H * self.v("caption", "y")
        rows = rows_of(ov.text)
        sub = rows_of(str(ov.d["sub"])) if ov.d.get("sub") else []
        ink, y2 = replace(PLAIN, emph=gradient(self.S["accent"])), y + (len(rows) - 1) * 1.15 * size + size * 0.95
        with grouped(c, a), placed(c, -(1 - G.ease_out((t - ov.t0) / 0.35)) * 60, 0):
            c.drawRRect(G.rrect(self.x0, y - size * 0.62, size * 0.2, y2 - y + size * (0.67 * len(sub) + 0.2 if sub else 0.8),
                                size * 0.1), G.paint(PINK, 1.0))
            draw_rows(c, rows, self.x0 + size * 0.45, y + (len(rows) - 1) * 0.575 * size, self.font("caption", size), ink,
                      1.15, 1.0, "left")
            draw_rows(c, sub, self.x0 + size * 0.45, y2 + (len(sub) - 1) * 0.33 * size, self.font("caption", size * 0.58),
                      ink, 1.15, 1.0, "left")

    def draw_note(self, c, ov, t) -> None:
        a = G.clamp01((t - ov.t0) / 0.25) * G.clamp01((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        size = self.v("sub", "size")
        rows = rows_of("※" + ov.text.lstrip("※"))
        fs = size * 0.58
        draw_rows(c, rows, self.x0, H * self.v("sub", "y") - size * 2.9 - 0.36 * fs - (len(rows) - 1) * 0.575 * fs,
                  G.font(fs, 900, self.v("sub", "font")), replace(PLAIN, emph=gradient(self.S["accent"])), 1.15, a, "left")

    # -------------------------------------------------------------- 9:16 card mode / band background
    def draw_card_mode(self, c, frame, box, k, face) -> None:
        super().draw_card_mode(c, frame, box, k, face)
        if k <= 0:
            return
        bx, by = W * self.BUBBLE[0], H * self.BUBBLE[1]
        rr = W * self.BUBBLE[2] * G.back(k, 1.6)
        c.drawCircle(bx, by, rr, G.paint(PINK, 1.0, stroke=18))      # also hides the base's accent ring
        c.drawCircle(bx, by, rr - 2, G.paint(WHITE, 1.0, stroke=8))

    def draw_band_bg(self, c, frame, box) -> None:
        """Behind 16:9 footage in 9:16: saturated blurred copy, pink-to-violet wash, soft bokeh dots."""
        ox, oy, ow, oh = box
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(28, 28, skia.TileMode.kClamp))
        p.setColorFilter(skia.ColorFilters.Matrix(saturate(1.5, 0.62)))
        with placed(c, W / 2, H / 2, max(W / ow, H / oh) * 1.1):
            c.translate(-ow / 2, -oh / 2)
            c.drawImage(frame, 0, 0, G.SAMPLING, p)
        wash = skia.Paint(AntiAlias=True)
        wash.setShader(G.lin_grad((0, 0), (0, H), [PINK, (80, 30, 160)], [0.32, 0.45]))
        c.drawRect(skia.Rect.MakeWH(W, H), wash)
        rng = random.Random(5)
        for _ in range(16):
            c.drawCircle(rng.random() * W, rng.random() * H, (0.015 + rng.random() * 0.05) * max(W, H),
                         G.paint(rng.choice((WHITE, (255, 230, 120), PINK)), 0.10))

    # -------------------------------------------------------------- full-screen cards
    def stage(self, c, bg, tt: float, grow: float = 0.05, dim: float = 0.8) -> None:
        """Violet stage: blurred (slowly zooming) picture, vignette, rotating light rays."""
        if bg is None:
            c.clear(G.col((40, 10, 60)))
        else:
            p = skia.Paint(AntiAlias=True)
            p.setImageFilter(skia.ImageFilters.Blur(9, 9, skia.TileMode.kClamp))
            with placed(c, W / 2, H / 2, 1.12 + grow * tt):
                c.drawImage(bg, -W / 2, -H / 2, G.SAMPLING, p)
        v = skia.Paint(AntiAlias=True)
        v.setShader(G.rad_grad((W / 2, H / 2), math.hypot(W, H) * 0.55, [(90, 20, 140), (20, 0, 40)], [0.25, dim]))
        c.drawRect(skia.Rect.MakeWH(W, H), v)
        rays(c, W / 2, H / 2, G.clamp01(tt / 0.4), tt * 0.18, n=48, alpha=0.07, spread=0.03, rgb=(255, 220, 255))

    def title_rows(self, part: dict) -> list[tuple[str, list[bool]]]:
        text = part.get("text") or self.brand.get("channel") or self.brand.get("name", "")
        return self.wrapped(text, self.font("chapter", 112 if TALL else 116), W - 220)

    def card_title(self, c, part, tt, dur, bg) -> None:
        self.stage(c, bg, tt)
        size, rows, ys = 112 if TALL else 116, self.title_rows(part), H / 2
        if self.brand.get("logo"):
            img = G.image(self.brand["logo"])
            lw = min(W * 0.34, H * 0.22 * img.width() / img.height())
            lh, k = lw * img.height() / img.width(), G.back(tt / 0.4)
            G.draw_image(c, img, W / 2 - lw / 2 * k, H * 0.30 - lh / 2 * k, lw * k, lh * k, alpha=G.clamp01(tt / 0.2))
            ys = H * 0.55
        for k, row in enumerate(rows):
            pk = G.clamp01((tt - 0.15 - 0.32 * k) / 0.16)
            f = self.font("chapter", size * (0.92 if k == 0 and len(rows) > 1 else 1.0))
            f = self.font("chapter", f.getSize() * self.fit([row], f, W - 140, size * 0.4))
            with grouped(c, pk), placed(c, W / 2, ys + (k - (len(rows) - 1) / 2) * size * 1.45, 1 + 0.9 * (1 - G.ease_out(pk, 3)),
                                        -2.0 + 0.6 * k):
                draw_rows(c, [row], 0, 0, f, TITLE_PINK if k == 0 else TITLE_YELLOW)
        rng = random.Random(3)
        for _ in range(26):
            x, y, ph, r = rng.random() * W, rng.random() * H, rng.random() * math.tau, 8 + rng.random() * 16
            star(c, x, y, r, (255, 240, 170), G.clamp01(math.sin(tt * 3.1 + ph) * 0.8 + 0.2) * G.clamp01(tt / 0.4), tt * 1.5 + ph)

    def card_text(self, c, part, tt, dur, bg) -> None:
        self.stage(c, bg, tt, dim=0.9)
        size = 104 if TALL else 112
        f = self.font("emphasis", size)
        rows = self.wrapped(part.get("text", ""), f, W - 200)
        with grouped(c, G.clamp01((dur - tt) / 0.25)), placed(c, W / 2, H / 2, G.back(tt / 0.3, 2.0) * self.fit(rows, f, W - 140, size * 0.4)):
            draw_rows(c, rows, 0, 0, f, replace(IMPACT["white"], emph=gradient(self.S["accent"])), 1.2)

    def card_eyecatch(self, c, part, tt, dur, bg) -> None:
        """Three colour bands swipe across; the logo or text pops in the middle."""
        c.clear(G.col((255, 244, 214)))
        u, slant = tt / max(dur, 0.1), H * 0.35
        for i, rgb in enumerate((PINK, (255, 200, 0), (110, 40, 190))):
            x = G.lerp(-W * 1.7, W * 1.2, G.ease_io5(G.clamp01((u - 0.07 * i) / 0.86)))
            path = skia.Path()
            path.moveTo(x, 0)
            for px, py in ((x + W * 1.1, 0), (x + W * 1.1 - slant, H), (x - slant, H)):
                path.lineTo(px, py)
            c.drawPath(path, G.paint(rgb, 1.0))
        a, k = G.win(tt, dur * 0.28, dur * 0.8, 0.12, 0.12), G.back((tt - dur * 0.28) / 0.25, 2.0)
        if self.brand.get("logo"):
            img = G.image(self.brand["logo"])
            lw = W * (0.6 if TALL else 0.34)
            lh = lw * img.height() / img.width()
            G.draw_image(c, img, W / 2 - lw / 2 * k, H / 2 - lh / 2 * k, lw * k, lh * k, alpha=a)
        elif part.get("text"):
            f = self.font("chapter", 100)
            with grouped(c, a), placed(c, W / 2, H / 2, k):
                draw_rows(c, self.wrapped(part["text"], f, W - 200), 0, 0, f, IMPACT["white"], 1.2)

    # -------------------------------------------------------------- end card: like / subscribe (or follow)
    LIKE_IN, SUB_IN, CLICKS, BELL = 0.2, 0.35, (0.9, 1.7), 2.35      # CLICKS match card_sfx

    def card_end(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg, 0.62)
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint((14, 4, 28), 0.35))
        yt = self.fmt == "youtube"
        f = self.font("chapter", 78 if TALL else 84)
        rows = self.wrapped(part.get("text") or ("ご視聴ありがとうございました！" if yt else "フォローして続きをチェック！"), f, W - 200)
        with grouped(c, G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.25)):
            with placed(c, W / 2, H * (0.25 if TALL else 0.19), (0.9 + 0.1 * G.back(tt / 0.35)) * self.fit(rows, f, W - 140, 40)):
                draw_rows(c, rows, 0, 0, f, TITLE_PINK, 1.2)
            self.buttons(c, tt, yt)

    def buttons(self, c, tt: float, yt: bool) -> None:
        """Two pill buttons pop in; a cursor clicks like (icon fills pink) then subscribe (turns 登録済み, bell rings)."""
        s = 1.1 if TALL else 1.0
        labels = ("高評価", "チャンネル登録", "登録済み") if yt else ("いいね", "フォロー", "フォロー中")
        f, bh, cy = self.font("card", 44 * s), 96 * s, H * 0.5
        lw = G.text_width(labels[0], f) + 150 * s
        sw = max(G.text_width(labels[1], f), G.text_width(labels[2], f) + 70 * s) + 100 * s
        x0 = W / 2 - (lw + 46 * s + sw) / 2
        like_c, sub_c = (x0 + lw / 2, cy), (x0 + lw + 46 * s + sw / 2, cy)
        ring = tt - self.BELL

        def label(text: str, x: float, rgb, align: str = "left") -> None:
            draw_rows(c, [(text, [False] * len(text))], x, 0, f, Ink(fill=rgb, strokes=(), shadow=(0, 0, 0, 0)), 1.0, 1.0, align)

        for i, (center, w, t_in, t_click) in enumerate(((like_c, lw, self.LIKE_IN, self.CLICKS[0]),
                                                       (sub_c, sw, self.SUB_IN, self.CLICKS[1]))):
            done, d = tt >= t_click, tt - t_click
            k = G.back(G.clamp01((tt - t_in) / 0.32), 2.4) if tt >= t_in else 0.0
            press = 1 - 0.08 * math.sin(math.pi * min(1.0, d / 0.12)) if 0 <= d < 0.12 else \
                1 + 0.06 * math.sin(math.pi * (d - 0.12) / 0.23) if 0.12 <= d < 0.35 else 1.0
            if k <= 0:
                continue
            with placed(c, *center, k * press):
                G.shadow(c, -w / 2 + 4, -bh / 2, w, bh, bh / 2, blur=10, dy=10, a=0.35)
                c.drawRRect(G.rrect(-w / 2, -bh / 2, w, bh, bh / 2),
                            G.paint(WHITE if i == 0 else (70, 64, 80) if done else (236, 28, 64), 0.97))
                if i == 0:
                    if done:
                        c.drawRRect(G.rrect(-w / 2 - 4, -bh / 2 - 4, w + 8, bh + 8, bh / 2 + 4),
                                    G.paint(PINK, G.clamp01(1 - d / 1.2) * 0.8 + 0.2, stroke=6))
                    ts = 50 * s * (1 + 0.35 * math.sin(math.pi * G.clamp01(d / 0.3)) if done else 1.0)
                    with placed(c, -w / 2 + 36 * s, -ts / 2 - 2):
                        c.drawPath(thumb_path(ts), G.paint((255, 40, 150), 1.0) if done else G.paint((30, 20, 40), 1.0, stroke=4.5 * s))
                    label(labels[0], -w / 2 + 110 * s, (30, 20, 40))
                elif not done:
                    label(labels[1], 0, WHITE, "center")
                else:
                    label(labels[2], -w / 2 + 50 * s, (235, 235, 240))
                    if yt:                      # bell: grey, then yellow and swinging
                        bs = 52 * s
                        with placed(c, w / 2 - 50 * s, 2, 1.0, 22 * math.sin(math.tau * 6.5 * ring) * math.exp(-ring * 2.8) if ring >= 0 else 0.0):
                            c.translate(-bs / 2, -bs * 0.44)
                            c.drawPath(bell_path(bs), G.paint((255, 225, 90) if ring >= 0 else (235, 235, 240), 1.0))
            if i == 0 and done and d < 0.7:     # sparkle burst from the icon
                for j in range(8):
                    ang, r = j * math.tau / 8 + 0.3, 40 + 90 * G.ease_out(d / 0.7)
                    star(c, center[0] - w / 2 + 61 * s + r * math.cos(ang), cy + r * math.sin(ang), 11 * (1 - d / 0.7) + 3,
                         (255, 230, 120), 1 - d / 0.7, d * 4)
        if yt and 0 <= ring < 0.8:              # shine lines around the ringing bell
            d, bx = ring / 0.8, sub_c[0] + sw / 2 - 76 * s
            for j in range(6):
                ang = -math.pi / 2 + (j - 2.5) * 0.45
                c.drawLine(bx + (46 + 30 * d) * math.cos(ang), cy + (46 + 30 * d) * math.sin(ang), bx + (62 + 46 * d) * math.cos(ang),
                           cy + (62 + 46 * d) * math.sin(ang), G.paint((255, 225, 90), 1 - d, stroke=5))
        if yt and tt >= self.BELL + 0.35:       # 「よろしくお願いします！」 bounces in under the buttons
            with placed(c, W / 2, cy + 112 * s, G.back((tt - self.BELL - 0.35) / 0.32, 2.4)):
                draw_rows(c, [("よろしくお願いします！", [False] * 11)], 0, 0, self.font("card", 46 * s),
                          Ink(strokes=((PINK, 0.2), ((40, 0, 50), 0.11)), shadow=(0.0, 0.09, 0.13, 0.45)))
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
                c.drawCircle(*at(tc), 14 + 40 * G.ease_out(d), G.paint(WHITE, 0.7 * (1 - d), stroke=4))
        self._cursor(c, *at(tt), G.clamp01((tt - keys[0][0]) / 0.2) * G.clamp01((keys[-1][0] - tt) / 0.3), tt, clicks)

    # -------------------------------------------------------------- sound
    def sfx(self, ov) -> list[tuple[float, str, float]]:
        """More than the base: pop on cards (+ a tick per item), whoosh + pop on chapters, whoosh into every impact word."""
        m = {"chapter": [(0.0, "whoosh", -3.0), (0.14, "pop", -6.0)],
             "card": [(0.0, "pop", -2.0)] + [(0.25 + 0.15 * j, "tick", -9.0) for j in range(len(ov.d.get("items", [])))],
             "emphasis": [(-0.1, "whoosh", -9.0), (0.0, "impact", -1.0)],
             "name": [(0.0, "tick", -6.0), (0.12, "pop", -10.0)], "caption": [(0.0, "tick", -8.0)],
             "note": [(0.0, "tick", -10.0)]}
        return [(max(0.0, ov.t0 + dt), name, g) for dt, name, g in m.get(ov.type, [])]

    def card_sfx(self, part: dict, t0: float, dur: float) -> list[tuple[float, str, float]]:
        card = part["card"]
        if card == "end":
            fx = [(t0 + 0.2, "pop", -6.0), (t0 + self.CLICKS[0], "click", -4.0), (t0 + self.CLICKS[1], "click", -4.0)]
            return fx + ([(t0 + self.CLICKS[1] + 0.05, "chime", -6.0), (t0 + self.BELL, "chime", -8.0)] if self.fmt == "youtube" else [])
        if card == "title":
            n = len(self.title_rows(part))
            return [(t0, "whoosh", -3.0)] + [(t0 + 0.15 + 0.32 * k + 0.12, "pop", -4.0) for k in range(n)] \
                + [(t0 + 0.3 + 0.32 * n, "chime", -8.0)]
        if card == "eyecatch":
            return [(t0, "whoosh", -3.0), (t0 + dur * 0.3, "pop", -4.0)]
        return [(t0, "whoosh", -4.0), (t0 + 0.12, "impact", -5.0)]


LOOK = Variety
