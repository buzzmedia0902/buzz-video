"""洗練・ミニマル (minimal): brand-film calm. Thin Hiragino with Avenir Next for Latin, small airy subtitles low in the
frame with no outline (a soft shadow only), monochrome plus one accent hairline for {emphasis}, chapters as thin text
on a rule, quiet white cards with a hairline, a faint cool grade, the camera never reframes, ambient BGM and no SFX.

Use for: brands, high-price services, interviews and talks that should feel quiet and considered rather than loud.
"""
from __future__ import annotations

import functools
import math

import skia

from cv import gfx as G
from cv.canvas import H, TALL, W
from cv.cues import JOIN
from cv.tones.base import Look, spans, strip_markup

WHITE, INK, GREY, RULE = (250, 250, 248), (22, 22, 24), (132, 132, 136), (208, 208, 210)
SQUARE = W == H                                 # 1:1 shares the "wide" values: the chapter moves below the title
TR = 0.04                                       # letter-spacing (em) of body text: generous, never tight
OPENERS = "【「『（〈《［｛"


def _flat(text: str) -> str:
    """Text for a one-line label: markup removed, a forced line break (｜) becomes a space."""
    return strip_markup(text.replace(JOIN, " "))


@functools.lru_cache(maxsize=4096)
def _split(s: str) -> tuple[tuple[str, bool], ...]:
    """Runs of (text, is_latin): ASCII goes to the Latin face, everything else to Hiragino."""
    out, cur, latin = [], "", None
    for ch in s:
        if latin is not None and ch.isascii() != latin:
            out.append((cur, latin))
            cur = ""
        cur, latin = cur + ch, ch.isascii()
    return tuple(out + [(cur, latin)]) if cur else ()


def _face(size: float, weight: int, latin: bool, jp: str = "gothic") -> skia.Font:
    """Avenir Next has no 300 and its 400 / 500 sit next to Hiragino W3 / W4."""
    if not latin:
        return G.font(size, weight, jp)
    return G.font(size, 400 if weight < 450 else 500 if weight < 550 else 600 if weight < 650 else 700, "avenir")


def width(s: str, size: float, weight: int, tr: float = TR) -> float:
    """Advance of `s`, tracking after the last character included."""
    return sum(G.text_width(t, _face(size, weight, latin), tr) + tr * size for t, latin in _split(s))


def put(c, s: str, x: float, y: float, size: float, weight: int, rgb, a: float, tr: float = TR, align: str = "left",
        shade: float = 0.0) -> float:
    """Text with its baseline at y; `shade` > 0 adds soft shadows (never an outline). Returns the width."""
    w = width(s, size, weight, tr)
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    passes = [(G.paint(rgb, a), 0.0)]
    if shade > 0:  # a soft dark cloud (blurred fat stroke) plus a tight contact shadow
        cloud = G.paint(G.BLACK, 0.42 * shade * a, stroke=size * 0.26, blur=size * 0.14, cap_round=True)
        passes = [(cloud, 3.0), (G.paint(G.BLACK, 0.5 * shade * a, blur=2.2), 2.0)] + passes
    for p, dy in passes:
        xx = x0
        for t, latin in _split(s):
            f = _face(size, weight, latin)
            G.draw_text(c, t, xx, y + dy, f, p, "left", tr)
            xx += G.text_width(t, f, tr) + tr * size
    return w


def _hair(c, x0: float, x1: float, y: float, rgb, a: float, th: float = 1.5) -> None:
    if x1 > x0:
        c.drawLine(x0, y, x1, y, G.paint(rgb, a, stroke=th))


def _desat(sat: float, k: float = 1.0) -> skia.ColorFilter:
    """Colour matrix: saturation `sat`, brightness `k`."""
    r, g, b = 0.2126 * (1 - sat), 0.7152 * (1 - sat), 0.0722 * (1 - sat)
    return skia.ColorFilters.Matrix([k * (r + sat), k * g, k * b, 0, 0, k * r, k * (g + sat), k * b, 0, 0,
                                     k * r, k * g, k * (b + sat), 0, 0, 0, 0, 0, 1, 0])


class Minimal(Look):
    key = "minimal"
    CUT = dict(pause=0.5, tail=0.18, lead=0.12, speed=1.0)
    CAMERA = dict(hold=10.0, med=1.0, punch_every=9999.0, punch_zoom=1.0, punch_dur=0.5)
    AUDIO = dict(bgm="ambient", music_db=-18.0, sfx="none", sfx_db=-2.0)
    GRADE = "eq=contrast=1.03:saturation=0.88:gamma_r=0.99:gamma_b=1.03"
    STYLE = {
        **Look.STYLE,
        "accent": (214, 190, 140),                  # champagne; a brand accent replaces it
        "ink": INK, "paper": (247, 246, 243),
        "sub": {"font": "gothic", "weight": 500, "size": {"wide": 50, "tall": 62}, "fill": WHITE, "tr": 0.03,
                "y": {"wide": 0.915, "tall": 0.735}, "gap": 1.5, "max_w": {"wide": 0.72, "tall": 0.82},
                "speaker": "label", "box": None, "box_a": 0.32},  # speaker: label (dot + name above) | none
        "title": {"font": "gothic", "weight": 500, "size": {"wide": 40, "tall": 56}, "fill": WHITE, "tr": 0.06,
                  "y": {"wide": 0.085, "tall": 0.125}},
        "chapter": {"font": "gothic", "weight": 400, "size": {"wide": 36, "tall": 42}, "fill": WHITE, "tr": 0.12,
                    "y": {"wide": 0.30 if SQUARE else 0.115, "tall": 0.30}},
        "card": {"font": "gothic", "weight": 400, "size": {"wide": 44, "tall": 56}, "fill": INK, "bg": (250, 250, 248),
                 "bg_a": 0.95, "w": {"wide": 0.40, "tall": 0.80}},
        "emphasis": {"font": "gothic", "weight": 600, "size": {"wide": 116, "tall": 104}, "fill": WHITE, "tr": 0.12,
                     "y": {"wide": 0.42, "tall": 0.44}},
        "name": {"font": "gothic", "weight": 500, "size": {"wide": 44, "tall": 50}, "fill": WHITE, "tr": 0.08,
                 "y": {"wide": 0.74, "tall": 0.56}},
        "caption": {"font": "gothic", "weight": 400, "size": {"wide": 34, "tall": 40}, "fill": WHITE, "tr": 0.12,
                    "y": {"wide": 0.11, "tall": 0.14}},
        "credit": {"font": "gothic", "weight": 400, "size": {"wide": 22, "tall": 26}, "fill": WHITE, "tr": 0.06,
                   "y": {"wide": 0.065, "tall": 0.262}},
        "band_bg": "blur",                          # blur (darkened, desaturated) | (r, g, b)
    }
    BUBBLE = (0.80, 0.335, 0.10)                    # a little lower and smaller than the base: clear of the credit line

    # -------------------------------------------------------------- helpers
    def _tr(self, el: str) -> float:
        return float(self.v(el, "tr", TR))

    def _fit(self, text: str, el: str, max_w: float, size: float | None = None) -> tuple[float, list]:
        """(size, lines) for `text`: forced breaks at ｜, else one line, else the best 2-line break; shrunk to max_w."""
        size, weight = size or float(self.v(el, "size")), self.v(el, "weight")
        plain, flags = spans(text)
        lines, cur, cf = [], "", []
        for ch, fl in zip(plain, flags):
            if ch in (JOIN, "\n"):
                lines.append((cur, tuple(cf)))
                cur, cf = "", []
            else:
                cur, cf = cur + ch, cf + [fl]
        lines.append((cur, tuple(cf)))
        if len(lines) == 1 and width(cur, size, weight, self._tr(el)) > max_w:
            lines = list(self._break(cur, cf, self.font(el, size), max_w, size))
        widest = max(width(s, size, weight, self._tr(el)) for s, _ in lines)
        return (size * max_w / widest if widest > max_w else size), lines

    def _marked(self, c, s: str, flags, x: float, base: float, size: float, weight: int, rgb, a: float, tr: float,
                shade: float = 0.0, align: str = "center", line=None) -> None:
        """One line: {emphasis} runs get a heavier weight and a hairline (accent colour, or `line`) underneath."""
        runs, start = [], 0
        for i in range(1, len(s) + 1):
            if i == len(s) or flags[i] != flags[start]:
                runs.append((s[start:i], flags[start]))
                start = i
        heavy = min(weight + 200, 700)
        ws = [width(t, size, heavy if hi else weight, tr) for t, hi in runs]
        xx = x - sum(ws) / 2 if align == "center" else x - (size * 0.5 if s[:1] in OPENERS else 0.0)
        for (t, hi), w in zip(runs, ws):
            put(c, t, xx, base, size, heavy if hi else weight, rgb, a, tr, "left", shade)
            if hi:
                _hair(c, xx, xx + w - tr * size, base + size * 0.2, line or self.S["accent"], 0.95 * a,
                      max(1.6, size * 0.035))
            xx += w

    # -------------------------------------------------------------- subtitles: small, low, no outline
    def sub_zone(self) -> tuple[float, float, float, float]:
        size = float(self.v("sub", "size"))
        y_last = H * self.v("sub", "y")
        top = y_last - size * self.v("sub", "gap") - size * 1.7  # two lines and the speaker label
        return self.x0, top, self.x1 - self.x0, y_last + size * 0.4 - top

    def draw_sub(self, c, sub, t: float) -> None:
        size, lines = self.sub_layout(sub.text)
        k = G.ease_out((t - sub.t0) / 0.22)
        a = G.clamp01((t - sub.t0) / 0.16) * G.clamp01((sub.t1 - t) / 0.14 + 0.3)
        if a <= 0:
            return
        gap, tr = size * self.v("sub", "gap"), self._tr("sub")
        y_first = H * self.v("sub", "y") - (len(lines) - 1) * gap
        weight, rgb = self.v("sub", "weight"), self.v("sub", "fill")
        c.save()
        c.translate(0, (1 - k) * 10)
        if self.v("sub", "box") is not None:
            bw = max(width(_flat(s), size, weight, tr) for s, _ in lines) + size * 1.3
            top, bh = y_first - size * 1.02, (len(lines) - 1) * gap + size * 1.55
            c.drawRRect(G.rrect(W / 2 - bw / 2, top, bw, bh, size * 0.2),
                        G.paint(self.v("sub", "box"), self.v("sub", "box_a", 0.32) * a))
        spk = self.speaker_rgb(sub.speaker)
        if spk is not None and self.v("sub", "speaker") != "none":
            f = size * 0.5
            w = width(sub.speaker, f, 500, 0.14)
            y = y_first - size * 1.3
            c.drawCircle(W / 2 - w / 2 - f * 0.6, y, f * 0.2, G.paint(spk, 0.95 * a))
            put(c, sub.speaker, W / 2 + f * 0.3, y + f * 0.36, f, 500, rgb, 0.95 * a, 0.14, "center", 1.0)
        for j, (s, fl) in enumerate(lines):
            self._marked(c, s, fl, W / 2, y_first + j * gap, size, weight, rgb, a, tr, 1.0)
        c.restore()

    # -------------------------------------------------------------- persistent title: left aligned, rule beneath
    def box(self, ov, center=None, scale: float = 1.0) -> tuple[float, float, float, float]:
        cx, cy = center or self.spots(ov)[0]
        if ov.type == "title_band":
            size, lines = self._fit(ov.text, "title", self.x1 - self.x0)
            return 0.0, H * self.v("title", "y") - size * 0.9, float(W), size * (1.5 * (len(lines) - 1) + 1.8)
        if ov.type == "name":
            size, sub = self.v("name", "size") * scale, _flat(ov.d.get("sub", ""))
            w = max(width(_flat(ov.text), size, self.v("name", "weight"), self._tr("name")),
                    width(sub, size * 0.5, 400, 0.14)) + size * 0.8
            return self.x0, H * cy - size, w, size * (2.0 if sub else 1.2)
        if ov.type == "emphasis":
            size = self.v("emphasis", "size") * scale
            sz, lines = self._fit(ov.text, "emphasis", self.x1 - self.x0 - size * 0.4, size)
            w = max(width(s, sz, self.v("emphasis", "weight"), self._tr("emphasis")) for s, _ in lines) + size * 0.4
            h = sz * 1.3 * len(lines) + size * 0.5
            return W * cx - w / 2, H * cy - h / 2, w, h
        x, y, w, h = super().box(ov, center, scale)
        return (x, y, w, h + self.v("card", "size") * scale * 0.4) if ov.type == "card" else (x, y, w, h)

    def draw_title_band(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.5) * G.clamp01((ov.t1 - t) / 0.4)
        if a <= 0:
            return
        size, lines = self._fit(ov.text, "title", self.x1 - self.x0)
        gap, tr, y0 = size * 1.5, self._tr("title"), H * self.v("title", "y")
        bottom = y0 + gap * (len(lines) - 1) + size * 0.9
        p = skia.Paint(AntiAlias=True)  # a very faint top shade keeps thin white type legible on bright footage
        p.setShader(G.lin_grad((0, 0), (0, bottom + H * 0.06), (G.BLACK, G.BLACK), (0.32 * a, 0.0)))
        c.drawRect(skia.Rect.MakeLTRB(0, 0, W, bottom + H * 0.06), p)
        weight = self.v("title", "weight")
        for j, (s, fl) in enumerate(lines):
            self._marked(c, s, fl, self.x0, y0 + j * gap, size, weight, self.v("title", "fill"), a, tr, 1.0, "left")
        widest = max(width(s, size, weight, tr) for s, _ in lines)
        k = G.ease_out((t - ov.t0) / 0.9)
        y = bottom - size * 0.05
        _hair(c, self.x0, self.x0 + widest * k, y, WHITE, 0.7 * a)
        _hair(c, self.x0, self.x0 + min(size * 0.9, widest) * k, y, self.S["accent"], a, 3.0)

    def draw_credit(self, c, ov, t: float) -> None:
        size, base = self.v("credit", "size"), H * self.v("credit", "y") + self.v("credit", "size") * 0.34
        put(c, _flat(ov.text), self.x1, base, size, self.v("credit", "weight"), WHITE, 0.78, self._tr("credit"),
            "right", 1.0)

    # -------------------------------------------------------------- timed elements
    def draw_chapter(self, c, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.6)
        a = min(G.clamp01((t - ov.t0) / 0.35), G.ease_io((ov.t1 - t) / 0.35))
        if a <= 0:
            return
        size, tr, text = self.v("chapter", "size"), self._tr("chapter"), _flat(ov.text)
        x, base = self.x0 + (1 - k) * 14, H * self.v("chapter", "y")
        w = put(c, text, x, base, size, self.v("chapter", "weight"), self.v("chapter", "fill"), a, tr, "left", 1.0)
        y, kk = base + size * 0.62, G.ease_out((t - ov.t0) / 0.8)
        _hair(c, self.x0, self.x0 + (w + size * 0.8) * kk, y, WHITE, 0.7 * a)
        _hair(c, self.x0, self.x0 + size * 0.9 * kk, y, self.S["accent"], a, 3.0)

    def draw_name(self, c, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.5)
        a = G.clamp01(k * 1.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        (_, fy), scale, see = self.placed(ov)
        a *= see
        size, sub = self.v("name", "size") * scale, _flat(ov.d.get("sub", ""))
        x, cy = self.x0 + (1 - k) * -14, H * fy
        h = size * (2.0 if sub else 1.2)
        c.drawLine(x, cy - h / 2, x, cy + h / 2, G.paint(self.S["accent"], a, stroke=3.0))
        base = cy - size * 0.12 if sub else cy + size * 0.3
        put(c, _flat(ov.text), x + size * 0.5, base, size, self.v("name", "weight"), WHITE, a, self._tr("name"),
            "left", 1.0)
        if sub:
            put(c, sub, x + size * 0.5, cy + size * 0.68, size * 0.5, 400, WHITE, 0.8 * a, 0.14, "left", 1.0)

    def draw_caption(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.5) * G.ease_io((ov.t1 - t) / 0.4)
        if a <= 0:
            return
        size, y = self.v("caption", "size"), H * self.v("caption", "y")
        _hair(c, self.x0, self.x0 + size * 0.9 * G.ease_out((t - ov.t0) / 0.7), y - size, self.S["accent"], a, 3.0)
        put(c, _flat(ov.text), self.x0, y, size, self.v("caption", "weight"), WHITE, a, self._tr("caption"),
            "left", 1.0)
        if ov.d.get("sub"):
            put(c, _flat(ov.d["sub"]), self.x0, y + size * 1.15, size * 0.62, 400, WHITE, 0.8 * a, 0.14, "left",
                1.0)

    def draw_note(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a > 0:
            size = self.v("credit", "size") * 1.15
            put(c, "※" + _flat(ov.text).lstrip("※"), self.x0, self.sub_zone()[1] - size * 0.4, size, 400, WHITE,
                0.85 * a, 0.06, "left", 1.0)

    # -------------------------------------------------------------- cards and the big word
    def draw_card(self, c, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.5)
        a = G.clamp01((t - ov.t0) / 0.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        x, y, w, h = self.card_rect(ov)
        _, scale, see = self.placed(ov)
        a *= see
        size, ink, r = self.v("card", "size") * scale, self.v("card", "fill"), 10.0 if TALL else 7.0
        y += (1 - k) * 14
        G.shadow(c, x, y, w, h, r, blur=28, dy=12, a=0.16 * a)
        c.drawRRect(G.rrect(x, y, w, h, r), G.paint(self.v("card", "bg"), self.v("card", "bg_a") * a))
        pad, hh = size * 0.9, size * 1.6
        title = _flat(ov.d.get("title", ""))
        if title:
            _hair(c, x + pad, x + pad + size * 0.7, y + size * 0.55, self.S["accent"], a, 3.0)
            put(c, title, x + pad, y + hh * 0.5 + size * 0.5, size * 0.66, 500, ink, a, 0.12)
            _hair(c, x + pad, x + w - pad, y + hh, RULE, a, 1.0)
        yy = y + hh + size * 0.55
        if ov.d.get("big"):
            put(c, _flat(str(ov.d["big"])), x + pad, yy + size * 1.4, size * 1.9, 500, ink, a, 0.0)
            yy += size * 1.75
        for j, item in enumerate(ov.d.get("items", [])):
            ka = G.clamp01((t - ov.t0 - 0.25 - 0.15 * j) / 0.3) * a
            base = yy + size * 0.6 + j * size * 1.55 + size * 0.3
            put(c, f"{j + 1:02d}", x + pad, base, size * 0.56, 500, GREY, ka, 0.1)
            plain, flags = spans(str(item).replace(JOIN, " "))
            self._marked(c, plain, flags, x + pad + size * 1.3, base, size * 0.8, self.v("card", "weight"), ink, ka,
                         TR, 0.0, "left", ink)
            if j:
                _hair(c, x + pad, x + w - pad, base - size * 0.95, RULE, 0.55 * ka, 1.0)

    def draw_emphasis(self, c, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.7)
        a = G.clamp01((t - ov.t0) / 0.4) * G.ease_io((ov.t1 - t) / 0.35)
        if a <= 0:
            return
        (fx, fy), scale, see = self.placed(ov)
        a *= see
        size0 = self.v("emphasis", "size") * scale
        size, lines = self._fit(ov.text, "emphasis", self.x1 - self.x0 - size0 * 0.4, size0)
        tr, gap = self._tr("emphasis") * (1.6 - 0.6 * k), size * 1.3  # the letters settle together as it appears
        cx, cy = W * fx, H * fy
        top = cy - gap * len(lines) / 2
        for j, (s, fl) in enumerate(lines):
            base = top + gap * (j + 0.78) + (1 - k) * 8
            self._marked(c, s, fl, cx, base, size, self.v("emphasis", "weight"), self.v("emphasis", "fill"), a, tr, 1.0)
        _hair(c, cx - size * 0.35 * k, cx + size * 0.35 * k, top - size * 0.12, self.S["accent"], a, 3.0)

    # -------------------------------------------------------------- footage + background
    def draw_card_mode(self, c, frame, box, k: float, face) -> None:
        """Cards over 9:16 close-ups: the picture goes dark and soft, the speaker sits in a thin-ringed circle."""
        if k <= 0:
            return
        ox, oy, ow, oh = box
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(20 * k, 20 * k, skia.TileMode.kClamp))
        p.setColorFilter(_desat(0.8))
        p.setAlphaf(G.clamp01(k * 1.2))
        c.drawImageRect(frame, skia.Rect.MakeXYWH(ox, oy, ow, oh), G.SAMPLING, p)
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint(G.BLACK, 0.5 * k))
        fx, fy, fh = face if face else (ox + ow / 2, oy + oh * 0.35, oh * 0.16)
        bx, by, br = W * self.BUBBLE[0], H * self.BUBBLE[1], W * self.BUBBLE[2]
        rr = br * G.ease_out(k)
        side = fh * 2.3
        sx, sy = (fx - ox) * frame.width() / ow, (fy - oy) * frame.height() / oh
        sside = side * frame.width() / ow
        c.save()
        path = skia.Path()
        path.addCircle(bx, by, rr)
        c.clipPath(path, doAntiAlias=True)
        c.drawImageRect(frame, skia.Rect.MakeXYWH(sx - sside / 2, sy - sside / 2, sside, sside),
                        skia.Rect.MakeXYWH(bx - rr, by - rr, 2 * rr, 2 * rr), G.SAMPLING, skia.Paint(AntiAlias=True))
        c.restore()
        c.drawCircle(bx, by, rr, G.paint(WHITE, 0.9 * k, stroke=2.5))
        c.drawCircle(bx, by, rr + 12, G.paint(self.S["accent"], 0.8 * k, stroke=1.5))

    def draw_band_bg(self, c, frame, box) -> None:
        """Behind footage that does not fill the canvas: the picture blurred, desaturated and darkened (or a colour).
        The blur runs on a 1/12-size copy that is scaled up: the same soft look at a fraction of the per-frame cost."""
        bg = self.S.get("band_bg", "blur")
        if bg != "blur":
            c.clear(G.col(G.hexrgb(bg)))
            return
        ow, oh = box[2], box[3]
        sw, sh = max(8, round(ow / 12)), max(8, round(oh / 12))
        small, sc = G.new_surface(sw, sh)
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(2.5, 2.5, skia.TileMode.kClamp))
        sc.drawImageRect(frame, skia.Rect.MakeWH(sw, sh), G.SAMPLING, p)
        z = max(W / ow, H / oh) * 1.1
        q = skia.Paint(AntiAlias=True)
        q.setColorFilter(_desat(0.35, 0.5))
        c.drawImageRect(small.makeImageSnapshot(), skia.Rect.MakeWH(sw, sh),
                        skia.Rect.MakeXYWH(W / 2 - ow * z / 2, H / 2 - oh * z / 2, ow * z, oh * z), G.SAMPLING, q)

    # -------------------------------------------------------------- full-screen cards: light paper
    def _card_bg(self, c, bg, dim: float = 0.55) -> None:
        """Warm light grey with a faint vignette: logos made for white grounds read on it."""
        p = skia.Paint(AntiAlias=True)
        p.setShader(G.rad_grad((W / 2, H / 2), max(W, H) * 0.72, ((248, 247, 245), (231, 230, 227))))
        c.drawRect(skia.Rect.MakeWH(W, H), p)

    def _centered(self, c, text: str, el: str, y: float, a: float, size: float, rise: float = 0.0) -> None:
        """Ink lines centred on y; `rise` shifts them down while they settle."""
        sz, lines = self._fit(text, el, W * 0.84, size)
        gap, tr = sz * 1.45, self._tr(el)
        top = y - gap * (len(lines) - 1) / 2
        for j, (s, fl) in enumerate(lines):
            self._marked(c, s, fl, W / 2, top + j * gap + sz * 0.34 + rise, sz, self.v(el, "weight"), INK, a, tr, 0.0,
                         line=INK)

    def _logo(self, c, y: float, a: float, rise: float = 0.0, at: float = 0.0) -> float:
        """The brand logo (at most 30 % of the height); `at` 0 / 0.5 / 1 puts y on its top / middle / bottom.
        Returns the bottom edge."""
        img = G.image(self.brand["logo"])
        lw = min(W * 0.34, H * 0.30 * img.width() / img.height())
        lh = lw * img.height() / img.width()
        top = y - lh * at
        G.draw_image(c, img, W / 2 - lw / 2, top + rise, lw, lh, alpha=a)
        return top + lh

    def card_title(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg)
        k = G.ease_out(tt / 0.9)
        a = G.clamp01(tt / 0.5) * G.clamp01((dur - tt) / 0.3)
        channel = self.brand.get("channel")
        text = part.get("text") or channel or self.brand.get("name", "")
        y = H * 0.46
        if self.brand.get("logo"):
            y = self._logo(c, H * 0.17, a, (1 - k) * -10) + H * 0.13
        _hair(c, W / 2 - 34 * k, W / 2 + 34 * k, y - H * 0.075, self.S["accent"], a, 3.0)
        if text:
            self._centered(c, text, "title", y, a, 64 if not TALL else 66, (1 - k) * 10)
        if channel and part.get("text"):
            put(c, channel, W / 2, y + H * (0.075 if TALL else 0.10), 28 if TALL else 26, 400, GREY, a, 0.3, "center")

    def card_text(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg)
        k = G.ease_out(tt / 0.8)
        a = G.clamp01(tt / 0.5) * G.clamp01((dur - tt) / 0.3)
        self._centered(c, part.get("text", ""), "emphasis", H / 2, a, 84, (1 - k) * 10)

    def card_eyecatch(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg)
        k = G.ease_io5(tt / max(dur, 0.1))
        half = W * 0.36 * math.sin(math.pi * k)  # a hairline opens from the centre and closes again
        _hair(c, W / 2 - half, W / 2 + half, H / 2, INK, 0.7)
        _hair(c, W / 2 - min(half, 34), W / 2 + min(half, 34), H / 2, self.S["accent"], 1.0, 3.0)
        a = G.win(tt, 0.15, dur - 0.15, 0.3, 0.3)
        if self.brand.get("logo"):
            self._logo(c, H / 2 - 40, a, at=1.0)
        elif part.get("text"):
            self._centered(c, part["text"], "title", H / 2 - H * 0.08, a, 56 if not TALL else 60)

    def card_end(self, c, part, tt, dur, bg) -> None:
        """Quiet thanks line, then two outlined buttons that a cursor clicks (like, subscribe)."""
        self._card_bg(c, bg)
        a = G.clamp01(tt / 0.5) * G.clamp01((dur - tt) / 0.25)
        thanks = part.get("text") or ("ご視聴ありがとうございました" if self.fmt == "youtube" else "フォローして続きをチェック")
        y = H * (0.28 if not TALL else 0.30)
        self._centered(c, thanks, "title", y, a, 54 if not TALL else 58, (1 - G.ease_out(tt / 0.8)) * 8)
        _hair(c, W / 2 - 28, W / 2 + 28, y + H * 0.06, self.S["accent"], a, 3.0)
        self._like_subscribe(c, tt, a)

    def _like_subscribe(self, c, tt: float, a: float) -> None:
        """Two hairline pills fade in; the cursor clicks like, then subscribe (the pill fills and says 登録済み)."""
        s = 1.0 if not TALL else 1.1
        bw, bh, gap, cy = 330 * s, 84 * s, 40 * s, H * 0.45
        xs = [W / 2 - bw - gap / 2, W / 2 + gap / 2]
        click = [0.9, 1.7]
        yt = self.fmt == "youtube"
        labels = (("高評価" if yt else "いいね", "高評価" if yt else "いいね"),
                  ("チャンネル登録" if yt else "フォロー", "登録済み" if yt else "フォロー中"))
        for i, (label, done) in enumerate(labels):
            ka = G.ease_out((tt - 0.2 - 0.15 * i) / 0.5) * a
            on = tt >= click[i]
            pulse = 1 + 0.05 * math.exp(-max(0.0, tt - click[i]) * 9) if on else 1.0
            c.save()
            c.translate(xs[i] + bw / 2, cy)
            c.scale(pulse, pulse)
            pill = G.paint(INK, 0.95 * ka) if on else G.paint(INK, 0.75 * ka, stroke=1.6)
            c.drawRRect(G.rrect(-bw / 2, -bh / 2, bw, bh, bh / 2), pill)
            txt = done if on else label
            tw = width(txt, 30 * s, 500, 0.1)
            put(c, txt, 0, 30 * s * 0.36, 30 * s, 500, WHITE if on else INK, ka, 0.1, "center")
            if on and i == 0:  # thin check before the label
                x0, ck = -tw / 2 - 26 * s, 9 * s
                path = skia.Path()
                path.moveTo(x0 - ck, 0)
                path.lineTo(x0 - ck * 0.3, ck * 0.7)
                path.lineTo(x0 + ck, -ck * 0.8)
                c.drawPath(path, G.paint(WHITE, ka, stroke=2.2 * s, cap_round=True))
            c.restore()
            q = (tt - click[i]) / 0.6  # a single ring spreads from the pill
            if 0 <= q < 1:
                grow = 14 + 36 * G.ease_out(q)
                c.drawRRect(G.rrect(xs[i] - grow, cy - bh / 2 - grow, bw + 2 * grow, bh + 2 * grow, bh / 2 + grow),
                            G.paint(INK, 0.4 * (1 - q) * a, stroke=1.4))
        pts = [(W * 0.8, H * 0.85, 0.3), (xs[0] + bw * 0.6, cy + bh * 0.2, click[0]),
               (xs[1] + bw * 0.6, cy + bh * 0.2, click[1]), (W * 0.85, H * 0.9, click[1] + 0.8)]
        for (x0, y0, t0), (x1, y1, t1) in zip(pts, pts[1:]):
            if t0 <= tt < t1:
                k = G.ease_io((tt - t0) / (t1 - t0))
                self._cursor(c, G.lerp(x0, x1, k), G.lerp(y0, y1, k), a, tt, click)
                break

    # -------------------------------------------------------------- sound: almost none
    def sfx(self, ov) -> list[tuple[float, str, float]]:
        return [(ov.t0, "tick", -20.0)] if ov.type == "card" else []

    def card_sfx(self, part: dict, t0: float, dur: float) -> list[tuple[float, str, float]]:
        if part["card"] == "end" and self.fmt == "youtube":
            return [(t0 + 0.9, "click", -14.0), (t0 + 1.7, "click", -14.0)]
        return []


LOOK = Minimal
