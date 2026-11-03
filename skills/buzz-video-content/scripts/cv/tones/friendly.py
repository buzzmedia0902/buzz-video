"""親しみ・やわらか (friendly): a soft look for talk-show clips and interviews. Cream / pastel panels, round
Tsukushi type in warm brown, white speech-bubble subtitles with a pastel speaker-colour border and name pill, a yellow
highlighter behind {key words} with small sparkles and a soft pop, calm camera, acoustic BGM, cute SFX.

Use for: people and personality (対談, interviews, vlogs, behind the scenes), anything that should feel warm and light.
"""
from __future__ import annotations

import functools
import math
import random

import skia

from cv import gfx as G
from cv.canvas import H, TALL, W
from cv.cues import JOIN
from cv.tones.base import LAYOUT, Look, spans, strip_markup

WHITE = (255, 255, 255)
BROWN, BROWN_SOFT = (84, 62, 52), (150, 126, 112)
CORAL, MINT, SKY, LAVENDER = (240, 132, 118), (110, 196, 164), (118, 180, 226), (184, 164, 228)
PEACH, BUTTER, WARM = (255, 196, 176), (255, 226, 120), (150, 110, 90)
CREAM, CREAM_LOW = (255, 249, 240), (253, 234, 220)
NEUTRAL = (222, 208, 198)                      # bubble border when nobody is tagged
SQUARE = W == H                                # 1:1 shares the "wide" values: a few anchors move down (below the title)
TRACK = -0.03                                  # Tsukushi kana sit a little loose: tighten by 3 % of the size
DOTS = (CORAL, MINT, SKY, LAVENDER)            # bullet colours of cards
# pastel blobs of the paper: (colour, radius as a share of the short side, x, y as shares of the canvas)
BLOBS = {"tall": ((PEACH, .35, .11, .135), (MINT, .30, .91, .27), (LAVENDER, .28, .13, .69), (BUTTER, .33, .89, .81),
                  (SKY, .24, .56, .94)),
         "wide": ((PEACH, .46, .07, .22), (MINT, .40, .94, .26), (LAVENDER, .38, .05, .88), (BUTTER, .44, .96, .86),
                  (SKY, .30, .30, 1.0), (PEACH, .26, .72, .02))}
_rng = random.Random(30)
SPARKS = tuple((_rng.random(), _rng.random(), _rng.randrange(3), _rng.uniform(0, 6.28), _rng.uniform(0.6, 1.4))
               for _ in range(18))             # (x, y, colour, phase, speed) of the paper's twinkles
SPARK_RGB = ((255, 205, 120), (240, 150, 140), (130, 200, 230))
LEAD_NAME, LEAD_CHAPTER, LEAD_CAPTION = ("dot", CORAL), ("spark", CORAL), ("spark", MINT)  # (kind, colour) of plates


def _flat(text: str) -> str:
    """Text for a one-line label: markup removed, a forced line break (｜) becomes a space."""
    return strip_markup(text.replace(JOIN, " "))


def _soft(rgb, k: float = 0.2):
    return G.mix_rgb(rgb, WHITE, k)


def _lum(rgb) -> float:
    lin = [(v / 255 / 12.92) if v / 255 <= 0.04045 else ((v / 255 + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _on(rgb):
    """Text colour for a fill: white (the approved look) unless the fill is so light that white would vanish."""
    return WHITE if 1.05 / (_lum(rgb) + 0.05) >= 1.8 else BROWN


def _w(s: str, f: skia.Font) -> float:
    """Width of `s` with the tone's tracking (the same spacing draw_text uses)."""
    return G.text_width(s, f, TRACK)


def _text(c, s: str, x: float, y: float, f: skia.Font, paint, align: str = "left") -> None:
    G.draw_text(c, s, x, y, f, paint, align, TRACK)


def _text_mid(c, s: str, x: float, cy: float, f: skia.Font, paint, align: str = "left") -> None:
    G.draw_text_mid(c, s, x, cy, f, paint, align, TRACK)


def _shadow(c, x, y, w, h, r, a, blur: float = 16.0, dy: float = 8.0) -> None:
    c.drawRRect(G.rrect(x, y + dy, w, h, r), G.paint(WARM, a, blur=blur))


def _sparkle(c, x, y, r, rgb, alpha) -> None:
    """Four-point star."""
    k = r * 0.16
    p = skia.Path()
    p.moveTo(x, y - r)
    for qx, qy, ex, ey in ((x + k, y - k, x + r, y), (x + k, y + k, x, y + r), (x - k, y + k, x - r, y),
                           (x - k, y - k, x, y - r)):
        p.quadTo(qx, qy, ex, ey)
    p.close()
    c.drawPath(p, G.paint(rgb, alpha))


def _marker(c, x, y, w, h, rgb, alpha, grow: float = 1.0) -> None:
    """Highlighter band, wiped in from the left by `grow` (0..1)."""
    if grow > 0:
        c.drawRRect(G.rrect(x, y, min(w, max(w * grow, h)), h, h / 2), G.paint(rgb, alpha))


def _squiggle(c, x0, x1, y, amp, rgb, alpha, grow: float = 1.0, wave: float = 34.0) -> None:
    """Hand-drawn wavy underline from x0 to x0 + (x1 - x0) * grow."""
    end = x0 + (x1 - x0) * grow
    if end - x0 < 2:
        return
    p = skia.Path()
    p.moveTo(x0, y)
    x, up = x0, 1
    while x < end:
        nx = min(x + wave, end)
        p.quadTo((x + nx) / 2, y - up * amp * 2, nx, y)
        x, up = nx, -up
    c.drawPath(p, G.paint(rgb, alpha, stroke=amp * 1.1, cap_round=True))


def _heart(c, x, y, u, rgb, alpha) -> None:
    p = skia.Path()
    p.moveTo(x, y + 0.9 * u)
    p.cubicTo(x - 1.35 * u, y - 0.1 * u, x - 0.85 * u, y - 1.05 * u, x, y - 0.4 * u)
    p.cubicTo(x + 0.85 * u, y - 1.05 * u, x + 1.35 * u, y - 0.1 * u, x, y + 0.9 * u)
    c.drawPath(p, G.paint(rgb, alpha))


class Friendly(Look):
    key = "friendly"
    CUT = dict(pause=0.45, tail=0.14, lead=0.10, speed=1.0)
    CAMERA = dict(hold=12.0, med=1.10, punch_every=60.0, punch_zoom=1.06, punch_dur=0.5)
    AUDIO = dict(bgm="acoustic", music_db=-15.0, sfx="cute", sfx_db=-2.0)
    GRADE = "eq=contrast=0.99:brightness=0.015:saturation=1.08:gamma_r=1.03:gamma_b=0.97"
    STYLE = {
        **Look.STYLE,
        "accent": (255, 214, 64),                  # highlighter yellow (a brand accent replaces it)
        "ink": BROWN, "paper": CREAM,
        "sub": {"font": "round", "weight": 700, "size": {"wide": 54, "tall": 64}, "fill": "ink",
                "y": {"wide": 0.885, "tall": 0.70}, "gap": 1.22, "max_w": {"wide": 0.80, "tall": 0.88},
                "speaker": "pill"},
        "title": {"font": "round", "weight": 700, "size": {"wide": 50, "tall": 74}, "fill": "ink", "band": CREAM,
                  "band_a": 0.95, "y": {"wide": 0.13, "tall": 0.155}},
        "chapter": {"font": "round", "weight": 700, "size": {"wide": 36, "tall": 44}, "fill": "ink", "bg": PEACH,
                    "y": {"wide": 0.285 if SQUARE else 0.10, "tall": 0.282}},
        "card": {"font": "round", "weight": 700, "size": {"wide": 44, "tall": 56}, "fill": "ink", "bg": WHITE,
                 "bg_a": 0.97, "head": (255, 226, 200), "head_fill": "ink", "w": {"wide": 0.40, "tall": 0.80}},
        "emphasis": {"font": "round", "weight": 700, "size": {"wide": 128, "tall": 140}, "fill": "ink",
                     "y": {"wide": 0.42, "tall": 0.44}},
        "name": {"font": "round", "weight": 700, "size": {"wide": 44, "tall": 50}, "fill": "ink",
                 "y": {"wide": 0.74, "tall": 0.56}},
        "caption": {"font": "round", "weight": 700, "size": {"wide": 38, "tall": 44}, "fill": "ink",
                    "y": {"wide": 0.11, "tall": 0.14}},
        "credit": {"font": "round", "weight": 700, "size": {"wide": 24, "tall": 28}, "fill": BROWN_SOFT,
                   "y": {"wide": 0.245 if SQUARE else 0.065, "tall": 0.262}},
        "band_bg": "paper",                        # paper | blur (soft blurred footage under a cream veil)
    }
    BUBBLE = (0.80, 0.335, 0.10)                   # a little lower and smaller than the base: clear of the credit line

    # -------------------------------------------------------------- helpers
    @property
    def marker(self):
        """Highlighter colour: the accent, lightened so brown text stays crisp on it."""
        return _soft(self.S["accent"], 0.22)

    def _fit(self, text: str, el: str, max_w: float, size: float | None = None) -> tuple[skia.Font, list]:
        """(font, lines) for `text`: forced breaks at ｜, else one line, else the best 2-line break; shrunk to max_w."""
        size = size or float(self.v(el, "size"))
        f = self.font(el, size)
        plain, flags = spans(text)
        lines, cur, cf = [], "", []
        for ch, fl in zip(plain, flags):
            if ch in (JOIN, "\n"):
                lines.append((cur, tuple(cf)))
                cur, cf = "", []
            else:
                cur, cf = cur + ch, cf + [fl]
        lines.append((cur, tuple(cf)))
        if len(lines) == 1 and _w(cur, f) > max_w:
            lines = list(self._break(cur, cf, f, max_w, size))
        widest = max(_w(s, f) for s, _ in lines)
        return (self.font(el, size * max_w / widest) if widest > max_w else f), lines

    def _marked(self, c, s: str, flags, x: float, base: float, f: skia.Font, alpha: float, grow: float = 1.0,
                fill=BROWN, align: str = "center") -> None:
        """One line on baseline `base`: highlighter bands behind the {runs}, then the text."""
        size = f.getSize()
        x = x - _w(s, f) / 2 if align == "center" else x
        runs, start = [], 0
        for i in range(1, len(s) + 1):
            if i == len(s) or flags[i] != flags[start]:
                run = s[start:i]
                runs.append((run, x, flags[start], f.measureText(run) + TRACK * size * len(run)))
                x += runs[-1][3]
                start = i
        for _, rx, hi, rw in runs:
            if hi:
                _marker(c, rx - size * 0.1, base - size * 0.56, rw + size * 0.2, size * 0.66, self.marker, 0.92 * alpha,
                        grow)
        for run, rx, _, _ in runs:
            _text(c, run, rx, base, f, G.paint(fill, alpha))

    @functools.cached_property
    def paper(self) -> skia.Image:
        """Cream gradient with soft pastel blobs, drawn once."""
        s, c = G.new_surface()
        p = skia.Paint(AntiAlias=True)
        p.setShader(G.lin_grad((0, 0), (0, H), (CREAM, CREAM_LOW)))
        c.drawRect(skia.Rect.MakeWH(W, H), p)
        short = min(W, H)
        for rgb, r, fx, fy in BLOBS[LAYOUT]:
            stops = [0.42 * (1 - x) ** 1.6 for x in (0, .25, .5, .75, 1)]
            p = skia.Paint(AntiAlias=True)
            p.setShader(G.rad_grad((W * fx, H * fy), r * short, [rgb] * 5, stops, [0, .25, .5, .75, 1]))
            c.drawCircle(W * fx, H * fy, r * short, p)
        return s.makeImageSnapshot()

    def draw_paper(self, c, t: float = 0.9) -> None:
        """The paper with twinkling sparkles (t = seconds)."""
        c.drawImage(self.paper, 0, 0)
        for fx, fy, ci, ph, sp in SPARKS:
            env = 0.5 + 0.5 * math.sin(t * 2.2 * sp + ph)
            _sparkle(c, W * fx, H * (0.03 + 0.94 * fy) + 12 * math.sin(t * 0.6 * sp + ph), 22 * (0.35 + 0.35 * env),
                     SPARK_RGB[ci], 0.25 + 0.5 * env)

    # -------------------------------------------------------------- subtitles: speech bubble
    def _bubble(self, lines, size):
        """(x, y, w, h, middle y of the last line) of the bubble that holds `lines`."""
        f = self.font("sub", size)
        gap = size * self.v("sub", "gap")
        pad_x, pad_y = size * 0.66, size * 0.40
        w = max(_w(s, f) for s, _ in lines) + 2 * pad_x
        mid = H * self.v("sub", "y") - G.cap_height(f) / 2
        n = len(lines) - 1
        return W / 2 - w / 2, mid - n * gap - size / 2 - pad_y, w, n * gap + size + 2 * pad_y, mid

    def sub_zone(self) -> tuple[float, float, float, float]:
        _, top, _, h, _ = self._bubble((("", ()),) * 2, float(self.v("sub", "size")))
        pill = float(self.v("sub", "size")) * 0.45  # the name pill sits on the bubble's top edge
        return self.x0, top - pill, self.x1 - self.x0, h + pill

    def _pill(self, c, name: str, rgb, x: float, y: float, size: float, alpha: float) -> None:
        """Speaker name pill tilted a little, centred on (x, y) = its left end."""
        f = G.font(size * 0.52, 700, self.v("sub", "font"))
        w, h = _w(name, f) + size * 0.8, size * 0.78
        c.save()
        c.translate(x + w / 2, y)
        c.rotate(-3.0)
        _shadow(c, -w / 2, -h / 2, w, h, h / 2, 0.14 * alpha, blur=6, dy=3)
        c.drawRRect(G.rrect(-w / 2, -h / 2, w, h, h / 2), G.paint(rgb, 0.98 * alpha))
        _text_mid(c, name, 0, 0, f, G.paint(_on(rgb), alpha), "center")
        c.restore()

    def draw_sub(self, c, sub, t: float) -> None:
        size, lines = self.sub_layout(sub.text)
        hi = any(any(fl) for _, fl in lines)
        p = (t - sub.t0) / 0.2
        a = G.clamp01(p * 1.8) * G.ease_out((sub.t1 - t) / 0.08 + 0.5)
        if a <= 0:
            return
        pop = 0.94 + 0.06 * G.back(p, 1.6) + (0.05 * math.sin(math.pi * G.clamp01((t - sub.t0) / 0.35)) if hi else 0)
        x, y, w, h, mid = self._bubble(lines, size)
        f, gap = self.font("sub", size), size * self.v("sub", "gap")
        spk = self.speaker_rgb(sub.speaker)
        edge, bw, r = (_soft(spk, 0.12) if spk else NEUTRAL), size * (0.11 if hi else 0.075), min(size * 0.7, h / 2)
        c.save()
        c.translate(W / 2, y + h / 2 + (1 - G.ease_out(p)) * 14)
        c.scale(pop, pop)
        c.translate(-W / 2, -(y + h / 2))
        _shadow(c, x, y, w, h, r, 0.18 * a)
        c.drawRRect(G.rrect(x, y, w, h, r), G.paint(WHITE, 0.98 * a))
        c.drawRRect(G.rrect(x + bw / 2, y + bw / 2, w - bw, h - bw, r - bw / 2), G.paint(edge, a, stroke=bw))
        grow = G.ease_out((t - sub.t0 - 0.05) / 0.25)
        for j, (s, fl) in enumerate(lines):
            base = mid - (len(lines) - 1 - j) * gap + G.cap_height(f) / 2
            self._marked(c, s, fl, W / 2, base, f, a, grow, self.v("sub", "fill"))
        if spk is not None and self.v("sub", "speaker") == "pill":
            self._pill(c, sub.speaker, spk, x + size * 0.45, y - size * 0.02, size, a)
        if hi:  # little sparkles pop around the bubble
            q = math.sin(math.pi * G.clamp01((t - sub.t0) / 0.6))
            for (dx, dy), rgb in zip(((-0.5, -0.5), (0.5, -0.6), (0.52, 0.45)), SPARK_RGB):
                _sparkle(c, W / 2 + dx * (w + 30), y + h / 2 + dy * (h + 24), size * 0.26 * (0.6 + 0.6 * q), rgb, q * a)
        c.restore()

    # -------------------------------------------------------------- plates (name, caption, chapter, note, credit)
    def _plate_size(self, text: str, sub: str, size: float, el: str, lead) -> tuple[float, float]:
        text, sub = _flat(text), _flat(sub)
        f, fs = self.font(el, size), self.font(el, size * 0.52)
        w = max(_w(text, f), _w(sub, fs)) + size * (1.25 + (0.65 if lead else 0))
        return w, size * (1.75 if sub else 1.3)

    def _plate(self, c, text: str, x: float, cy: float, size: float, alpha: float, sub: str = "", el: str = "name",
               fill=WHITE, lead=None, align: str = "left") -> None:
        """Rounded label with an optional second line and a leading dot / sparkle ((kind, rgb))."""
        text, sub = _flat(text), _flat(sub)
        f, fs = self.font(el, size), self.font(el, size * 0.52)
        w, h = self._plate_size(text, sub, size, el, lead)
        left, top, r = (x if align == "left" else x - w), cy - h / 2, min(h / 2, size * 0.75)
        _shadow(c, left, top, w, h, r, 0.16 * alpha, blur=12, dy=6)
        c.drawRRect(G.rrect(left, top, w, h, r), G.paint(fill, 0.96 * alpha))
        tx = left + size * 0.62
        if lead:
            kind, rgb = lead
            if kind == "dot":
                c.drawCircle(tx, cy, size * 0.2, G.paint(rgb, alpha))
            else:
                _sparkle(c, tx, cy, size * 0.34, rgb, alpha)
            tx += size * 0.65
        ink = self.v(el, "fill")
        if sub:
            _text_mid(c, text, tx, cy - size * 0.36, f, G.paint(ink, alpha))
            _text_mid(c, sub, tx + size * 0.02, cy + size * 0.46, fs, G.paint(BROWN_SOFT, alpha))
        else:
            _text_mid(c, text, tx, cy, f, G.paint(ink, alpha))

    def box(self, ov, center=None, scale: float = 1.0) -> tuple[float, float, float, float]:
        cx, cy = center or self.spots(ov)[0]
        if ov.type == "name":
            w, h = self._plate_size(ov.text, ov.d.get("sub", ""), self.v("name", "size") * scale, "name", LEAD_NAME)
            return self.x0, H * cy - h / 2, w, h
        if ov.type == "emphasis":
            size = self.v("emphasis", "size") * scale
            f, lines = self._fit(ov.text, "emphasis", self.x1 - self.x0 - size * 0.6, size)
            w = max(_w(s, f) for s, _ in lines) + size * 0.6
            h = f.getSize() * 1.15 * len(lines) + size * 0.2
            return W * cx - w / 2, H * cy - h / 2, w, h
        x, y, w, h = super().box(ov, center, scale)
        return (x, y, w, h + self.v("card", "size") * scale * 0.4) if ov.type == "card" else (x, y, w, h)

    def draw_chapter(self, c, ov, t: float) -> None:
        k = G.back((t - ov.t0) / 0.45, 1.5)
        a = min(G.clamp01((t - ov.t0) / 0.2), G.ease_io((ov.t1 - t) / 0.3))
        if a <= 0:
            return
        size = self.v("chapter", "size")
        w, h = self._plate_size(ov.text, "", size, "chapter", LEAD_CHAPTER)
        x, cy = self.x0 + (k - 1) * (w + 40), H * self.v("chapter", "y") + h / 2
        self._plate(c, ov.text, x, cy, size, a, "", "chapter", self.v("chapter", "bg"), LEAD_CHAPTER)

    def draw_name(self, c, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.4)
        a = G.clamp01(k * 1.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        (_, fy), scale, see = self.placed(ov)
        self._plate(c, ov.text, self.x0 - (1 - k) * 40, H * fy, self.v("name", "size") * scale, a * see,
                    ov.d.get("sub", ""), "name", WHITE, LEAD_NAME)

    def draw_caption(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.35) * G.ease_io((ov.t1 - t) / 0.35)
        if a > 0:
            self._plate(c, ov.text, self.x0, H * self.v("caption", "y"), self.v("caption", "size"), a,
                        ov.d.get("sub", ""), "caption", WHITE, LEAD_CAPTION)

    def draw_note(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.2) * G.ease_io((ov.t1 - t) / 0.2)
        if a > 0:
            size = self.v("credit", "size") * 1.25
            self._plate(c, "※" + _flat(ov.text).lstrip("※"), self.x0, self.sub_zone()[1] - size * 1.1, size, a,
                        "", "credit", CREAM)

    def draw_credit(self, c, ov, t: float) -> None:
        self._plate(c, ov.text, self.x1, H * self.v("credit", "y"), self.v("credit", "size"), 0.92, "", "credit", CREAM,
                    align="right")

    # -------------------------------------------------------------- persistent title
    def draw_title_band(self, c, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.3) * G.clamp01((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        f, lines = self._fit(ov.text, "title", self.x1 - self.x0)
        size = f.getSize()
        _, top, _, h = self.box(ov)
        x, w, r = 36.0, W - 72.0, size * 0.62
        _shadow(c, x, top, w, h, r, 0.2 * a, blur=18, dy=9)
        c.drawRRect(G.rrect(x, top, w, h, r), G.paint(self.v("title", "band"), self.v("title", "band_a") * a))
        c.drawRRect(G.rrect(x + 3, top + 3, w - 6, h - 6, r - 3), G.paint(PEACH, 0.85 * a, stroke=5))
        gap = size * 1.22
        for j, (s, fl) in enumerate(lines):
            base = top + h / 2 + (j - (len(lines) - 1) / 2) * gap + G.cap_height(f) / 2
            self._marked(c, s, fl, W / 2, base, f, a, 1.0, self.v("title", "fill"))
        _sparkle(c, x + w - size * 0.5, top - size * 0.05, size * 0.34, CORAL, 0.9 * a)
        _sparkle(c, x + w - size, top + size * 0.1, size * 0.2, SKY, 0.9 * a)

    # -------------------------------------------------------------- cards and big words
    def draw_card(self, c, ov, t: float) -> None:
        k = G.back((t - ov.t0) / 0.4, 1.8)
        a = G.clamp01((t - ov.t0) / 0.15) * G.ease_io((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        x, y, w, h = self.card_rect(ov)
        _, scale, see = self.placed(ov)
        a *= see
        size, ink = self.v("card", "size") * scale, self.v("card", "fill")
        r, hh = size * 0.7, size * 1.6
        z = 0.88 + 0.12 * k
        c.save()
        c.translate(x + w / 2, y + h / 2)
        c.scale(z, z)
        c.translate(-(x + w / 2), -(y + h / 2))
        _shadow(c, x, y, w, h, r, 0.22 * a, blur=20, dy=10)
        panel = G.rrect(x, y, w, h, r)
        c.drawRRect(panel, G.paint(self.v("card", "bg"), self.v("card", "bg_a") * a))
        title = _flat(ov.d.get("title", ""))
        if title:
            c.save()
            c.clipRRect(panel, doAntiAlias=True)
            c.drawRect(skia.Rect.MakeXYWH(x, y, w, hh), G.paint(self.v("card", "head"), a))
            c.restore()
            f = self.font("card", size * 0.9)
            _text_mid(c, title, x + w / 2, y + hh / 2, f, G.paint(self.v("card", "head_fill"), a), "center")
            _sparkle(c, x + w / 2 - _w(title, f) / 2 - size * 0.55, y + hh / 2, size * 0.3, CORAL, a)
        c.drawRRect(G.rrect(x + 2.5, y + 2.5, w - 5, h - 5, r - 2.5), G.paint(PEACH, 0.9 * a, stroke=5))
        yy = y + hh + size * 0.55
        if ov.d.get("big"):
            fb = G.font(size * 1.8, 700, self.v("card", "font"))
            _text_mid(c, _flat(str(ov.d["big"])), x + w / 2, yy + size * 0.65, fb, G.paint(CORAL, a), "center")
            yy += size * 1.6
        f = self.font("card", size * 0.82)
        for j, item in enumerate(ov.d.get("items", [])):
            ka = G.clamp01((t - ov.t0 - 0.25 - 0.18 * j) / 0.2) * a
            cy = yy + size * 0.6 + j * size * 1.55
            c.drawCircle(x + size * 0.85, cy, size * 0.17, G.paint(DOTS[j % len(DOTS)], ka))
            plain, flags = spans(str(item).replace(JOIN, " "))
            self._marked(c, plain, flags, x + size * 1.35, cy + G.cap_height(f) / 2, f, ka, 1.0, ink, "left")
        c.restore()

    def draw_emphasis(self, c, ov, t: float) -> None:
        k = G.back((t - ov.t0) / 0.32, 2.0)
        a = G.clamp01((t - ov.t0) / 0.08) * G.ease_io((ov.t1 - t) / 0.2)
        if a <= 0:
            return
        (fx, fy), scale, see = self.placed(ov)
        a *= see
        size0 = self.v("emphasis", "size") * scale
        f, lines = self._fit(ov.text, "emphasis", self.x1 - self.x0 - size0 * 0.6, size0)
        size, ink = f.getSize(), self.v("emphasis", "fill")
        grow = G.ease_out((t - ov.t0 - 0.04) / 0.3)
        c.save()
        c.translate(W * fx, H * fy)
        z = 0.7 + 0.3 * k
        c.scale(z, z)
        c.rotate(-2.5 * (1 - G.clamp01((t - ov.t0) / 0.35)))
        widest = max(_w(s, f) for s, _ in lines)
        for j, (s, _) in enumerate(lines):
            base = (j - (len(lines) - 1) / 2) * size * 1.15 + G.cap_height(f) / 2
            _text(c, s, 0, base + 6, f, G.paint(WARM, 0.35 * a, stroke=size * 0.2, blur=7, cap_round=True), "center")
            _text(c, s, 0, base, f, G.paint(WHITE, a, stroke=size * 0.16, cap_round=True), "center")
            w = _w(s, f)
            _marker(c, -w / 2 - size * 0.26, base - size * 0.5, w + size * 0.52, size * 0.62, self.marker, 0.95 * a,
                    grow)
            _text(c, s, 0, base, f, G.paint(ink, a), "center")
        q = math.sin(math.pi * G.clamp01((t - ov.t0) / 0.7))
        span_w, span_h = widest + size * 0.9, size * 1.15 * len(lines) + size * 0.5
        for (dx, dy), r, rgb in (((-0.5, -0.5), 0.3, SPARK_RGB[1]), ((0.5, -0.55), 0.24, SPARK_RGB[0]),
                                 ((0.5, 0.5), 0.2, SPARK_RGB[2])):
            _sparkle(c, dx * span_w, dy * span_h, size * r * (0.6 + 0.6 * q), rgb, a * (0.4 + 0.6 * q))
        c.restore()

    # -------------------------------------------------------------- footage + paper
    def draw_card_mode(self, c, frame, box, k: float, face) -> None:
        """Cards over 9:16 close-ups: the picture goes soft and creamy, the speaker sits in a round sticker."""
        if k <= 0:
            return
        ox, oy, ow, oh = box
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(22 * k, 22 * k, skia.TileMode.kClamp))
        p.setAlphaf(G.clamp01(k * 1.2))
        c.drawImageRect(frame, skia.Rect.MakeXYWH(ox, oy, ow, oh), G.SAMPLING, p)
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint(CREAM, 0.55 * k))
        fx, fy, fh = face if face else (ox + ow / 2, oy + oh * 0.35, oh * 0.16)
        bx, by, br = W * self.BUBBLE[0], H * self.BUBBLE[1], W * self.BUBBLE[2]
        rr = br * G.back(k, 1.6)
        side = fh * 2.3
        sx, sy = (fx - ox) * frame.width() / ow, (fy - oy) * frame.height() / oh
        sside = side * frame.width() / ow
        c.save()
        _shadow(c, bx - rr, by - rr, 2 * rr, 2 * rr, rr, 0.3 * k, blur=18, dy=8)
        path = skia.Path()
        path.addCircle(bx, by, rr)
        c.clipPath(path, doAntiAlias=True)
        c.drawImageRect(frame, skia.Rect.MakeXYWH(sx - sside / 2, sy - sside / 2, sside, sside),
                        skia.Rect.MakeXYWH(bx - rr, by - rr, 2 * rr, 2 * rr), G.SAMPLING, skia.Paint(AntiAlias=True))
        c.restore()
        c.drawCircle(bx, by, rr, G.paint(WHITE, k, stroke=16))
        c.drawCircle(bx, by, rr, G.paint(CORAL, k, stroke=7))
        _sparkle(c, bx + rr * 0.85, by - rr * 0.8, rr * 0.2 * k, SPARK_RGB[0], k)

    def draw_band_bg(self, c, frame, box) -> None:
        """Behind footage that does not fill the canvas: pastel paper (or a creamy blur) and a white frame."""
        ox, oy, ow, oh = box
        if self.S.get("band_bg") == "blur":  # blur a 1/12-size copy and scale it up: same look, far cheaper per frame
            sw, sh = max(8, round(ow / 12)), max(8, round(oh / 12))
            small, sc = G.new_surface(sw, sh)
            p = skia.Paint(AntiAlias=True)
            p.setImageFilter(skia.ImageFilters.Blur(2.5, 2.5, skia.TileMode.kClamp))
            sc.drawImageRect(frame, skia.Rect.MakeWH(sw, sh), G.SAMPLING, p)
            z = max(W / ow, H / oh) * 1.1
            c.drawImageRect(small.makeImageSnapshot(), skia.Rect.MakeWH(sw, sh),
                            skia.Rect.MakeXYWH(W / 2 - ow * z / 2, H / 2 - oh * z / 2, ow * z, oh * z), G.SAMPLING,
                            skia.Paint(AntiAlias=True))
            c.drawRect(skia.Rect.MakeWH(W, H), G.paint(CREAM, 0.6))
        else:
            self.draw_paper(c)
        pad = 14.0
        _shadow(c, ox - pad, oy - pad, ow + 2 * pad, oh + 2 * pad, 38, 0.24, blur=22, dy=12)
        c.drawRRect(G.rrect(ox - pad, oy - pad, ow + 2 * pad, oh + 2 * pad, 38), G.paint(WHITE))

    # -------------------------------------------------------------- full-screen cards
    def _heading(self, c, text: str, el: str, y: float, tt: float, dur: float, size: float, wave: bool = False) -> None:
        """Brown heading lines centred on y with highlighter wipes and an optional squiggle underneath."""
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.3)
        f, lines = self._fit(text, el, W * 0.88, size)
        gap = f.getSize() * 1.3
        grow = G.ease_out((tt - 0.25) / 0.35)
        pop = 0.9 + 0.1 * G.back(tt / 0.5)
        c.save()
        c.translate(W / 2, y)
        c.scale(pop, pop)
        for j, (s, fl) in enumerate(lines):
            base = (j - (len(lines) - 1) / 2) * gap + G.cap_height(f) / 2
            self._marked(c, s, fl, 0, base, f, a, grow, BROWN)
        if wave:
            half = max(_w(s, f) for s, _ in lines) / 2
            _squiggle(c, -half, half, (len(lines) / 2) * gap + f.getSize() * 0.1, f.getSize() * 0.07, CORAL, a, grow)
        c.restore()

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
        self.draw_paper(c, tt)
        k = G.ease_out(tt / 0.5)
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.3)
        channel = self.brand.get("channel")
        text = part.get("text") or channel or self.brand.get("name", "")
        y = H * 0.46
        if self.brand.get("logo"):
            y = self._logo(c, H * 0.17, a, (1 - k) * -20) + H * 0.12
        elif channel and part.get("text"):
            cw = self._plate_size(channel, "", 40, "title", None)[0]
            self._plate(c, channel, W / 2 - cw / 2, y - H * 0.09 - (1 - k) * 12, 40, a, "", "title", PEACH)
        if text:
            self._heading(c, text, "title", y, tt, dur, 96 if not TALL else 92, wave=True)

    def card_text(self, c, part, tt, dur, bg) -> None:
        self.draw_paper(c, tt)
        self._heading(c, part.get("text", ""), "emphasis", H / 2, tt, dur, 108 if not TALL else 104, wave=True)

    def card_eyecatch(self, c, part, tt, dur, bg) -> None:
        self.draw_paper(c, tt)
        k = G.ease_io5(tt / max(dur, 0.1))
        x = -W * 1.15 + 2.3 * W * k
        c.drawRRect(G.rrect(x, -H * 0.1, W * 1.15, H * 1.2, min(W, H) * 0.3), G.paint(PEACH))
        if self.brand.get("logo"):
            self._logo(c, H / 2, G.win(tt, 0.15, dur - 0.15, 0.2, 0.2), at=0.5)
        elif part.get("text"):
            self._heading(c, part["text"], "title", H / 2, tt, dur, 84)

    def card_end(self, c, part, tt, dur, bg) -> None:
        """Thanks line, then like / subscribe pop in and a cursor clicks them (drawn from scratch, no logos)."""
        self.draw_paper(c, tt)
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.25)
        thanks = part.get("text") or ("ご視聴ありがとうございました" if self.fmt == "youtube" else "フォローして続きをチェック")
        f, lines = self._fit(thanks, "title", W * 0.88, 72 if not TALL else 70)
        y = H * (0.26 if not TALL else 0.30)
        for j, (s, fl) in enumerate(lines):
            self._marked(c, s, fl, W / 2, y + j * f.getSize() * 1.3 + G.cap_height(f) / 2, f, a, 1.0, BROWN)
        _heart(c, W / 2 + _w(lines[-1][0], f) / 2 + f.getSize() * 0.9, y + (len(lines) - 1) * f.getSize() * 1.3,
               f.getSize() * 0.3, CORAL, a * G.back((tt - 0.4) / 0.4))
        self._like_subscribe(c, tt, a)

    def _like_subscribe(self, c, tt: float, a: float) -> None:
        """Two pastel pill buttons pop in; a cursor clicks like (heart fills) then subscribe (turns 登録済み)."""
        s = 1.2 if not TALL else 1.1
        bw = min(380 * s, (self.x1 - self.x0 - 40 * s) / 2)
        bh, gap, cy = 100 * s, 40 * s, H * 0.45
        xs = [W / 2 - bw - gap / 2, W / 2 + gap / 2]
        click = [0.9, 1.7]
        f = G.font(38 * s, 700, self.v("title", "font"))
        yt = self.fmt == "youtube"
        buttons = (  # (label, label after the click, colour)
            ("高評価" if yt else "いいね", "高評価" if yt else "いいね", CORAL),
            ("チャンネル登録" if yt else "フォロー", "登録済み" if yt else "フォロー中", SKY),
        )
        for i, (label, done, rgb) in enumerate(buttons):
            k = G.back((tt - 0.2 - 0.15 * i) / 0.35)
            on = tt >= click[i]
            pulse = 1 + 0.08 * math.exp(-max(0.0, tt - click[i]) * 9) if on else 1.0
            ink = WHITE if on else BROWN
            c.save()
            c.translate(xs[i] + bw / 2, cy)
            c.scale(k * pulse, k * pulse)
            _shadow(c, -bw / 2, -bh / 2, bw, bh, bh / 2, 0.18 * a, blur=12, dy=7)
            c.drawRRect(G.rrect(-bw / 2, -bh / 2, bw, bh, bh / 2), G.paint(rgb if on else WHITE, a))
            c.drawRRect(G.rrect(-bw / 2 + 2.5, -bh / 2 + 2.5, bw - 5, bh - 5, bh / 2 - 2.5), G.paint(rgb, a, stroke=5))
            label_w = _w(done if on else label, f)
            if i == 0:
                _heart(c, -label_w / 2 - 30 * s, 0, 15 * s, WHITE if on else rgb, a)
            _text_mid(c, done if on else label, 14 * s if i == 0 else 0, 0, f, G.paint(ink, a), "center")
            if on and i == 1:  # check mark before the label
                x0, ck = -label_w / 2 - 28 * s, 12 * s
                path = skia.Path()
                path.moveTo(x0 - ck, 0)
                path.lineTo(x0 - ck * 0.3, ck * 0.75)
                path.lineTo(x0 + ck, -ck * 0.8)
                c.drawPath(path, G.paint(WHITE, a, stroke=5 * s, cap_round=True))
            c.restore()
            q = (tt - click[i]) / 0.6  # sparkle burst on the click
            if 0 <= q < 1:
                for n in range(6):
                    ang = n * math.pi / 3 + 0.4 * i
                    d = bh * (0.9 + 0.7 * G.ease_out(q))
                    _sparkle(c, xs[i] + bw / 2 + math.cos(ang) * d * 1.5, cy + math.sin(ang) * d,
                             bh * 0.24 * (1 - q * 0.7), SPARK_RGB[n % 3], a * (1 - q))
        pts = [(W * 0.8, H * 0.85, 0.3), (xs[0] + bw * 0.6, cy + bh * 0.2, click[0]),
               (xs[1] + bw * 0.6, cy + bh * 0.2, click[1]), (W * 0.85, H * 0.9, click[1] + 0.8)]
        for (x0, y0, t0), (x1, y1, t1) in zip(pts, pts[1:]):
            if t0 <= tt < t1:
                k = G.ease_io((tt - t0) / (t1 - t0))
                self._cursor(c, G.lerp(x0, x1, k), G.lerp(y0, y1, k), a, tt, click)
                break

    # -------------------------------------------------------------- sound
    def sfx(self, ov) -> list[tuple[float, str, float]]:
        m = {"chapter": [(0.0, "pop", -6.0)], "card": [(0.0, "pop", -4.0)],
             "emphasis": [(0.0, "pop", -3.0), (0.1, "sparkle", -2.0)], "name": [(0.0, "pop", -9.0)],
             "caption": [(0.0, "pop", -11.0)]}
        return [(ov.t0 + dt, name, g) for dt, name, g in m.get(ov.type, [])]

    def card_sfx(self, part: dict, t0: float, dur: float) -> list[tuple[float, str, float]]:
        if part["card"] in ("title", "eyecatch"):
            return [(t0, "pop", -5.0), (t0 + 0.2, "sparkle", -3.0)]
        return super().card_sfx(part, t0, dur)


LOOK = Friendly
