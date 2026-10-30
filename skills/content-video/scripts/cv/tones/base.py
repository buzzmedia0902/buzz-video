"""Look: everything a tone (トンマナ) decides. Tones subclass it and mostly change the class attributes:

  CUT      jet cut: pause (s) that gets shortened, tail / lead kept around words, speed of talk parts
  CAMERA   hold (s) before wide <-> medium, med zoom (1.0 = never), punch_every / punch_zoom / punch_dur
  AUDIO    bgm style, music_db (BGM under speech, relative to the voice), sfx set name, sfx gain
  GRADE    ffmpeg filter applied while decoding (eq=...), "" for none
  SUBS     min_show (s), tidy (trim fillers in the text), max_lines
  STYLE    fonts / sizes / colours of every element, per layout ("wide" = 16:9 / 1:1, "tall" = 9:16 / 4:5)

Every draw_* method gets the skia canvas and the output time t; the base class draws a clean, readable default for
every element, so a tone only overrides what makes it different. Positions are fractions of the canvas, kept
inside the text-safe area (cv.canvas.safe_rect).
"""
from __future__ import annotations

import copy
import functools
import math
import re

import skia

from cv import gfx as G
from cv import linebreak
from cv.canvas import ASPECT, FPS, H, TALL, W, safe_rect  # noqa: F401
from cv.cues import JOIN

LAYOUT = "tall" if TALL else "wide"
SQUARE = W == H  # 1:1 takes a "square" value when a style gives one, else the "wide" one


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def spans(text: str) -> tuple[str, list[bool]]:
    """Plain text + per-char emphasis flags from {..} markup."""
    out, flags, on = [], [], False
    for ch in text:
        if ch == "{":
            on = True
        elif ch == "}":
            on = False
        else:
            out.append(ch)
            flags.append(on)
    return "".join(out), flags


def _inside_latin(s: str, i: int) -> bool:
    left, right = s[:i].rstrip(), s[i:].lstrip()
    return bool(left) and bool(right) and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() \
        and right[0].isalnum()


class Look:
    key = "base"
    CUT = dict(pause=0.32, tail=0.11, lead=0.08, speed=1.0)
    CAMERA = dict(hold=10.0, med=1.14, punch_every=40.0, punch_zoom=1.08, punch_dur=0.5)
    AUDIO = dict(bgm="lofi", music_db=-14.0, sfx="soft", sfx_db=-2.0)
    GRADE = "eq=contrast=1.03:saturation=1.04"
    SUBS = dict(min_show=1.2, tidy=True, max_lines=2)
    STYLE = {
        "accent": (254, 219, 46),
        "ink": (24, 24, 27),
        "paper": (250, 250, 248),
        "sub": {"font": "gothic", "weight": 800, "size": {"wide": 62, "tall": 76}, "fill": (255, 255, 255),
                "stroke": (13, 13, 13), "stroke_w": 0.18, "outer": None, "outer_w": 0.0, "shadow": 0.45,
                "emph": "accent", "box": None, "y": {"wide": 0.935, "tall": 0.70}, "gap": 1.28, "max_w": 0.92,
                "speaker": "outer"},  # speaker colour as: outer | pill | none
        "title": {"font": "gothic", "weight": 900, "size": {"wide": 56, "tall": 84, "square": 60},
                  "fill": (255, 255, 255), "band": (13, 13, 13), "band_a": 0.88, "emph": "accent", "stroke_w": 0.0,
                  "y": {"wide": 0.08, "tall": 0.155, "square": 0.11}},
        "chapter": {"font": "gothic", "weight": 800, "size": {"wide": 40, "tall": 46}, "fill": (13, 13, 13),
                    "bg": "accent", "y": {"wide": 0.10, "tall": 0.265, "square": 0.24}},
        "card": {"font": "gothic", "weight": 800, "size": {"wide": 44, "tall": 60}, "fill": (24, 24, 27),
                 "bg": (255, 255, 255), "bg_a": 0.95, "head": (13, 13, 13), "head_fill": (255, 255, 255),
                 "w": {"wide": 0.40, "tall": 0.86}},
        "emphasis": {"font": "midashi_go", "weight": 800, "size": {"wide": 150, "tall": 150}, "fill": "accent",
                     "stroke": (13, 13, 13), "stroke_w": 0.12, "y": {"wide": 0.42, "tall": 0.44}},
        "name": {"font": "gothic", "weight": 800, "size": {"wide": 46, "tall": 50}, "fill": (255, 255, 255),
                 "bar": "accent", "y": {"wide": 0.74, "tall": 0.56}},
        "caption": {"font": "gothic", "weight": 700, "size": {"wide": 40, "tall": 46}, "fill": (255, 255, 255),
                    "y": {"wide": 0.11, "tall": 0.14}},
        "credit": {"font": "gothic", "weight": 600, "size": {"wide": 24, "tall": 28}, "fill": (235, 235, 235),
                   "y": {"wide": 0.065, "tall": 0.262}},  # 9:16: between the title band and a 16:9 band
        "band_bg": "blur",       # behind 16:9 footage in a 9:16 clip: blur | (r, g, b)
        # centre of 16:9 footage in a 9:16 frame: under the title band, above two-line subtitles
        "band_y": {"short": 0.43, "clip": 0.43, "vlog": 0.50, "youtube": 0.50},
    }

    CARD_MODE_CENTER = (0.5, 0.50)                  # 9:16 card mode: card here, face bubble at BUBBLE
    BUBBLE = (0.80, 0.31, 0.12)                      # (cx, cy fractions, radius as a fraction of the width)

    def __init__(self, brand: dict | None = None, look: dict | None = None, speakers: dict | None = None,
                 fmt: str = "youtube"):
        self.brand, self.speakers, self.fmt = brand or {}, speakers or {}, fmt
        self.S = _merge(copy.deepcopy(self.STYLE), (look or {}).get("style", {}))
        for k in ("CUT", "CAMERA", "AUDIO", "SUBS"):
            setattr(self, k, {**getattr(type(self), k), **(look or {}).get(k.lower(), {})})
        if (look or {}).get("grade") is not None:
            self.GRADE = look["grade"]
        if self.brand.get("accent"):
            self.S["accent"] = G.hexrgb(self.brand["accent"])
        for role, fam in (look or {}).get("fonts", {}).items():
            if role in self.S and isinstance(self.S[role], dict):
                self.S[role]["font"] = fam
        self.x0, self.y0, self.x1, self.y1 = safe_rect()

    # -------------------------------------------------------------- style helpers
    def v(self, el: str, key: str, default=None):
        """Style value of element `el`, resolving per-layout dicts and colour names ("accent", "ink", "paper")."""
        val = self.S.get(el, {}).get(key, default)
        if isinstance(val, dict) and LAYOUT in val:
            val = val["square"] if SQUARE and "square" in val else val[LAYOUT]
        if isinstance(val, str) and val in ("accent", "ink", "paper"):
            val = self.S[val]
        elif isinstance(val, str) and val.startswith("#"):
            val = G.hexrgb(val)
        return val

    def font(self, el: str, size: float | None = None) -> skia.Font:
        return G.font(size or self.v(el, "size"), self.v(el, "weight", 700), self.v(el, "font", "gothic"))

    def band_y(self, fmt: str) -> float:
        return float(self.S["band_y"].get(fmt, 0.5))

    def speaker_rgb(self, name: str | None):
        if not name or name not in self.speakers:
            return None
        c = self.speakers[name].get("color")
        return G.hexrgb(c) if c else None

    # -------------------------------------------------------------- subtitles
    @functools.lru_cache(maxsize=4096)
    def sub_layout(self, text: str) -> tuple[float, tuple[tuple[str, tuple[bool, ...]], ...]]:
        """(font size, lines) for a subtitle: one line if it fits, else the best balanced 2-line break."""
        plain, flags = spans(text)
        size = float(self.v("sub", "size"))
        max_w = (self.x1 - self.x0) * self.v("sub", "max_w", 0.92)
        f = self.font("sub", size)
        if JOIN in plain:
            i = plain.index(JOIN)
            a, b = plain[:i], plain[i + 1:].replace(JOIN, "")
            fa, fb = flags[:i], [fl for ch, fl in zip(plain[i + 1:], flags[i + 1:]) if ch != JOIN]
            lines = ((a, tuple(fa)), (b, tuple(fb)))
        elif G.text_width(plain, f) <= max_w:
            lines = ((plain, tuple(flags)),)
        else:
            lines = self._break(plain, flags, f, max_w, size)
        widest = max(G.text_width(s, f) for s, _ in lines)
        if widest > max_w:  # still too wide: shrink
            size *= max_w / widest
        return size, lines

    def _break(self, plain: str, flags: list[bool], f: skia.Font, max_w: float, size: float):
        costs = linebreak.break_costs(plain)
        best, best_i = None, len(plain) // 2
        for i, c in costs.items():
            if _inside_latin(plain, i) or plain[i] == "ん" or plain[:i].count("「") > plain[:i].count("」"):
                continue
            a, b = plain[:i], plain[i:]
            wa, wb = G.text_width(a, f), G.text_width(b, f)
            over = max(0.0, max(wa, wb) - max_w) / size * 50
            sentence_end = plain[:i].rstrip()[-1:] in "。！？!?"
            score = c + abs(wa - wb) / size * (1.5 if sentence_end else 3.5) + over
            if best is None or score < best:
                best, best_i = score, i
        i = best_i
        a = plain[:i].rstrip("、 　")
        j = i
        while j < len(plain) and plain[j] in " 　":
            j += 1
        return (a, tuple(flags[: len(a)])), (plain[j:], tuple(flags[j:]))

    def sub_line(self, c: skia.Canvas, s: str, flags, cx: float, y: float, size: float, alpha: float,
                 speaker_rgb=None) -> None:
        """One subtitle line centred at cx on baseline y: shadow, outer edge, outline, then fills run by run."""
        f = self.font("sub", size)
        w = G.text_width(s, f)
        x = cx - w / 2
        sw = self.v("sub", "stroke_w", 0.18) * size
        outer = self.v("sub", "outer")
        if speaker_rgb is not None and self.v("sub", "speaker") == "outer":
            outer = speaker_rgb
        ow = self.v("sub", "outer_w", 0.0) * size if outer is not None else 0.0
        if outer is not None and ow <= 0:
            ow = 0.12 * size
        if self.v("sub", "shadow", 0) > 0:
            G.draw_text(c, s, x + 2, y + 5, f, G.paint(G.BLACK, self.v("sub", "shadow") * alpha, stroke=sw + ow + 4,
                                                       blur=5, cap_round=True))
        if outer is not None:
            G.draw_text(c, s, x, y, f, G.paint(outer, alpha, stroke=sw + ow, cap_round=True))
        if sw > 0:
            G.draw_text(c, s, x, y, f, G.paint(self.v("sub", "stroke"), alpha, stroke=sw, cap_round=True))
        fill, emph = self.v("sub", "fill"), self.v("sub", "emph")
        xx, start = x, 0
        for i in range(1, len(s) + 1):
            if i == len(s) or flags[i] != flags[start]:
                run = s[start:i]
                G.draw_text(c, run, xx, y, f, G.paint(emph if flags[start] else fill, alpha))
                xx += f.measureText(run)
                start = i

    def draw_sub(self, c: skia.Canvas, sub, t: float) -> None:
        size, lines = self.sub_layout(sub.text)
        k = G.ease_out((t - sub.t0) / 0.12)
        alpha = G.clamp01(k * 1.5) * G.clamp01((sub.t1 - t) / 0.06 + 0.2)
        gap = size * self.v("sub", "gap", 1.28)
        y_last = H * self.v("sub", "y")
        y_first = y_last - (len(lines) - 1) * gap
        if self.v("sub", "box") is not None:
            self.sub_box(c, lines, size, y_first, gap, alpha)
        spk = self.speaker_rgb(sub.speaker)
        if spk is not None and self.v("sub", "speaker") == "pill":
            self.speaker_pill(c, sub.speaker, spk, y_first - size * 1.05, alpha)
        for j, (s, fl) in enumerate(lines):
            self.sub_line(c, s, fl, W / 2, y_first + j * gap + (1 - k) * 8, size, alpha, spk)

    def sub_box(self, c, lines, size, y_first, gap, alpha) -> None:
        f = self.font("sub", size)
        w = max(G.text_width(s, f) for s, _ in lines) + size * 1.0
        top = y_first - size * 0.95
        h = (len(lines) - 1) * gap + size * 1.35
        c.drawRRect(G.rrect(W / 2 - w / 2, top, w, h, size * 0.3), G.paint(self.v("sub", "box"), 0.62 * alpha))

    def speaker_pill(self, c, name: str, rgb, y: float, alpha: float) -> None:
        f = G.font(self.v("sub", "size") * 0.42, 800, self.v("sub", "font"))
        w = G.text_width(name, f) + 28
        x = W / 2 - w / 2
        c.drawRRect(G.rrect(x, y - 30, w, 40, 20), G.paint(rgb, 0.95 * alpha))
        G.draw_text_mid(c, name, W / 2, y - 10, f, G.paint((255, 255, 255), alpha), "center")

    # -------------------------------------------------------------- persistent elements
    def draw_title_band(self, c: skia.Canvas, ov, t: float) -> None:
        """Title of a short / clip at the top (stays on)."""
        a = G.clamp01((t - ov.t0) / 0.25) * G.clamp01((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        size = self.v("title", "size")
        plain, flags = spans(ov.text.replace(JOIN, "\n"))
        lines = plain.split("\n")
        f = self.font("title", size)
        max_w = (self.x1 - self.x0)
        widest = max(G.text_width(s, f) for s in lines)
        if widest > max_w:
            size *= max_w / widest
            f = self.font("title", size)
        gap = size * 1.22
        cy = H * self.v("title", "y")
        top = cy - gap * len(lines) / 2
        band = self.v("title", "band")
        if band is not None:
            pad = size * 0.45
            c.drawRect(skia.Rect.MakeLTRB(0, top - pad, W, top + gap * len(lines) + pad * 0.6),
                       G.paint(band, self.v("title", "band_a", 0.88) * a))
        k = 0
        for j, s in enumerate(lines):
            fl = flags[k:k + len(s)]
            k += len(s) + 1
            y = top + gap * (j + 0.82)
            self._runs(c, s, fl, W / 2, y, f, self.v("title", "fill"), self.v("title", "emph"), a,
                       self.v("title", "stroke_w", 0.0) * size)

    def _runs(self, c, s, flags, cx, y, f, fill, emph, alpha, stroke_w=0.0, stroke=(13, 13, 13)):
        w = G.text_width(s, f)
        x = cx - w / 2
        if stroke_w > 0:
            G.draw_text(c, s, x, y, f, G.paint(stroke, alpha, stroke=stroke_w, cap_round=True))
        xx, start = x, 0
        for i in range(1, len(s) + 1):
            if i == len(s) or flags[i] != flags[start]:
                run = s[start:i]
                G.draw_text(c, run, xx, y, f, G.paint(emph if flags[start] else fill, alpha))
                xx += f.measureText(run)
                start = i

    def draw_credit(self, c: skia.Canvas, ov, t: float) -> None:
        f = self.font("credit")
        G.outlined_text(c, ov.text, self.x1, H * self.v("credit", "y"), f, self.v("credit", "fill"), G.BLACK,
                        self.v("credit", "size") * 0.12, "right", alpha=0.9)

    # -------------------------------------------------------------- timed elements
    def draw_chapter(self, c: skia.Canvas, ov, t: float) -> None:
        k_in = G.ease_out((t - ov.t0) / 0.35)
        k_out = G.ease_io((ov.t1 - t) / 0.3)
        a = min(G.clamp01(k_in * 1.4), k_out)
        if a <= 0:
            return
        f = self.font("chapter")
        size = self.v("chapter", "size")
        w = G.text_width(ov.text, f) + size * 1.4
        h = size * 1.7
        x = self.x0 - (1 - k_in) * (w + 40)
        y = H * self.v("chapter", "y")
        c.drawRRect(G.rrect(x, y, w, h, h * 0.18), G.paint(self.v("chapter", "bg"), 0.96 * a))
        G.draw_text_mid(c, ov.text, x + size * 0.7, y + h / 2, f, G.paint(self.v("chapter", "fill"), a))

    # -------------------------------------------------------------- placement (cv.layout keeps faces clear)
    def spots(self, ov) -> list[tuple[float, float]]:
        """Candidate centres (fractions of the canvas) for a card / name / emphasis, best first."""
        if ov.type == "card":
            return ([(0.73, 0.40), (0.27, 0.40), (0.73, 0.27), (0.27, 0.27), (0.5, 0.30)] if LAYOUT == "wide"
                    else [(0.5, 0.40), (0.5, 0.54), (0.5, 0.30), (0.5, 0.58)])
        if ov.type == "name":
            return ([(0.0, 0.74), (0.0, 0.20)] if LAYOUT == "wide" else [(0.0, 0.56), (0.0, 0.22), (0.0, 0.46)])
        if ov.type == "emphasis":
            return [(0.5, self.v("emphasis", "y")), (0.5, 0.56), (0.5, 0.25)]
        return [(0.5, 0.5)]

    def box(self, ov, center=None, scale: float = 1.0) -> tuple[float, float, float, float]:
        """Rectangle (x, y, w, h) the element covers at `center` (fractions) and `scale`."""
        cx, cy = center or self.spots(ov)[0]
        if ov.type == "card":
            items = ov.d.get("items", [])
            size = self.v("card", "size") * scale
            w = min(W * self.v("card", "w"), self.x1 - self.x0) * scale
            h = size * (1.9 + 1.55 * len(items)) + (size * 1.4 if ov.d.get("big") else 0)
            x = min(max(W * cx - w / 2, self.x0), self.x1 - w)
            y = min(max(H * cy - h / 2, self.y0), self.y1 - h)
            return x, y, w, h
        if ov.type == "name":
            size = self.v("name", "size") * scale
            f = self.font("name", size)
            w = G.text_width(ov.text, f) + size * 0.6
            return self.x0, H * cy - size * 1.05, w, size * 1.8
        if ov.type == "emphasis":
            size = self.v("emphasis", "size") * scale
            w = min(G.text_width(ov.text, self.font("emphasis", size)), self.x1 - self.x0)
            return W * cx - w / 2, H * cy - size * 0.6, w, size * 1.2
        if ov.type == "title_band":
            n = len(spans(ov.text.replace(JOIN, "\n"))[0].split("\n"))
            size = self.v("title", "size")
            top = H * self.v("title", "y") - size * 1.22 * n / 2 - size * 0.45
            return 0.0, top, float(W), size * 1.22 * n + size * 0.72
        return 0.0, 0.0, 0.0, 0.0

    def sub_zone(self) -> tuple[float, float, float, float]:
        """Where two-line subtitles sit (kept free of cards)."""
        size = self.v("sub", "size")
        y_last = H * self.v("sub", "y")
        top = y_last - size * self.v("sub", "gap", 1.28) - size * 0.95
        return self.x0, top, self.x1 - self.x0, y_last + size * 0.35 - top

    def placed(self, ov) -> tuple[tuple[float, float], float, float]:
        """(centre, scale, alpha) chosen by cv.layout (default spot otherwise)."""
        p = ov.d.get("_place")
        return (p["center"], p["scale"], p["alpha"]) if p else (self.spots(ov)[0], 1.0, 1.0)

    def card_rect(self, ov) -> tuple[float, float, float, float]:
        center, scale, _ = self.placed(ov)
        return self.box(ov, center, scale)

    def draw_card(self, c: skia.Canvas, ov, t: float) -> None:
        k = G.back((t - ov.t0) / 0.35)
        a = G.clamp01((t - ov.t0) / 0.15) * G.ease_io((ov.t1 - t) / 0.25)
        if a <= 0:
            return
        x, y, w, h = self.card_rect(ov)
        _, scale, see = self.placed(ov)
        a *= see
        size = self.v("card", "size") * scale
        c.save()
        c.translate(x + w / 2, y + h / 2)
        c.scale(0.85 + 0.15 * k, 0.85 + 0.15 * k)
        c.translate(-(x + w / 2), -(y + h / 2))
        G.shadow(c, x, y, w, h, size * 0.4, a=0.25 * a)
        c.drawRRect(G.rrect(x, y, w, h, size * 0.4), G.paint(self.v("card", "bg"), self.v("card", "bg_a") * a))
        hh = size * 1.6
        title = ov.d.get("title", "")
        if title:
            c.save()
            c.clipRRect(G.rrect(x, y, w, h, size * 0.4), doAntiAlias=True)
            c.drawRect(skia.Rect.MakeXYWH(x, y, w, hh), G.paint(self.v("card", "head"), a))
            c.restore()
            f = self.font("card", size * 0.9)
            G.draw_text_mid(c, title, x + w / 2, y + hh / 2, f, G.paint(self.v("card", "head_fill"), a), "center")
        yy = y + hh + size * 0.55
        if ov.d.get("big"):
            fb = G.font(size * 2.0, 900, self.v("card", "font"))
            G.draw_text_mid(c, str(ov.d["big"]), x + w / 2, yy + size * 0.7, fb, G.paint(self.S["accent"], a), "center")
            yy += size * 1.6
        f = self.font("card", size * 0.82)
        for j, item in enumerate(ov.d.get("items", [])):
            ka = G.clamp01((t - ov.t0 - 0.25 - 0.18 * j) / 0.2) * a
            cy = yy + size * 0.6 + j * size * 1.55
            c.drawCircle(x + size * 0.85, cy, size * 0.16, G.paint(self.S["accent"], ka))
            G.draw_text_mid(c, str(item), x + size * 1.35, cy, f, G.paint(self.v("card", "fill"), ka))
        c.restore()

    def draw_emphasis(self, c: skia.Canvas, ov, t: float) -> None:
        k = G.back((t - ov.t0) / 0.28, 2.2)
        a = G.clamp01((t - ov.t0) / 0.08) * G.ease_io((ov.t1 - t) / 0.2)
        if a <= 0:
            return
        (fx, fy), scale, see = self.placed(ov)
        a *= see
        size = self.v("emphasis", "size") * scale
        f = self.font("emphasis", size)
        w = G.text_width(ov.text, f)
        if w > (self.x1 - self.x0):
            size *= (self.x1 - self.x0) / w
            f = self.font("emphasis", size)
        cx, cy = W * fx, H * fy
        c.save()
        c.translate(cx, cy)
        z = 0.6 + 0.4 * k
        c.scale(z, z)
        c.rotate(-3.0 * (1 - G.clamp01((t - ov.t0) / 0.3)))
        G.outlined_text(c, ov.text, 0, G.cap_height(f) / 2, f, self.v("emphasis", "fill"), self.v("emphasis", "stroke"),
                        self.v("emphasis", "stroke_w") * size, "center", alpha=a, shadow_a=0.4)
        c.restore()

    def draw_name(self, c: skia.Canvas, ov, t: float) -> None:
        k = G.ease_out((t - ov.t0) / 0.4)
        a = G.clamp01(k * 1.3) * G.ease_io((ov.t1 - t) / 0.3)
        if a <= 0:
            return
        (_, fy), scale, see = self.placed(ov)
        a *= see
        size = self.v("name", "size") * scale
        f, fs = self.font("name", size), self.font("name", size * 0.55)
        x = self.x0 + (1 - k) * -30
        y = H * fy
        c.drawRect(skia.Rect.MakeXYWH(x, y - size * 1.05, size * 0.14, size * 1.75), G.paint(self.v("name", "bar"), a))
        G.outlined_text(c, ov.text, x + size * 0.4, y, f, self.v("name", "fill"), G.BLACK, size * 0.14, alpha=a)
        if ov.d.get("sub"):
            G.outlined_text(c, ov.d["sub"], x + size * 0.42, y + size * 0.72, fs, self.v("name", "fill"), G.BLACK,
                            size * 0.1, alpha=a)

    def draw_caption(self, c: skia.Canvas, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.35) * G.ease_io((ov.t1 - t) / 0.35)
        if a <= 0:
            return
        size = self.v("caption", "size")
        f, fs = self.font("caption", size), self.font("caption", size * 0.6)
        x, y = self.x0, H * self.v("caption", "y")
        G.outlined_text(c, ov.text, x, y, f, self.v("caption", "fill"), G.BLACK, 0, alpha=a, shadow_a=0.6)
        if ov.d.get("sub"):
            G.outlined_text(c, ov.d["sub"], x, y + size * 0.95, fs, self.v("caption", "fill"), G.BLACK, 0, alpha=a,
                            shadow_a=0.6)

    def draw_note(self, c: skia.Canvas, ov, t: float) -> None:
        a = G.clamp01((t - ov.t0) / 0.2) * G.ease_io((ov.t1 - t) / 0.2)
        if a <= 0:
            return
        f = G.font(self.v("credit", "size") * 1.2, 700, self.v("sub", "font"))
        G.outlined_text(c, "※" + ov.text.lstrip("※"), self.x0, H * self.v("sub", "y") - self.v("sub", "size") * 2.9,
                        f, (255, 255, 255), G.BLACK, 5, alpha=a)

    def draw_overlay(self, c: skia.Canvas, ov, t: float) -> None:
        getattr(self, f"draw_{ov.type}")(c, ov, t)

    # -------------------------------------------------------------- 9:16 card mode
    def draw_card_mode(self, c: skia.Canvas, frame: skia.Image, box, k: float, face) -> None:
        """While a card is up in a tall layout: blur + dim the picture, show the speaker's face in a round bubble
        (pops in with k 0..1). face = (x, y, h) in canvas px or None."""
        if k <= 0:
            return
        ox, oy, ow, oh = box
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(18 * k, 18 * k, skia.TileMode.kClamp))
        p.setAlphaf(G.clamp01(k * 1.2))
        c.drawImageRect(frame, skia.Rect.MakeXYWH(ox, oy, ow, oh), G.SAMPLING, p)
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint(G.BLACK, 0.45 * k))
        fx, fy, fh = face if face else (ox + ow / 2, oy + oh * 0.35, oh * 0.16)
        bx, by, br = W * self.BUBBLE[0], H * self.BUBBLE[1], W * self.BUBBLE[2]
        rr = br * G.back(k, 1.6)
        side = fh * 2.3
        sx, sy = (fx - ox) * frame.width() / ow, (fy - oy) * frame.height() / oh
        sside = side * frame.width() / ow
        src = skia.Rect.MakeXYWH(sx - sside / 2, sy - sside / 2, sside, sside)
        c.save()
        G.shadow(c, bx - rr, by - rr, 2 * rr, 2 * rr, rr, blur=18, dy=8, a=0.45 * k)
        path = skia.Path()
        path.addCircle(bx, by, rr)
        c.clipPath(path, doAntiAlias=True)
        c.drawImageRect(frame, src, skia.Rect.MakeXYWH(bx - rr, by - rr, 2 * rr, 2 * rr), G.SAMPLING,
                        skia.Paint(AntiAlias=True))
        c.restore()
        c.drawCircle(bx, by, rr, G.paint(G.hexrgb(self.S.get("ring", self.S["accent"])), k, stroke=8))

    # -------------------------------------------------------------- band layout background
    _SMALL: dict = {}

    def draw_band_bg(self, c: skia.Canvas, frame: skia.Image, box) -> None:
        """Behind footage that does not fill the canvas (16:9 in 9:16): blurred, darkened copy or a colour.
        The blur runs on a 1/12-size copy (a full-size blur cost ~95 ms a frame) and is scaled up."""
        bg = self.S.get("band_bg", "blur")
        if bg != "blur":
            c.clear(G.col(G.hexrgb(bg)))
            return
        ox, oy, ow, oh = box
        z = max(W / ow, H / oh) * 1.1
        sw, sh = max(8, int(ow / 12)), max(8, int(oh / 12))
        surf = self._SMALL.get((sw, sh))
        if surf is None:
            surf = self._SMALL[(sw, sh)] = skia.Surface(sw, sh)
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(2.4, 2.4, skia.TileMode.kClamp))
        p.setColorFilter(skia.ColorFilters.Matrix([0.55, 0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 1, 0]))
        surf.getCanvas().drawImageRect(frame, skia.Rect.MakeWH(sw, sh), G.SAMPLING, p)
        small = surf.makeImageSnapshot()
        c.drawImageRect(small, skia.Rect.MakeXYWH(W / 2 - ow * z / 2, H / 2 - oh * z / 2, ow * z, oh * z), G.SAMPLING)

    # -------------------------------------------------------------- full-screen cards
    def draw_card_part(self, c: skia.Canvas, part: dict, tt: float, dur: float, bg: skia.Image | None) -> None:
        """Title / end / eyecatch / text card; tt = seconds since the card started."""
        getattr(self, f"card_{part['card']}")(c, part, tt, dur, bg)

    def _card_bg(self, c, bg, dim: float = 0.55) -> None:
        if bg is None:
            c.clear(G.col(self.S["ink"]))
            return
        p = skia.Paint(AntiAlias=True)
        p.setImageFilter(skia.ImageFilters.Blur(18, 18, skia.TileMode.kClamp))
        c.drawImageRect(bg, skia.Rect.MakeWH(W, H), G.SAMPLING, p)
        c.drawRect(skia.Rect.MakeWH(W, H), G.paint(G.BLACK, dim))

    def card_title(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg)
        k = G.ease_out(tt / 0.5)
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.3)
        y = H * 0.46
        logo = self.brand.get("logo")
        if logo:
            img = G.image(logo)
            lw = min(W * 0.42, H * 0.5 * img.width() / img.height())
            lh = lw * img.height() / img.width()
            G.draw_image(c, img, W / 2 - lw / 2, y - lh / 2 - (1 - k) * 20, lw, lh, alpha=a)
            y += lh / 2 + 70
        text = part.get("text") or self.brand.get("channel") or self.brand.get("name", "")
        if text:
            f = G.font(64 if not TALL else 72, 900, self.v("title", "font"))
            G.draw_text_mid(c, text, W / 2, y, f, G.paint((255, 255, 255), a), "center")

    def card_text(self, c, part, tt, dur, bg) -> None:
        self._card_bg(c, bg, 0.65)
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.3)
        plain, flags = spans(part.get("text", ""))
        f = G.font(96 if not TALL else 104, 900, self.v("emphasis", "font"))
        lines = plain.split("\n")
        k = 0
        for j, s in enumerate(lines):
            fl = flags[k:k + len(s)]
            k += len(s) + 1
            y = H / 2 + (j - (len(lines) - 1) / 2) * f.getSize() * 1.3 + G.cap_height(f) / 2
            self._runs(c, s, fl, W / 2, y, f, (255, 255, 255), self.S["accent"], a)

    def card_eyecatch(self, c, part, tt, dur, bg) -> None:
        c.clear(G.col(self.S["accent"]))
        k = G.ease_io5(tt / max(dur, 0.1))
        x = -W + 2 * W * k
        c.drawRect(skia.Rect.MakeXYWH(x, 0, W, H), G.paint(self.S["ink"]))
        logo = self.brand.get("logo")
        a = G.win(tt, 0.15, dur - 0.15, 0.2, 0.2)
        if logo:
            img = G.image(logo)
            lw = W * 0.3
            lh = lw * img.height() / img.width()
            G.draw_image(c, img, W / 2 - lw / 2, H / 2 - lh / 2, lw, lh, alpha=a)
        elif part.get("text"):
            G.draw_text_mid(c, part["text"], W / 2, H / 2, G.font(90, 900), G.paint((255, 255, 255), a), "center")

    def card_end(self, c, part, tt, dur, bg) -> None:
        """End card. YouTube: animated like / subscribe (drawn from scratch, no YouTube logo) with a cursor click,
        lower third left free for end-screen elements. Short / clip / vlog: follow call-to-action."""
        self._card_bg(c, bg, 0.62)
        a = G.clamp01(tt / 0.3) * G.clamp01((dur - tt) / 0.25)
        thanks = part.get("text") or ("ご視聴ありがとうございました" if self.fmt == "youtube" else "フォローして続きをチェック")
        f = G.font(60 if not TALL else 66, 900, self.v("title", "font"))
        G.draw_text_mid(c, thanks, W / 2, H * (0.26 if not TALL else 0.30), f, G.paint((255, 255, 255), a), "center")
        self._like_subscribe(c, tt, a)

    def _like_subscribe(self, c, tt, a) -> None:
        """Two pill buttons pop in; a cursor clicks like (thumb fills) then subscribe (turns 登録済み)."""
        s = 1.0 if not TALL else 1.15
        bw, bh = 300 * s, 96 * s
        gap = 40 * s
        cy = H * (0.45 if not TALL else 0.45)
        xs = [W / 2 - bw - gap / 2, W / 2 + gap / 2]
        click = [0.9, 1.7]
        f = G.font(40 * s, 800)
        buttons = (  # (label, label after the click, fill before, fill after)
            ("高評価", "高評価", (255, 255, 255), (40, 40, 40)),
            ("チャンネル登録", "登録済み", (230, 33, 23), (90, 90, 90)),
        )
        for i, (label, done, fill0, fill1) in enumerate(buttons):
            k = G.back((tt - 0.2 - 0.15 * i) / 0.35)
            on = tt >= click[i]
            pulse = 1 + 0.08 * math.exp(-max(0.0, tt - click[i]) * 9) if on else 1.0
            fill = fill1 if on else fill0
            c.save()
            c.translate(xs[i] + bw / 2, cy)
            c.scale(k * pulse, k * pulse)
            c.drawRRect(G.rrect(-bw / 2, -bh / 2, bw, bh, bh / 2), G.paint(fill, a))
            txt_rgb = (20, 20, 20) if fill == (255, 255, 255) else (255, 255, 255)
            G.draw_text_mid(c, done if on else label, 0, 0, f, G.paint(txt_rgb, a), "center")
            if on and i == 0:  # check mark before the label
                tw = G.text_width(done, f)
                x0, ck = -tw / 2 - 34 * s, 14 * s
                path = skia.Path()
                path.moveTo(x0 - ck, 0)
                path.lineTo(x0 - ck * 0.3, ck * 0.75)
                path.lineTo(x0 + ck, -ck * 0.8)
                c.drawPath(path, G.paint(txt_rgb, a, stroke=5 * s, cap_round=True))
            c.restore()
        # cursor path: in from bottom right -> like -> subscribe -> away
        pts = [(W * 0.8, H * 0.85, 0.3), (xs[0] + bw * 0.6, cy + bh * 0.2, click[0]),
               (xs[1] + bw * 0.6, cy + bh * 0.2, click[1]), (W * 0.85, H * 0.9, click[1] + 0.8)]
        for (x0, y0, t0), (x1, y1, t1) in zip(pts, pts[1:]):
            if t0 <= tt < t1:
                k = G.ease_io((tt - t0) / (t1 - t0))
                self._cursor(c, G.lerp(x0, x1, k), G.lerp(y0, y1, k), a, tt, click)
                break

    def _cursor(self, c, x, y, a, tt, click) -> None:
        press = any(0 <= tt - t < 0.12 for t in click)
        z = 0.85 if press else 1.0
        path = skia.Path()
        pts = [(0, 0), (0, 46), (12, 35), (21, 55), (29, 51), (20, 32), (36, 32)]
        path.moveTo(x + pts[0][0] * z, y + pts[0][1] * z)
        for px, py in pts[1:]:
            path.lineTo(x + px * z, y + py * z)
        path.close()
        c.drawPath(path, G.paint((20, 20, 20), a, stroke=5, cap_round=True))
        c.drawPath(path, G.paint((255, 255, 255), a))

    # -------------------------------------------------------------- sound
    def sfx(self, ov) -> list[tuple[float, str, float]]:
        """Sound effects an overlay triggers: (output seconds, name, gain dB)."""
        m = {"chapter": [(0.0, "whoosh", -4.0)], "card": [(0.0, "pop", -3.0)], "emphasis": [(0.0, "impact", -2.0)],
             "name": [(0.0, "tick", -8.0)], "caption": [(0.0, "tick", -10.0)]}
        return [(ov.t0 + dt, name, g) for dt, name, g in m.get(ov.type, [])]

    def card_sfx(self, part: dict, t0: float, dur: float) -> list[tuple[float, str, float]]:
        if part["card"] == "end" and self.fmt == "youtube":
            return [(t0 + 0.9, "click", -4.0), (t0 + 1.7, "click", -4.0), (t0 + 1.75, "chime", -6.0)]
        if part["card"] in ("title", "eyecatch"):
            return [(t0, "whoosh", -4.0), (t0 + 0.25, "chime", -8.0)]
        return []


def strip_markup(text: str) -> str:
    return re.sub(r"[{}" + JOIN + "]", "", text)
