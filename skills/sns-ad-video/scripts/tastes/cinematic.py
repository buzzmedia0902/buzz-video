"""Cinematic taste: a slow, filmic brand film.

Slow-motion shots joined by long crossfades with a warm light leak blooming over every cut, film grain and
halation (the "cine" grade), vertical (tategaki) lines revealed character by character, chapter marks
(I 静 / II 整 / III 還), a fade in from black and a slow crossfade into a dark logo card whose CTA is one
quiet line of text. Lines and chapters are pinned to shots (shot + offsets in seconds), so the film keeps its
shape at any length; the end card takes whatever the shots leave.

Version spec (see references/tastes.md):
  {"taste": "cinematic", "duration": 24,
   "shots": [{"name": "open", "clip": "IMG_2742.MOV", "in": 0.6, "dur": 3.2, "speed": 0.5,
              "zoom": [1.5, 1.4], "center": [[0.40, 0.33], [0.40, 0.32]]}, ...],
   "lines": [{"text": "ほどける。", "shot": 1, "at": 0.72},
             {"text": "わたしに、還る。", "shot": "light", "at": 0.08, "side": "left", "color": "ink", "top": 0.11,
              "aspect": {"1:1": {"side": "right"}}}],
   "chapters": [{"num": "I", "text": "静", "shot": 2}, {"num": "II", "text": "整", "shot": 4}],
   "xfade": 0.8, "leak": 0.38, "grain": 0.035, "fade_in": 1.12, "grade": "cine",
   "card_dur": 3.6, "tagline": false, "bgm": {"style": "cinematic"}}

  shots     shot dicts of spec.py (clip, in, speed, zoom, center, focus, patches, soft_below, ...). "dur" is the
            time on screen in output seconds (>= xfade + 0.5); "name" is optional (default s1, s2, ...).
            Slow motion = "speed": 0.5. "pre"/"post" handles for the crossfades are set here; no handheld shake.
  duration  total length in s; the end card gets what the shots leave (2.8-8 s). Omit it to use card_dur.
  card_dur  end card length when "duration" is omitted (default 3.6). Give one of the two, not both.
  xfade     crossfade length centred on every cut (0.8).   leak  peak light leak per cut (0.38, 0 = off).
  grain     film grain amount (0.035).   fade_in  fade from black at the start, seconds (1.12, 0 = none).
  grade     grade of all shots (default "cine"; a shot's own "grade" wins).
  tagline   also show the brand tagline on the end card (false).
  lines[]   vertical text. text: a string, or a list of strings = columns read right to left (max 3).
            shot: 1-based shot number or shot name.  at: seconds after that shot starts (0.4).
            until: seconds relative to the END of that shot, <= 0 is before it (-0.16).
            side: right | left | center (right).  top: fraction of the safe height (0.16).
            color: white | ink | a brand colour name | "#RRGGBB" | [r, g, b] (white).
            size: font size in px (72 tall / 70 portrait / 64 square / 60 wide); shrinks to fit the safe zone.
            "ー" and brackets are swapped for their vertical forms. In 16:9 (portrait footage in a blurred
            frame) the column sits in the blurred side band, and dark colours turn white there.
  chapters[] small mark at the top left. num: roman numeral, text: one kanji, shot, at (0.24), until (-0.3).
  Lines, chapters and shots accept "aspect": {"16:9": {...}} with overrides of any of their keys for one
  output format (e.g. move a line to the other side where the subject sits in 4:5); lines and chapters also
  take "off": true there to drop them in that format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from core import engine as E
from core.canvas import ASPECT, H, W
from core.cards import EndCard
from core.spec import SpecError, make_shot
from core.timeline import Timeline

# ---- timing (output seconds) -------------------------------------------------------------------------------
CARD_LEAD = 0.36  # the blend into the end card starts this long before the last shot ends ...
CARD_AFTER = 0.28  # ... and finishes this long after it (the card, and the music's arrival, start here)
CARD_BLEND = CARD_LEAD + CARD_AFTER
FOOT_TAIL = 0.24  # the last shot is drawn until this long after it ends; the blend is 99 % done by then
MIN_CARD, MAX_CARD, DEFAULT_CARD = 2.8, 8.0, 3.6  # the CTA shows 0.8 s into the card and needs >= 2 s
MIN_CLEAR = 0.5  # a shot must stay clear of both of its crossfades for this long
TEXT_PACE = 1.25  # text animations run this much faster than the engine's natural pace (tuned on the 24 s master)
STAGGER, CHAR_FADE = 0.09 / TEXT_PACE, 0.6 / TEXT_PACE  # vertical text: delay per character, time to resolve
MARK_IN, MARK_OUT = 0.8 / TEXT_PACE, 0.5 / TEXT_PACE  # chapter mark fade in / out
MIN_HOLD = 0.45  # reveal + this = shortest time a line may stay on screen
BASE_LEAK = 0.05  # faint light leak that drifts over the whole film
LEAK_SEEDS = (4.0, 8.0, 12.0, 15.5, 18.0, 23.0)  # blob layouts of the six cuts of the 24 s master, cycled
LEAK_DRIFT = 1.25  # how fast the blobs drift

# ---- type --------------------------------------------------------------------------------------------------
TRACKING = 0.28  # airy spacing between vertical characters
COLUMN_PITCH = 1.9  # distance between neighbouring columns in units of the font size
EDGE_LEFT, EDGE_RIGHT = 1.4, 1.6  # column centre to the safe edge, in font sizes (room for the glow)
MIN_SIZE = 40
LIGHT = (255, 255, 255)
# the mincho font has no rotated glyphs without OpenType 'vert': swap ー, dashes, dots and brackets by hand
VERTICAL_FORMS = str.maketrans("ー—…「」『』（）", "丨︱︙﹁﹂﹃﹄︵︶")


# ---- small helpers -----------------------------------------------------------------------------------------
def _num(d: dict, key: str, default: float, where: str, lo: float = -1e9, hi: float = 1e9) -> float:
    x = d.get(key, default)
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not lo <= x <= hi:
        span = f" from {lo:g} to {hi:g}" if (lo, hi) != (-1e9, 1e9) else ""
        raise SpecError(f"{where}: '{key}' must be a number{span} (got {x!r})")
    return float(x)


def _items(v: dict, key: str) -> list[dict]:
    """v[key] as a list of dicts with their "aspect" overrides for this output format applied ("off" drops one)."""
    out = v.get(key, [])
    if not isinstance(out, list) or not all(isinstance(d, dict) for d in out):
        raise SpecError(f"cinematic: '{key}' must be a list of objects")
    merged = [{**d, **d.get("aspect", {}).get(ASPECT, {})} for d in out]
    return [d for d in merged if not d.get("off")]


def _columns(d: dict, where: str) -> list[str]:
    text = d.get("text")
    cols = [text] if isinstance(text, str) else text
    if not isinstance(cols, list) or not 1 <= len(cols) <= 3 or not all(isinstance(c, str) and c for c in cols):
        raise SpecError(f"{where}: 'text' is required: a string, or a list of 1-3 strings (one per column)")
    return cols


def _cells(text: str) -> float:
    """Height of a vertical column in character cells (、。 take 0.6 of a cell)."""
    return sum(0.6 if ch in E.SQUEEZE else 1.0 for ch in text)


def _delays(cols: list[str]) -> list[float]:
    """Start offset of every column: the next one begins while the previous is mostly revealed."""
    out, t = [], 0.0
    for c in cols:
        out.append(t)
        t += 0.7 * len(c) * STAGGER + 0.2
    return out


def _reveal(cols: list[str]) -> float:
    """Seconds until the last character of the last column is fully visible."""
    return max(d + (len(c) - 1) * STAGGER + CHAR_FADE for d, c in zip(_delays(cols), cols))


def _color(spec, brand, where: str) -> tuple[int, int, int]:
    if spec in (None, "white"):
        return LIGHT
    try:
        if isinstance(spec, str) and spec.startswith("#") and len(spec) == 7:
            return E.rgb(spec)
        if isinstance(spec, (list, tuple)) and len(spec) == 3:
            return E.rgb(spec)
        return brand.color(spec)
    except (SpecError, ValueError, TypeError) as e:
        raise SpecError(f"{where}: color {spec!r}: use white, ink, a brand colour name, \"#RRGGBB\" "
                        f"or [r, g, b]") from e


def _check_overlap(spans: list[tuple[str, float, float, str]], fix: str) -> None:
    """spans: (label, t_in, t_out, place). Two things at the same place must not be on screen together."""
    for i, (la, a0, a1, pa) in enumerate(spans):
        for lb, b0, b1, pb in spans[i + 1:]:
            if pa == pb and min(a1, b1) - max(a0, b0) > 0.1:
                raise SpecError(f"cinematic: {la} and {lb} are both at the {pa} {max(a0, b0):.2f}-{min(a1, b1):.2f}s: "
                                f"move one with {fix}")


def _card_shade(depth: float = 0.55) -> np.ndarray:
    """Radial darkening for the end card: 1 at the centre, 1 - depth at the far corners (H x W x 1)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.42) / H) ** 2)
    return (1 - depth * np.clip(r / 0.7, 0, 1) ** 1.3)[..., None]


# ---- drawables ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VLine:
    """One vertical line (one to three columns) with its place and time on the output timeline."""
    columns: tuple  # E.VText, rightmost first
    xs: tuple[float, ...]  # column centres
    top: float
    delays: tuple[float, ...]
    t_in: float
    t_out: float
    reveal: float
    side: str

    def draw(self, frame: np.ndarray, t: float) -> None:
        if not self.t_in <= t <= self.t_out:
            return
        for vt, x, d in zip(self.columns, self.xs, self.delays):
            vt.draw(frame, x, self.top, t * TEXT_PACE, (self.t_in + d) * TEXT_PACE, self.t_out * TEXT_PACE)


@dataclass(frozen=True)
class Chapter:
    """Chapter mark: roman numeral, rule and kanji, drifting in at the top left."""
    img: Image.Image
    shadow: Image.Image
    x: float  # left edge
    y: float  # vertical centre
    t_in: float
    t_out: float

    def draw(self, frame: np.ndarray, t: float) -> None:
        dur = self.t_out - self.t_in
        e = E.window(t, self.t_in, self.t_out, min(MARK_IN, 0.45 * dur), min(MARK_OUT, 0.3 * dur))
        if e <= 0:
            return
        cx, cy = self.x + self.img.width / 2, self.y - (1 - e) * 8
        E.blit_img(frame, self.shadow, cx, cy, opacity=e * 0.8)
        E.blit_img(frame, self.img, cx, cy, opacity=e)


# ---- the timeline ------------------------------------------------------------------------------------------
class Cinematic(Timeline):
    bgm_style = "cinematic"

    def __init__(self, ctx):
        v = ctx.v
        self.L, self.b = ctx.layout, ctx.brand
        self.xf = _num(v, "xfade", 0.8, "cinematic", 0.3, 1.6)
        self.leak = _num(v, "leak", 0.38, "cinematic", 0.0, 0.9)
        self.grain = _num(v, "grain", 0.035, "cinematic", 0.0, 0.12)
        self.fade_in = _num(v, "fade_in", 1.12, "cinematic", 0.0, 3.0)
        self.leak_w = 0.9 * self.xf  # half-width of the light leak around a cut
        shots = self._shots(ctx)
        super().__init__(ctx, shots)
        self.end_shots = shots[-1].t1
        self.card_at = self.end_shots + CARD_AFTER
        self.duration = self._duration(v)
        self.cut_times = [s.t0 for s in shots[1:]]
        self.lines = [self._line(d, f"cinematic: lines[{n}]") for n, d in enumerate(_items(v, "lines"), 1)]
        self.chapters = [self._chapter(d, f"cinematic: chapters[{n}]") for n, d in enumerate(_items(v, "chapters"), 1)]
        _check_overlap([(f"lines[{n}]", ln.t_in, ln.t_out, f"{ln.side} edge") for n, ln in enumerate(self.lines, 1)],
                       "'at' / 'until' or 'side'")
        _check_overlap([(f"chapters[{n}]", c.t_in, c.t_out, "top left") for n, c in enumerate(self.chapters, 1)],
                       "'at' / 'until'")
        self.card = EndCard(ctx, theme="dark", cta_style="text", show_tagline=bool(v.get("tagline", False)),
                            times={"logo": 0.0, "label": 0.36, "sub": 0.4, "glint": 0.44, "cta": 0.52})
        self.shade = _card_shade()
        self.cover_at = self._cover()

    # -------------------------------------------------------------- spec -> timeline
    def _shots(self, ctx) -> list[E.Shot]:
        specs = ctx.v["shots"]
        if not isinstance(specs, list) or not specs or not all(isinstance(d, dict) for d in specs):
            raise SpecError("cinematic: 'shots' must be a list of shot objects ({\"clip\", \"in\", \"dur\", ...})")
        out, t = [], 0.0
        for i, d in enumerate(specs):
            where = f"cinematic: shot {i + 1}"
            if "dur" not in d:
                raise SpecError(f"{where}: 'dur' (seconds on screen) is required")
            dur = _num(d, "dur", 0.0, where, 0.0, 600.0)
            if dur < self.xf + MIN_CLEAR:
                raise SpecError(f"{where}: dur {dur:g}s is too short for xfade {self.xf:g}s "
                                f"(needs >= {self.xf + MIN_CLEAR:g}s)")
            pre = 0.0 if i == 0 else self.xf / 2  # handles carry the crossfades
            post = FOOT_TAIL if i == len(specs) - 1 else self.xf / 2
            out.append(make_shot(ctx.project, {**d, "pre": pre, "post": post}, str(d.get("name", f"s{i + 1}")),
                                 t, t + dur, grade=ctx.v.get("grade", "cine")))
            t += dur
        names = [s.name for s in out]
        if len(set(names)) != len(names):
            raise SpecError(f"cinematic: shot names must be unique (got {', '.join(names)})")
        return out

    def _duration(self, v: dict) -> float:
        card = _num(v, "card_dur", DEFAULT_CARD, "cinematic", 0.0, 600.0)
        if "duration" in v:
            total = _num(v, "duration", 0.0, "cinematic", 0.0, 600.0)
            if "card_dur" in v and abs(self.end_shots + card - total) > 0.05:
                raise SpecError(f"cinematic: duration {total:g}s does not match shots {self.end_shots:g}s + "
                                f"card_dur {card:g}s: give one of them")
            card = total - self.end_shots
        if not MIN_CARD <= card <= MAX_CARD:
            raise SpecError(f"cinematic: the end card would last {card:.2f}s (allowed {MIN_CARD:g}-{MAX_CARD:g}s; "
                            f"the shots add up to {self.end_shots:.2f}s): change the shot lengths or the duration")
        return self.end_shots + card

    def _shot_index(self, ref, where: str) -> int:
        names = [s.name for s in self.shots]
        have = f"1..{len(names)} or a name ({', '.join(names)})"
        if ref is None:
            raise SpecError(f"{where}: 'shot' is required: {have}")
        if isinstance(ref, int) and not isinstance(ref, bool):
            if 1 <= ref <= len(names):
                return ref - 1
            raise SpecError(f"{where}: shot {ref} does not exist: {have}")
        if ref in names:
            return names.index(ref)
        raise SpecError(f"{where}: unknown shot {ref!r}: {have}")

    def _window(self, d: dict, where: str, at: float, until: float) -> tuple[int, float, float]:
        """(shot index, t_in, t_out) of something pinned to a shot."""
        i = self._shot_index(d.get("shot"), where)
        s = self.shots[i]
        t_in, t_out = s.t0 + _num(d, "at", at, where), s.t1 + _num(d, "until", until, where)
        if t_in < 0 or t_out > self.end_shots or t_out <= t_in:
            raise SpecError(f"{where}: runs {t_in:.2f}-{t_out:.2f}s but shot {i + 1} lasts {s.t0:.2f}-{s.t1:.2f}s "
                            f"(check 'at' / 'until'; it must stay within 0-{self.end_shots:.2f}s)")
        return i, t_in, t_out

    # -------------------------------------------------------------- vertical lines
    def _line(self, d: dict, where: str) -> VLine:
        L, cols = self.L, _columns(d, where)
        i, t_in, t_out = self._window(d, where, 0.4, -0.16)
        reveal = _reveal(cols)
        if t_out - t_in < reveal + MIN_HOLD:
            raise SpecError(f"{where} {cols[0]!r}: on screen {t_out - t_in:.2f}s, needs >= {reveal + MIN_HOLD:.2f}s "
                            f"(lengthen shot {i + 1}, lower 'at', or shorten the text)")
        side = d.get("side", "right")
        if side not in ("right", "left", "center"):
            raise SpecError(f"{where}: side must be right, left or center (got {side!r})")
        top = L.y(_num(d, "top", 0.16, where, 0.0, 0.9))
        size0 = int(_num(d, "size", L.pick(tall=72, portrait=70, square=64, wide=60), where, MIN_SIZE, 400))
        size = self._fit_size(cols, size0, top, where)
        x_first, in_band = self._first_column_x(side, i, size, len(cols))
        color = _color(d.get("color"), self.b, f"{where}.color")
        if in_band and sum(color) < 384:  # dark ink would vanish on the dimmed blur band
            color = LIGHT
        columns = tuple(E.VText(c.translate(VERTICAL_FORMS), self.b.font("vertical"), size, color, TRACKING)
                        for c in cols)
        xs = tuple(x_first - j * COLUMN_PITCH * size for j in range(len(cols)))
        return VLine(columns, xs, top, tuple(_delays(cols)), t_in, t_out, reveal, side)

    def _fit_size(self, cols: list[str], size: int, top: float, where: str) -> int:
        """Largest size <= `size` at which the tallest column ends inside the safe zone."""
        avail, cells = self.L.y1 - top, max(_cells(c) for c in cols)
        while size > MIN_SIZE and cells * size * (1 + TRACKING) > avail:
            size -= 2
        if cells * size * (1 + TRACKING) > avail:
            raise SpecError(f"{where}: the column is too long for {ASPECT} even at {MIN_SIZE}px: "
                            f"split the text into several columns or move 'top' up")
        return size

    def _first_column_x(self, side: str, shot: int, size: int, n: int) -> tuple[float, bool]:
        """Centre x of the rightmost column, and whether it sits in the blurred side band of a blur-fit shot."""
        L, span = self.L, (n - 1) * COLUMN_PITCH * size
        band = self._band_centre(shot, side, 2.6 * size + span)
        if band is not None:
            return band + span / 2, True
        if side == "left":
            return L.x0 + EDGE_LEFT * size + span, False
        if side == "center":
            return L.cx + span / 2, False
        return L.x1 - EDGE_RIGHT * size, False

    def _band_centre(self, shot: int, side: str, need: float) -> float | None:
        """Centre x of the blurred band beside the footage box (blur-fit shots in wide frames) if it is wide enough."""
        if side == "center" or self.shots[shot].fit != "blur":
            return None
        band = (W - self.sources[shot].view_w) / 2
        if band < need:
            return None
        return band / 2 if side == "left" else W - band / 2

    # -------------------------------------------------------------- chapters
    def _chapter(self, d: dict, where: str) -> Chapter:
        for key in ("num", "text"):
            if not isinstance(d.get(key), str) or not d[key]:
                raise SpecError(f"{where}: '{key}' is required (e.g. \"num\": \"I\", \"text\": \"静\")")
        _, t_in, t_out = self._window(d, where, 0.24, -0.3)
        img, shadow = self._mark(d["num"], d["text"], self.L.pick(tall=1.0, portrait=1.0, default=0.92))
        return Chapter(img, shadow, self.L.x0 + 30, self.L.y(0.062), t_in, t_out)

    def _mark(self, num: str, kanji: str, k: float) -> tuple[Image.Image, Image.Image]:
        gold = self.b.color("accent_light")
        n = E.text_img(num, "didot", int(54 * k), gold, pad=6)
        h = E.text_img(kanji, self.b.font("serif"), int(46 * k), LIGHT, pad=6)
        rule0, rule1, gap = int(12 * k), int(46 * k), int(54 * k)
        img = Image.new("RGBA", (n.width + h.width + int(70 * k), max(n.height, h.height) + 10), (0, 0, 0, 0))
        img.alpha_composite(n, (0, (img.height - n.height) // 2))
        ImageDraw.Draw(img).line((n.width + rule0, img.height // 2, n.width + rule1, img.height // 2),
                                 fill=(*gold, 255), width=2)
        img.alpha_composite(h, (n.width + gap, (img.height - h.height) // 2 + 2))
        return img, E.shadow_for(img, 9, 1.4, pad=30)

    def _cover(self) -> float:
        """Cover frame: the first line fully revealed, else a little way into the first shot."""
        if self.lines:
            ln = min(self.lines, key=lambda x: x.t_in)
            t = min(ln.t_in + ln.reveal + 0.3, ln.t_out - 0.6)
        else:
            t = 0.45 * self.shots[0].t1
        return round(min(max(t, self.fade_in + 0.2), self.card_at - 1.0), 3)

    # -------------------------------------------------------------- drawing
    def _footage(self, t: float) -> np.ndarray:
        """The shots at time t: one, or two crossfading around a cut."""
        live = [i for i, s in enumerate(self.shots) if s.t0 - s.pre - 1e-6 <= t < s.t1 + s.post - 1e-6]
        if not live:
            live = [self.index_at(t)]
        frame = self.shot_frame(live[0], t)
        if len(live) > 1:
            nxt = self.shots[live[1]]
            w = E.smoothstep((t - (nxt.t0 - self.xf / 2)) / self.xf)
            frame = frame * (1 - w) + self.shot_frame(live[1], t) * w
        return frame

    def _leaks(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Warm light leaks: a bloom centred on every cut plus a faint drift over everything."""
        for j, cut in enumerate(self.cut_times):
            d = abs(t - cut)
            if self.leak > 0 and d < self.leak_w:
                seed = LEAK_SEEDS[j % len(LEAK_SEEDS)]
                k = self.leak * math.cos(d / self.leak_w * math.pi / 2) ** 2
                frame = E.screen(frame, E.light_leak(seed + (t - cut) * LEAK_DRIFT, k, seed=seed))
        return E.screen(frame, E.light_leak(t * LEAK_DRIFT, BASE_LEAK, seed=7.0))

    def _card_frame(self, t: float) -> np.ndarray:
        frame = self.card.frame(t, self.card_at)
        frame *= self.shade
        return frame

    def frame(self, t: float) -> np.ndarray:
        if t >= self.end_shots + FOOT_TAIL:
            frame = self._card_frame(t)
        else:
            frame = self._leaks(self._footage(t), t)
            for ln in self.lines:
                ln.draw(frame, t)
            for ch in self.chapters:
                ch.draw(frame, t)
            if self.fade_in > 0:
                frame *= E.ease_in_out_sine(t / self.fade_in)
            k = E.smoothstep((t - (self.card_at - CARD_BLEND)) / CARD_BLEND)
            if k > 0:
                frame = frame * (1 - k) + self._card_frame(t) * k
        if self.grain > 0:
            E.add_grain(frame, t, self.grain)
        return frame


def build(ctx) -> Timeline:
    if "shots" not in ctx.v:
        raise SpecError("cinematic: 'shots' is required (list of {\"clip\", \"in\", \"dur\", \"speed\", ...})")
    return Cinematic(ctx)
