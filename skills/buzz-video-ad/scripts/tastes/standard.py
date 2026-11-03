"""Standard taste: the main brand ad. Ken Burns shots, per-character captions, a white-out into a light end card.

Every shot is a slow push-in (beauty grade by default). Shots are joined by cuts, cross-dissolves or a dissolve
with a warm flash. Each shot may carry one caption (a small lead-in row + a large row, optionally a gold label
above): the letters fade in one by one over a soft shadow (light theme: white text) or a soft glow (dark theme:
ink text, for bright backgrounds). The first shot can show the brand mark above its caption. The last shot
whites out into the end card (rising dust, label, logo with glint, tagline, outlined CTA pill); with a pale logo
(logo_prep "pale logo") the card is dark by default and the shots dissolve into it.

Version spec (see references/tastes.md):
  {"taste": "standard", "duration": 24, "hook_logo": true, "bgm": {"style": "piano"},
   "shots": [
     {"clip": "IMG_2742.MOV", "dur": 1.8, "in": 2.85, "speed": 0.5, "zoom": [1.42, 1.30],
      "center": [[0.46, 0.36], [0.46, 0.35]],
      "caption": {"rows": [{"text": "顔に、鍼。", "size": 140, "font": "title", "tracking": 0.06}],
                  "label": "BEAUTY ACUPUNCTURE", "pos": 0.3}},
     {"clip": "IMG_1454 2.mov", "dur": 1.8, "in": 7.5, "zoom": [1.4, 1.46], "center": [0.42, 0.42],
      "grade": "daily", "transition": "cut",
      "caption": {"rows": ["画面と、", "にらめっこの毎日。"], "theme": "dark", "pos": "upper"}},
     {"clip": "IMG_0996.MOV", "dur": 2.4, "in": 3.4, "zoom": [1.36, 1.44], "center": [0.42, 0.5],
      "grade": "counsel", "transition": "flash", "tdur": 0.48,
      "caption": {"rows": ["あなたの顔を、", "知ることから。"], "label": "COUNSELING", "theme": "dark",
                  "pos": "top", "aspect": {"4:5": {"pos": 0.12}}}},
     ...],
   "card": {"dur": 3.6, "transition": "whiteout", "tdur": 0.48, "push": 0.03}}

Version keys
  shots       list of shot dicts (below); at least one.
  grade       default grade of the shots (default "beauty"); a shot's own "grade" wins.
  card        {"dur": 3.6, "theme": "light" | "dark", "transition": "whiteout" | "dissolve" | "cut", "tdur": 0.48,
              "push": 0.03, "times": {...EndCard times, e.g. "cta": 1.17...}}. "push" = slow zoom-in of the card
              (fraction). theme defaults to "dark" for a pale logo, else "light"; transition defaults to "whiteout"
              on a light card and "dissolve" on a dark one.
  duration    optional total length in s: the end card absorbs the difference between it and the shots (it must
              come out at 3.2-6.0 s so the CTA stays on screen 2 s+). Wins over card.dur. Without it:
              duration = sum of the shot "dur"s + card.dur.
  hook_logo   true (default) | false | {"variant": "on_dark" | "on_light", "halo": 0.45}: the brand mark above
              the first shot's caption. on_dark (default) = the dark-background logo on a dark halo; on_light = the
              light-background logo on a pale glow (bright footage). halo = halo opacity 0-1 (0.75 for a pale logo).
              A pale logo needs a dark part of the first shot behind it: check the stills.
  sfx         true (default) | false: soft chime when the end-card logo glints.
Shot keys (core/spec.py: clip, dur, in, speed, zoom, center, grade, patches, focus, soft_below, aspect, ...; the
shots play back to back, "dur" = output seconds; a shot's own "soft_below" replaces the clip's, its "patches"
add to them) plus
  transition  how the shot is entered: "cut" | "dissolve" | "flash" (dissolve + warm flash) | "whiteout" (dip to
              white). Default: cut for the first shot, dissolve for the others.
  tdur        transition length in s, centred on the cut (defaults: dissolve 0.32, flash 0.4, whiteout 0.48);
              the two transitions around a shot must fit inside it.
  caption     optional, one per shot:
    rows      1-3 rows, top to bottom; a row is a string or {"text", "size", "font", "tracking"} ("font" = brand
              font role such as serif / title / body, or an engine font key). Default sizes at 9:16: last row 76
              (serif, the message), earlier rows 48 (lead-in). Sizes shrink with the format and to fit the picture.
    label     small tracked Latin line above the rows, e.g. "COUNSELING" (brand "latin" font, accent colour).
    theme     "light" (default: white text, dark shadow) | "dark" (ink text, pale glow, for bright shots).
    pos       block position: a fraction 0-1 of the safe zone height (the block's centre), or "top" / "upper"
              (0.22) / "middle" / "lower" (0.8) / "bottom". "top" and "bottom" snap the block to the safe-zone edge.
              Always clamped into the safe zone. Default "upper". Choose the side that keeps faces clear and check
              every aspect ratio (the subject sits mid-frame in all of them).
    x         horizontal centre, fraction of the safe width (default 0.5).
    scale     extra text scale (default 1).
    in / out  seconds after the shot's start / before its end. Defaults: 0.04 + incoming tdur / 4, and the start
              of the outgoing transition. "at" / "until" give absolute times instead (e.g. to span two shots).
    aspect    per-aspect overrides of any caption key: {"4:5": {"pos": "top"}}.
Grades: any engine preset, including the original ad's "daily" (muted, for everyday scenes), "counsel" and
"daylight" (the finale look, a little darker than "bright").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from core import engine as E
from core.canvas import H, W
from core.cards import TIMES as CARD_TIMES
from core.cards import EndCard
from core.spec import SpecError, make_shot
from core.timeline import Timeline

KINDS = ("cut", "dissolve", "flash", "whiteout")
CARD_KINDS = ("whiteout", "dissolve", "cut")
DEFAULT_TDUR = {"cut": 0.0, "dissolve": 0.32, "flash": 0.40, "whiteout": 0.48}
THEMES = ("light", "dark")
POS_WORDS = {"upper": 0.22, "middle": 0.5, "lower": 0.8}  # block centre as a fraction of the safe zone height
LEAD_SIZE, MAIN_SIZE = 48, 76  # caption row sizes at 9:16
TEXT_SCALE = {"tall": 1.0, "portrait": 0.94, "square": 0.9, "wide": 0.8}
LOGO_WIDTH = {"tall": 338, "portrait": 320, "square": 285, "wide": 265}  # px; the wordmark is ~95 % of it
CARD_MIN, CARD_MAX = 3.2, 6.0  # the CTA appears ~1.2 s into the card and must stay on screen >= 2 s
MIN_CAPTION_S = 0.8
CTA_MIN_S = 2.0
EPS = 1e-6  # frame times come from n / fps and can sit a hair below an exact boundary
FLASH_TINT = np.array([1.0, 0.93, 0.82], np.float32)
EDGE = 0.02  # gap between a snapped block ("top" / "bottom" / hook logo) and the safe-zone edge, x safe height

# ---------------------------------------------------------------- spec parsing
@dataclass(frozen=True)
class Transition:
    kind: str
    at: float  # centre of the transition = start of the incoming shot
    dur: float

    @property
    def half(self) -> float:
        return self.dur / 2


@dataclass(frozen=True)
class Row:
    text: str
    size: int
    font: str  # engine font key
    tracking: float


@dataclass(frozen=True)
class CaptionSpec:
    shot: int
    rows: tuple[Row, ...]
    label: str
    theme: str
    pos: float | str
    x: float
    scale: float
    t_in: float
    t_out: float


def _num(v, where: str, key: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise SpecError(f"{where}: '{key}' must be a number, got {v!r}") from None
    if not math.isfinite(x):
        raise SpecError(f"{where}: '{key}' must be a finite number, got {v!r}")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise SpecError(f"{where}: '{key}' = {x:g} is outside {lo}..{hi}")
    return x


def _flag(v: dict, key: str, default: bool = True) -> bool:
    x = v.get(key, default)
    if not isinstance(x, bool):
        raise SpecError(f"standard: '{key}' must be true or false, got {x!r}")
    return x


def _aspect_override(d: dict, aspect: str, where: str) -> dict:
    """The per-aspect overrides {"4:5": {...}} of a shot / caption dict ({} if none)."""
    a = d.get("aspect", {})
    if not isinstance(a, dict) or not isinstance(a.get(aspect, {}), dict):
        raise SpecError(f"{where}: 'aspect' must look like {{\"{aspect}\": {{...overrides...}}}}")
    return a.get(aspect, {})


def parse_transition(d: dict, i: int, at: float) -> Transition:
    """Transition INTO shot i (the first shot is always a plain start)."""
    kind, where = d.get("transition", "cut" if i == 0 else "dissolve"), f"standard: shot {i + 1}"
    if kind not in KINDS:
        raise SpecError(f"{where}: transition {kind!r} unknown (use {' | '.join(KINDS)})")
    if i == 0 or kind == "cut":
        return Transition("cut", at, 0.0)
    return Transition(kind, at, _num(d.get("tdur", DEFAULT_TDUR[kind]), where, "tdur", 0.1, 1.5))


def parse_card(v: dict, shots_end: float, pale: bool) -> tuple[Transition, float, float, dict, str]:
    """(transition into the card, card length, push-in amount, EndCard time overrides, theme)."""
    c = v.get("card", {})
    if not isinstance(c, dict):
        raise SpecError("standard: 'card' must be an object like {\"dur\": 3.6}")
    where = "standard: card"
    theme = c.get("theme", "dark" if pale else "light")
    if theme not in ("light", "dark"):
        raise SpecError(f"{where}: theme {theme!r} unknown (use light | dark)")
    kind = c.get("transition", "whiteout" if theme == "light" else "dissolve")
    if kind not in CARD_KINDS:
        raise SpecError(f"{where}: transition {kind!r} unknown (use {' | '.join(CARD_KINDS)})")
    tdur = 0.0 if kind == "cut" else _num(c.get("tdur", DEFAULT_TDUR[kind]), where, "tdur", 0.1, 1.5)
    if "duration" in v:  # the card absorbs the difference; card.dur is not used
        total = _num(v["duration"], "standard", "duration", 4)
        dur = total - shots_end
        if not CARD_MIN - 1e-6 <= dur <= CARD_MAX + 1e-6:
            raise SpecError(f"standard: duration {total:g} s with shots totalling {shots_end:.2f} s leaves a "
                            f"{dur:.2f} s end card; it needs {CARD_MIN}-{CARD_MAX} s (so the CTA stays on screen "
                            f"2 s+): make the shots total {total - CARD_MAX:.1f}-{total - CARD_MIN:.1f} s")
    else:
        dur = _num(c.get("dur", 3.6), where, "dur", CARD_MIN, CARD_MAX)
    push = _num(c.get("push", 0.03), where, "push", 0.0, 0.1)
    return Transition(kind, shots_end, tdur), dur, push, parse_times(c.get("times", {}), where), theme


def parse_times(t, where: str) -> dict:
    """EndCard time overrides {"cta": 1.2, ...} (seconds after the card starts)."""
    if not isinstance(t, dict):
        raise SpecError(f"{where}: 'times' must be an object like {{\"cta\": 1.2}}")
    unknown = [k for k in t if k not in CARD_TIMES]
    if unknown:
        raise SpecError(f"{where}: unknown times {unknown} (use {', '.join(CARD_TIMES)})")
    return {k: _num(v, where, f"times.{k}", 0.0, 10.0) for k, v in t.items()}


def _font(b, name: str, where: str) -> str:
    if not isinstance(name, str):
        raise SpecError(f"{where}: 'font' must be a font role or key (text), got {name!r}")
    if name in b.fonts:
        return b.font(name)
    if name in E.FONT_KEYS:
        return name
    raise SpecError(f"{where}: font {name!r} is neither a brand font role ({', '.join(b.fonts)}) "
                    f"nor an engine font key")


def parse_rows(b, rows, where: str) -> tuple[Row, ...]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise SpecError(f"{where}: 'rows' must be a list of 1-3 rows (strings or {{\"text\": ...}})")
    out = []
    for k, r in enumerate(rows):
        if not isinstance(r, (str, dict)):
            raise SpecError(f"{where}: row {k + 1} must be a string or {{\"text\": ...}}, got {r!r}")
        d = {"text": r} if isinstance(r, str) else r
        if not isinstance(d.get("text"), str) or not d["text"]:
            raise SpecError(f"{where}: row {k + 1} needs a text")
        size = _num(d.get("size", MAIN_SIZE if k == len(rows) - 1 else LEAD_SIZE), where, "size", 16, 400)
        out.append(Row(d["text"], int(size), _font(b, d.get("font", "serif"), where),
                       _num(d.get("tracking", 0.08), where, "tracking", -0.1, 1)))
    return tuple(out)


def parse_caption(ctx, c: dict, i: int, shot: E.Shot, tr_in: Transition, tr_out: Transition) -> CaptionSpec:
    where = f"standard: shot {i + 1} caption"
    if not isinstance(c, dict):
        raise SpecError(f"{where}: must be an object with 'rows'")
    c = {**c, **_aspect_override(c, ctx.layout.aspect, where)}
    theme = c.get("theme", "light")
    if theme not in THEMES:
        raise SpecError(f"{where}: theme {theme!r} must be 'light' or 'dark'")
    pos = c.get("pos", "upper")
    if not isinstance(pos, str):
        pos = _num(pos, where, "pos", 0.0, 1.0)
    elif pos not in ("top", "bottom", *POS_WORDS):
        raise SpecError(f"{where}: pos {pos!r} unknown (a fraction 0-1 or top | upper | middle | lower | bottom)")
    lead = _num(c.get("in", 0.04 + tr_in.dur / 4), where, "in", 0.0)
    tail = _num(c.get("out", max(0.04, tr_out.half)), where, "out", 0.0)
    t_in = _num(c["at"], where, "at", 0.0) if "at" in c else shot.t0 + lead
    t_out = _num(c["until"], where, "until", 0.0) if "until" in c else shot.t1 - tail
    if t_out - t_in < MIN_CAPTION_S:
        raise SpecError(f"{where}: on screen only {t_out - t_in:.2f} s ({t_in:.2f}-{t_out:.2f}); it needs "
                        f"{MIN_CAPTION_S} s+ (lengthen the shot or reduce 'in' / 'out')")
    label = c.get("label") or ""
    if not isinstance(label, str):
        raise SpecError(f"{where}: 'label' must be text, got {label!r}")
    x, scale = _num(c.get("x", 0.5), where, "x", 0.0, 1.0), _num(c.get("scale", 1.0), where, "scale", 0.3, 3.0)
    return CaptionSpec(i, parse_rows(ctx.brand, c.get("rows"), where), label, theme, pos, x, scale, t_in, t_out)


# ---------------------------------------------------------------- captions
def _paste_max(canvas: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """canvas = max(canvas, sprite alpha) with the sprite's top-left at (x, y); clipped to the canvas."""
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], alpha[y0 - y:y1 - y, x0 - x:x1 - x])


class CaptionBlock:
    """Label + rows + short rule laid out around a centre point, over a soft shadow (light) or glow (dark).

    Everything is built relative to the block centre, so the block can be (re)placed with place()."""

    def __init__(self, ctx, spec: CaptionSpec, max_w: float):
        L, b = ctx.layout, ctx.brand
        self.spec, self.L = spec, L
        dark = spec.theme == "dark"
        latin = b.font("latin")
        k = TEXT_SCALE[L.kind] * spec.scale
        widths = [E.text_width(r.text, r.font, max(10, int(r.size * k)), r.tracking) for r in spec.rows]
        if spec.label:
            widths.append(E.text_width(spec.label, latin, max(10, int(30 * k)), 0.36))
        k *= min(1.0, max_w / max(widths))  # the whole block shrinks together to fit the picture ...
        unit = sum(r.size * 1.22 + 8 for r in spec.rows) + (30 * 1.22 + 20 if spec.label else 0) + 34  # height, k = 1
        k = min(k, 0.94 * L.sh / unit)  # ... and the safe height
        ink = b.color("ink") if dark else (255, 255, 255)
        gold = b.color("accent_deep") if dark else b.color("accent_light")
        self.rule_color = np.array(gold, np.float32) / 255
        self.lines = [E.TextLine(r.text, r.font, max(10, int(r.size * k)), ink, r.tracking) for r in spec.rows]
        self.label = E.TextLine(spec.label, latin, max(10, int(30 * k)), gold, tracking=0.36) if spec.label else None
        self.stagger = [0.06 if r.size >= 80 else 0.025 for r in spec.rows]  # reveal speed follows the nominal size
        self._layout(k)
        self._shadow(dark, k, E.mix(b.color("paper"), (255, 255, 255), 0.5))
        self.cx = self.cy = 0

    def _layout(self, k: float) -> None:
        gap_label, gap_row, rule_h = 20 * k, 8 * k, 34 * k
        heights = ([self.label.box_h + gap_label] if self.label else []) + [ln.box_h + gap_row for ln in self.lines]
        self.height = sum(heights) + rule_h
        self.width = max([ln.width for ln in self.lines] + ([self.label.width] if self.label else []))
        y, self.items = -self.height / 2, []  # (kind, TextLine, dx, dy) relative to the block centre
        if self.label:
            self.items.append(("label", self.label, -self.label.width / 2, y))
            y += self.label.box_h + gap_label
        for ln in self.lines:
            self.items.append(("line", ln, -ln.width / 2, y))
            y += ln.box_h + gap_row
        self.rule_dy, self.rule_half = y + 22 * k, 48 * k

    def _shadow(self, dark: bool, k: float, glow_color) -> None:
        """Soft shadow / glow sprite from the union of all glyph alphas (built once, relative to the centre)."""
        m = int(100 * k) + 30
        x0, y0 = -self.width / 2 - m, -self.height / 2 - m
        mask = np.zeros((int(self.height + 2 * m) + 1, int(self.width + 2 * m) + 1), np.float32)
        for _, ln, dx, dy in self.items:
            for c in ln.chars:
                _paste_max(mask, c["levels"][0][..., 3], int(round(dx + c["x"] - x0)), int(round(dy + c["y"] - y0)))
        if dark:
            near, wide = gaussian_filter(mask, 9 * k), gaussian_filter(mask, 30 * k)
            alpha, color, self.shadow_dy = np.clip(near * 2.2, 0, 1) * 0.75 + wide * 0.9, glow_color, 0
        else:
            near, wide = gaussian_filter(mask, 5 * k), gaussian_filter(mask, 34 * k)
            alpha = np.clip(near * 1.6, 0, 1) * 0.6 + np.clip(wide * 2.6, 0, 1) * 0.4
            color, self.shadow_dy = (10, 8, 6), 3
        self.shadow = E.solid_sprite(alpha, color)
        self.shadow_off = (x0, y0)

    # ------------------------------------------------------------ placement
    @property
    def top(self) -> float:
        return self.cy - self.height / 2

    @property
    def bottom(self) -> float:
        return self.cy + self.height / 2

    def place(self, cx: float, cy: float) -> None:
        self.cx, self.cy = round(cx), round(cy)

    def default_centre(self) -> tuple[float, float]:
        """Centre for the spec's pos / x, kept inside the safe zone."""
        L, s, h = self.L, self.spec, self.height
        lo, hi = L.y0 + h / 2 + EDGE * L.sh, L.y1 - h / 2 - EDGE * L.sh
        c = lo if s.pos == "top" else hi if s.pos == "bottom" else L.y(POS_WORDS.get(s.pos, s.pos))
        half_w = self.width / 2
        return min(max(L.x(s.x), L.x0 + half_w), L.x1 - half_w), min(max(c, lo), hi)

    # ------------------------------------------------------------ drawing
    def _fade(self, t: float) -> float:
        return E.ease_out_cubic((self.spec.t_out - t) / 0.22)

    def visible(self, t: float) -> bool:
        return self.spec.t_in - 0.01 <= t <= self.spec.t_out + 0.01

    def draw_shadow(self, frame: np.ndarray, t: float) -> None:
        if not self.visible(t):
            return
        fade = self._fade(t)
        reveal = E.ease_out_cubic((t - self.spec.t_in) / 0.5)
        E.blit(frame, self.shadow, self.cx + self.shadow_off[0],
               self.cy + self.shadow_off[1] + self.shadow_dy - (1 - fade) * 10, reveal * fade)

    def draw(self, frame: np.ndarray, t: float) -> None:
        if not self.visible(t):
            return
        t_in = self.spec.t_in
        fade = self._fade(t)
        drift = (1 - fade) * 10
        start, stagger = t_in + (0.08 if self.label else 0.0), iter(self.stagger)
        for kind, ln, dx, dy in self.items:
            ox, oy = self.cx + dx, self.cy + dy
            if kind == "label":
                ln.draw(frame, ox, oy, t, t_in, 0.01, 0.35, fade, drift)
                continue
            step = next(stagger)
            ln.draw(frame, ox, oy, t, start, step, 0.32, fade, drift)
            start += 0.08 + len(ln.chars) * step * 0.5
        self._rule(frame, t, fade, drift)

    def _rule(self, frame: np.ndarray, t: float, fade: float, drift: float) -> None:
        e = E.ease_in_out_sine((t - self.spec.t_in - 0.25) / 0.5)
        y = int(self.cy + self.rule_dy - drift)
        if e <= 0 or not 0 <= y < H - 2:
            return
        half = self.rule_half * e
        region = frame[y:y + 2, max(0, int(self.cx - half)):min(W, int(self.cx + half))]
        a = 0.95 * fade
        region *= 1 - a
        region += self.rule_color * a


def parse_hook(v: dict, pale: bool) -> dict | None:
    """hook_logo: true | false | {"variant": "on_dark" | "on_light", "halo": 0-1} -> HookLogo kwargs or None."""
    h = v.get("hook_logo", True)
    if h is False:
        return None
    if h is True:
        h = {}
    if not isinstance(h, dict):
        raise SpecError("standard: 'hook_logo' must be true, false or {\"variant\": \"on_light\", \"halo\": 0.6}")
    variant = h.get("variant", "on_dark")
    if variant not in ("on_dark", "on_light"):
        raise SpecError(f"standard: hook_logo variant {variant!r} unknown (use on_dark | on_light)")
    return {"variant": variant, "halo": _num(h.get("halo", 0.75 if pale else 0.45), "standard: hook_logo", "halo",
                                             0.0, 1.0)}


class HookLogo:
    """Small brand mark with a soft halo, shown during the first shot: the dark-background logo on a dark halo,
    or (variant "on_light") the light-background logo on a pale glow for bright footage."""

    def __init__(self, ctx, t_end: float, variant: str = "on_dark", halo: float = 0.45):
        L, b = ctx.layout, ctx.brand
        img = Image.open(b.logo_on_light if variant == "on_light" else b.logo_on_dark).convert("RGBA")
        w = LOGO_WIDTH[L.kind]
        self.img = img.resize((w, max(1, int(img.height * w / img.width))), Image.LANCZOS)
        glow = (250, 246, 238) if variant == "on_light" else (12, 10, 8)
        self.halo = E.shadow_for(self.img, 16 * w / 338, 1.8, glow, pad=60)
        self.halo_opacity = halo
        self.h, self.cx, self.top, self.t_end = self.img.height, L.cx, L.y0, t_end

    def draw(self, frame: np.ndarray, t: float) -> None:
        if t >= self.t_end:
            return
        e = E.ease_out_cubic(t / 0.5) * E.ease_out_cubic((self.t_end - 0.1 - t) / 0.3)
        cy = self.top + self.h / 2 - (1 - e) * 10
        E.blit_img(frame, self.halo, self.cx, cy, opacity=self.halo_opacity * e)
        E.blit_img(frame, self.img, self.cx, cy, opacity=e)


# ---------------------------------------------------------------- timeline
def _push_in(frame: np.ndarray, z: float) -> np.ndarray:
    """Centre zoom of a finished frame by factor z (the end card's slow push-in)."""
    wc, hc = W / z, H / z
    box = ((W - wc) / 2, (H - hc) / 2, (W + wc) / 2, (H + hc) / 2)
    return np.asarray(Image.fromarray(E.to_u8(frame)).resize((W, H), Image.BICUBIC, box=box), np.float32) / 255


class Standard(Timeline):
    bgm_style = "piano"

    def __init__(self, ctx):
        v, self.L = ctx.v, ctx.layout
        specs = v["shots"]
        if not isinstance(specs, list) or not specs or not all(isinstance(d, dict) for d in specs):
            raise SpecError("standard: 'shots' must be a non-empty list of shot dicts (each with 'clip' and 'dur')")
        self.trans, ends, t = [], [], 0.0
        for i, d in enumerate(specs):
            _aspect_override(d, ctx.layout.aspect, f"standard: shot {i + 1}")
            if "dur" not in d:
                raise SpecError(f"standard: shot {i + 1}: 'dur' (output seconds) is required")
            self.trans.append(parse_transition(d, i, t))
            t += _num(d["dur"], f"standard: shot {i + 1}", "dur", 0.3)
            ends.append(t)
        pale = ctx.brand.pale
        self.card_tr, card_dur, self.push, card_times, theme = parse_card(v, t, pale)
        self._check_transitions(specs)
        shots = []
        for i, d in enumerate(specs):
            post = self.trans[i + 1].half if i + 1 < len(specs) else self.card_tr.half
            name, t0 = str(d.get("name", f"s{i + 1}")), ends[i] - float(d["dur"])
            shots.append(make_shot(ctx.project, d, name, t0, ends[i], grade=v.get("grade", "beauty"),
                                   pre=self.trans[i].half, post=post))
            self._check_grade(shots[-1].grade, i)  # after merging clip / shot / aspect settings
        super().__init__(ctx, shots)
        self.card_at, self.duration = t, t + card_dur
        self.captions = self._captions(ctx, specs)
        self.cover_at = self._cover_at()
        hook = parse_hook(v, pale)
        self.hook = self._hook_logo(ctx, hook) if hook is not None else None
        times = {"cta": 1.17, "glint": 1.2, "tagline": 1.07, **card_times}
        if card_dur - times["cta"] < CTA_MIN_S - 1e-6:
            raise SpecError(f"standard: the CTA would be on screen only {card_dur - times['cta']:.2f} s "
                            f"(card {card_dur:.2f} s, CTA at +{times['cta']:g} s); it needs {CTA_MIN_S:g} s+")
        self.card = EndCard(ctx, theme=theme, cta_style="outline", particles=True, times=times)
        if _flag(v, "sfx"):
            self.sfx.append([round(self.card_at + self.card.t["glint"], 3), "chime", 0.5])

    # -------------------------------------------------------------- spec checks
    def _check_transitions(self, specs: list) -> None:
        """The transitions at both ends of a shot must fit inside it (so at most two shots overlap)."""
        for i, d in enumerate(specs):
            nxt = self.trans[i + 1].half if i + 1 < len(specs) else self.card_tr.half
            if self.trans[i].half + nxt > float(d["dur"]) - 0.05:
                raise SpecError(f"standard: shot {i + 1} ({d['dur']} s) is too short for its transitions "
                                f"({self.trans[i].half + nxt:.2f} s): shorten 'tdur' or lengthen the shot")

    @staticmethod
    def _check_grade(name, i: int) -> None:
        if not isinstance(name, str) or name not in E.PRESETS:
            raise SpecError(f"standard: shot {i + 1}: grade {name!r} unknown (use {' | '.join(E.PRESETS)})")

    def _cover_at(self) -> float:
        """Frame for the cover image: the hook with its caption and logo fully in."""
        first = next((c for c in self.captions if c.spec.shot == 0), None)
        if first is None:
            return min(self.shots[0].t0 + 1.3, self.shots[0].t1 - 0.1)
        return max(0.0, min(first.spec.t_in + 1.1, first.spec.t_out - 0.3))

    # -------------------------------------------------------------- assets
    def _captions(self, ctx, specs: list) -> list[CaptionBlock]:
        blocks = []
        for i, d in enumerate(specs):
            if not d.get("caption"):
                continue
            tr_out = self.trans[i + 1] if i + 1 < len(specs) else self.card_tr
            spec = parse_caption(ctx, d["caption"], i, self.shots[i], self.trans[i], tr_out)
            max_w = min(self.L.text_w, 0.94 * self.sources[i].view_w)  # stay on the picture when pillar-boxed
            block = CaptionBlock(ctx, spec, max_w)
            block.place(*block.default_centre())
            blocks.append(block)
        return blocks

    def _hook_logo(self, ctx, opts: dict) -> HookLogo:
        """Brand mark above the first shot's caption (the pair moves down together if it would leave the zone)."""
        L = self.L
        hook = HookLogo(ctx, self.shots[0].t1, **opts)
        gap, top_min = 22 * TEXT_SCALE[L.kind], L.y0 + EDGE * L.sh
        first = next((c for c in self.captions if c.spec.shot == 0), None)
        if first is None:
            hook.top = top_min
            return hook
        short = top_min - (first.top - gap - hook.h)
        if short > 0:
            first.place(first.cx, first.cy + min(short, L.y1 - EDGE * L.sh - first.bottom))
        hook.top = max(top_min, first.top - gap - hook.h)
        return hook

    # -------------------------------------------------------------- drawing
    def _active(self, t: float) -> list[int]:
        act = [i for i, s in enumerate(self.sources) if s.start - 1e-6 <= t < s.end - 1e-6]
        return act or [len(self.sources) - 1]

    def _picture(self, t: float) -> np.ndarray:
        act = self._active(t)
        out = self.shot_frame(act[0], t)
        if len(act) > 1:
            out = self._join(out, self.shot_frame(act[1], t), self.trans[act[1]], t)
        return out

    @staticmethod
    def _join(a: np.ndarray, b: np.ndarray, tr: Transition, t: float) -> np.ndarray:
        w = E.smoothstep((t - (tr.at - tr.half)) / tr.dur)
        out = a * (1 - w) + b * w
        if tr.kind == "flash":
            f = math.exp(-((t - tr.at) / (tr.dur * 0.28)) ** 2) * 0.75
            out = 1 - (1 - out) * (1 - f * FLASH_TINT)
        elif tr.kind == "whiteout":
            out = 1 - (1 - out) * (1 - 0.8 * math.sin(math.pi * w))
        return out

    def _to_card(self, out: np.ndarray, t: float) -> np.ndarray:
        """Last shot -> end card (lifting to white first for 'whiteout'); the card's own elements fade in with it."""
        tr = self.card_tr
        w = E.smoothstep((t - (tr.at - tr.half)) / tr.dur)
        lifted = 1 - (1 - out) * (1 - 0.7 * w) if tr.kind == "whiteout" else out
        return lifted * (1 - w) + self._card_frame(t) * w

    def _card_frame(self, t: float) -> np.ndarray:
        frame = self.card.frame(t, self.card_at)
        z = 1 + self.push * E.ease_in_out_sine((t - self.card_at) / max(self.duration - self.card_at, 1e-6))
        return _push_in(frame, z) if z > 1.0005 else frame

    def frame(self, t: float) -> np.ndarray:
        if t >= self.card_at + self.card_tr.half - EPS:
            return self._card_frame(t)
        out = self._picture(t)
        if self.hook is not None:
            self.hook.draw(out, t)
        for cap in self.captions:
            cap.draw_shadow(out, t)
            cap.draw(out, t)
        if t >= self.card_at - self.card_tr.half - EPS and self.card_tr.kind != "cut":
            out = self._to_card(out, t)
        return out


def build(ctx) -> Timeline:
    if "shots" not in ctx.v:
        raise SpecError("standard: 'shots' is required (list of shot dicts with 'clip' and 'dur')")
    return Standard(ctx)
