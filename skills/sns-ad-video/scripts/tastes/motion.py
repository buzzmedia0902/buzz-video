"""Motion taste: a fast, animation-heavy cut driven by a beat grid (BGM style "pop", ~120-150 BPM).

Everything is measured in BEATS: the length of the ad follows from the beat counts (or the tempo follows from a
requested duration). Structure: hook -> steps (1..n) -> closing -> end card.
  hook     a gold circle iris opens on the footage; the brand logo and caption pills pop in on the beat
  step     a numbered badge (disc, "STEP n", title pill, sub pill) pops in and later shrinks into a chip at the
           top left. A step with several shots is a beat montage: each cut is a "hit" (zoom punch, white flash,
           sparkle burst, twinkles). A shot can mark a detail with a gold ring (e.g. the iPad screen).
  closing  caption pills (the reward line) over the last shot
  card     the end card opens as a circle iris: logo letters pop, tagline, CTA pill that pulses to the beat
Every animation finishes before its section ends (exits start early), so nothing vanishes abruptly at a cut.

Version spec (see references/tastes.md).  * = required.  Times called "beat(s)" are beats of the tempo.
  {"taste": "motion",
   "bpm": 138.5,                      # tempo (default 138) -- or "duration": 13 -> bpm = total beats * 60 / duration
   "grade": "pop", "iris": true,      # shot grade; iris false = no opening iris (hard cut in)
   "cta": "ご予約はリンクから",        # optional, default: the project cta
   "fx": {"flash": 0.45, "punch": 0.12, "sparkle": true},   # hit strength (0 = off); sparkle: bursts + twinkles
   "hook*": {"lines": ["顔に鍼って、", {"text": "どんな感じ？", "style": "gold", "size": 88, "tilt": 2.5,
                                       "font": "goth8", "at": 1.5}],   # at: beats after the section start
             "logo": true | "on_light" | false,   # "on_light" = the dark logo, for bright footage
             "y": 0.46,                           # centre of the caption stack (fraction of the safe zone height)
             "shot*": {"clip": "IMG_2742.MOV", "in": 2.85, "speed": 0.5, "zoom": [1.4, 1.28], "beats": 4}},
   "steps*": [
     {"title*": "カウンセリング", "sub": "お顔をじっくり確認", "label": "STEP 1", "num": "01",
      "badge_y": 0.154,               # badge disc centre (fraction of the safe zone height)
      "chip": true, "chip_at": 2.9,   # shrink into the top-left chip (default: if >= 4 beats, after ~3 beats)
      "shot": {"clip": "IMG_0996.MOV", "in": 3.0, "beats": 4,
               "cam_beats": [[0, 1.36, 0.42, 0.5], [1.9, 1.4, 0.42, 0.5], [2.6, 1.8, 0.4, 0.5], [4, 1.86, 0.4, 0.5]],
               "ring": {"at": [0.39, 0.495], "r": 0.125, "beat": 2.4}}},   # source coords; r in source widths
     {"title": "施術", "sub": "横になって、おまかせ", "beats": 8,           # montage: 8 beats / 4 shots = 2 each
      "shots": [{"clip": "IMG_2722.MOV", "in": 1.0, "speed": 0.5, "zoom": [1.25, 1.15]}, {...}, {...}, {...}]}],
   "closing": {"lines": ["がんばる毎日に、", "ごほうびを。"], "y": 0.27, "shot": {"clip": "...", "beats": 3}},
   "card*": {"beats": 8, "theme": "light", "cta_style": "solid"},
   "bgm": {"style": "pop"}}
Sections hook / step / closing take "shot" (+ "beats") or "shots" (each with "beats", or the section "beats" shared
out evenly) and last at least 2 beats; the card needs enough beats for a 2 s CTA. A shot dict is a normal
make_shot dict plus: "beats"; "cam_beats" (camera keys [beat, zoom, cx, cy] instead of "cam" in seconds); "ring";
"hit" (flash + punch + sparkles on the cut; default: every step shot except the first one of the first step);
"twinkle" (default: the shots of a multi-shot step). Any section can carry "aspect": {"1:1": {"y": 0.34}} overrides.
Lines: "style" dark | gold | light (default: the last line gold, the others dark), "size" (px at 1080 wide, shrunk
to fit), "font" (brand font role or engine font key), "tilt" (degrees). Brand fonts: bold (pills), latin (STEP
label), "numeral" (default didot), "sub" (default goth6).
Hits closer than 1/3 s are refused (no more than three flashes per second).
Audio cues: bgm.bpm / bgm.beats (beats before the card) and sfx at every pop, hit and iris (time = the visible
moment; swish / whoosh peak there). The BGM can only keep its bars on the cuts when the beats before the card are a
multiple of 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from core import engine as E
from core.canvas import ASPECT, FPS, H, W
from core.cards import EndCard
from core.spec import SpecError, make_shot
from core.timeline import Timeline

# ---------------------------------------------------------------- timing constants (beats unless noted)
DEFAULT_BPM = 138.0
BPM_RANGE = (90.0, 190.0)
IRIS_OPEN = 0.9               # opening iris: from the small circle to the full frame
IRIS_LEAD, IRIS_TAIL = 0.5, 0.2  # iris into the card starts / ends this long before / after the card start
IRIS_OVER = 1.15              # circle grows to this x the radius that covers the frame (finishes covering early)
POP = 0.7                     # a pill or badge pops in over this long
PUNCH = 0.6                   # zoom punch after a hit
FLASH_SIGMA, FLASH_SPAN = 0.14, (-0.04, 0.4)
BURST = 1.0                   # sparkle burst length
EXIT = 0.45                   # exit animation length; always finishes before the section boundary
LINE_START = {"hook": (0.5, 1.0), "closing": (0.16, 0.54)}  # first entrance / spacing of caption lines
EXIT_LEAD = {"hook": 0.04, "step": 0.06, "closing": 0.3}     # exits finish this long before the section ends
CARD_AT = {"logo": 0.0, "sub": 0.6, "glint": 0.6, "label": 0.9, "tagline": 1.2, "cta": 2.0}  # beats after card_at
MIN_CTA_SECONDS = 2.0         # ads that cut away sooner lose the click
MIN_SECTION = 2.0             # shortest hook / step / closing (its pop-in and exit have to fit)
MIN_HIT_GAP = 1 / 3           # s between flashes: at most three per second (photosensitivity guideline)
GAP = 36                      # px between stacked caption pills (1080 px wide frame)
SIZES = {"num": 64, "label": 38, "title": 76, "sub": 42, "chip": 36, "disc": 76, "ring": 70}
STEP_Y = {"num": 2, "label": 100, "title": 196, "sub": 300}   # offsets from the disc centre
STEP_REACH = {"top": 80, "bottom": 255, "bottom_sub": 340}    # stack extent around the disc centre

HOOK_KEYS = {"lines", "logo", "shot", "shots", "beats", "y", "aspect", "note"}
STEP_KEYS = {"title", "sub", "label", "num", "chip", "chip_at", "badge_y", "shot", "shots", "beats", "aspect", "note"}
CLOSING_KEYS = {"lines", "shot", "shots", "beats", "y", "aspect", "note"}
CARD_KEYS = {"beats", "theme", "cta_style", "note"}
VERSION_KEYS = {"taste", "bpm", "duration", "grade", "hook", "steps", "closing", "card", "iris", "fx", "cta",
                "bgm", "note"}
FX_DEFAULTS = {"flash": 0.45, "punch": 0.12, "sparkle": True}


def snap(t: float) -> float:
    """Nearest frame time: cuts land on a frame, within 1/60 s of the beat."""
    return round(t * FPS) / FPS


def leave(t: float, t_end: float, dur: float) -> float:
    """Exit envelope: 1 until `dur` before t_end, then a smooth fade that reaches exactly 0 at t_end.
    (An ease-out fade would drop most of its alpha in the last frame or two, which reads as a pop.)"""
    return 1.0 - E.smoothstep((t - (t_end - dur)) / dur)


# ---------------------------------------------------------------- spec -> plan
def _check_keys(d: dict, allowed: set, where: str) -> None:
    for k in d:
        if k not in allowed:
            raise SpecError(f"{where}: unknown key {k!r} (allowed: {', '.join(sorted(allowed))})")


def _num(d: dict, key: str, default, where: str, lo: float | None = None, hi: float | None = None) -> float:
    """d[key] as a float within lo..hi (default None = required)."""
    raw = d.get(key, default)
    if raw is None:
        raise SpecError(f"{where}.{key}: is required")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise SpecError(f"{where}.{key}: number expected, got {raw!r}") from None
    if lo is not None and v < lo:
        raise SpecError(f"{where}.{key}: {v:g} is too small (minimum {lo:g})")
    if hi is not None and v > hi:
        raise SpecError(f"{where}.{key}: {v:g} is too large (maximum {hi:g})")
    return v


def _section(d, allowed: set, where: str) -> dict:
    """Section dict with the override for the current aspect ratio applied."""
    if not isinstance(d, dict):
        raise SpecError(f"{where}: must be an object")
    _check_keys(d, allowed, where)
    over = d.get("aspect", {}).get(ASPECT, {})
    _check_keys(over, allowed - {"aspect"}, f"{where}.aspect[{ASPECT}]")
    return {**{k: v for k, v in d.items() if k != "aspect"}, **over}


def _shot_dicts(sec: dict, where: str) -> list[dict]:
    """The section's shots as dicts that all carry 'beats'."""
    raw = sec.get("shots") or ([sec["shot"]] if "shot" in sec else None)
    if not raw:
        raise SpecError(f"{where}: needs 'shot' or 'shots'")
    shots = [dict(s) for s in raw]
    total = sec.get("beats")
    given = sum(float(s["beats"]) for s in shots if "beats" in s)
    missing = [s for s in shots if "beats" not in s]
    if missing:
        if total is None:
            raise SpecError(f"{where}: give 'beats' on every shot, or on the section to share it out")
        rest = float(total) - given
        if rest <= 0:
            raise SpecError(f"{where}: beats {total} leave nothing for the shots without 'beats'")
        for s in missing:
            s["beats"] = rest / len(missing)
    elif total is not None and abs(float(total) - given) > 1e-6:
        raise SpecError(f"{where}: beats {total} but the shots add up to {given:g}")
    return shots


@dataclass(frozen=True)
class Slot:
    """One shot on the beat grid."""
    name: str
    b0: float
    beats: float
    section: int
    hit: bool
    twinkle: bool
    ring: dict | None
    shot: dict

    @property
    def b1(self) -> float:
        return self.b0 + self.beats


@dataclass(frozen=True)
class Section:
    kind: str  # hook | step | closing
    n: int  # step number (1-based), 0 for hook / closing
    b0: float
    b1: float
    spec: dict


@dataclass(frozen=True)
class Plan:
    bpm: float
    sections: tuple[Section, ...]
    slots: tuple[Slot, ...]
    card_beats: float
    card: dict
    fx: dict

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def lead(self) -> float:
        """Beats before the end card."""
        return self.sections[-1].b1

    @property
    def total(self) -> float:
        return self.lead + self.card_beats


def _ring(shot: dict, beats: float, where: str) -> dict | None:
    r = shot.pop("ring", None)
    if not r:
        return None
    if not isinstance(r, dict):
        raise SpecError(f"{where}.ring: object like {{\"at\": [x, y], \"r\": 0.12, \"beat\": 2}} expected")
    at = r.get("at")
    if not (isinstance(at, (list, tuple)) and len(at) == 2 and all(0 <= float(c) <= 1 for c in at)):
        raise SpecError(f"{where}.ring.at: [x, y] in source coordinates 0..1 expected, got {at!r}")
    return {"at": (float(at[0]), float(at[1])), "r": _num(r, "r", 0.12, f"{where}.ring", 0.02, 0.6),
            "beat": _num(r, "beat", 0.6 * beats, f"{where}.ring", 0.0, beats)}


def _add_section(slots: list, sections: list, kind: str, n: int, sec: dict, b: float, where: str) -> float:
    """Append the section's shots after beat b; returns the beat where the section ends."""
    first, shots = b, _shot_dicts(sec, where)
    length = sum(_num(d, "beats", 0, f"{where}.shots[{j}]", 0.25) for j, d in enumerate(shots))
    if length < MIN_SECTION:
        raise SpecError(f"{where}: {length:g} beats is too short for its animations (minimum {MIN_SECTION:g})")
    multi = len(shots) > 1
    base = f"step{n}" if kind == "step" else kind
    for j, d in enumerate(shots):
        beats = _num(d, "beats", 0, f"{where}.shots[{j}]", 0.25)
        del d["beats"]
        hit_default = kind == "step" and (n > 1 or j > 0)
        slots.append(Slot(f"{base}.{j + 1}" if multi else base, b, beats, len(sections),
                          bool(d.pop("hit", hit_default)), bool(d.pop("twinkle", kind == "step" and multi)),
                          _ring(d, beats, where), d))
        b += beats
    sections.append(Section(kind, n, first, b, sec))
    return b


def parse(v: dict) -> Plan:
    """Version spec -> beat plan (sections, shot slots, tempo). Raises SpecError with a pointer to the fault."""
    _check_keys(v, VERSION_KEYS, "motion")
    for key in ("hook", "steps", "card"):
        if key not in v:
            raise SpecError(f"motion: {key!r} is required (see the docstring of tastes/motion.py)")
    if not isinstance(v["steps"], list) or not v["steps"]:
        raise SpecError("motion: 'steps' must be a list with at least one step")
    sections, slots = [], []
    b = _add_section(slots, sections, "hook", 0, _section(v["hook"], HOOK_KEYS, "hook"), 0.0, "hook")
    for n, raw in enumerate(v["steps"], 1):
        where = f"steps[{n}]"
        sec = _section(raw, STEP_KEYS, where)
        if not sec.get("title"):
            raise SpecError(f"{where}: 'title' is required")
        b = _add_section(slots, sections, "step", n, sec, b, where)
    if "closing" in v:
        b = _add_section(slots, sections, "closing", 0, _section(v["closing"], CLOSING_KEYS, "closing"), b, "closing")
    card = _section(v["card"], CARD_KEYS, "card")
    for key, ok in (("theme", ("light", "dark")), ("cta_style", ("solid", "outline"))):
        if card.get(key, ok[0]) not in ok:
            raise SpecError(f"card.{key}: one of {', '.join(ok)}, got {card[key]!r}")
    card_beats = _num(card, "beats", None, "card", 1.0)
    bpm = _tempo(v, b + card_beats)
    plan = Plan(bpm, tuple(sections), tuple(slots), card_beats, card, _fx(v.get("fx", {})))
    _check_flash_rate(plan)
    dwell = (card_beats - CARD_AT["cta"]) * plan.beat
    if dwell < MIN_CTA_SECONDS:
        need = CARD_AT["cta"] + MIN_CTA_SECONDS / plan.beat
        raise SpecError(f"card.beats {card_beats:g} = {card_beats * plan.beat:.1f} s: the CTA would stay on screen "
                        f"only {max(dwell, 0):.1f} s (needs >= {MIN_CTA_SECONDS:g} s); use at least {need:.1f} beats")
    return plan


def _fx(fx) -> dict:
    """Hit effects with defaults: flash (white peak 0..1), punch (extra zoom 0..0.5), sparkle (bursts + twinkles)."""
    if not isinstance(fx, dict):
        raise SpecError("fx: object like {\"flash\": 0.45, \"punch\": 0.12, \"sparkle\": true} expected")
    _check_keys(fx, set(FX_DEFAULTS), "fx")
    out = {**FX_DEFAULTS, **fx}
    out["flash"], out["punch"] = _num(out, "flash", None, "fx", 0.0, 1.0), _num(out, "punch", None, "fx", 0.0, 0.5)
    out["sparkle"] = bool(out["sparkle"])
    return out


def _check_flash_rate(plan: Plan) -> None:
    """Refuse flashes faster than three per second (hit cuts closer than MIN_HIT_GAP seconds)."""
    if not plan.fx["flash"]:
        return
    starts = [sl.b0 for sl in plan.slots if sl.hit]
    for a, b in zip(starts, starts[1:]):
        if (b - a) * plan.beat < MIN_HIT_GAP - 1e-9:
            raise SpecError(f"hits {(b - a) * plan.beat:.2f} s apart flash more than 3 times per second: use shots "
                            f"of >= {MIN_HIT_GAP / plan.beat:.1f} beats, set \"hit\": false on some, "
                            f"or \"fx\": {{\"flash\": 0}}")


def _tempo(v: dict, total_beats: float) -> float:
    lo, hi = BPM_RANGE
    if "duration" in v:
        dur = _num(v, "duration", None, "motion", 4.0, 90.0)
        bpm = 60.0 * total_beats / dur
        if not lo <= bpm <= hi:
            raise SpecError(f"motion: {total_beats:g} beats in {dur:g} s is {bpm:.0f} BPM (allowed {lo:g}-{hi:g}); "
                            f"about {dur * DEFAULT_BPM / 60:.0f} beats fit {dur:g} s at {DEFAULT_BPM:g} BPM")
        if "bpm" in v and abs(_num(v, "bpm", None, "motion") - bpm) > 0.01 * bpm:
            raise SpecError(f"motion: bpm {v['bpm']:g} and duration {dur:g} s disagree for {total_beats:g} beats "
                            f"(that is {bpm:.1f} BPM): give only one of them")
        return bpm
    return _num(v, "bpm", DEFAULT_BPM, "motion", lo, hi)


def _cam_seconds(d: dict, bt: float) -> dict:
    """cam_beats [[beat, zoom, cx, cy], ...] -> cam in seconds, also inside per-aspect overrides."""
    out = dict(d)
    if "cam_beats" in out:
        keys = out.pop("cam_beats")
        if not all(len(k) == 4 for k in keys):
            raise SpecError(f"cam_beats: keys are [beat, zoom, cx, cy], got {keys}")
        out["cam"] = [[k[0] * bt, *k[1:]] for k in keys]
    if "aspect" in out:
        out["aspect"] = {a: _cam_seconds(o, bt) for a, o in out["aspect"].items()}
    return out


# ---------------------------------------------------------------- overlay building blocks
@dataclass(frozen=True)
class Caption:
    """A pill that pops in at t_in and shrinks away so that it is gone at t_out."""
    img: Image.Image
    x: float
    y: float
    tilt: float
    t_in: float
    t_out: float
    pop: float  # seconds
    fade: float

    def draw(self, frame: np.ndarray, t: float) -> None:
        p = (t - self.t_in) / self.pop
        if p <= 0 or t >= self.t_out:
            return
        out = leave(t, self.t_out, self.fade)
        E.blit_img(frame, self.img, self.x, self.y, scale=max(0.05, E.ease_out_back(p, 2.2)) * (0.85 + 0.15 * out),
                   opacity=E.clamp01(p * 2) * out, angle=self.tilt)


@dataclass(frozen=True)
class Badge:
    """STEP badge: big stack that shrinks into a chip."""
    sprites: dict
    y: float
    t_in: float
    t_end: float
    sub_at: float | None
    chip_at: float | None


def _disc(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    return np.clip(r - np.sqrt(xx ** 2 + yy ** 2) + 0.5, 0, 1)


class Motion(Timeline):
    bgm_style = "pop"

    def __init__(self, ctx) -> None:
        self.plan = parse(ctx.v)
        self.bt = self.plan.beat
        super().__init__(ctx, self._build_shots(ctx))
        self.L, self.b = ctx.layout, ctx.brand
        self.fx = self.plan.fx
        self.open_iris = bool(ctx.v.get("iris", True))
        bt = self.bt
        self.card_at = snap(self.plan.lead * bt)
        self.duration = snap(self.plan.total * bt)
        self.iris_span = (self.card_at - IRIS_LEAD * bt, self.card_at + IRIS_TAIL * bt)
        self.iris_c = (W / 2, H * self.L.pick(tall=0.47, default=0.5))
        stage_w = min(self.L.sw, 960.0)
        self.stage = (self.L.cx - stage_w / 2, self.L.cx + stage_w / 2)  # where sparkles and the chip live
        self.ks = min(1.0, self.L.sh / 979.0, stage_w / 950.0)  # sparkle reach vs the 9:16 design
        self.hits = [s.t0 for s, sl in zip(self.shots, self.plan.slots) if sl.hit]
        self._assets(ctx)
        self._plan_sfx()
        self._check_bgm_grid()
        hook_end = self.time_at(self.plan.sections[0].b1)
        last_in = max((c.t_in for c in self.hook_caps), default=0.0)  # cover: captions in, before their exit
        self.cover_at = snap(min(max(last_in + 0.7 * bt, hook_end - 0.8 * bt), hook_end - 0.5 * bt))

    # -------------------------------------------------------------- setup
    def time_at(self, beat: float) -> float:
        """Time in seconds of a beat on the grid (beat 0 = the first frame)."""
        return beat * self.bt

    def _build_shots(self, ctx) -> list[E.Shot]:
        """Engine shots on the beat grid (cuts snapped to frames); the last one keeps footage under the iris."""
        bt, grade, last = self.bt, ctx.v.get("grade", "pop"), len(self.plan.slots) - 1
        return [make_shot(ctx.project, _cam_seconds(sl.shot, bt), sl.name, snap(sl.b0 * bt), snap(sl.b1 * bt),
                          grade=grade, post=IRIS_TAIL * bt + 0.1 if i == last else 0.0)
                for i, sl in enumerate(self.plan.slots)]

    def _font(self, name: str) -> str:
        key = self.b.fonts.get(name, name)  # brand role or engine font key
        if key not in E.FONT_KEYS:
            raise SpecError(f"unknown font {name!r}: use a brand font role or one of {', '.join(E.FONT_KEYS)}")
        return key

    def _styles(self) -> dict:
        b = self.b
        return {"dark": dict(fg=(255, 255, 255), bg=b.color("night"), alpha=205, size=64, tilt=0.0),
                "gold": dict(fg=b.color("ink"), bg=b.color("accent"), alpha=245, size=88, tilt=2.5),
                "light": dict(fg=b.color("ink"), bg=(255, 255, 255), alpha=235, size=60, tilt=0.0)}

    def _assets(self, ctx) -> None:
        b, L, plan = self.b, self.L, self.plan
        self.gold, self.gold_light = b.color("accent"), b.color("accent_light")
        self.styles = self._styles()
        self.spark = E.sparkle_img(46)
        self.spark_gold = E.sparkle_img(46, self.gold_light)
        self.disc = E.solid_sprite(_disc(SIZES["disc"]), b.color("night"), 0.78)
        self.hook_caps, self.close_caps, self.badges, self.hook_logo = [], [], {}, None
        for si, sec in enumerate(plan.sections):
            if sec.kind == "step":
                self.badges[si] = self._badge(sec)
            elif sec.kind == "hook":
                self.hook_caps = self._captions(sec, "hook")
                self.hook_logo = self._logo(sec)
            else:
                self.close_caps = self._captions(sec, "closing")
        self._twinkles()
        self._card(ctx)

    def _logo(self, sec: Section):
        want = sec.spec.get("logo", True)
        if not want:
            return None
        path = self.b.logo_on_light if want == "on_light" else self.b.logo_on_dark
        img = Image.open(path).convert("RGBA")
        w = int(min(340, 0.36 * self.L.sw))
        img = img.resize((w, max(1, int(img.height * w / img.width))), Image.LANCZOS)
        return img, E.shadow_for(img, 14, 1.6, (252, 248, 240) if want == "on_light" else (10, 8, 6))

    def _captions(self, sec: Section, kind: str) -> list[Caption]:
        raw = sec.spec.get("lines") or []
        if not raw:
            return []
        L, bt = self.L, self.bt
        first, gap = LINE_START[kind]
        pills = []
        for i, ln in enumerate(raw):
            ln = {"text": ln} if isinstance(ln, str) else ln
            if "text" not in ln:
                raise SpecError(f"{kind}.lines[{i}]: 'text' is required")
            style = ln.get("style", "gold" if i == len(raw) - 1 else "dark")
            if style not in self.styles:
                raise SpecError(f"{kind}.lines[{i}].style: one of {', '.join(self.styles)}, got {style!r}")
            st = self.styles[style]
            font = self._font(ln.get("font", "bold"))
            size = E.fit_size([ln["text"]], font, int(ln.get("size", st["size"])), L.text_w - 76, 0.04)
            img = E.pill_img(ln["text"], font, size, st["fg"], st["bg"], bg_alpha=st["alpha"])
            pills.append((img, float(ln.get("tilt", st["tilt"])), float(ln.get("at", first + i * gap))))
        total = sum(p[0].height for p in pills) + GAP * (len(pills) - 1)
        y = L.y(float(sec.spec.get("y", 0.46 if kind == "hook" else 0.27))) - total / 2
        t_out = self.time_at(sec.b1 - EXIT_LEAD[kind])
        caps = []
        for img, tilt, at in pills:
            caps.append(Caption(img, L.cx, y + img.height / 2, tilt, self.time_at(sec.b0 + at), t_out,
                                POP * bt, EXIT * bt))
            y += img.height + GAP
        return caps

    def _badge(self, sec: Section) -> Badge:
        b, L, bt, spec = self.b, self.L, self.bt, sec.spec
        n, title = sec.n, str(spec["title"])
        label = spec.get("label", f"STEP {n}")
        bold = self._font("bold")
        fit = L.text_w - 76
        tsize = E.fit_size([title], bold, SIZES["title"], fit, 0.04)
        white, night = (255, 255, 255), b.color("night")
        sp = {"num": E.text_img(str(spec.get("num", f"{n:02d}")), b.fonts.get("numeral", "didot"), SIZES["num"],
                                white, pad=6),
              "title": E.pill_img(title, bold, tsize, white, night, bg_alpha=215)}
        if label:
            sp["label"] = E.text_img(str(label), b.font("latin"), SIZES["label"], self.gold_light, tracking=0.3, pad=6)
        if spec.get("sub"):
            ssize = E.fit_size([str(spec["sub"])], b.fonts.get("sub", "goth6"), SIZES["sub"], fit, 0.04)
            sp["sub"] = E.pill_img(str(spec["sub"]), b.fonts.get("sub", "goth6"), ssize, b.color("ink"), white,
                                   bg_alpha=235)
        length = sec.b1 - sec.b0
        use_chip = spec.get("chip", length >= 4)
        chip_at = None
        if use_chip:
            at = _num(spec, "chip_at", min(3.2, 0.725 * length), f"steps[{n}]", 0.5, length - 0.5)
            chip_at = self.time_at(sec.b0 + at)
            text = f"{label}  {title}" if label else title
            sp["chip"] = E.pill_img(text, bold, SIZES["chip"], white, night, bg_alpha=200)
        reach = STEP_REACH["bottom_sub" if "sub" in sp else "bottom"]
        y = L.y(float(spec.get("badge_y", 0.154)))
        y = min(max(y, L.y0 + STEP_REACH["top"]), L.y1 - reach)
        return Badge(sp, y, self.time_at(sec.b0 + 0.1), self.time_at(sec.b1 - EXIT_LEAD["step"]),
                     self.time_at(sec.b0 + 0.4) if "sub" in sp else None, chip_at)

    def _twinkles(self) -> None:
        """Fixed random positions/phases of the twinkles of each multi-shot step shot."""
        L, (s0, s1) = self.L, self.stage
        w = s1 - s0
        self.tw = {}
        for i, sl in enumerate(self.plan.slots):
            if sl.twinkle:
                rng = np.random.default_rng(40 + i)
                self.tw[i] = [(rng.uniform(s0 + 0.12 * w, s1 - 0.12 * w), rng.uniform(L.y(0.36), L.y(1.0)),
                               rng.uniform(0, 1)) for _ in range(6)]
        rng = np.random.default_rng(9)
        self.card_tw = [(rng.uniform(s0 + 0.06 * w, s1 - 0.06 * w), rng.uniform(L.y(0.154), L.y(1.0)),
                         rng.uniform(0, 1)) for _ in range(10)]

    def _card(self, ctx) -> None:
        """End card; its logo and CTA are drawn by this taste (faster letter pops, pulsing CTA)."""
        bt, spec = self.bt, self.plan.card
        times = {k: v * bt for k, v in CARD_AT.items()}
        times["logo"] = times["cta"] = 1e6
        self.card = EndCard(ctx, theme=spec.get("theme", "light"), cta_style=spec.get("cta_style", "solid"),
                            logo_style="pop", note=spec.get("note"), particles=False, times=times)

    # -------------------------------------------------------------- audio cues
    def _check_bgm_grid(self) -> None:
        """Warn when bgm.py would fit a different tempo than the cut grid (it prefers whole bars near the card)."""
        try:  # advisory only: never let the optional music module break a render
            from core.composer.plan import fit_tempo
            beats, bpm = fit_tempo(self.card_at, self.plan.bpm)
        except Exception:  # noqa: BLE001
            return
        lead = self.plan.lead
        if beats != lead:
            print(f"[motion] warning: the cuts sit on {lead:g} beats at {self.plan.bpm:.1f} BPM before the card, "
                  f"the BGM would fit {beats} beats at {bpm:.1f} BPM and drift off them; use a multiple of 4 beats "
                  f"before the card (or let bgm.py honour cues bgm.beats)", flush=True)

    def _plan_sfx(self) -> None:
        """SFX at the visible moments (level 0..1 relative). swish / whoosh peak at the given time."""
        add, bt, plan = self.sfx.append, self.bt, self.plan
        if self.open_iris:
            add([round(IRIS_OPEN * bt, 3), "swish", 0.7])
        for cap in self.hook_caps + self.close_caps:
            add([round(cap.t_in + 0.05, 3), "pop", 0.8])
        for bd in self.badges.values():
            add([round(bd.t_in + 0.06, 3), "pop", 0.9])
            if bd.sub_at is not None:
                add([round(bd.sub_at + 0.05, 3), "pop", 0.6])
            if bd.chip_at is not None:
                add([round(bd.chip_at, 3), "swish", 0.5])
        run = False
        for s, sl in zip(self.shots, plan.slots):
            if sl.hit:
                if not run:
                    add([round(s.t0, 3), "whoosh", 0.7])
                add([round(s.t0, 3), "impact", 0.5 if run else 0.7])
            run = sl.hit
            if sl.ring:
                add([round(s.t0 + self.time_at(sl.ring["beat"]), 3), "chime", 0.6])
        add([round(self.iris_span[1], 3), "whoosh", 0.8])
        add([round(self.card_at, 3), "pop", 0.8])
        add([round(self.card_at + self.time_at(CARD_AT["cta"]) + 0.05, 3), "pop", 0.8])
        self.sfx.sort(key=lambda e: e[0])

    def cues(self) -> dict:
        out = super().cues()
        out["bgm"] = {"bpm": round(self.plan.bpm, 3), "beats": self.plan.lead, **out["bgm"]}
        return out

    # -------------------------------------------------------------- drawing
    def _punch(self, i: int, t: float) -> float:
        if not self.plan.slots[i].hit:
            return 1.0
        p = (t - self.shots[i].t0) / (PUNCH * self.bt)
        return 1 + self.fx["punch"] * (1 - E.ease_out_cubic(p)) if 0 <= p < 1 else 1.0

    def _src_point(self, i: int, t: float, x: float, y: float, extra: float) -> tuple[float, float, float]:
        """Source point -> output px plus the px per source width at that moment."""
        win = E.crop_window(self.sources[i], t, extra)
        vx, vy = win.to_view(x, y)
        return vx + (W - win.vw) // 2, vy + (H - win.vh) // 2, win.scale()

    def _burst(self, frame: np.ndarray, t: float, t0: float, cx: float, cy: float, n: int = 9) -> None:
        p = (t - t0) / (BURST * self.bt)
        if not self.fx["sparkle"] or not 0 <= p < 1:
            return
        e = E.ease_out_cubic(p)
        for k in range(n):
            ang = k / n * 2 * math.pi + t0
            dist = (90 + 430 * e * (0.7 + 0.3 * math.sin(k * 2.3))) * self.ks
            sc = (0.5 + 0.7 * (1 - e)) * (0.7 + 0.3 * math.cos(k * 1.7)) * max(self.ks, 0.8)
            E.blit_img(frame, self.spark_gold if k % 2 else self.spark, cx + math.cos(ang) * dist,
                       cy + math.sin(ang) * dist * 1.1, scale=sc, opacity=1 - e)

    def _twinkle(self, frame: np.ndarray, t: float, i: int) -> None:
        if i not in self.tw or not self.fx["sparkle"]:
            return
        for x, y, ph in self.tw[i]:
            p = ((t - self.shots[i].t0) / (1.4 * self.bt) + ph) % 1.0
            env = math.sin(math.pi * p)
            E.blit_img(frame, self.spark, x, y, scale=0.25 + 0.35 * env, opacity=0.85 * env)

    def _flash(self, frame: np.ndarray, t: float) -> np.ndarray:
        if not self.fx["flash"]:
            return frame
        bt = self.bt
        for c in self.hits:
            if c + FLASH_SPAN[0] * bt <= t < c + FLASH_SPAN[1] * bt:
                f = self.fx["flash"] * math.exp(-((t - c) / (FLASH_SIGMA * bt)) ** 2)
                frame = E.screen(frame, np.full((1, 1, 3), f, np.float32))
        return frame

    def _iris_reach(self) -> float:
        """Final iris radius (a little beyond the radius that covers the whole frame)."""
        cx, cy = self.iris_c
        return (math.hypot(max(cx, W - cx), max(cy, H - cy)) + 8) * IRIS_OVER

    def _open_iris(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Hook: the footage shows through a circle that grows from a small gold-rimmed disc over the card paper."""
        if not self.open_iris or t >= IRIS_OPEN * self.bt:
            return frame
        r0 = 0.14 * min(W, H)
        r = r0 + (self._iris_reach() - r0) * E.ease_in_cubic(t / (IRIS_OPEN * self.bt))
        return E.iris(self.card.bg, frame, *self.iris_c, r, ring=(7, self.gold))

    def _hook(self, frame: np.ndarray, t: float) -> None:
        sec = self.plan.sections[0]
        if self.hook_logo is not None and t < self.time_at(sec.b1):
            logo, shadow = self.hook_logo
            e = (E.ease_out_cubic((t - self.time_at(0.6)) / (0.8 * self.bt))
                 * leave(t, self.time_at(sec.b1 - 0.1), 0.4 * self.bt))
            y = self.L.y(0.0676)
            E.blit_img(frame, shadow, self.L.cx, y, opacity=e * 0.9)
            E.blit_img(frame, logo, self.L.cx, y, opacity=e)
        for cap in self.hook_caps:
            cap.draw(frame, t)

    def _ring(self, frame: np.ndarray, t: float, i: int) -> None:
        sl, s = self.plan.slots[i], self.shots[i]
        if not sl.ring:
            return
        t_in, t_end = s.t0 + self.time_at(sl.ring["beat"]), s.t1 - 0.04 * self.bt
        if not t_in <= t < t_end:
            return
        x, y, scale = self._src_point(i, t, *sl.ring["at"], self._punch(i, t))
        ring = E.ring_sprite(sl.ring["r"] * scale, 6, self.gold,
                             progress=E.ease_out_cubic((t - t_in) / (0.8 * self.bt)))
        E.blit(frame, ring, x - ring.shape[1] / 2, y - ring.shape[0] / 2, leave(t, t_end, 0.4 * self.bt))
        self._burst(frame, t, t_in + 0.3 * self.bt, x, y, n=7)

    def _step(self, frame: np.ndarray, t: float, bd: Badge) -> None:
        if not bd.t_in <= t < bd.t_end:
            return
        bt, L, sp = self.bt, self.L, bd.sprites
        out = leave(t, bd.t_end, EXIT * bt)
        chip_p = E.ease_in_out_sine((t - bd.chip_at) / (POP * bt)) if bd.chip_at is not None else 0.0
        big = 1 - chip_p
        if big > 0.01:
            p, cx, y = (t - bd.t_in) / (POP * bt), L.cx, bd.y
            E.blit(frame, self.disc, cx - self.disc.shape[1] / 2, y - self.disc.shape[0] / 2,
                   0.9 * E.clamp01(p * 2) * out * big)
            ring = E.ring_sprite(SIZES["ring"], 5, self.gold, progress=E.ease_out_cubic((t - bd.t_in) / (0.9 * bt)))
            E.blit(frame, ring, cx - ring.shape[1] / 2, y - ring.shape[0] / 2, out * big)
            E.blit_img(frame, sp["num"], cx, y + STEP_Y["num"], scale=max(0.05, E.ease_out_back(p, 2.0)),
                       opacity=E.clamp01(p * 2) * out * big)
            if "label" in sp:
                E.blit_img(frame, sp["label"], cx, y + STEP_Y["label"],
                           opacity=E.ease_out_cubic((t - bd.t_in - 0.2 * bt) / (0.6 * bt)) * out * big)
            pt = (t - bd.t_in - 0.2 * bt) / (POP * bt)
            E.blit_img(frame, sp["title"], cx, y + STEP_Y["title"],
                       scale=max(0.05, E.ease_out_back(pt, 2.0) * (1 - 0.45 * chip_p)),
                       opacity=E.clamp01(pt * 2) * out * big)
            if bd.sub_at is not None:
                ps = (t - bd.sub_at) / (0.6 * bt)
                E.blit_img(frame, sp["sub"], cx, y + STEP_Y["sub"], scale=max(0.05, E.ease_out_back(ps, 1.6)),
                           opacity=E.clamp01(ps * 2) * out * big)
        if chip_p > 0:
            chip = sp["chip"]  # top left of the stage (centred over the picture column in 16:9)
            x0 = L.cx - chip.width / 2 if L.wide else self.stage[0] + 27
            E.blit_img(frame, chip, x0 + chip.width / 2 + (1 - chip_p) * 60, L.y(0.0505), opacity=chip_p * out)

    def _card_logo(self, frame: np.ndarray, t: float) -> None:
        c, bt, t0 = self.card, self.bt, self.card_at
        c.logo.draw(frame, t, self.L.cx, c.pos["logo"] - c.logo.h / 2, t0 + CARD_AT["logo"] * bt, style="pop",
                    gap=0.1 * bt, sub_at=t0 + CARD_AT["sub"] * bt, glint_at=t0 + CARD_AT["glint"] * bt)

    def _cta(self, frame: np.ndarray, t: float) -> None:
        dt = (t - self.card_at) / self.bt - CARD_AT["cta"]
        p = dt / POP
        if p <= 0:
            return
        pulse = 1 + 0.035 * max(0.0, math.sin(math.pi * (dt - 1.0))) if dt > 1.0 else 1.0  # one pulse per 2 beats
        c = self.card
        E.blit_img(frame, c.cta, self.L.cx, c.pos["cta"], scale=max(0.05, E.ease_out_back(p, 2.0)) * pulse,
                   opacity=E.clamp01(p * 2))

    def _card_frame(self, t: float) -> np.ndarray:
        c, t0, bt = self.card, self.card_at, self.bt
        frame = c.background(t, t0)
        for x, y, ph in self.card_tw if self.fx["sparkle"] else ():  # floating gold twinkles
            p = ((t - t0) / (2.8 * bt) + ph) % 1
            env = math.sin(math.pi * p) * E.clamp01((t - t0 - 0.4 * bt) / (0.8 * bt))
            E.blit_img(frame, self.spark_gold, x, y - 30 * self.ks * p, scale=0.18 + 0.2 * env, opacity=0.55 * env)
        c.draw_on(frame, t, t0)
        self._card_logo(frame, t)
        self._cta(frame, t)
        return frame

    def frame(self, t: float) -> np.ndarray:
        if t >= self.iris_span[1]:
            return self._card_frame(t)
        i = self.index_at(t)
        frame = self._open_iris(self.shot_frame(i, t, self._punch(i, t)), t)
        frame = self._flash(frame, t)
        self._hook(frame, t)
        self._ring(frame, t, i)
        for bd in self.badges.values():
            self._step(frame, t, bd)
        for c in self.hits:
            self._burst(frame, t, c, self.iris_c[0], self.iris_c[1])
        self._twinkle(frame, t, i)
        for cap in self.close_caps:
            cap.draw(frame, t)
        t_a, t_b = self.iris_span
        if t >= t_a:  # the end card opens as a circle over the last shot
            r = E.ease_in_cubic((t - t_a) / (t_b - t_a)) * self._iris_reach()
            frame = E.iris(frame, self._card_frame(t), *self.iris_c, r, ring=(7, self.gold))
        return frame


def build(ctx) -> Timeline:
    return Motion(ctx)
