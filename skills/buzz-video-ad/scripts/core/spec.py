"""project.json loading, validation and Shot construction (schema: references/spec.md).

Shot dicts (all tastes):
  {"clip": "IMG_2722.MOV", "in": 0.85, "dur": 2.0, "speed": 0.5, "zoom": [1.15, 1.22],
   "center": [0.5, 0.42] | [[0.5, 0.42], [0.5, 0.40]], "grade": "beauty", "fit": "auto",
   "patches": [[cx, cy, rx, ry(, strength)] | {"track": [[src_t, cx, cy], ...], "r": [rx, ry]}],
   "focus": [cx, cy, rx, ry(, strength)], "soft_below": 0.83, "shake": 3, "exposure": 1.0, "tint": 0.0,
   "aspect": {"1:1": {"center": [0.5, 0.35]}}}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

from core import engine as E
from core.canvas import ASPECT, FPS, H, W

TASTES = ("standard", "motion", "cinematic", "vlog", "chat")
TASTE_JA = {"standard": "スタンダード", "motion": "モーション", "cinematic": "シネマティック", "vlog": "Vlog",
            "chat": "トーク"}
DEFAULT_COLORS = {"accent": "#C9A86A", "ink": "#24201C", "paper": "#F7F3EC", "night": "#1C1815"}
DEFAULT_FONTS = {"title": "midashi", "body": "maru", "latin": "optima_b", "hand": "klee", "vertical": "hmin6",
                 "ui": "goth4", "bold": "goth8", "serif": "mincho"}
PATCH_STRENGTH = 16.0


class SpecError(ValueError):
    pass


_WARNED: set = set()


def _warn(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[spec] warning: {msg}", flush=True)


@dataclass(frozen=True)
class Brand:
    name: str
    logo: str  # original file
    label: str
    tagline: str
    colors: dict
    fonts: dict

    def _stem(self) -> str:
        return os.path.splitext(self.logo)[0]

    @property
    def logo_on_light(self) -> str:
        return self._stem() + "_on_light.png"

    @property
    def logo_on_dark(self) -> str:
        return self._stem() + "_on_dark.png"

    @property
    def parts(self) -> str:
        return self._stem() + "_parts.json"

    @property
    def pale(self) -> bool:
        """logo_prep found the logo hard to read on light backgrounds (contrast < 3:1 for most of its ink)."""
        with open(self.parts, encoding="utf-8") as f:
            return json.load(f).get("low_contrast", {}).get("on_paper", 0.0) >= 0.8

    def color(self, name: str) -> tuple[int, int, int]:
        c = self.colors
        derived = {"accent_deep": lambda: E.mix(c["accent"], (0, 0, 0), 0.22),
                   "accent_light": lambda: E.mix(c["accent"], (255, 255, 255), 0.35)}
        if name in c:
            return E.rgb(c[name])
        if name in derived:
            return derived[name]()
        raise SpecError(f"brand.colors: unknown colour {name!r}")

    def font(self, role: str) -> str:
        """Font key for a role (title, body, latin, hand, vertical, ui, bold, serif); engine keys pass through."""
        if role in self.fonts:
            return self.fonts[role]
        if role in DEFAULT_FONTS:
            return DEFAULT_FONTS[role]
        if role in E.FONT_KEYS:
            return role
        _warn(f"unknown font role {role!r}: using goth6 (roles: {', '.join(DEFAULT_FONTS)})")
        return "goth6"


@dataclass(frozen=True)
class Project:
    path: str
    root: str
    name: str
    footage_dir: str
    out_dir: str
    work_dir: str
    brand: Brand
    cta: str
    platform: str
    clips: dict
    versions: dict
    raw: dict = field(repr=False, default_factory=dict)

    def clip_path(self, clip: str) -> str:
        p = os.path.expanduser(clip)
        return p if os.path.isabs(p) else os.path.join(self.footage_dir, clip)

    def version(self, name: str) -> dict:
        if name not in self.versions:
            raise SpecError(f"version {name!r} not in project.json (have: {', '.join(self.versions)})")
        return self.versions[name]


def _abs(root: str, p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def load(path: str) -> Project:
    path = os.path.abspath(os.path.expanduser(path))
    root = os.path.dirname(path)
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: JSON error line {e.lineno} col {e.colno}: {e.msg}") from e
    for key in ("project", "footage_dir", "brand", "versions"):
        if key not in raw:
            raise SpecError(f"project.json: missing {key!r}")
    b = raw["brand"]
    if "logo" not in b:
        raise SpecError("brand.logo: path of the client logo is required")
    brand = Brand(name=b.get("name", raw["project"]), logo=_abs(root, b["logo"]), label=b.get("label", ""),
                  tagline=b.get("tagline", ""), colors={**DEFAULT_COLORS, **b.get("colors", {})},
                  fonts={**DEFAULT_FONTS, **b.get("fonts", {})})
    for p in (brand.logo_on_light, brand.logo_on_dark, brand.parts):
        if not os.path.exists(p):
            raise SpecError(f"{p} missing: run  logo_prep.py {brand.logo}  first")
    footage = _abs(root, raw["footage_dir"])
    if not os.path.isdir(footage):
        raise SpecError(f"footage_dir not found: {footage}")
    versions = raw["versions"]
    for name, v in versions.items():
        if v.get("taste") not in TASTES:
            raise SpecError(f"versions.{name}.taste must be one of {TASTES}")
    return Project(path=path, root=root, name=raw["project"], footage_dir=footage,
                   out_dir=_abs(root, raw.get("out_dir", "納品")), work_dir=_abs(root, raw.get("work_dir", "work")),
                   brand=brand, cta=raw.get("cta", "ご予約はリンクから"), platform=raw.get("platform", "meta"),
                   clips=raw.get("clips", {}), versions=versions, raw=raw)


# ---------------------------------------------------------------- shots
def _patch(p, where: str) -> tuple:
    if isinstance(p, dict):
        keys = tuple(tuple(float(v) for v in k) for k in p["track"])
        rx, ry = (float(v) for v in p["r"])
        return keys, rx, ry, float(p.get("strength", PATCH_STRENGTH))
    if len(p) not in (4, 5):
        raise SpecError(f"{where}: patch needs [cx, cy, rx, ry(, strength)], got {p}")
    vals = [float(v) for v in p]
    return tuple(vals) if len(vals) == 5 else (*vals, PATCH_STRENGTH)


def _focus(f, where: str) -> tuple | None:
    if not f:
        return None
    if len(f) not in (4, 5):
        raise SpecError(f"{where}: focus needs [cx, cy, rx, ry(, strength)]")
    vals = [float(v) for v in f]
    return tuple(vals) if len(vals) == 5 else (*vals, 16.0)


def _pair(v, where: str) -> tuple[tuple, tuple]:
    """center: [x, y] or [[x0, y0], [x1, y1]] -> (start, end)."""
    if isinstance(v[0], (int, float)):
        return (float(v[0]), float(v[1])), (float(v[0]), float(v[1]))
    if len(v) != 2:
        raise SpecError(f"{where}: center must be [x, y] or [[x0, y0], [x1, y1]]")
    return tuple(float(x) for x in v[0]), tuple(float(x) for x in v[1])


def auto_fit(path: str, rot: int, where: str = "") -> str:
    """cover unless it would show less than 45 % of the source (e.g. vertical clip in 16:9) -> blur, with a
    warning: a pillar-box on a blurred copy is weak in an ad, a per-aspect cover crop usually reads better."""
    info = E.probe(path)
    sw, sh = (info["h"], info["w"]) if rot in (90, 270) else (info["w"], info["h"])
    c = max(W / sw, H / sh)
    shown = (W / c) * (H / c) / (sw * sh)
    if shown >= 0.45:
        return "cover"
    sharp = ("the source is sharp enough" if c <= 1.05 else
             f"cover enlarges the source x{c:.2f}{', a little soft' if c <= 1.5 else ', soft'}")
    _warn(f"{where}: {ASPECT} shows it small on a blurred copy (fit auto -> blur). Usually better: "
          f'"aspect": {{"{ASPECT}": {{"fit": "cover", "center": [x, y]}}}} ({sharp})')
    return "blur"


def make_shot(project: Project, d: dict, name: str, t0: float, t1: float, **defaults) -> E.Shot:
    """Build an engine Shot from a shot dict (taste defaults < clip settings < shot < shot.aspect[ASPECT])."""
    where = f"shot {name} ({d.get('clip')})"
    if "clip" not in d:
        raise SpecError(f"{where}: 'clip' is required")
    clip = project.clips.get(d["clip"], {})
    merged = {**defaults, **{k: v for k, v in clip.items() if k not in ("patches", "note")}, **d,
              **d.get("aspect", {}).get(ASPECT, {})}
    path = project.clip_path(d["clip"])
    if not os.path.exists(path):
        raise SpecError(f"{where}: file not found {path}")
    rot = int(merged.get("rot", 0))
    speed = float(merged.get("speed", 1.0))
    src_in = float(merged.get("in", 0.0))
    pre, post = float(merged.get("pre", 0.0)), float(merged.get("post", 0.0))
    info = E.probe(path)
    if not info["image"] and info["fps"] * speed < FPS * 0.95:
        _warn(f"{where}: x{speed} slow motion of a {info['fps']:.0f} fps clip repeats frames (stutter) - "
              f"use a 60 fps clip or speed >= {FPS / info['fps']:.2f}")
    start, need = src_in - pre * speed, src_in + (t1 - t0 + post) * speed
    if start < -0.01 or need > info["duration"] + 0.05:
        why = []
        if start < -0.01 and pre:
            why.append(f"the transition into the shot uses {pre * speed:.2f} s of source before 'in', "
                       f"so 'in' must be >= {pre * speed:.2f}")
        if need > info["duration"] + 0.05 and post:
            why.append(f"the transition out uses {post * speed:.2f} s of source after the shot")
        raise SpecError(f"{where}: needs source {start:.2f}-{need:.2f}s but clip is {info['duration']:.2f}s "
                        f"(adjust in / dur / speed{''.join('; ' + w for w in why)})")
    z = merged.get("zoom", 1.0)
    z0, z1 = (float(z), float(z)) if isinstance(z, (int, float)) else (float(z[0]), float(z[1]))
    if min(z0, z1) < 1.0:
        raise SpecError(f"{where}: zoom must be >= 1.0 (1.0 = source just fills the frame)")
    c0, c1 = _pair(merged.get("center", [0.5, 0.5]), where)
    cam = (tuple(tuple(float(x) for x in k) for k in merged["cam"]) if "cam" in merged
           else E.cam_lin(z0, z1, c0, c1, t1 - t0, pre, post))
    per_aspect = d.get("aspect", {}).get(ASPECT, {})
    patches = tuple(_patch(p, where) for p in [*clip.get("patches", []), *d.get("patches", []),
                                                *per_aspect.get("patches", [])])
    fit = merged.get("fit", "auto")
    if fit not in ("auto", "cover", "blur"):
        raise SpecError(f"{where}: fit must be auto | cover | blur")
    if merged.get("grade", "natural") not in E.PRESETS:
        raise SpecError(f"{where}: grade {merged.get('grade')!r} unknown (use {' | '.join(E.PRESETS)})")
    return E.Shot(name=name, path=path, t0=t0, t1=t1, src_in=src_in, speed=speed, rot=rot, cam=cam,
                  fit=auto_fit(path, rot, where) if fit == "auto" else fit, grade=merged.get("grade", "natural"),
                  exposure=float(merged.get("exposure", 1.0)), tint=float(merged.get("tint", 0.0)),
                  patches=patches, focus=_focus(merged.get("focus"), where),
                  soft_below=float(merged.get("soft_below", 0.0)), pre=pre, post=post,
                  shake=float(merged.get("shake", 0.0)), hdr=bool(merged.get("hdr", False)),
                  flash_guard=bool(merged.get("flash_guard", False)))


def sequence(project: Project, shots: list[dict], start: float = 0.0, **defaults) -> list[E.Shot]:
    """Back-to-back shots from their 'dur' fields."""
    out, t = [], start
    for i, d in enumerate(shots):
        if "dur" not in d:
            raise SpecError(f"shot {i + 1}: 'dur' (output seconds) is required")
        out.append(make_shot(project, d, d.get("name", f"s{i + 1}"), t, t + float(d["dur"]), **defaults))
        t += float(d["dur"])
    return out


@dataclass
class Ctx:
    """Everything a taste template needs."""
    project: Project
    name: str
    v: dict
    layout: object
    still: bool = False

    @property
    def brand(self) -> Brand:
        """Project brand; a version may override font roles with "fonts": {"latin": "optima"}."""
        b = self.project.brand
        if "fonts" in self.v:
            return replace(b, fonts={**b.fonts, **self.v["fonts"]})
        return b

    @property
    def cta(self) -> str:
        return self.v.get("cta", self.project.cta)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.project.work_dir, self.name, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
