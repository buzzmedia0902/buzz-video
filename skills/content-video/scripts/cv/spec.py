"""edit.json loading and validation (schema: references/edit-spec.md).

  {"project": "...", "tone": "trust", "brand": {...}, "sources": {"talk": {"path": "..."}},
   "cues": "cues.json", "speakers": {...}, "look": {...},
   "versions": {"short": {"format": "short", "aspect": "9:16", "title": "...", "parts": [...], "overlays": [...]}}}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

FORMATS = ("youtube", "short", "clip", "vlog")
TONES = ("trust", "friendly", "variety", "wa", "minimal")
TONE_JA = {"trust": "信頼", "friendly": "親しみ", "variety": "バラエティ", "wa": "和", "minimal": "ミニマル"}
FORMAT_JA = {"youtube": "YouTube", "short": "ショート", "clip": "切り抜き", "vlog": "Vlog"}
DEFAULT_ASPECT = {"youtube": "16:9", "short": "9:16", "clip": "9:16", "vlog": "9:16"}
PART_KINDS = ("talk", "clip", "card")
CARD_KINDS = ("title", "end", "eyecatch", "text")
OVERLAY_TYPES = ("chapter", "card", "emphasis", "name", "caption", "credit", "title_band", "note")


class SpecError(ValueError):
    pass


_WARNED: set = set()


def warn(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[spec] warning: {msg}", flush=True)


@dataclass
class Project:
    path: str
    root: str
    name: str
    tone: str
    brand: dict
    sources: dict          # id -> {"path": abs path, "rot": int, ...}
    speakers: dict
    cues_path: str | None
    look: dict
    out_dir: str
    work_dir: str
    versions: dict
    raw: dict = field(repr=False, default_factory=dict)

    def version(self, name: str) -> dict:
        if name not in self.versions:
            raise SpecError(f"version {name!r} not in edit.json (have: {', '.join(self.versions)})")
        return self.versions[name]

    def src_path(self, sid: str) -> str:
        if sid not in self.sources:
            raise SpecError(f"source {sid!r} not in 'sources' (have: {', '.join(self.sources)})")
        return self.sources[sid]["path"]

    def work(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


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
    for key in ("project", "sources", "versions"):
        if key not in raw:
            raise SpecError(f"edit.json: missing {key!r}")
    tone = raw.get("tone", "trust")
    if tone not in TONES:
        raise SpecError(f"tone must be one of {', '.join(TONES)} (got {tone!r})")
    sources = {}
    for sid, s in raw["sources"].items():
        s = {"path": s} if isinstance(s, str) else dict(s)
        s["path"] = _abs(root, s["path"])
        if not os.path.exists(s["path"]):
            raise SpecError(f"sources.{sid}: file not found {s['path']}")
        s["rot"] = int(s.get("rot", 0))
        if s["rot"] not in (0, 90, 180, 270):
            raise SpecError(f"sources.{sid}.rot must be 0 / 90 / 180 / 270")
        sources[sid] = s
    brand = dict(raw.get("brand", {}))
    if brand.get("logo"):
        brand["logo"] = _abs(root, brand["logo"])
        if not os.path.exists(brand["logo"]):
            raise SpecError(f"brand.logo not found: {brand['logo']}")
    versions = raw["versions"]
    for name, v in versions.items():
        _check_version(name, v, sources)
    cues = raw.get("cues")
    return Project(path=path, root=root, name=raw["project"], tone=tone, brand=brand, sources=sources,
                   speakers=raw.get("speakers", {}), cues_path=_abs(root, cues) if cues else None,
                   look=raw.get("look", {}), out_dir=_abs(root, raw.get("out_dir", "納品")),
                   work_dir=_abs(root, raw.get("work_dir", "work")), versions=versions, raw=raw)


def _check_version(name: str, v: dict, sources: dict) -> None:
    where = f"versions.{name}"
    fmt = v.get("format")
    if fmt not in FORMATS:
        raise SpecError(f"{where}.format must be one of {', '.join(FORMATS)}")
    if v.get("tone", "trust") not in TONES:
        raise SpecError(f"{where}.tone must be one of {', '.join(TONES)}")
    if v.get("aspect", DEFAULT_ASPECT[fmt]) not in ("9:16", "4:5", "1:1", "16:9"):
        raise SpecError(f"{where}.aspect must be 9:16 / 4:5 / 1:1 / 16:9")
    parts = v.get("parts")
    if not isinstance(parts, list) or not parts:
        raise SpecError(f"{where}.parts: a non-empty list of talk / clip / card parts is required")
    for i, p in enumerate(parts):
        w = f"{where}.parts[{i}]"
        kind = p.get("kind")
        if kind not in PART_KINDS:
            raise SpecError(f"{w}.kind must be one of {', '.join(PART_KINDS)}")
        if kind in ("talk", "clip"):
            if p.get("src") not in sources:
                raise SpecError(f"{w}.src must name a source ({', '.join(sources)})")
            if "from" in p and "to" in p and float(p["to"]) <= float(p["from"]):
                raise SpecError(f"{w}: 'to' must be after 'from'")
            for r in p.get("remove", []):
                if len(r) != 2 or float(r[1]) <= float(r[0]):
                    raise SpecError(f"{w}.remove: ranges are [start, end] source seconds")
        else:
            if p.get("card") not in CARD_KINDS:
                raise SpecError(f"{w}.card must be one of {', '.join(CARD_KINDS)}")
            if float(p.get("dur", 0)) <= 0:
                raise SpecError(f"{w}.dur (seconds) is required for a card")
    for i, o in enumerate(v.get("overlays", [])):
        w = f"{where}.overlays[{i}]"
        if o.get("type") not in OVERLAY_TYPES:
            raise SpecError(f"{w}.type must be one of {', '.join(OVERLAY_TYPES)}")
        if "out" not in o and ("src" not in o or "at" not in o) and o["type"] not in ("title_band", "credit"):
            raise SpecError(f"{w}: give 'src' + 'at' (source seconds) or 'out' (output seconds)")
        if "src" in o and o["src"] not in sources:
            raise SpecError(f"{w}.src {o['src']!r} is not a source")
