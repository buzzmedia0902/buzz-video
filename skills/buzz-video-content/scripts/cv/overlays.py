"""Overlay events of a version in output seconds: the authored "overlays" plus the format's defaults.

  chapter    banner when a topic starts                {"text"}
  card       explanation panel                         {"title", "items": [...], "dur"} (or "big": "3倍")
  emphasis   big impact word                           {"text"}
  name       name plate                                {"text", "sub"}
  caption    small place / date / feeling line (vlog)  {"text", "sub"}
  note       ※ small note                               {"text"}
  title_band persistent title (short / clip)           {"text"}        (default: the version "title")
  credit     source credit (clip)                      {"text"}        (default: the version "credit")
Timing: "src" + "at" (source seconds of that source) or "out" (output seconds); "dur" in output seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cv.spec import warn

DEFAULT_DUR = {"chapter": 2.4, "card": 4.0, "emphasis": 1.4, "name": 3.2, "caption": 2.8, "note": 3.0}
PERSISTENT = ("title_band", "credit")


@dataclass
class Ov:
    type: str
    t0: float
    t1: float
    d: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.d.get("text", ""))


def content_span(tl) -> tuple[float, float]:
    """Output span of the talk / clip parts (persistent overlays live here, not over title / end cards)."""
    spans = tl.talk_spans()
    return (spans[0][0], spans[-1][1]) if spans else (0.0, tl.duration)


def build(version: dict, tl) -> list[Ov]:
    fmt = version["format"]
    c0, c1 = content_span(tl)
    out = []
    for i, o in enumerate(version.get("overlays", [])):
        kind = o["type"]
        if kind in PERSISTENT:
            out.append(Ov(kind, float(o.get("out", c0)), float(o.get("until", c1)), o))
            continue
        if "out" in o:
            t0 = float(o["out"])
        else:  # authored against the talk: prefer a talk part over a cold open replaying the same words
            talk = [i for i, p in enumerate(version["parts"]) if p["kind"] == "talk" and p.get("src") == o["src"]]
            t0 = None
            for part in talk + [None]:
                t0 = tl.src_to_out(o["src"], float(o["at"]), "next", part)
                if t0 is not None:
                    break
        if t0 is None:
            warn(f"overlays[{i}] ({kind} {o.get('text', o.get('title', ''))!r}): its source time is cut out of "
                 f"this version, skipped")
            continue
        dur = float(o.get("dur", DEFAULT_DUR.get(kind, 3.0)))
        span = next(((a, b) for a, b in tl.part_spans if a - 1e-6 <= t0 < b), (t0, t0 + dur))
        out.append(Ov(kind, t0, min(t0 + dur, span[1]), o))
    have = {o.type for o in out}
    if fmt in ("short", "clip") and version.get("title") and "title_band" not in have:
        out.append(Ov("title_band", c0, c1, {"text": version["title"]}))
    if version.get("credit") and "credit" not in have:
        out.append(Ov("credit", c0, c1, {"text": version["credit"]}))
    out.sort(key=lambda o: (o.t0, o.type))
    return out


def active(ovs: list[Ov], t: float, pad: float = 0.0) -> list[Ov]:
    return [o for o in ovs if o.t0 - pad <= t < o.t1 + pad]
