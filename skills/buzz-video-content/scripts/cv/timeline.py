"""Output timeline of one version: parts -> placed shots, and source <-> output time mapping.

  talk  jet-cut speech of a source (pauses >= cut.pause shortened, `remove` ranges dropped), at cut.speed
  clip  a source range as it is (b-roll, vlog clips, cold open), optional speed; "dur" (output seconds) instead
        of "to" takes that much from "from"; with version "snap": "beat" every clip lasts a whole number of beats
        of the BGM tempo (version "bgm": {"bpm": ...}, else the tone's AUDIO bpm, else 100) - the vlog rhythm
  card  a generated full-screen card (title / end / eyecatch / text) of `dur` seconds

A shot shows source time a + k / FPS * speed on its k-th frame; its audio is the same [a, b) range time-stretched
to exactly n / FPS seconds, so picture and voice stay in sync whatever the speed.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from cv import media, vad
from cv.canvas import FPS
from cv.spec import SpecError, warn


@dataclass
class Shot:
    kind: str                 # talk | clip | card
    part: int                 # index into version["parts"]
    src: str | None
    a: float                  # source seconds
    b: float
    speed: float
    f0: int                   # first output frame
    n: int                    # output frames
    card: str | None = None
    data: dict = field(default_factory=dict, repr=False)   # the part dict
    view: dict = field(default_factory=dict)               # framing (cv.framing)

    @property
    def t0(self) -> float:
        return self.f0 / FPS

    @property
    def t1(self) -> float:
        return (self.f0 + self.n) / FPS

    def src_time(self, k: int) -> float:
        return self.a + k / FPS * self.speed


def _subtract(ranges: list[tuple[float, float]], cuts: list) -> list[tuple[float, float]]:
    out = list(ranges)
    for c0, c1 in sorted((float(a), float(b)) for a, b in cuts):
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a >= 0.1]


class Timeline:
    def __init__(self, project, version: dict, look):
        self.project, self.version, self.look = project, version, look
        self.shots: list[Shot] = []
        self.part_spans: list[tuple[float, float]] = []
        f = 0
        cut = {**look.CUT, **version.get("cut", {})}
        for i, p in enumerate(version["parts"]):
            start = f
            for shot in self._part_shots(i, p, cut, f):
                shot.f0 = f
                f += shot.n
                self.shots.append(shot)
            self.part_spans.append((start / FPS, f / FPS))
        if not self.shots:
            raise SpecError("the version has no frames (check parts from / to)")
        self.total = f
        self.duration = f / FPS
        self._starts = [s.f0 for s in self.shots]

    def _part_shots(self, i: int, p: dict, cut: dict, f0: int) -> list[Shot]:
        kind = p["kind"]
        if kind == "card":
            return [Shot("card", i, None, 0.0, 0.0, 1.0, 0, max(1, round(float(p["dur"]) * FPS)), card=p["card"],
                         data=p)]
        sid = p["src"]
        path = self.project.src_path(sid)
        info = media.probe(path)
        speed = float(p.get("speed", cut["speed"] if kind == "talk" else 1.0))
        if speed <= 0:
            raise SpecError(f"parts[{i}]: speed must be > 0")
        if info["image"] and "dur" not in p:
            raise SpecError(f"parts[{i}]: {sid} is a picture: give the clip a \"dur\" (seconds on screen)")
        a0 = float(p.get("from", 0.0))
        if kind == "clip" and "dur" in p:
            dur = float(p["dur"])
            if self.version.get("snap") == "beat":  # end on the beat grid counted from 0 (no drift clip after clip)
                beat = 60.0 / self.bpm()
                t_end = max(f0 / FPS + beat, round((f0 / FPS + dur) / beat) * beat)
                dur = (round(t_end * FPS) - f0) / FPS
            b0 = a0 + dur * speed if info["image"] else min(a0 + dur * speed, info["duration"])
        else:
            b0 = min(float(p.get("to", info["duration"])), info["duration"])
        if b0 - a0 < 0.1:
            raise SpecError(f"parts[{i}]: {sid} range {a0:.2f}-{b0:.2f} s is empty (source is {info['duration']:.2f} s)")
        if kind == "talk":
            if not info["audio"]:
                raise SpecError(f"parts[{i}]: talk part needs a source with audio ({sid} has none); use kind 'clip'")
            sp = vad.for_source(path, self.project.work_dir, sid)
            ranges = sp.tighten(a0, b0, float(p.get("pause", cut["pause"])), float(p.get("tail", cut["tail"])),
                                float(p.get("lead", cut["lead"])))
        else:
            ranges = [(a0, b0)]
        ranges = _subtract(ranges, p.get("remove", []))
        if info["fps"] and info["fps"] * speed < FPS * 0.95 and speed < 1:
            warn(f"parts[{i}]: x{speed} slow motion of a {info['fps']:.0f} fps clip repeats frames (stutter)")
        shots = []
        for a, b in ranges:
            n = round((b - a) / speed * FPS)
            if n >= 2:
                shots.append(Shot(kind, i, sid, a, b, speed, 0, n, data=p))
        return shots

    def bpm(self) -> float:
        return float(self.version.get("bgm", {}).get("bpm") or self.look.AUDIO.get("bpm") or 100.0)

    # -------------------------------------------------------------- lookup
    def at(self, n: int) -> Shot:
        i = max(0, bisect.bisect_right(self._starts, n) - 1)
        return self.shots[i]

    def src_to_out(self, src: str, t: float, side: str = "nearest", part: int | None = None) -> float | None:
        """Output seconds where source time t of `src` plays (in `part` only, if given). In a cut-out gap: the next
        shot's start (side='next'), the previous shot's end ('prev') or the closer of the two ('nearest'); None if
        src never plays near t."""
        best = None
        for s in self.shots:
            if s.src != src or (part is not None and s.part != part):
                continue
            if s.a - 1e-6 <= t < s.b + 1e-6:
                return s.t0 + (t - s.a) / s.speed
            if t < s.a:
                cand = ("next", s.t0, s.a - t)
            else:
                cand = ("prev", s.t1, t - s.b)
            if side != "nearest" and cand[0] != side:
                continue
            if best is None or cand[2] < best[2]:
                best = cand
        if best is None or best[2] > 3.0:
            return None
        return best[1]

    def out_to_src(self, t: float) -> tuple[str | None, float]:
        s = self.at(int(t * FPS))
        return s.src, s.a + (t - s.t0) * s.speed

    def parts_playing(self, src: str, t0: float, t1: float) -> list[int]:
        """Parts in which [t0, t1] of `src` plays (the same words can be in a cold open and in the talk)."""
        return sorted({s.part for s in self.shots if s.src == src and s.a < t1 and t0 < s.b})

    def part_of(self, t: float) -> int:
        return self.at(int(t * FPS)).part

    def talk_spans(self) -> list[tuple[float, float]]:
        """Output spans of talk / clip parts (subtitles never stretch across a part boundary)."""
        return [span for span, p in zip(self.part_spans, self.version["parts"]) if p["kind"] != "card"]
