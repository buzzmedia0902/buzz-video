"""Tempo fitting, song form and chord plan.

Time is measured in beats from t=0 (``Grid.t(beat) = beat * Grid.beat``). The end card sits on beat
``n_beats`` which is always a bar line; if ``n_beats`` is not a multiple of 4 the first bar is a partial
"pick-up" bar. The form (intro / a / b / brk / a / build / card) and the chord for every beat are derived
from the number of full bars, a style config and a seeded RNG, so any duration 8-60 s gets a composed
arc: intro -> groove -> variation (b, break) -> V-chord build -> arrival on the tonic chord at the card.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from core.composer.theory import Chord, Key, make_chord

STYLE_ID = {"piano": 1, "pop": 2, "cinematic": 3, "lofi": 4}
KEYS = (0, 2, 3, 4, 5, 7, 9, 10)  # C D Eb E F G A Bb
ENERGY = {"intro": 0.40, "a": 0.60, "b": 0.82, "brk": 0.35, "build": 0.95, "card": 1.0}


@dataclass(frozen=True)
class Grid:
    bpm: float
    beat: float          # seconds per beat
    n_beats: int         # beats before the card
    pickup: int          # beats in the partial first bar (0-3)
    card_at: float
    dur: float

    @property
    def bar(self) -> float:
        return 4 * self.beat

    @property
    def bars(self) -> float:
        return self.n_beats / 4

    @property
    def tail_beats(self) -> float:
        return (self.dur - self.card_at) / self.beat

    def t(self, beat: float) -> float:
        return beat * self.beat


def fit_tempo(card_at: float, bpm0: float, tol: float = 0.08, exact: bool = False) -> tuple[int, float]:
    """(beats before the card, fitted bpm). Prefers whole bars within +-tol of bpm0, then any whole
    number of beats within tol (partial first bar), else the nearest whole number of beats.
    With `exact`, a tempo that already puts the card on a whole beat (+-0.02 beat) is kept as it is -
    for callers whose cuts sit on that beat grid."""
    if exact:
        m = round(card_at * bpm0 / 60)
        if m >= 2 and abs(card_at * bpm0 / 60 - m) < 0.02:
            return m, 60 * m / card_at
    lo_m = max(2, int(math.floor(card_at * bpm0 * (1 - tol) / 60)))
    hi_m = int(math.ceil(card_at * bpm0 * (1 + tol) / 60))
    inside = [m for m in range(lo_m, hi_m + 1) if abs(60 * m / card_at / bpm0 - 1) <= tol + 1e-9]

    def dev(m: int) -> float:
        return abs(math.log(60 * m / card_at / bpm0))

    whole = [m for m in inside if m % 4 == 0]
    pool = whole or inside or [max(2, round(card_at * bpm0 / 60))]
    m = min(pool, key=dev)
    return m, 60 * m / card_at


@dataclass(frozen=True)
class Section:
    kind: str       # intro | a | b | brk | build | card
    nth: int        # how many sections of this kind came before
    beat0: float
    beats: float
    energy: float

    @property
    def end(self) -> float:
        return self.beat0 + self.beats


@dataclass(frozen=True)
class ChordEvent:
    beat: float
    length: float
    chord: Chord
    kind: str
    nth: int
    k: int          # bar index inside its section (-1 for the pick-up bar)
    energy: float

    @property
    def end(self) -> float:
        return self.beat + self.length


@dataclass(frozen=True)
class StyleCfg:
    bpm: float
    loops_a: tuple[tuple[int, ...], ...]   # degrees, one chord per bar
    loops_b: tuple[tuple[int, ...], ...]
    ext: dict[int, str]                    # chord extension per scale degree
    card_ext: str = "9"
    dom_ext: str = "add9"
    sus: bool = False                      # V sus4 -> V inside the build bar


@dataclass(frozen=True)
class Plan:
    grid: Grid
    key: Key
    sections: tuple[Section, ...]
    events: tuple[ChordEvent, ...]
    loop_a: tuple[int, ...]
    loop_b: tuple[int, ...]

    def event_index_at(self, beat: float) -> int:
        """Index of the chord event sounding at `beat`."""
        return max(0, bisect_right([e.beat for e in self.events], beat + 1e-6) - 1)

    def event_at(self, beat: float) -> ChordEvent:
        """The chord event sounding at `beat`."""
        return self.events[self.event_index_at(beat)]

    def chord_at(self, beat: float) -> Chord:
        """The chord sounding at `beat`."""
        return self.event_at(beat).chord

    def section_at(self, beat: float) -> Section:
        """The form section containing `beat`."""
        for s in self.sections:
            if s.beat0 - 1e-6 <= beat < s.end - 1e-6:
                return s
        return self.sections[-1]

    def of_kind(self, *kinds: str) -> list[Section]:
        """All sections whose kind is one of `kinds`, in time order."""
        return [s for s in self.sections if s.kind in kinds]

    @property
    def card(self) -> ChordEvent:
        """The tonic chord event that starts on the end card."""
        return self.events[-1]


def split_body(body: int) -> list[tuple[str, int]]:
    """Body bars -> [(kind, bars)]: a | a b | a b a | a b brk a | ... in 4-bar chunks."""
    if body <= 0:
        return []
    if body <= 4:
        return [("a", body)]
    if body <= 7:
        a = (body + 1) // 2
        return [("a", a), ("b", body - a)]
    chunks, rem = divmod(body, 4)
    names = ["a", "b", "a", "brk", "a", "b", "a", "brk"]
    forms = {2: ["a", "b"], 3: ["a", "b", "a"], 4: ["a", "b", "brk", "a"], 5: ["a", "b", "a", "brk", "a"]}
    kinds = list(forms.get(chunks) or (names * 4)[:chunks])
    if kinds[-1] == "brk":
        kinds[-1] = "a"  # never end the body on a break: the build needs the groove
    lens = [4] * chunks
    lens[-1] += rem
    return list(zip(kinds, lens))


def plan_sections(n_full: int) -> list[tuple[str, int]]:
    """Form for `n_full` bars before the card: [(kind, bars)] ending in the build."""
    build = 2 if n_full >= 12 else 1
    if n_full <= 1:
        return [("build", 1)] if n_full == 1 else []
    intro = {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2}.get(n_full, 3 if n_full >= 20 else 2)
    body = max(0, n_full - intro - build)
    return [("intro", intro)] + split_body(body) + [("build", build)]


def _loop_degrees(loop: tuple[int, ...], kind: str, nbars: int, last: bool) -> list[int]:
    """One degree per bar from a 4-chord loop; the last body section is aligned to END on the loop's
    final chord (so the V of the build follows a pre-dominant), others start on its first chord."""
    if kind == "brk":
        loop = loop[2:] + loop[:2]
    off = (-nbars) % len(loop) if last else 0
    return [loop[(j + off) % len(loop)] for j in range(nbars)]


def make_plan(cfg: StyleCfg, grid: Grid, rng: np.random.Generator) -> Plan:
    """Pick key and loops with `rng` (independent of the duration) and lay out the form."""
    key = Key(int(rng.choice(KEYS)))
    la = cfg.loops_a[int(rng.integers(len(cfg.loops_a)))]
    lb = cfg.loops_b[int(rng.integers(len(cfg.loops_b)))]
    alt = int(rng.choice([6, 4]))
    n_full = grid.n_beats // 4

    def ch(degree: int, ext: str | None = None) -> Chord:
        return make_chord(key, degree, ext or cfg.ext[degree])

    sections: list[Section] = []
    events: list[ChordEvent] = []
    counts: dict[str, int] = {}
    beat = float(grid.pickup)
    forms = plan_sections(n_full)
    if grid.pickup and (not forms or forms[0][0] != "intro"):
        forms.insert(0, ("intro", 0))  # the pick-up beats alone form the intro
    last_body = max((i for i, (k, _) in enumerate(forms) if k in ("a", "b", "brk")), default=-1)
    prev_deg = 1
    if grid.pickup:
        events.append(ChordEvent(0.0, float(grid.pickup), ch(1), "intro", 0, -1, ENERGY["intro"]))
    for fi, (kind, nbars) in enumerate(forms):
        nth = counts.get(kind, 0)
        counts[kind] = nth + 1
        beat0 = 0.0 if (fi == 0 and grid.pickup) else beat
        sec_beats = nbars * 4 + (grid.pickup if fi == 0 else 0)
        energy = min(1.0, ENERGY[kind] + (0.08 * nth if kind == "a" else 0.05 * nth if kind == "b" else 0.0))
        sections.append(Section(kind, nth, beat0, float(sec_beats), energy))
        for j in range(nbars):
            bt = beat + 4 * j
            if kind == "intro":
                cs = [(0.0, 1 if j % 2 == 0 else alt, None, 4.0)]
            elif kind == "build":
                cs = _build_chords(cfg, prev_deg, j, nbars)
            else:
                degs = _loop_degrees(lb if kind == "b" else la, kind, nbars, fi == last_body)
                cs = [(0.0, degs[j], None, 4.0)]
            for off, deg, ext, ln in cs:
                events.append(ChordEvent(bt + off, ln, ch(deg, ext), kind, nth, j, energy))
                prev_deg = deg
        beat += 4 * nbars
    tail = max(grid.tail_beats, 1.0)
    sections.append(Section("card", 0, float(grid.n_beats), tail, ENERGY["card"]))
    events.append(ChordEvent(float(grid.n_beats), tail, ch(1, cfg.card_ext), "card", 0, 0, 1.0))
    return Plan(grid, key, tuple(sections), tuple(events), la, lb)


def _build_chords(cfg: StyleCfg, prev_deg: int, j: int, nbars: int) -> list[tuple[float, int, str | None, float]]:
    """(offset, degree, ext, beats) chords of build bar j: [pre, V] (or vi-IV | ii-V for a 2-bar build)."""
    if nbars == 2 and j == 0:
        return [(0.0, 6, None, 2.0), (2.0, 4, None, 2.0)]
    pre = 2 if (prev_deg == 4 or (nbars == 2 and j == 1)) else 4
    if cfg.sus:
        return [(0.0, pre, None, 2.0), (2.0, 5, "7sus4", 1.0), (3.0, 5, cfg.dom_ext, 1.0)]
    return [(0.0, pre, None, 2.0), (2.0, 5, cfg.dom_ext, 2.0)]
