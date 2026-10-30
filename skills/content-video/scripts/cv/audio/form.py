"""Musical form of one piece: tempo grid, blocks, chords and the energy of every bar.

Time is counted in beats from t = 0, which is always a downbeat. The piece lands on beat `n_beats` (the tonic chord
that rings out to the end): on the end card when the video has one, else a couple of seconds before the end.
  - tempo free (no `bpm`): whole bars, the tempo is fitted within a few percent so the landing is on a bar line;
  - tempo given (beat-snapped vlogs): exactly that tempo, and a shortened bar before the landing when needed.
The body between the intro and the build is a chain of 8-bar blocks (a, a2, b, brk) picked by a small Markov chain;
every block gets its own eight-chord progression (two different four-chord loops), so a 10-minute track keeps
changing instead of looping. Title /
end / eyecatch cards raise the energy of the bars they cover ("swell").
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from cv.audio.theory import Chord, Key, make_chord

STYLE_ID = {"lofi": 1, "acoustic": 2, "pop": 3, "wa": 4, "ambient": 5, "piano": 6}
KEYS = (0, 2, 3, 4, 5, 7, 9, 10)      # C D Eb E F G A Bb
ENERGY = {"intro": 0.40, "a": 0.55, "a2": 0.68, "b": 0.82, "brk": 0.32, "build": 0.90, "land": 1.0}
SWELL = {"title": 0.90, "end": 1.0, "eyecatch": 0.85, "text": 0.70}
TRANSITIONS = {"start": (("a", 1.0),),
               "a": (("a2", 0.50), ("b", 0.30), ("brk", 0.10), ("a", 0.10)),
               "a2": (("b", 0.45), ("a", 0.30), ("brk", 0.15), ("a2", 0.10)),
               "b": (("a", 0.40), ("a2", 0.30), ("brk", 0.20), ("b", 0.10)),
               "brk": (("a", 0.55), ("a2", 0.45))}


@dataclass(frozen=True)
class StyleCfg:
    bpm: float
    loops_a: tuple[tuple[int, ...], ...]   # four-chord loops (scale degrees), one chord per bar
    loops_b: tuple[tuple[int, ...], ...]
    ext: dict[int, str]                    # chord extension per scale degree
    card_ext: str = "9"
    dom_ext: str = "add9"
    sus: bool = False                      # V sus4 -> V inside the build bar


@dataclass(frozen=True)
class Block:
    kind: str       # intro | a | a2 | b | brk | build
    i: int
    bar0: int
    bars: int
    energy: float


@dataclass(frozen=True)
class Bar:
    i: int
    beat0: int
    nbeats: int                 # 4, or 1-3 for the shortened bar before the landing
    t0: float
    t1: float
    block: int                  # block index, -1 for the landing bar
    k: int                      # bar index inside its block
    kind: str                   # block kind, "land" for the landing bar
    energy: float
    chord: Chord
    chord2: Chord | None        # chord of the second half (build bars)
    card: str | None            # title | end | eyecatch | text when a card covers the bar


@dataclass
class Plan:
    style: str
    cfg: StyleCfg
    bpm: float
    beat: float
    duration: float
    t_land: float
    n_beats: int                # beat index of the landing
    key: Key
    blocks: list[Block]
    bars: list[Bar]
    exact: bool = False         # tempo was given (beat-snapped), not fitted
    extra: dict = field(default_factory=dict)   # what a style composes up front (melody, patterns, voicings)

    def __post_init__(self) -> None:
        self._starts = [b.beat0 for b in self.bars]

    def t(self, beat: float) -> float:
        return beat * self.beat

    def bar_index(self, beat: float) -> int:
        return max(0, bisect.bisect_right(self._starts, beat + 1e-6) - 1)

    def chord_at(self, beat: float) -> Chord:
        b = self.bars[self.bar_index(beat)]
        return b.chord2 if b.chord2 is not None and beat - b.beat0 >= b.nbeats / 2 - 1e-6 else b.chord

    @property
    def bar_len(self) -> float:
        return 4 * self.beat

    def beat_times(self) -> list[float]:
        n = int(np.floor(self.duration / self.beat))
        return [round(k * self.beat, 4) for k in range(n + 1)]


# =================================================================== grid
def landing_time(duration: float, sections: list[dict]) -> float:
    """Where the tonic chord lands: the end card's start, else a little before the end of the picture."""
    end = next((s for s in reversed(sections) if s["kind"] == "card" and s.get("card") == "end"
                and s["t0"] > 0.25 * duration), None)
    if end is not None:
        return float(end["t0"])
    return duration - float(np.clip(0.05 * duration, 1.5, 3.0))


def fit_grid(t_land: float, bpm0: float, exact: bool) -> tuple[float, int]:
    """(bpm, landing beat). Free tempo: whole bars; exact tempo: the beat nearest to t_land (never more than 0.12 s late)."""
    if exact:
        beat = 60.0 / bpm0
        n = max(4, int(round(t_land / beat)))
        if n * beat > t_land + 0.12:
            n = max(4, n - 1)
        return bpm0, n
    bars = max(2, int(round(t_land * bpm0 / 240.0)))
    return 240.0 * bars / t_land, 4 * bars


# =================================================================== blocks
def _body_sizes(body: int) -> list[int]:
    """Body bars -> 8-bar blocks; a remainder of 4-7 bars is a block of its own, 1-3 join the last block."""
    if body <= 0:
        return []
    sizes = [8] * (body // 8)
    r = body % 8
    if r >= 4 or not sizes:
        sizes.append(r) if r else None
    elif r:
        sizes[-1] += r
    return sizes


def _body_kinds(sizes: list[int], rng: np.random.Generator) -> list[str]:
    kinds: list[str] = []
    prev = "start"
    for j, size in enumerate(sizes):
        opts = [(k, p) for k, p in TRANSITIONS[prev] if not (k == "brk" and (size < 8 or j == len(sizes) - 1))]
        if len(kinds) >= 2 and kinds[-1] == kinds[-2]:
            opts = [(k, p) for k, p in opts if k != kinds[-1]] or opts
        w = np.array([p for _, p in opts])
        kinds.append(opts[int(rng.choice(len(opts), p=w / w.sum()))][0])
        prev = kinds[-1]
    if kinds and kinds[-1] in ("a", "brk"):
        kinds[-1] = "a2"                       # the build needs a groove with some energy under it
    return kinds


def _loop_degrees(loop: tuple[int, ...], kind: str, nbars: int, last: bool) -> list[int]:
    """One degree per bar from a four-chord loop; a break starts on the loop's third chord, the last body block ends
    on the loop's final chord (so the V of the build follows a pre-dominant)."""
    if kind == "brk":
        loop = loop[2:] + loop[:2]
    off = (-nbars) % len(loop) if last else 0
    return [loop[(j + off) % len(loop)] for j in range(nbars)]


def _build_chords(cfg: StyleCfg, prev_deg: int, j: int, nbars: int) -> tuple[int, int | None, str | None]:
    """(degree, second-half degree, second-half extension) of build bar j."""
    if nbars == 2 and j == 0:
        return 6, 4, None
    pre = 2 if (prev_deg == 4 or (nbars == 2 and j == 1)) else 4
    return pre, 5, cfg.dom_ext


def build_plan(style: str, cfg: StyleCfg, duration: float, sections: list[dict], bpm: float | None, seed: int
               ) -> Plan:
    rng = np.random.default_rng([abs(int(seed)), STYLE_ID[style], 1])
    key = Key(int(rng.choice(KEYS)))
    t_land = landing_time(duration, sections)
    bpm_fit, n_beats = fit_grid(t_land, float(bpm or cfg.bpm), exact=bool(bpm))
    beat = 60.0 / bpm_fit
    n_full, rem = divmod(n_beats, 4)
    title = next((s for s in sections if s["kind"] == "card" and s.get("card") == "title" and s["t0"] < 0.5), None)
    intro = int(np.ceil(title["t1"] / (4 * beat) - 0.3)) if title else (2 if n_full >= 14 else 1 if n_full >= 5 else 0)
    intro = max(1 if title else 0, min(intro, n_full // 3))
    build = 2 if n_full >= 20 else 1 if n_full >= 6 else 0
    build = max(0, min(build, n_full - intro))
    if n_full >= 14:                                  # phrases of 4 bars: let the intro / build absorb the remainder
        for d_in, d_bd in ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, 1), (0, -2), (2, 0), (-2, 0)):
            i2, b2 = intro + d_in, build + d_bd
            if i2 >= (1 if title else 0) and 0 <= b2 <= 3 and (n_full - i2 - b2) % 4 == 0 and (title is None or d_in >= 0):
                intro, build = i2, b2
                break
    sizes = _body_sizes(n_full - intro - build)
    kinds = _body_kinds(sizes, rng)

    def progression(pool: tuple, prev: tuple[int, ...] | None) -> tuple[int, ...]:
        """Eight chords: two different four-chord loops of the pool, so no block repeats the chords of the last one
        and the first chord does not repeat the previous block's last chord."""
        first = [p for p in pool if prev is None or p[0] != prev[-1]] or list(pool)
        a = first[int(rng.integers(len(first)))]
        rest = [p for p in pool if p != a and p[0] != a[-1]] or [p for p in pool if p != a] or list(pool)
        return a + rest[int(rng.integers(len(rest)))]

    layout = ([("intro", intro)] if intro else []) + list(zip(kinds, sizes)) + ([("build", build)] if build else [])
    last_body = len(layout) - 1 - (1 if build else 0)
    blocks: list[Block] = []
    bars: list[Bar] = []
    prev_a = prev_b = None
    kind_n, prev_deg = {}, 1
    for bi, (kind, nb) in enumerate(layout):
        kind_n[kind] = kind_n.get(kind, 0) + 1
        energy = min(1.0, ENERGY[kind] + (0.04 * (kind_n[kind] - 1) if kind in ("a", "a2") else 0.0))
        blocks.append(Block(kind, bi, len(bars), nb, energy))
        if kind == "intro":
            alt = int(rng.choice([6, 4]))
            degs = [1 if j % 2 == 0 else alt for j in range(nb)]
        elif kind != "build":
            if kind == "b":
                prev_b = progression(cfg.loops_b, prev_b)
            else:
                prev_a = progression(cfg.loops_a, prev_a)
            degs = _loop_degrees(prev_b if kind == "b" else prev_a, kind, nb, bi == last_body)
            if nb >= 8 and bi != last_body and rng.random() < 0.5:
                degs[-1] = 5                                      # turnaround into the next block
        for j in range(nb):
            c2 = None
            if kind == "build":
                d1, d2, e2 = _build_chords(cfg, prev_deg, j, nb)
                c1, c2 = make_chord(key, d1, cfg.ext[d1]), make_chord(key, d2, e2 or cfg.ext[d2])
                prev_deg, e = d2, min(1.0, 0.7 + 0.2 * (j + 1) / nb)
            else:
                d1 = degs[j]
                c1, prev_deg, e = make_chord(key, d1, cfg.ext[d1]), d1, energy
            bars.append(Bar(len(bars), 4 * len(bars), 4, 0.0, 0.0, bi, j, kind, e, c1, c2, None))
    if rem:                                                       # shortened bar: the dominant, rising into the landing
        if blocks and blocks[-1].kind == "build":
            last = blocks[-1]
            blocks[-1] = Block("build", last.i, last.bar0, last.bars + 1, last.energy)
        else:
            blocks.append(Block("build", len(blocks), len(bars), 1, ENERGY["build"]))
        bars.append(Bar(len(bars), n_full * 4, rem, 0.0, 0.0, len(blocks) - 1, 0, "build", 0.95,
                        make_chord(key, 5, cfg.dom_ext), None, None))
    tail_beats = max(4, int(np.ceil((duration - n_beats * beat) / beat)))
    bars.append(Bar(len(bars), n_beats, tail_beats, 0.0, 0.0, -1, 0, "land", 1.0,
                    make_chord(key, 1, cfg.card_ext), None, None))
    return Plan(style, cfg, bpm_fit, beat, duration, n_beats * beat, n_beats, key, blocks,
                _stamp(bars, beat, duration, sections), exact=bool(bpm))


def _stamp(bars: list[Bar], beat: float, duration: float, sections: list[dict]) -> list[Bar]:
    """Seconds for every bar and the card swell (energy up, card kind noted) for bars a card covers."""
    cards = [s for s in sections if s["kind"] == "card" and s.get("card") in SWELL]
    out = []
    for b in bars:
        t0 = b.beat0 * beat
        t1 = min(duration, (b.beat0 + b.nbeats) * beat) if b.kind != "land" else duration
        card = next((c for c in cards if c["t0"] < t1 - 0.05 and c["t1"] > t0 + 0.05), None)
        kind = card["card"] if card else None
        out.append(Bar(b.i, b.beat0, b.nbeats, t0, t1, b.block, b.k, b.kind,
                       max(b.energy, SWELL[kind]) if kind else b.energy, b.chord, b.chord2, kind))
    return out
