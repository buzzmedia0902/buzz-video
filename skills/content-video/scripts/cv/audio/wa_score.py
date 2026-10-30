"""wa_score - music theory, time map and the seeded melody writer (original material only).

- 4/4 on an 8th-note grid ("slots": 8 per bar). The tempo is nudged so that a whole
  number of bars lands exactly on the final tonic stroke, with a gentle ritardando
  over the last beats (TimeMap).
- calm   = 都節 miyako-bushi  D Eb G A Bb  (oshide = semitone)
  bright = 陽音階 yo / min'yo D E G A B   (oshide = whole tone)
  Nuclear tones D-G-A (Koizumi tetrachord frame): half cadences on A / G, full on D.
- The melody koto is tuned like 平調子 (string 1 doubles string 5); melody lines are
  generate-and-score random walks over its open strings, phrase pairs are
  call (2 bars, ends on A/G) -> response (2 bars, ends on D), with motif reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax_pick(rng: np.random.Generator, costs: np.ndarray, temp: float) -> int:
    """Index sampled with probability ~ exp(-cost / temp) (low cost = likely)."""
    z = -(costs - costs.min()) / max(temp, 1e-6)
    p = np.exp(z)
    return int(rng.choice(len(costs), p=p / p.sum()))


TONIC = 62                      # D4
NUCLEAR = frozenset({0, 5, 7})  # D, G, A (semitones above the tonic)
SLOTS_PER_BAR = 8

# rhythm cells in 8th-note slots; negative = rest
_CELLS_COMMON = ((2, 2, 4), (3, 1, 4), (2, 1, 1, 4), (1, 1, 2, 4), (4, 2, 2), (3, 1, 2, 2),
                 (-2, 2, 4), (-2, 1, 1, 4), (2, 2, 2, 2), (-1, 1, 2, 2, 2), (6, 1, 1))
_CELLS_BRIGHT = ((3, 1, 3, 1), (1, 1, 1, 1, 4), (2, 1, 1, 2, 2), (-1, 1, 1, 1, 4), (3, 1, 2, 1, 1))
HALF_CELLS = ((2, 2, 4), (3, 1, 4), (2, 6), (1, 1, 6), (4, 4), (2, 1, 1, 4), (1, 1, 2, 4))
FULL_CELLS = ((2, 2, 4), (3, 1, 4), (2, 6), (4, 4), (1, 1, 2, 4), (2, 1, 1, 4))
FINAL_CELLS = ((2, 2, 2, 2), (4, 2, 2), (3, 1, 2, 2), (2, 2, 4))
LEADIN_CELLS = ((-4, 2, 2), (-4, 1, 1, 2), (-5, 1, 2), (-6, 1, 1), (-2, 2, 2, 2))
CONTOURS = ("rise", "fall", "arch", "valley", "free")
PLANS_A = (("T", "D", "S", "T"), ("T", "S", "D", "T"), ("T", "T", "S", "T"),
           ("T", "D", "T", "T"), ("S", "D", "S", "T"))
PLANS_B = (("S", "D", "S", "T"), ("D", "S", "D", "T"), ("S", "T", "D", "T"), ("S", "S", "D", "T"))

STEPS = np.array([-3, -2, -1, 0, 1, 2, 3])
STEP_P = np.array([0.04, 0.13, 0.30, 0.08, 0.30, 0.12, 0.03])
STEP_COST = {0: 0.5, 1: 0.0, 2: 0.25, 3: 0.9, 4: 1.6}


@dataclass(frozen=True)
class Mood:
    name: str
    degrees: tuple[int, ...]
    press: int
    koto: tuple[int, ...]                     # 13 open strings (平調子-like)
    voicing: dict = field(hash=False)        # frame -> 2nd-koto (low, mid, high)
    pad: dict = field(hash=False)            # frame -> drone pitches
    bass: dict = field(hash=False)           # frame -> 17-string bass pitch
    frame_pcs: dict = field(hash=False)      # frame -> preferred pcs on strong beats
    half_pc: dict = field(hash=False)        # frame -> pc of the half-cadence note
    approach: dict = field(hash=False)       # (pc_prev, pc_last) -> cost bonus
    cells: tuple = ()
    acc_a: tuple = ()
    acc_b: tuple = ()
    vel: float = 0.56
    shami_p: float = 0.5

    @property
    def mel(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.koto[1:])))

    def pc(self, pitch: int) -> int:
        return (pitch - TONIC) % 12


_FRAMES = {"T": frozenset({0, 7}), "S": frozenset({5, 0}), "D": frozenset({7, 0})}
MOODS = {
    "calm": Mood(
        name="calm", degrees=(0, 1, 5, 7, 8), press=1,
        koto=(62, 55, 57, 58, 62, 63, 67, 69, 70, 74, 75, 79, 81),
        voicing={"T": (50, 57, 62), "S": (55, 62, 67), "D": (57, 62, 69)},
        pad={"T": (50, 57, 62), "S": (55, 62), "D": (57, 62)},
        bass={"T": 50, "S": 43, "D": 45}, frame_pcs=_FRAMES,
        half_pc={"T": 7, "S": 5, "D": 7},
        approach={(1, 0): -0.7, (8, 7): -0.5, (7, 0): -0.2, (7, 5): -0.15, (5, 1): -0.35},
        cells=_CELLS_COMMON, acc_a=("lilt", "pulse", "sparse"), acc_b=("flow", "pulse"),
        vel=0.56, shami_p=0.45),
    "bright": Mood(
        name="bright", degrees=(0, 2, 5, 7, 9), press=2,
        koto=(62, 55, 57, 59, 62, 64, 67, 69, 71, 74, 76, 79, 81),
        voicing={"T": (50, 57, 62), "S": (55, 62, 67), "D": (57, 64, 69)},
        pad={"T": (50, 57, 62), "S": (55, 62), "D": (57, 64)},
        bass={"T": 50, "S": 43, "D": 45},
        frame_pcs={"T": frozenset({0, 7}), "S": frozenset({5, 0}), "D": frozenset({7, 2})},
        half_pc={"T": 7, "S": 5, "D": 7},
        approach={(2, 0): -0.6, (9, 7): -0.45, (7, 0): -0.25, (7, 5): -0.15, (5, 2): -0.35,
                  (7, 2): -0.15},
        cells=_CELLS_COMMON + _CELLS_BRIGHT, acc_a=("lilt", "pulse", "dotted"),
        acc_b=("flow", "dotted"), vel=0.6, shami_p=0.6),
}


@dataclass(frozen=True)
class Note:
    """Abstract note in beats (converted to seconds by the arranger)."""
    inst: str
    beat: float
    beats: float
    pitch: int
    vel: float
    orn: str = ""
    open_pitch: int | None = None
    hold_beats: float = 0.0
    extra: tuple = ()


@dataclass(frozen=True)
class TimeMap:
    """Beat -> seconds with a quadratic ritardando into the final stroke."""
    n_bars: int
    spb: float
    rit_beats: float
    rit: float

    @property
    def final_beat(self) -> float:
        return 4.0 * self.n_bars

    def sec(self, beat: float) -> float:
        b0 = self.final_beat - self.rit_beats
        if beat <= b0:
            return self.spb * beat
        x, rb, r = beat - b0, self.rit_beats, self.rit
        if x <= rb:
            return self.spb * (b0 + x + r * x ** 3 / (3 * rb ** 2))
        return self.spb * (b0 + rb + r * rb / 3 + (1 + r) * (x - rb))

    @staticmethod
    def fit(t_final: float, bpm: float, rit: float = 0.22) -> "TimeMap":
        spb_nom = 60.0 / bpm
        n = max(1, int(round((t_final / spb_nom - rit * 2.0) / 4)))
        rit_beats = min(6.0, 2.0 * n)
        spb = t_final / (4 * n + rit * rit_beats / 3)
        return TimeMap(n, spb, rit_beats, rit)


@dataclass(frozen=True)
class Theme:
    """Seed-level choices shared by every phrase of one piece."""
    cell_a: tuple
    half_a: tuple
    full_a: tuple
    cell_b: tuple
    half_b: tuple
    full_b: tuple
    final_cell: tuple
    bar3_mode: str
    acc_a: str
    acc_b: str
    opening: str
    final_stroke: str
    shaku_intro: bool
    shaku_end: bool
    shaku_end_pitch: int
    taiko_start: bool
    taiko_end: bool
    plan_a: tuple
    plan_b: tuple
    contour_a: str
    contour_b: str
    leadin: tuple


def make_theme(rng: np.random.Generator, mood: Mood) -> Theme:
    cells = mood.cells
    ia, ib = rng.choice(len(cells), size=2, replace=False)
    opening = str(rng.choice(["sararin", "kororin", "drone"], p=[0.45, 0.3, 0.25]))
    return Theme(
        cell_a=cells[ia], half_a=HALF_CELLS[rng.integers(len(HALF_CELLS))],
        full_a=FULL_CELLS[rng.integers(len(FULL_CELLS))],
        cell_b=cells[ib], half_b=HALF_CELLS[rng.integers(len(HALF_CELLS))],
        full_b=FULL_CELLS[rng.integers(len(FULL_CELLS))],
        final_cell=FINAL_CELLS[rng.integers(len(FINAL_CELLS))],
        bar3_mode=str(rng.choice(["echo", "shift", "invert"], p=[0.45, 0.3, 0.25])),
        acc_a=str(rng.choice(mood.acc_a)), acc_b=str(rng.choice(mood.acc_b)),
        opening=opening, final_stroke=str(rng.choice(["awase", "roll"])),
        shaku_intro=bool(opening == "drone" or rng.random() < 0.5),
        shaku_end=bool(rng.random() < 0.75),
        shaku_end_pitch=int(rng.choice([62, 74]) if mood.name == "calm" else 74),
        taiko_start=bool(opening == "drone" or rng.random() < 0.4),
        taiko_end=bool(rng.random() < 0.6),
        plan_a=PLANS_A[rng.integers(len(PLANS_A))], plan_b=PLANS_B[rng.integers(len(PLANS_B))],
        contour_a=str(rng.choice(CONTOURS)), contour_b=str(rng.choice(CONTOURS)),
        leadin=LEADIN_CELLS[rng.integers(len(LEADIN_CELLS))])


def cell_notes(cell: tuple) -> tuple[tuple[int, int], ...]:
    """Rhythm cell -> ((slot, length), ...) for the sounding notes."""
    out, pos = [], 0
    for v in cell:
        if v > 0:
            out.append((pos, v))
        pos += abs(v)
    return tuple(out)


# =================================================================== line writer
def _contour_cost(seq: list[int], contour: str) -> float:
    """Mismatch between the line's step directions and a contour template."""
    if contour == "free" or len(seq) < 2:
        return 0.0
    m = len(seq) - 1
    want = {"rise": [1] * m, "fall": [-1] * m,
            "arch": [1 if k < (m + 1) // 2 else -1 for k in range(m)],
            "valley": [-1 if k < (m + 1) // 2 else 1 for k in range(m)]}[contour]
    got = np.sign(np.diff(seq))
    return float(sum(0.5 if g == 0 else (0.35 if g != w else 0.0) for g, w in zip(got, want)))


def _shape_cost(seq: list[int], avoid_tonic: bool, pcs: list[int]) -> float:
    """Variety: enough distinct pitches, no x-y-x-y see-saw, some span, no early tonic."""
    m = len(seq)
    cost = 0.8 if len(set(seq)) < max(2, int(np.ceil(0.6 * m))) else 0.0
    cost += sum(0.8 for k in range(3, m) if seq[k] == seq[k - 2] and seq[k - 1] == seq[k - 3])
    cost += 0.5 if (m >= 3 and max(seq) - min(seq) < 2) else 0.0
    if avoid_tonic:
        cost += 0.3 * sum(1 for pc in pcs[:-1] if pc == 0)
    return cost


def _line_cost(seq: list[int], rhythm: tuple, frame: str, prev: int | None, mood: Mood,
               contour: str = "free", avoid_tonic: bool = False) -> float:
    mel = mood.mel
    pcs = [mood.pc(mel[i]) for i in seq]
    full = ([prev] if prev is not None else []) + list(seq)
    cost = 0.0
    for k in range(1, len(full)):
        d = full[k] - full[k - 1]
        semis = abs(mel[full[k]] - mel[full[k - 1]])
        cost += STEP_COST.get(abs(d), 3.0) + (0.5 if semis == 6 else 0.0) + (0.4 if semis > 9 else 0.0)
        if k >= 2:
            pd = full[k - 1] - full[k - 2]
            cost += 1.0 if (abs(pd) >= 3 and d * pd > 0) else 0.0
            cost += 1.5 if (pd == 0 and d == 0) else 0.0
    for (slot, length), pc in zip(rhythm, pcs):
        cost += 0.7 if (slot % 4 == 0 and pc not in mood.frame_pcs[frame]) else 0.0
        cost += 0.6 if (length >= 4 and pc not in NUCLEAR) else 0.0
    if len(pcs) >= 2:
        cost += mood.approach.get((pcs[-2], pcs[-1]), 0.0) + (0.6 if seq[-2] == seq[-1] else 0.0)
    span = max(seq) - min(seq)
    cost += 0.5 * (span - 5) if span > 5 else 0.0
    return cost + _contour_cost(seq, contour) + _shape_cost(seq, avoid_tonic, pcs)


def search_line(rng: np.random.Generator, mood: Mood, rhythm: tuple, frame: str,
                prev: int | None, target: int | None, lo: int, hi: int,
                n_cand: int = 240, temp: float = 0.35, contour: str = "free",
                avoid_tonic: bool = False) -> tuple[int, ...]:
    """Pick scale-index pitches for `rhythm` by sampling walks and scoring them."""
    m = len(rhythm)
    mel = mood.mel
    starts = [i for i in range(lo, hi + 1) if mood.pc(mel[i]) in mood.frame_pcs[frame]]
    cands = []
    for _ in range(n_cand):
        cur = prev if prev is not None else int(rng.choice(starts))
        seq: list[int] = [] if prev is not None else [cur]
        while len(seq) < m - (target is not None):
            nxt = cur + int(rng.choice(STEPS, p=STEP_P))
            if not lo <= nxt <= hi:
                nxt = 2 * cur - nxt
            cur = int(np.clip(nxt, lo, hi))
            seq.append(cur)
        if target is not None:
            seq.append(target)
        cands.append(tuple(seq[-m:] if len(seq) > m else seq))
    uniq = sorted(set(cands))            # dedupe: selection depends on cost only
    costs = np.array([_line_cost(list(s), rhythm, frame, prev, mood, contour, avoid_tonic)
                      for s in uniq])
    return uniq[softmax_pick(rng, costs, temp)]


def _nearest(mood: Mood, pc: int, ref: int, lo: int, hi: int) -> int:
    cands = [i for i in range(lo, hi + 1) if mood.pc(mood.mel[i]) == pc]
    return min(cands, key=lambda i: (abs(i - ref), i))


def _shifted(line: tuple[int, ...], s: int, lo: int, hi: int) -> tuple[int, ...]:
    for step in (s, -s, 0):
        cand = tuple(i + step for i in line)
        if all(lo <= i <= hi for i in cand):
            return cand
    return tuple(int(np.clip(i, lo, hi)) for i in line)


def _variant(rng: np.random.Generator, line: tuple[int, ...], mode: str, lo: int,
             hi: int) -> tuple[int, ...]:
    if mode == "shift":
        return _shifted(line, int(rng.choice([-1, 1])), lo, hi)
    if mode == "invert":
        inv = tuple(2 * line[0] - i for i in line)
        return inv if all(lo <= i <= hi for i in inv) else line
    return line


@dataclass(frozen=True)
class PairResult:
    notes: tuple            # ((slot, length, idx, vel), ...) slots from the pair start
    motif: tuple            # bar-1 line actually used
    call2: tuple            # bar-2 line actually used
    last: int               # index of the last note


def write_pair(rng: np.random.Generator, mood: Mood, theme: Theme, kind: str, plan: tuple,
               prev: int | None, memo: dict, final: bool = False) -> PairResult:
    """Call (bars 1-2, half cadence) + response (bars 3-4, full cadence or final approach)."""
    is_a = kind.startswith("A")
    lo, hi = (3, 9) if is_a else (4, 10)          # A: D4..Eb5/E5, B: Eb4/E4..G5
    r1 = cell_notes(theme.cell_a if is_a else theme.cell_b)
    r2 = cell_notes(theme.half_a if is_a else theme.half_b)
    r4 = cell_notes(theme.final_cell if final else (theme.full_a if is_a else theme.full_b))
    key = "A" if is_a else "B"
    base = memo.get(key)
    if base is None:
        contour = theme.contour_a if is_a else theme.contour_b
        b1 = search_line(rng, mood, r1, plan[0], prev, None, lo, hi, contour=contour,
                         avoid_tonic=True)
    elif kind in ("A2", "B2"):
        b1 = _shifted(base[0], int(rng.choice([-1, 1])), lo, hi)
    else:
        b1 = base[0]
    half_t = _nearest(mood, mood.half_pc[plan[1]], b1[-1], lo, hi)
    reuse = base is not None and kind not in ("A2", "B2")
    b2 = base[1] if reuse else search_line(rng, mood, r2, plan[1], b1[-1], half_t, lo, hi,
                                           avoid_tonic=True)
    b3 = _variant(rng, b1, theme.bar3_mode, lo, hi)
    if final:
        tonic = _nearest(mood, 0, b3[-1], max(lo, 3), min(hi, 8))
        target = tonic + 1
    else:
        target = _nearest(mood, 0, b3[-1], lo, hi)
    b4 = search_line(rng, mood, r4, plan[3], b3[-1], target, lo, hi)
    notes = []
    for bar, (rhy, line) in enumerate(((r1, b1), (r2, b2), (r1, b3), (r4, b4))):
        for (slot, length), idx in zip(rhy, line):
            notes.append((bar * SLOTS_PER_BAR + slot, length, idx, _vel(rng, mood, bar, slot, length)))
    return PairResult(tuple(notes), b1, b2, b4[-1])


def _vel(rng: np.random.Generator, mood: Mood, bar: int, slot: int, length: int) -> float:
    pos = (bar * SLOTS_PER_BAR + slot) / 32.0
    v = mood.vel + 0.06 * (slot % 4 == 0) + 0.08 * np.sin(np.pi * pos) - 0.03 * (bar >= 2)
    v -= 0.04 * (length >= 4 and bar in (1, 3))
    return float(np.clip(v + rng.normal(0.0, 0.035), 0.2, 0.9))
