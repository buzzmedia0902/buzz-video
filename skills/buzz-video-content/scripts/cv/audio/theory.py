"""Music-theory helpers: major keys, diatonic chord stacks and smooth voice leading.

Pitches are MIDI numbers (C4 = 60). A ``Key`` maps scale indices to MIDI (index 0 = tonic in the octave
starting at C4, 7 = tonic + octave, -1 = the leading tone below), so melodies and chords are built
from scale steps and are always diatonic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

MAJOR = (0, 2, 4, 5, 7, 9, 11)
PC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")
# scale steps stacked above the root (0 = root, 2 = third, 4 = fifth, 6 = seventh, 8 = ninth, ...)
EXT_STEPS = {
    "tri": (0, 2, 4), "6": (0, 2, 4, 5), "7": (0, 2, 4, 6), "9": (0, 2, 4, 6, 8), "add9": (0, 2, 4, 8),
    "sus2": (0, 1, 4), "sus4": (0, 3, 4), "7sus4": (0, 3, 4, 6), "13": (0, 2, 6, 12),
}


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0..11

    @property
    def name(self) -> str:
        return PC_NAMES[self.tonic]

    def midi(self, idx: int) -> int:
        """MIDI of scale index `idx` (0 = tonic in the C4 octave)."""
        return 60 + self.tonic + MAJOR[idx % 7] + 12 * (idx // 7)

    def nearest_idx(self, midi: float) -> int:
        """Scale index whose pitch is closest to `midi`."""
        guess = round((midi - 60 - self.tonic) / 12 * 7)
        return min(range(guess - 4, guess + 5), key=lambda i: abs(self.midi(i) - midi))

    def idx_range(self, lo: int, hi: int) -> tuple[int, int]:
        """Scale indices whose pitch lies inside [lo, hi] MIDI (inclusive)."""
        a, b = self.nearest_idx(lo), self.nearest_idx(hi)
        return (a if self.midi(a) >= lo else a + 1), (b if self.midi(b) <= hi else b - 1)


@dataclass(frozen=True)
class Chord:
    degree: int                # 1..7
    ext: str
    root_pc: int
    stack: tuple[int, ...]     # semitones above the root, ascending (e.g. 0, 4, 7, 11, 14)

    @property
    def pcs(self) -> tuple[int, ...]:
        return tuple((self.root_pc + s) % 12 for s in self.stack)

    @property
    def label(self) -> str:
        minor = len(self.stack) > 1 and self.stack[1] == 3
        base = ROMAN[self.degree - 1].lower() if minor else ROMAN[self.degree - 1]
        return base + ("" if self.ext == "tri" else self.ext)


def make_chord(key: Key, degree: int, ext: str = "tri") -> Chord:
    """Diatonic chord on scale degree `degree` with extension `ext` (see EXT_STEPS)."""
    if ext in ("9",) and degree in (3, 7):  # the diatonic 9th of iii / vii is a flat 9: drop it
        ext = "7"
    steps = EXT_STEPS[ext]
    midis = [key.midi(degree - 1 + s) for s in steps]
    return Chord(degree, ext, midis[0] % 12, tuple(m - midis[0] for m in midis))


def pitch_near(pc: int, target: float, lo: int, hi: int) -> int:
    """MIDI note with pitch class `pc` inside [lo, hi] that is closest to `target`."""
    return min((m for m in range(lo, hi + 1) if m % 12 == pc), key=lambda m: abs(m - target))


def _voicing_cost(notes: Sequence[int], prev: Sequence[int] | None, center: float) -> float:
    cost = 0.0
    if prev:
        if len(notes) == len(prev):
            cost += sum(abs(a - b) for a, b in zip(notes, prev))
        else:
            cost += sum(min(abs(a - b) for b in prev) for a in notes)
        cost += 0.6 * abs(notes[-1] - prev[-1])  # keep the top voice smooth
    else:
        cost += 2.0 * abs(sum(notes) / len(notes) - center)
    cost += 0.15 * max(0, notes[-1] - notes[0] - 10)
    for a, b in zip(notes, notes[1:]):
        gap = b - a
        if gap == 1:
            cost += 8
        elif gap < 3 and a < 55:
            cost += 6
        elif gap == 2 and a < 64:
            cost += 3
    return cost


def voice_chord(pcs: Sequence[int], lo: int, hi: int, prev: Sequence[int] | None = None,
                center: float | None = None, max_span: int = 16) -> list[int]:
    """Pick one octave per pitch class (duplicates allowed, in different octaves) inside [lo, hi]
    so that the chord is close to the previous voicing (or to `center`). Returns ascending MIDI notes."""
    center = (lo + hi) / 2 if center is None else center
    cands = [[m for m in range(lo, hi + 1) if m % 12 == pc] for pc in pcs]
    for span in (max_span, max_span + 6, 99):
        best, best_cost = None, 1e9
        for combo in itertools.product(*cands):
            notes = sorted(combo)
            if len(set(notes)) < len(notes) or notes[-1] - notes[0] > span:
                continue
            cost = _voicing_cost(notes, prev, center)
            if cost < best_cost:
                best, best_cost = notes, cost
        if best:
            return best
    return sorted(c[0] for c in cands)
