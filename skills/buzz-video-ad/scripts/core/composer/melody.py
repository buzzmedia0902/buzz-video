"""Seeded, motif-based melody generator and build-up gestures.

A phrase is a few bars that share rhythm cells (so the ear hears a motif), move mostly by scale step,
land chord tones on strong beats, follow a gentle arch and end on a stable tone (closed) or on the
2nd/5th degree (open). All pitches are diatonic scale indices converted to MIDI through the ``Key``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.composer.plan import Plan

Cell = tuple[tuple[float, float], ...]  # ((offset in bar, length) in beats, ...)
STEP_W = {0: 0.45, 1: 1.0, 2: 0.55, 3: 0.14}  # weight by |scale steps| (no leaps beyond a 4th)


@dataclass(frozen=True)
class Note:
    beat: float
    length: float
    midi: int
    vel: float


class Composer:
    """Generates melodic material over a ``Plan`` inside the MIDI register [lo, hi]."""

    def __init__(self, plan: Plan, rng: np.random.Generator, lo: int, hi: int) -> None:
        self.plan, self.rng, self.key = plan, rng, plan.key
        self.lo_i, self.hi_i = self.key.idx_range(lo, hi)
        self.center = (self.lo_i + self.hi_i) // 2

    # ------------------------------------------------------------ helpers
    def is_chord_tone(self, idx: int, beat: float) -> bool:
        """True if scale index `idx` is a tone of the chord sounding at `beat`."""
        return self.key.midi(idx) % 12 in self.plan.chord_at(beat).pcs

    def clashes(self, idx: int, beat: float) -> bool:
        """True for a non-chord tone a semitone above a chord tone (e.g. the 4th over a major chord)."""
        pcs = self.plan.chord_at(beat).pcs
        pc = self.key.midi(idx) % 12
        return pc not in pcs and any((pc - p) % 12 == 1 for p in pcs)

    def nearest_chord_tone(self, idx: int, beat: float) -> int:
        """Closest chord-tone scale index within three steps of `idx` (else `idx`)."""
        cands = [c for c in range(max(self.lo_i, idx - 3), min(self.hi_i, idx + 3) + 1)
                 if self.is_chord_tone(c, beat)]
        return min(cands, key=lambda c: abs(c - idx)) if cands else idx

    def _pick(self, cands: Sequence[int], weights: Sequence[float]) -> int:
        w = np.asarray(weights, dtype=float) + 1e-9
        return int(cands[int(self.rng.choice(len(cands), p=w / w.sum()))])

    def start_idx(self, beat: float) -> int:
        """A chord tone near the middle of the register to start a phrase on."""
        near = [c for c in range(self.center - 3, self.center + 4) if self.is_chord_tone(c, beat)]
        return self._pick(near, [1.0] * len(near)) if near else self.center

    # ------------------------------------------------------------ walk
    def _walk(self, cell: Cell, b0: float, idx: int, prog0: float, span: float, state: dict) -> list[int]:
        """Random walk over scale indices for one bar. The weights favour steps, chord tones on strong
        beats and a gentle arch (`guide`) that rises toward the middle of the phrase and settles at its end."""
        out = []
        for off, ln in cell:
            strong = off in (0.0, 2.0) or ln >= 1.5
            guide = self.center + 2.0 * np.sin(np.pi * min(1.0, prog0 + span * off / 4))
            cands = [c for c in range(idx - 3, idx + 4) if self.lo_i <= c <= self.hi_i]
            w = []
            for c in cands:
                d = c - idx
                wt = STEP_W.get(abs(d), 0.0) * np.exp(-abs(c - guide) / 2.5)
                if d * state["dir"] < 0 and abs(state["step"]) >= 2:
                    wt *= 1.8  # after a leap, step back
                if d == 0 and state["same"] >= 2:
                    wt *= 0.15
                if strong and not self.is_chord_tone(c, b0 + off):
                    wt *= 0.22
                if self.clashes(c, b0 + off):  # avoid notes: tolerated only as quick passing tones
                    wt *= 0.04 if (strong or ln >= 1) else 0.4
                w.append(wt)
            new = self._pick(cands, w)
            state["same"] = state["same"] + 1 if new == idx else 0
            state["step"], state["dir"] = new - idx, int(np.sign(new - idx)) or state["dir"]
            idx = new
            out.append(idx)
        return out

    def _cadence(self, idx: int, closed: bool, beat: float) -> int:
        """Final note of a phrase: tonic-triad tone (closed) or 2nd/5th degree (open), preferably also a tone
        of the chord sounding at `beat` and never a clash with it."""
        degs = (0, 2, 4) if closed else (1, 4)
        wts = {0: 1.0, 2: 0.9, 4: 0.7, 1: 0.8}
        cands = [c for c in range(max(self.lo_i, idx - 3), min(self.hi_i, idx + 3) + 1) if c % 7 in degs]
        if not cands:
            return idx
        fit = [3.0 if self.is_chord_tone(c, beat) else 0.1 if self.clashes(c, beat) else 1.0 for c in cands]
        return self._pick(cands, [wts[c % 7] * np.exp(-abs(c - idx) / 2.0) * f for c, f in zip(cands, fit)])

    # ------------------------------------------------------------ phrases
    def phrase(self, bar_beats: Sequence[float], cells: Sequence[Cell], cad_cells: Sequence[Cell], *,
               closed: bool = True, vel: float = 0.5, repeat_p: float = 0.7,
               start: int | None = None) -> list[Note]:
        """Melody for bars starting at `bar_beats`: bar 2 repeats bar 1's rhythm (and, sequenced over
        the new chord, its contour) with probability `repeat_p`; the last bar uses a cadence cell."""
        k = len(bar_beats)
        cell0 = cells[int(self.rng.integers(len(cells)))]
        rhythm = [cell0]
        for i in range(1, k):
            if i == 1 and self.rng.random() < repeat_p:
                rhythm.append(cell0)
            elif i == 2 and k >= 4 and self.rng.random() < 0.4:
                rhythm.append(cell0)
            else:
                rhythm.append(cells[int(self.rng.integers(len(cells)))])
        if k >= 2:
            rhythm[-1] = cad_cells[int(self.rng.integers(len(cad_cells)))]
        idx = self.start_idx(bar_beats[0]) if start is None else start
        state = {"dir": 1, "step": 0, "same": 0}
        notes: list[Note] = []
        motif: list[int] | None = None
        for i, (b0, cell) in enumerate(zip(bar_beats, rhythm)):
            prog = i / k
            if i == 1 and cell is rhythm[0] and motif:
                shift = int(self.rng.choice([0, 1, -1, 2], p=[0.35, 0.3, 0.2, 0.15]))
                idxs = [min(self.hi_i, max(self.lo_i, motif[0] + shift + d)) for d in
                        (m - motif[0] for m in motif)]
                idxs = [self.nearest_chord_tone(c, b0 + o) if (o in (0.0, 2.0)) else c
                        for c, (o, _) in zip(idxs, cell)]
            else:
                idxs = self._walk(cell, b0, idx, prog, 1.0 / k, state)
            if i == 0:
                motif = list(idxs)
            if i == k - 1 and idxs:
                idxs[-1] = self._cadence(idxs[-1], closed, b0 + cell[-1][0])
            idx = idxs[-1]
            for j, ((off, ln), c) in enumerate(zip(cell, idxs)):
                accent = 1.0 if off == 0.0 else 0.92 if off == 2.0 else 0.84
                arch_v = 0.88 + 0.12 * np.sin(np.pi * min(1.0, (i + off / 4) / k))
                v = vel * accent * arch_v * float(self.rng.uniform(0.94, 1.06))
                notes.append(Note(b0 + off, ln, self.key.midi(c), min(1.0, v)))
        return notes

    def run(self, beat0: float, beats: float, step: float, end_idx: int, *, arp: bool = False,
            vel0: float = 0.28, vel1: float = 0.6) -> list[Note]:
        """Ascending build gesture that arrives on scale index `end_idx` on its last note."""
        n = max(1, int(round(beats / step)))
        gaps = [(2 if (arp and j % 2 == 0) else 1) for j in range(n - 1)][::-1]
        idxs = [end_idx]
        for g in gaps:
            idxs.append(idxs[-1] - g)
        idxs = [c for c in idxs[::-1] if c >= self.lo_i]  # drop notes that fall below the register
        m = len(idxs)
        first = beat0 + (n - m) * step
        return [Note(first + j * step, step * 1.4, self.key.midi(c), vel0 + (vel1 - vel0) * (j + 1) / m)
                for j, c in enumerate(idxs)]
