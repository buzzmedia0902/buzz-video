"""Arrangement helpers shared by the four styles: voicings, section melodies, card bells, timing."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from core.composer.melody import Cell, Composer, Note
from core.composer.plan import Grid, Plan, Section
from core.composer.theory import Chord, Key, pitch_near, voice_chord
from core.synth import SR


def voice_events(plan: Plan, lo: int, hi: int, tones: Callable[[Chord], Sequence[int]]) -> list[list[int]]:
    """Smoothly voice-led notes for every chord event; `tones(chord)` returns the pitch classes to play."""
    out: list[list[int]] = []
    prev = None
    for e in plan.events:
        prev = voice_chord(tones(e.chord), lo, hi, prev)
        out.append(prev)
    return out


def bass_midi(chord: Chord, lo: int, hi: int) -> int:
    """Chord root inside [lo, hi] (closest to the middle of the range)."""
    return pitch_near(chord.root_pc, (lo + hi) / 2, lo, hi)


def section_bars(sec: Section, pickup: int) -> list[float]:
    """Beat positions of the full bars inside a section (the pick-up beats are excluded)."""
    start = sec.beat0 + (pickup if sec.beat0 == 0.0 else 0)
    return [start + 4 * i for i in range(int((sec.end - start) // 4))]


def all_bars(grid: Grid) -> list[float]:
    """Beat position of every bar line from the (possibly negative) start of the partial first bar."""
    start = grid.pickup - 4 if grid.pickup else 0
    return [float(start + 4 * i) for i in range((grid.n_beats - start) // 4)]


def chunk_sizes(n: int) -> list[int]:
    """Phrase lengths for n bars: 4s, a remainder of 2-3 stays alone, a remainder of 1 joins the last."""
    sizes = [4] * (n // 4)
    r = n % 4
    if r == 1 and sizes:
        sizes[-1] += 1
    elif r:
        sizes.append(r)
    return sizes


def section_melody(comp: Composer, plan: Plan, cells: Sequence[Cell], cad: Sequence[Cell], *,
                   vel: float = 0.5, kinds: Sequence[str] = ("a", "b")) -> list[Note]:
    """Melody for every section of `kinds`: phrase A is composed once and replayed (with a freshly
    composed final bar) whenever the chords match; phrase B is composed separately and starts higher."""
    last_body = max((s.beat0 for s in plan.of_kind("a", "b", "brk")), default=-1)
    cache: dict[tuple, tuple[float, list[Note]]] = {}
    notes: list[Note] = []
    for sec in plan.of_kind(*kinds):
        bars = section_bars(sec, plan.grid.pickup)
        pos = 0
        for size in chunk_sizes(len(bars)):
            chunk = bars[pos:pos + size]
            pos += size
            labels = (sec.kind, tuple(plan.chord_at(b).label for b in chunk))
            closed = not (pos >= len(bars) and (sec.kind == "b" or sec.beat0 == last_body))
            if labels in cache and size >= 2:
                start0, old = cache[labels]
                head = [Note(n.beat - start0 + chunk[0], n.length, n.midi, n.vel)
                        for n in old if n.beat < start0 + (size - 1) * 4]
                start = comp.key.nearest_idx(head[-1].midi) if head else None
                notes += head + comp.phrase([chunk[-1]], cad, cad, closed=closed, vel=vel, start=start)
                continue
            start = comp.center + 2 if sec.kind == "b" else None
            ph = comp.phrase(chunk, cells, cad, closed=closed, vel=vel, start=start)
            cache.setdefault(labels, (chunk[0], ph))
            notes += ph
    return sorted(notes, key=lambda n: n.beat)


def card_target(key: Key, near_midi: int) -> int:
    """Scale index of the tonic-triad tone (root favoured) closest to `near_midi`: where the melody lands on the card."""
    top = key.nearest_idx(near_midi)
    return min((c for c in range(top - 4, top + 3) if c % 7 in (0, 2, 4)),
               key=lambda c: abs(c - top) - (0.5 if c % 7 == 0 else 0.0))


def card_bells(key: Key, scale: float = 1.0) -> list[tuple[int, float]]:
    """(midi, level) for the end-card chime: tonic, fifth, upper tonic, third (all < ~2.9 kHz)."""
    t, fifth, third = key.tonic, (key.tonic + 7) % 12, (key.tonic + 4) % 12
    r = pitch_near(t, 86, 82, 93)
    spec = [(r, 0.17), (pitch_near(fifth, 93, 88, 99), 0.125), (r + 12, 0.09), (pitch_near(third, 94, 88, 99), 0.07)]
    return [(m, lv * scale) for m, lv in spec]


def sparkle_idx(key: Key) -> list[int]:
    """Scale indices (pentatonic, ascending) for the shimmer after the chime."""
    base = key.nearest_idx(86)
    return [base + o for o in (0, 1, 2, 4, 5, 7)]


def human(rng: np.random.Generator, ms: float = 6.0) -> float:
    """Small positive timing offset in seconds (laid-back feel)."""
    return float(rng.uniform(0.0, ms * 1e-3))


def breath(st: np.ndarray, card_at: float, depth: float = 0.35, pre: float = 0.075, ramp: float = 0.004) -> np.ndarray:
    """Micro-pause right before the card: the mix dips to `depth` (linear) for `pre` s and is back at unity
    exactly on `card_at`, so the chime/chord onset stands out clearly (and is easy to verify)."""
    t = np.arange(st.shape[1]) / SR
    down = np.clip((t - (card_at - pre - 0.010)) / 0.010, 0.0, 1.0)        # 1 -> depth over 10 ms
    up = np.clip((t - (card_at - ramp)) / ramp, 0.0, 1.0)                  # depth -> 1 by the downbeat
    g = 1.0 - (1.0 - depth) * down * (1.0 - up)
    return st * g
