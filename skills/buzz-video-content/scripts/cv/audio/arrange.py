"""Arrangement helpers shared by the BGM styles: voicings, bass notes, timing feel, the landing bells."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from cv.audio.form import Plan
from cv.audio.theory import Chord, Key, pitch_near, voice_chord


def voicings(plan: Plan, lo: int, hi: int, tones: Callable[[Chord], Sequence[int]]) -> dict[tuple[int, int], list[int]]:
    """Smoothly voice-led notes for every chord of the plan: {(bar index, 0 | 1 for the second half): [midi]}."""
    out: dict[tuple[int, int], list[int]] = {}
    prev = None
    for bar in plan.bars:
        for half, ch in enumerate((bar.chord, bar.chord2)):
            if ch is None:
                continue
            prev = voice_chord(tones(ch), lo, hi, prev)
            out[(bar.i, half)] = prev
    return out


def bass_midi(chord: Chord, lo: int, hi: int) -> int:
    """Chord root inside [lo, hi] (closest to the middle of the range)."""
    return pitch_near(chord.root_pc, (lo + hi) / 2, lo, hi)


def human(rng: np.random.Generator, ms: float = 6.0) -> float:
    """Small positive timing offset in seconds (laid-back feel)."""
    return float(rng.uniform(0.0, ms * 1e-3))


def card_bells(key: Key, scale: float = 1.0) -> list[tuple[int, float]]:
    """(midi, level) of the landing chime: tonic, fifth, upper tonic, third (all < ~2.9 kHz)."""
    t, fifth, third = key.tonic, (key.tonic + 7) % 12, (key.tonic + 4) % 12
    r = pitch_near(t, 86, 82, 93)
    spec = [(r, 0.17), (pitch_near(fifth, 93, 88, 99), 0.125), (r + 12, 0.09), (pitch_near(third, 94, 88, 99), 0.07)]
    return [(m, lv * scale) for m, lv in spec]


def sparkle_idx(key: Key) -> list[int]:
    """Scale indices (pentatonic, ascending) for the shimmer after the chime."""
    base = key.nearest_idx(86)
    return [base + o for o in (0, 1, 2, 4, 5, 7)]


def breath(st: np.ndarray, at: float, depth: float = 0.4, pre: float = 0.07) -> np.ndarray:
    """Micro-pause right before the landing: the mix dips to `depth` (linear) for `pre` s and is back at unity exactly
    on `at` (seconds into `st`), so the landing chord and chime stand out."""
    from cv.audio.dsp import SR
    t = np.arange(st.shape[1]) / SR
    down = np.clip((t - (at - pre - 0.010)) / 0.010, 0.0, 1.0)
    up = np.clip((t - (at - 0.004)) / 0.004, 0.0, 1.0)
    return st * (1.0 - (1.0 - depth) * down * (1.0 - up))


def padded(x: np.ndarray, extra: int) -> np.ndarray:
    """x with `extra` samples of silence appended (to add a reverb return that rings past it)."""
    return np.pad(x, ((0, 0), (0, extra)))
