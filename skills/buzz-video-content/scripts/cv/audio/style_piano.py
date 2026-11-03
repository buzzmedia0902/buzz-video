"""Style "piano" (~84 BPM): felt piano + warm pad + soft sub + FM bell, no drums.

Sparse intro -> arpeggiated groove with a sung-like melody -> brighter b blocks -> a rising arpeggio over a pre-dominant
and V build with a noise swell -> tonic 9th chord, FM bell chime and a sparkle on the landing, then the piano settles on
a short falling figure while the hall rings out. Ported from buzz-video-ad's piano composer, re-planned for any length.
"""
from __future__ import annotations

import numpy as np

from cv.audio import arrange as A
from cv.audio import dsp
from cv.audio import instruments as S
from cv.audio.dsp import Bus
from cv.audio.form import Bar, Plan, StyleCfg
from cv.audio.melody import Composer, block_melody

CFG = StyleCfg(
    bpm=84.0,
    loops_a=((1, 5, 6, 4), (6, 4, 1, 5), (1, 3, 6, 4), (4, 1, 5, 6), (1, 4, 6, 5)),
    loops_b=((4, 5, 3, 6), (6, 5, 4, 1), (4, 1, 2, 5), (2, 5, 1, 4)),
    ext={1: "9", 2: "9", 3: "7", 4: "9", 5: "add9", 6: "9", 7: "7"},
    card_ext="9", dom_ext="add9", sus=True,
)
CELLS = (((0, 1.5), (1.5, .5), (2, 1), (3, 1)), ((0, 1), (1, .5), (1.5, .5), (2, 1), (3, 1)),
         ((0, .5), (.5, .5), (1, 1), (2, .5), (2.5, .5), (3, 1)), ((.5, .5), (1, 1), (2, 1.5), (3.5, .5)),
         ((0, 2), (2, .5), (2.5, .5), (3, 1)), ((0, 1), (1, 1), (2, 1.5), (3.5, .5)))
CAD = (((0, 1), (1, 1), (2, 2)), ((0, .5), (.5, .5), (1, 1), (2, 2)), ((0, 1.5), (1.5, .5), (2, 2)))
ARP = ((0, 2, 1, 3, 2, 1, 3, 2), (0, 1, 2, 3, 2, 1, 2, 3), (0, 2, 3, 1, 2, 3, 1, 2), (0, 1, 3, 2, 1, 3, 2, 1))
MASKS = ((1, 0, 1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 0, 1, 0), (1, 0, 1, 0, 1, 1, 1, 0))


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    plan.extra["voic"] = A.voicings(plan, 52, 76, lambda c: c.pcs[1:] if len(c.pcs) > 3 else c.pcs)
    plan.extra["pat"] = {b.i: (ARP[int(rng.integers(len(ARP)))], MASKS[int(rng.integers(len(MASKS)))]) for b in plan.blocks}
    plan.extra["melody"] = block_melody(plan, rng, CELLS, CAD, lo=66, hi=83, vel=0.48, kinds=("a", "a2", "b"))


def make_ctx(plan: Plan, seed: int) -> dict:
    return {"ir": dsp.make_ir(np.random.default_rng([seed, 6, 99]), 3.2, 1.4, 0.022)}


def _chords(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, piano: Bus, pad: dict, low: Bus) -> None:
    """Per chord: warm pad, soft sub, rolled piano strike and a left-hand root."""
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        beat = bar.beat0 + half * bar.nbeats / 2
        length = bar.nbeats / (2 if bar.chord2 is not None else 1)
        t0, dur = plan.t(beat) - tc, plan.t(length)
        notes = plan.extra["voic"][(bar.i, half)]
        freqs = [S.midi_hz(m) for m in notes]
        pad.add(S.warm_pad(rng, freqs, dur + 0.15, 0.08 + 0.10 * bar.energy), max(0.0, t0 - 0.05))
        root = A.bass_midi(ch, 40, 52)
        low.add(S.sub_sine(S.midi_hz(root), dur, 0.05 + 0.04 * bar.energy), t0)
        spread = 0.006 if bar.kind == "land" else 0.018
        for i, f in enumerate(freqs):
            piano.add(S.felt_piano(f, dur * 0.95, 0.17 + 0.08 * bar.energy), t0 + i * spread, pan=-0.25 + 0.15 * i)
        piano.add(S.felt_piano(S.midi_hz(root + 12), dur * 0.9, 0.22), t0, pan=-0.3)


def _arps(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, piano: Bus) -> None:
    pat, mask = plan.extra["pat"].get(bar.block, (ARP[0], MASKS[0]))
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        notes = plan.extra["voic"][(bar.i, half)]
        b0 = bar.beat0 + half * bar.nbeats / 2
        for j in range(int(bar.nbeats * 2 / (2 if bar.chord2 is not None else 1))):
            if (bar.kind in ("a", "a2") and not mask[j % 8]) or (bar.kind in ("brk", "intro") and j % 2):
                continue
            if bar.i == 0 and j < 2:
                continue                                        # let the opening chord ring alone for a beat
            v = (0.095 + 0.035 * (j % 2 == 0)) * (0.8 + 0.4 * bar.energy) + rng.uniform(-0.012, 0.012)
            f = S.midi_hz(notes[pat[j % 8] % len(notes)] + 12)
            piano.add(S.felt_piano(f, 0.7 * plan.beat, v), plan.t(b0 + j * 0.5) - tc + A.human(rng, 8), pan=0.3)


def _landing(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, piano: Bus, fx: Bus) -> None:
    """Melody lands on the tonic-triad tone, bells chime on the downbeat, sparkle, falling settle."""
    t0, tail, key = bar.t0 - tc, bar.t1 - bar.t0, plan.key
    top = key.nearest_idx(77)
    target = min((c for c in range(top - 4, top + 3) if c % 7 in (0, 2, 4)), key=lambda c: abs(c - top) - (0.5 if c % 7 == 0 else 0.0))
    piano.add(S.felt_piano(S.midi_hz(key.midi(target)), min(tail * 0.9, 4.0), 0.56), t0, pan=0.08)
    for k, idx in enumerate((target - 2, target - 4)):
        t = t0 + (1.6 + 1.5 * k) * plan.beat
        if t < t0 + tail - 0.9:
            piano.add(S.felt_piano(S.midi_hz(key.midi(idx)), 1.5 * plan.beat, 0.4 - 0.08 * k), t, pan=0.08)
    for i, (m, lv) in enumerate(A.card_bells(key)):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 3.4, index=2.0, a_tau=1.25), t0 + i * 0.045, pan=-0.3 + 0.2 * i)
    for k, idx in enumerate(A.sparkle_idx(key)):
        fx.add(S.fm_bell(S.midi_hz(key.midi(idx)), 0.045 * (1 - 0.07 * k), 1.2, index=0.6, i_tau=0.1, a_tau=0.35),
               t0 + 0.3 + k * 0.04, pan=-0.6 + 0.24 * k)


def _build(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, piano: Bus, rise: Bus) -> None:
    comp = Composer(plan, rng, 66, 83)
    top = comp.key.nearest_idx(77)
    target = min((c for c in range(top - 4, top + 3) if c % 7 in (0, 2, 4)), key=lambda c: abs(c - top) - (0.5 if c % 7 == 0 else 0.0))
    approach = target - 1 if rng.random() < 0.6 else target + 1
    b_end = bar.beat0 + bar.nbeats
    for n in comp.run(b_end - 2, 2.0, 0.5, approach, arp=bool(rng.random() < 0.5), vel0=0.3, vel1=0.58):
        piano.add(S.felt_piano(S.midi_hz(n.midi), n.length * plan.beat, n.vel), plan.t(n.beat) - tc, pan=0.1)
    swell = min(plan.bar_len, 2.6)
    rise.add(S.riser(rng, swell, 300.0, 4500.0, 2.2), plan.t(b_end) - tc - swell, gain=0.10)


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    ir = ctx["ir"]
    tc, span = bars[0].t0, bars[-1].t1 - bars[0].t0
    piano, pad, low, fx, rise = (Bus(span) for _ in range(5))
    for bar in bars:
        if bar.kind == "land":
            _landing(plan, bar, rng, tc, piano, fx)
            _chords(plan, bar, rng, tc, piano, pad, low)
            continue
        _chords(plan, bar, rng, tc, piano, pad, low)
        _arps(plan, bar, rng, tc, piano)
        if bar.kind == "build" and plan.bars[bar.i + 1].kind == "land":
            _build(plan, bar, rng, tc, piano, rise)
        if bar.kind == "b" and bar.k == 0 and bar.beat0 > 4:                 # light lift into every b block
            rise.add(S.riser(rng, 0.9, 400.0, 4000.0, 2.0), bar.t0 - tc - 0.9, gain=0.05)
    lo, hi = bars[0].beat0, bars[-1].beat0 + bars[-1].nbeats
    for n in plan.extra["melody"]:
        if lo <= n.beat < hi:
            piano.add(S.felt_piano(S.midi_hz(n.midi), n.length * plan.beat * 0.98, n.vel), plan.t(n.beat) - tc + A.human(rng, 10), pan=0.08)
    tail = ir.shape[1] - 1

    def rv(x: np.ndarray) -> np.ndarray:
        return dsp.reverb(x, ir, keep_tail=True)

    dry = piano.st + pad.st + low.st + rise.st + fx.st
    body = A.padded(dry, tail) + 0.44 * rv(piano.st) + 0.68 * rv(pad.st) + 0.2 * rv(rise.st) + 0.76 * rv(fx.st)
    if bars[-1].kind == "land":
        body = A.breath(body, bars[-1].t0 - tc, 0.35, 0.075)
    return body
