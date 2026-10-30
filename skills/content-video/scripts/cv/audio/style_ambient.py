"""Style "ambient" (洗練・ミニマル tone, ~64 BPM): slow pads, a soft sub, sparse piano and bell notes, a breath of air.

No drums and no pulse: the grid only times the chord changes (one chord per bar, pads overlap) and the scattered
notes, which are drawn per bar from the chord and the scale. Energy adds strings and more notes (lifts, card
swells); a break leaves the pad alone. Everything sits in a long hall, so it stays out of the speech band.
"""
from __future__ import annotations

import numpy as np

from cv.audio import arrange as A
from cv.audio import dsp
from cv.audio import instruments as S
from cv.audio.dsp import Bus
from cv.audio.form import Bar, Plan, StyleCfg
from cv.audio.theory import pitch_near

CFG = StyleCfg(
    bpm=64.0,
    loops_a=((1, 4, 6, 5), (6, 4, 1, 5), (1, 5, 6, 4), (4, 1, 5, 6), (1, 6, 4, 5)),
    loops_b=((4, 1, 5, 6), (6, 5, 4, 1), (1, 3, 4, 5)),
    ext={1: "9", 2: "sus2", 3: "7", 4: "9", 5: "sus2", 6: "9", 7: "7"},
    card_ext="9", dom_ext="sus2", sus=False,
)
PENTA = (0, 2, 4, 7, 9)           # semitones above the tonic


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    plan.extra["voic"] = A.voicings(plan, 48, 76, lambda c: c.pcs)


def make_ctx(plan: Plan, seed: int) -> dict:
    return {"hall": dsp.make_ir(np.random.default_rng([seed, 5, 99]), 4.2, 1.6, 0.03)}


def _scatter(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, notes_bus: Bus, bells: Bus) -> None:
    """A few soft notes per bar on the 8th grid: chord tones favoured, pentatonic colour tones otherwise."""
    if bar.kind in ("brk",) or bar.nbeats < 4:
        return
    count = int(rng.poisson(0.6 + 2.2 * bar.energy))
    slots = rng.choice(8, size=min(count, 6), replace=False) if count else []
    pcs = list(bar.chord.pcs)
    for s in sorted(slots):
        if rng.random() < 0.6:
            pc = int(rng.choice(pcs))
        else:
            pc = (plan.key.tonic + int(rng.choice(PENTA))) % 12
        midi = pitch_near(pc, 74 if rng.random() < 0.5 else 67, 60, 84)
        t = plan.t(bar.beat0 + s / 2) - tc + rng.uniform(0.0, 0.04)
        pan = float(rng.uniform(-0.55, 0.55))
        if rng.random() < 0.62:
            notes_bus.add(S.felt_piano(S.midi_hz(midi), 1.2, float(rng.uniform(0.16, 0.3))), t, pan=pan)
        else:
            bells.add(S.fm_bell(S.midi_hz(min(midi + 12, 88)), float(rng.uniform(0.025, 0.05)), 3.0, index=0.6, i_tau=0.15, a_tau=1.1), t, pan=pan)


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    ir = ctx["hall"]
    tc, span = bars[0].t0, bars[-1].t1 - bars[0].t0
    pad, low, keys, bells, strings, fx = (Bus(span) for _ in range(6))
    for bar in bars:
        for half, ch in enumerate((bar.chord, bar.chord2)):
            if ch is None:
                continue
            notes = plan.extra["voic"][(bar.i, half)]
            beat = bar.beat0 + half * bar.nbeats / 2
            length = plan.t(bar.nbeats) / (2 if bar.chord2 is not None else 1)
            t0 = plan.t(beat) - tc
            lvl = (0.10 + 0.12 * bar.energy) * (1.5 if bar.kind in ("build", "land") else 1.0)
            dur = (min(length + 1.6, 6.0) if bar.kind != "land" else bar.t1 - bar.t0)
            pad.add(S.warm_pad(rng, [S.midi_hz(m) for m in notes], dur, lvl), max(0.0, t0 - 0.3))
            low.add(S.sub_sine(S.midi_hz(A.bass_midi(ch, 40, 52)), min(length + 0.8, 5.0), 0.05 + 0.05 * bar.energy), t0)
            if bar.energy >= 0.7 or bar.kind == "land":
                strings.add(S.strings(rng, [S.midi_hz(m + 12) for m in notes[1:4]], min(length + 1.0, 6.0), 0.9, 1.3),
                            max(0.0, t0 - 0.2), gain=0.55 * bar.energy)
        if bar.kind != "land":
            _scatter(plan, bar, rng, tc, keys, bells)
    last = bars[-1]
    if last.kind == "land":
        t0, tail = last.t0 - tc, last.t1 - last.t0
        for i, (m, lv) in enumerate(A.card_bells(plan.key, 0.9)):
            fx.add(S.fm_bell(S.midi_hz(m), lv, 4.0, index=1.3, a_tau=1.4), t0 + 0.05 * i, pan=-0.3 + 0.2 * i)
        for i, m in enumerate(plan.extra["voic"][(last.i, 0)][:3]):
            keys.add(S.felt_piano(S.midi_hz(m + 12), min(tail, 4.0), 0.3), t0 + 0.03 * i, pan=-0.2 + 0.2 * i)
    prev = plan.bars[last.i - 1] if last.kind == "land" and last.i else None
    if prev is not None and prev.kind == "build":
        rl = min(plan.bar_len, 3.0)
        fx.add(S.lp(S.riser(rng, rl, 250.0, 3000.0, 2.2), 3500.0), last.t0 - tc - rl, gain=0.07)
    air = S.air(rng, pad.st.shape[1] / dsp.SR) * (0.004 + 0.004 * float(np.mean([b.energy for b in bars])))
    tail = ir.shape[1] - 1

    def rv(x: np.ndarray) -> np.ndarray:
        return dsp.reverb(x, ir, hp_fc=160.0, keep_tail=True)

    wet = S.lp(keys.st + bells.st + strings.st, 6500.0)
    dry = S.hp(pad.st, 70.0) + S.hp(low.st, 40.0) + S.lp(keys.st + bells.st + strings.st, 7500.0) + air + S.lp(fx.st, 9000.0)
    body = A.padded(dry, tail) + 0.5 * rv(pad.st) + 0.85 * rv(wet) + 0.7 * rv(fx.st)
    if bars[-1].kind == "land":
        body = A.breath(body, bars[-1].t0 - tc, 0.5, 0.08)
    return body
