"""Style "acoustic" (親しみ tone, ~96 BPM): nylon-string guitar, marimba, soft shaker / snap percussion, warm bass.

Arpeggiated fingerpicking in the verses (a, a2), strummed chords in the lifts (b), only shaker and a thin arpeggio
in a break; the melody is a marimba (and a glock in the lifts). Everything is plucked or tapped, so the bed stays
transparent under speech. The build is a rising marimba run with a brush swell; the landing is a rolled full chord,
a glock chime and a held bass note.
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
    bpm=96.0,
    loops_a=((1, 5, 6, 4), (1, 4, 5, 4), (6, 4, 1, 5), (1, 6, 4, 5), (4, 1, 5, 6)),
    loops_b=((4, 5, 1, 6), (6, 5, 4, 5), (4, 1, 5, 1), (1, 3, 4, 5)),
    ext={1: "add9", 2: "tri", 3: "tri", 4: "add9", 5: "tri", 6: "tri", 7: "tri"},
    card_ext="add9", dom_ext="tri", sus=False,
)
ARP = ((0, 2, 3, 2, 1, 2, 3, 2), (0, 2, 1, 3, 2, 3, 1, 2), (0, 1, 2, 3, 2, 1, 2, 3), (0, 3, 2, 1, 3, 2, 1, 2))
STRUM = ((0, "d", 1.0), (2, "d", 0.7), (3, "u", 0.55), (5, "u", 0.6), (6, "d", 0.85), (7, "u", 0.55))    # 8th, direction, vel
CELLS = (((0, 1.5), (1.5, .5), (2, 1), (3, 1)), ((0, 1), (1, .5), (1.5, .5), (2, 1), (3, 1)),
         ((0, .5), (.5, .5), (1, 1), (2, .5), (2.5, .5), (3, 1)), ((.5, .5), (1, 1), (2, 1.5), (3.5, .5)))
CAD = (((0, 1), (1, 1), (2, 2)), ((0, .5), (.5, .5), (1, 1), (2, 2)), ((0, 1.5), (1.5, .5), (2, 2)))
GUITAR_GAIN, MELODY_GAIN, BASS_GAIN, PERC_GAIN, PAD_GAIN = 0.34, 0.26, 0.30, 1.0, 0.07


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    plan.extra["voic"] = A.voicings(plan, 50, 74, lambda c: c.pcs)
    plan.extra["arp"] = {b.i: ARP[int(rng.integers(len(ARP)))] for b in plan.blocks}
    plan.extra["melody"] = block_melody(plan, rng, CELLS, CAD, lo=69, hi=86, vel=0.55, kinds=("a", "a2", "b"))


def make_ctx(plan: Plan, seed: int) -> dict:
    rng = np.random.default_rng([seed, 2, 99])
    return {"room": dsp.make_ir(rng, 1.3, hf_rt60=0.6, predelay=0.015),
            "kick": S.lp(S.kick(rng, f0=120.0, f1=75.0, p_tau=0.03, a_tau=0.14, drive=1.2, click=0.0), 900.0),
            "shakers": [S.lp(S.shaker(rng, tau=0.035, lo=4500.0, hi=11000.0), 11000.0) for _ in range(4)],
            "snaps": [S.snap(rng) for _ in range(3)], "brushes": [S.brush_hit(rng, 0.12) for _ in range(3)]}


def _guitar_bar(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, guitar: Bus) -> None:
    """Fingerpicked arpeggio (a, a2, brk, intro) or strummed chord (b, swells) of one bar; split in halves for builds."""
    arp = plan.extra["arp"].get(bar.block, ARP[0])
    strum = bar.kind == "b" or (bar.energy >= 0.8 and bar.kind != "build")
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        notes = plan.extra["voic"][(bar.i, half)]
        b0 = bar.beat0 + half * bar.nbeats / 2
        span8 = bar.nbeats * 2 // (2 if bar.chord2 is not None else 1)
        thin = bar.kind in ("brk", "intro") and bar.energy < 0.8
        for s in range(span8):
            tt = plan.t(b0 + s / 2) - tc
            if strum:
                for sp, d, v in STRUM:
                    if sp == s % 8:
                        seq = notes if d == "d" else notes[::-1]
                        for i, m in enumerate(seq[:5]):
                            guitar.add(S.nylon_guitar(S.midi_hz(m), 0.7 * plan.beat, 0.55 * v * (1 - 0.06 * i), 0.7),
                                       tt + 0.014 * i + A.human(rng, 5), pan=-0.3 + 0.12 * i)
            elif not (thin and s % 2):
                i = arp[s % 8] % len(notes)
                v = (0.62 if s % 2 == 0 else 0.45) * (0.8 + 0.4 * bar.energy) * rng.uniform(0.9, 1.05)
                guitar.add(S.nylon_guitar(S.midi_hz(notes[i] + (12 if s % 8 in (3, 5, 7) and i == 0 else 0)),
                                          0.9 * plan.beat, v, 0.45), tt + A.human(rng, 7), pan=-0.25 + 0.08 * i)


def _bass_bar(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, bass: Bus) -> None:
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        root = A.bass_midi(ch, 38, 50)
        b0 = bar.beat0 + half * bar.nbeats / 2
        pat = [(0, 1.6, 0.95)] + ([(2, 1.2, 0.7)] if bar.nbeats == 4 and bar.chord2 is None and bar.kind != "brk" else [])
        for off, ln, v in pat:
            m = root + (7 if off == 2 and bar.kind in ("a2", "b") else 0)
            bass.add(S.round_bass(S.midi_hz(m), ln * plan.beat, v * (0.7 + 0.4 * bar.energy)),
                     plan.t(b0 + off) - tc + A.human(rng, 4))


def _perc_bar(plan: Plan, bar: Bar, ctx: dict, rng: np.random.Generator, tc: float, perc: Bus, room: Bus) -> None:
    kind, e = bar.kind, bar.energy
    lvl = 0.0 if (kind == "intro" and bar.i == 0 and e < 0.8) else (0.4 if kind in ("intro", "brk") else 1.0)
    for s in range(bar.nbeats * 2):
        t = plan.t(bar.beat0 + s / 2) - tc
        if lvl and (kind != "brk" or s % 2 == 0):
            perc.add(ctx["shakers"][(bar.i + s) % 4], t + rng.uniform(-0.004, 0.004), pan=0.3,
                     gain=0.06 * lvl * (1.0 if s % 2 == 0 else 0.6) * rng.uniform(0.85, 1.1))
        beat = s / 2
        if kind in ("a2", "b", "build") or e >= 0.8:
            if beat in (0.0, 2.0):
                perc.add(ctx["kick"], t, gain=0.42 if kind == "b" else 0.3)
            if beat in (1.0, 3.0):
                sn = ctx["snaps"][(bar.i + s) % 3]
                perc.add(sn, t + 0.004, pan=-0.15, gain=0.16)
                room.add(sn, t + 0.004, gain=0.2)


def _landing(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, guitar: Bus, bass: Bus, fx: Bus,
             perc: Bus) -> None:
    t0, tail = bar.t0 - tc, bar.t1 - bar.t0
    notes = plan.extra["voic"][(bar.i, 0)]
    for i, m in enumerate(notes):
        guitar.add(S.nylon_guitar(S.midi_hz(m), min(2.8, tail), 0.6, 0.6), t0 + 0.03 * i, pan=-0.3 + 0.12 * i)
    bass.add(S.round_bass(S.midi_hz(A.bass_midi(bar.chord, 38, 50)), min(2.4, tail), 0.9), t0)
    perc.add(S.round_bass(S.midi_hz(48), 0.3, 0.8), t0, gain=0.3)                  # soft thump under the chord
    for i, (m, lv) in enumerate(A.card_bells(plan.key, 0.8)):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 3.0, index=1.2, a_tau=1.1), t0 + 0.04 * i, pan=-0.3 + 0.2 * i)
    for k, i in enumerate(A.sparkle_idx(plan.key)):
        fx.add(S.marimba(S.midi_hz(plan.key.midi(i)), 0.4, 0.28 * (1 - 0.08 * k)), t0 + 0.25 + k * 0.06, pan=-0.5 + 0.2 * k)


def _build(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, melody: Bus, perc: Bus, ctx: dict) -> None:
    comp = Composer(plan, rng, 67, 86)
    end = comp.key.nearest_idx(79)
    b_end = bar.beat0 + bar.nbeats
    for n in comp.run(b_end - 2, 2.0, 0.25, end, vel0=0.3, vel1=0.65):
        melody.add(S.marimba(S.midi_hz(n.midi), n.length * plan.beat, n.vel), plan.t(n.beat) - tc, pan=0.15)
    for i in range(8):
        perc.add(ctx["brushes"][i % 3], plan.t(b_end - 2 + i / 4) - tc, gain=0.05 + 0.025 * i)


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    tc, span = bars[0].t0, bars[-1].t1 - bars[0].t0
    guitar, melody, bass, perc, room, fx, pad = (Bus(span) for _ in range(7))
    for bar in bars:
        if bar.kind == "land":
            _landing(plan, bar, rng, tc, guitar, bass, fx, perc)
            continue
        _guitar_bar(plan, bar, rng, tc, guitar)
        _bass_bar(plan, bar, rng, tc, bass)
        _perc_bar(plan, bar, ctx, rng, tc, perc, room)
        for half, ch in enumerate((bar.chord, bar.chord2)):
            if ch is not None:
                pad.add(S.warm_pad(rng, [S.midi_hz(m) for m in plan.extra["voic"][(bar.i, half)]],
                                   plan.t(bar.nbeats) / (2 if bar.chord2 is not None else 1) + 0.2, PAD_GAIN * (0.6 + 0.8 * bar.energy)),
                        plan.t(bar.beat0 + half * bar.nbeats / 2) - tc)
        if plan.bars[bar.i + 1].kind == "land" and bar.kind == "build":
            _build(plan, bar, rng, tc, melody, perc, ctx)
    lo, hi = bars[0].beat0, bars[-1].beat0 + bars[-1].nbeats
    for n in plan.extra["melody"]:
        if lo <= n.beat < hi:
            bar = plan.bars[plan.bar_index(n.beat)]
            if bar.kind == "b":
                melody.add(S.fm_bell(S.midi_hz(n.midi + 12), 0.1 * n.vel, 1.4, index=0.9, i_tau=0.12, a_tau=0.5), plan.t(n.beat) - tc, pan=0.3)
            melody.add(S.marimba(S.midi_hz(n.midi), n.length * plan.beat * 0.95, n.vel), plan.t(n.beat) - tc, pan=0.15)
    n = guitar.st.shape[1]
    ir = ctx["room"]
    tail = ir.shape[1] - 1

    def rv(x: np.ndarray) -> np.ndarray:
        return dsp.reverb(x, ir, hp_fc=200.0, keep_tail=True)

    g = GUITAR_GAIN * S.hp(S.lp(guitar.st, 7000.0), 90.0)
    m = MELODY_GAIN * S.hp(S.lp(melody.st, 8000.0), 250.0)
    dry = g + m + BASS_GAIN * S.hp(S.lp(bass.st, 600.0), 50.0) + PERC_GAIN * S.hp(perc.st, 60.0) + pad.st + S.lp(fx.st, 12000.0)
    body = A.padded(dry, tail) + 0.5 * rv(g) + 0.55 * rv(m) + 0.3 * rv(room.st) + 0.4 * rv(fx.st)
    if bars[-1].kind == "land":
        body = A.breath(body, bars[-1].t0 - tc, 0.5, 0.06)
    return body
