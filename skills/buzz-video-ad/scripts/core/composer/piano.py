"""Style "piano" (~100 BPM): felt piano + warm pad + soft sub + FM bell + noise swell into the end card.

Form (see plan.py): sparse intro -> arpeggiated groove with a sung-like melody -> brighter b section ->
rising arpeggio/run over a IV-V(sus4)-V build with a noise swell -> tonic 9th chord, FM bell chime and a
sparkle on the card, then the piano settles on a short falling figure while the hall rings out.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core import synth as S
from core.composer import arrange as A
from core.composer.context import Ctx
from core.composer.melody import Composer
from core.composer.plan import StyleCfg
from core.composer.theory import Chord, pitch_near

CFG = StyleCfg(
    bpm=100.0,
    loops_a=((1, 5, 6, 4), (6, 4, 1, 5), (1, 3, 6, 4), (4, 1, 5, 6), (1, 4, 6, 5)),
    loops_b=((4, 5, 3, 6), (6, 5, 4, 1), (4, 1, 2, 5), (2, 5, 1, 4)),
    ext={1: "9", 2: "9", 3: "7", 4: "9", 5: "add9", 6: "9", 7: "7"},
    card_ext="9", dom_ext="add9", sus=True,
)
MASTER = {"glue_db": 6.0, "glue_ratio": 1.7}
CELLS = (((0, 1.5), (1.5, .5), (2, 1), (3, 1)), ((0, 1), (1, .5), (1.5, .5), (2, 1), (3, 1)),
         ((0, .5), (.5, .5), (1, 1), (2, .5), (2.5, .5), (3, 1)), ((.5, .5), (1, 1), (2, 1.5), (3.5, .5)),
         ((0, 2), (2, .5), (2.5, .5), (3, 1)), ((0, 1), (1, 1), (2, 1.5), (3.5, .5)))
CAD = (((0, 1), (1, 1), (2, 2)), ((0, .5), (.5, .5), (1, 1), (2, 2)), ((0, 1.5), (1.5, .5), (2, 2)))
INTRO = (((0.5, 1), (1.5, 1), (2.5, 1.5)), ((0, 1), (1, 1), (2, 2)), ((1, 1), (2, .5), (2.5, 1.5)))
ARP = ([0, 2, 1, 3, 2, 1, 3, 2], [0, 1, 2, 3, 2, 1, 2, 3], [0, 2, 3, 1, 2, 3, 1, 2], [0, 1, 3, 2, 1, 3, 2, 1])
MASKS = ([1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0])


def _tones(chord: Chord) -> Sequence[int]:
    """Rootless upper structure (the root is played by the sub and the left hand)."""
    return chord.pcs[1:] if len(chord.pcs) > 3 else chord.pcs


def _arps(ctx: Ctx, piano: S.Bus, voic: list[list[int]], pat: list[int], mask: list[int]) -> None:
    g, rng = ctx.g, ctx.rng
    build0 = min((s.beat0 for s in ctx.plan.of_kind("build")), default=1e9)
    for ev, notes in zip(ctx.plan.events, voic):
        if ev.kind == "card" or ev.length < 2:
            continue
        for j in range(int(ev.length * 2)):
            if (ev.kind == "a" and not mask[j % 8]) or (ev.kind in ("brk", "intro") and j % 2):
                continue
            if ev.kind == "intro" and ev.beat == 0 and j < 2:
                continue  # let the opening chord and bell glint ring alone for a beat
            tb = ev.beat + j * 0.5
            v = 0.095 + 0.035 * (j % 2 == 0) + (0.07 * (tb - build0) / 8 if ev.kind == "build" else 0.0)
            v *= 0.8 + 0.4 * ev.energy
            f = S.midi_hz(notes[pat[j % 8] % len(notes)] + 12)
            piano.add(S.felt_piano(f, 0.7 * g.beat, v + rng.uniform(-0.012, 0.012)), ctx.t(tb) + A.human(rng, 8),
                      pan=0.3)


def _melody(ctx: Ctx, piano: S.Bus) -> int:
    """Section melodies + build run + card landing. Returns the scale index of the card note."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    comp = Composer(plan, ctx.comp, 66, 83)
    melody = A.section_melody(comp, plan, CELLS, CAD, vel=0.48)
    intro = [s for s in plan.of_kind("intro") if A.section_bars(s, g.pickup)]
    if intro:  # foreshadow the theme: one sparse bar at the end of the intro
        melody += comp.phrase([A.section_bars(intro[0], g.pickup)[-1]], INTRO, INTRO, vel=0.3)
    ctx.log["melody"] = melody
    for n in melody:
        piano.add(S.felt_piano(S.midi_hz(n.midi), n.length * g.beat * 0.98, n.vel), ctx.t(n.beat) + A.human(rng, 10),
                  pan=0.08)
    target = A.card_target(comp.key, 77)
    approach = target - 1 if ctx.comp.random() < 0.6 else target + 1
    if plan.of_kind("build"):  # rising figure that arrives one scale step beside the card note
        for n in comp.run(g.n_beats - 2, 2.0, 0.5, approach, arp=bool(ctx.comp.random() < 0.5), vel0=0.3, vel1=0.58):
            piano.add(S.felt_piano(S.midi_hz(n.midi), n.length * g.beat, n.vel), ctx.t(n.beat), pan=0.1)
    return target


def _card(ctx: Ctx, piano: S.Bus, fx: S.Bus, target: int) -> None:
    """Card: melody lands on `target`, bells chime on the downbeat, sparkle, falling settle; glint at t=0."""
    g, key = ctx.g, ctx.plan.key
    piano.add(S.felt_piano(S.midi_hz(key.midi(target)), min((g.dur - g.card_at) * 0.9, 4.0), 0.56), g.card_at, pan=0.08)
    for k, idx in enumerate((target - 2, target - 4)):  # settle: 3rd/1st (or 5th/3rd) falling
        t = g.card_at + (1.6 + 1.5 * k) * g.beat
        if t < g.dur - 0.9:
            piano.add(S.felt_piano(S.midi_hz(key.midi(idx)), 1.5 * g.beat, 0.4 - 0.08 * k), t, pan=0.08)
    for i, (m, lv) in enumerate(A.card_bells(key)):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 3.4, index=2.0, a_tau=1.25), g.card_at + i * 0.045, pan=-0.3 + 0.2 * i)
    for k, idx in enumerate(A.sparkle_idx(key)):
        fx.add(S.fm_bell(S.midi_hz(key.midi(idx)), 0.045 * (1 - 0.07 * k), 1.2, index=0.6, i_tau=0.1, a_tau=0.35),
               g.card_at + 0.3 + k * 0.04, pan=-0.6 + 0.24 * k)
    for i, (pc, lv) in enumerate(((key.tonic + 4, 0.10), (key.tonic + 11, 0.05))):  # glint: major 3rd and 7th
        fx.add(S.fm_bell(S.midi_hz(pitch_near(pc % 12, 90 + 7 * i, 84, 103)), lv, 3.5), 0.10 + i * 0.07, pan=0.3)


def _chords(ctx: Ctx, piano: S.Bus, pad: S.Bus, low: S.Bus, voic: list[list[int]]) -> None:
    """Per chord: warm pad, soft sub, rolled piano strike and a left-hand root."""
    rng = ctx.rng
    for ev, notes in zip(ctx.plan.events, voic):
        t0, dur = ctx.t(ev.beat), ctx.t(ev.length)
        freqs = [S.midi_hz(m) for m in notes]
        pad.add(S.warm_pad(rng, freqs, dur + 0.15, 0.08 + 0.10 * ev.energy), max(0.0, t0 - 0.05))
        root = A.bass_midi(ev.chord, 40, 52)
        low.add(S.sub_sine(S.midi_hz(root), dur, 0.05 + 0.04 * ev.energy), t0)
        spread = 0.006 if ev.kind == "card" else 0.018
        for i, f in enumerate(freqs):
            piano.add(S.felt_piano(f, dur * 0.95, 0.17 + 0.08 * ev.energy), t0 + i * spread, pan=-0.25 + 0.15 * i)
        piano.add(S.felt_piano(S.midi_hz(root + 12), dur * 0.9, 0.22), t0, pan=-0.3)


def render(ctx: Ctx) -> np.ndarray:
    """Render the whole piece (pre-master stereo mix) from the plan in `ctx`."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    piano, pad, low, fx, rise = (S.Bus(g.dur) for _ in range(5))
    pat, mask = ARP[int(ctx.comp.integers(len(ARP)))], MASKS[int(ctx.comp.integers(len(MASKS)))]
    voic = A.voice_events(plan, 52, 76, _tones)
    _chords(ctx, piano, pad, low, voic)
    _arps(ctx, piano, voic, pat, mask)
    _card(ctx, piano, fx, _melody(ctx, piano))
    swell = min(g.bar, 2.6)
    rise.add(S.riser(rng, swell, 300.0, 4500.0, 2.2), g.card_at - swell, gain=0.10)
    for sec in plan.of_kind("b"):  # light lift into every b section
        if sec.beat0 > 4:
            rise.add(S.riser(rng, 0.9, 400.0, 4000.0, 2.0), ctx.t(sec.beat0) - 0.9, gain=0.05)
    ir = S.make_ir(rng, 3.2, 1.4, 0.022)
    mix = (piano.st + 0.44 * S.reverb(piano.st, ir) + pad.st + 0.68 * S.reverb(pad.st, ir) + low.st
           + rise.st + 0.2 * S.reverb(rise.st, ir) + fx.st + 0.76 * S.reverb(fx.st, ir))
    return A.breath(mix, g.card_at, 0.35, 0.075)
