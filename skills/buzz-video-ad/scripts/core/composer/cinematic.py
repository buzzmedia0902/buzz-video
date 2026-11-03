"""Style "cinematic" (~70 BPM): string ensemble + felt piano + low pedal + air, swelling into a warm chime.

The strings breathe through a dynamics arc (-6 dB -> +3 dB just before the card); an octave-up string layer
joins as the piece builds; sparse piano plays a slow motif; soft chime markers open the main sections; a
light sweep leads into the card where a tonic chord and a warm rolled chime arrive, then it rings out.
"""
from __future__ import annotations

import numpy as np

from core import synth as S
from core.composer import arrange as A
from core.composer.context import Ctx
from core.composer.melody import Composer
from core.composer.plan import Grid, StyleCfg
from core.composer.theory import pitch_near

CFG = StyleCfg(
    bpm=70.0,
    loops_a=((1, 5, 6, 4), (1, 6, 4, 5), (6, 4, 1, 5), (1, 3, 4, 5), (1, 4, 6, 5)),
    loops_b=((4, 1, 5, 6), (4, 5, 3, 6), (6, 5, 4, 1), (2, 5, 1, 4)),
    ext={1: "9", 2: "9", 3: "7", 4: "9", 5: "add9", 6: "9", 7: "7"},
    card_ext="9", dom_ext="add9", sus=True,
)
MASTER = {"glue_db": 7.0, "glue_ratio": 1.6, "hp_fc": 38.0, "hp_order": 2}
CELLS = (((0, 1), (1, 1), (2, 2)), ((0, 2), (2, 1), (3, 1)), ((0, 1.5), (1.5, .5), (2, 2)), ((.5, 1.5), (2, 2)),
         ((0, 1), (1, .5), (1.5, .5), (2, 1), (3, 1)), ((0, .5), (.5, .5), (1, 2), (3, 1)))
CAD = (((0, 2), (2, 2)), ((0, 1), (1, 1), (2, 2)), ((0, 4),))
INTRO = (((1, .5), (1.5, .5), (2, 1.9)), ((0.5, .5), (1, .5), (1.5, 2.4)))


def _string_events(ctx: Ctx, strings: S.Bus, voic: list[list[int]]) -> None:
    """Per chord: string ensemble (+ cello root, + an octave-up layer that opens as the piece builds)."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    for ev, notes in zip(plan.events, voic):
        prog = ev.beat / max(g.n_beats, 1)
        t0 = ctx.t(ev.beat)
        start = max(0.0, t0 - 0.35)
        dur = ctx.t(ev.end) - start
        strings.add(S.strings(rng, [S.midi_hz(m) for m in notes], dur), start)
        root = pitch_near(ev.chord.root_pc, 44, 38, 50)
        strings.add(S.strings(rng, [S.midi_hz(root)], dur, cutoff=1200.0, voices=4, spread=6.0), start, gain=0.45)
        hl = 0.85 if ev.kind == "card" else min(0.85, 0.85 * max(0.0, prog - 0.12) / 0.8)
        if hl > 0.02:
            strings.add(S.strings(rng, [S.midi_hz(m + 12) for m in notes], dur, attack=1.0, cutoff=2600.0), start,
                        gain=0.75 * hl)


def _dynamics(g: Grid, n: int) -> np.ndarray:
    """Slow dynamics arc for the strings: -6 dB -> +3 dB just before the card, relaxing in the tail."""
    c = g.card_at
    pts = [(0.0, -6.0), (0.15 * c, -5.5), (0.4 * c, -4.5), (0.65 * c, -3.0), (0.85 * c, -1.5), (max(c - 1.0, 0.9 * c), 0.5),
           (c - 0.4, 3.0), (c + 0.7, 3.0), (c + 1.9, 0.0), (g.dur, -4.0), (g.dur + 4.0, -8.0)]
    return S.dyn_curve(pts, n)


def _piano(ctx: Ctx, piano: S.Bus) -> None:
    """Pedal-like left hand, slow right-hand motif/melody, and the note the melody lands on at the card."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    for ev in plan.events:  # left hand: root, then the fifth halfway through long chords
        root = pitch_near(ev.chord.root_pc, 48, 41, 53)
        t0 = ctx.t(ev.beat) + (0.5 if ev.beat == 0 else 0.0)
        piano.add(S.felt_piano(S.midi_hz(root), min(ctx.t(ev.length) * 0.95, 4.0), 0.22 + 0.1 * ev.energy), t0, pan=-0.15)
        if ev.length >= 4 and ev.kind != "card":
            piano.add(S.felt_piano(S.midi_hz(root + 7), ctx.t(ev.length) * 0.4, 0.16), ctx.t(ev.beat + 2), pan=-0.15)
    comp = Composer(plan, ctx.comp, 64, 83)
    mel = A.section_melody(comp, plan, CELLS, CAD, vel=0.38)
    intro = [s for s in plan.of_kind("intro") if A.section_bars(s, g.pickup)]
    if intro:
        mel += comp.phrase([A.section_bars(intro[0], g.pickup)[-1]], INTRO, INTRO, vel=0.3)
    ctx.log["melody"] = mel
    for n in mel:
        v = n.vel * (0.78 + 0.45 * n.beat / max(g.n_beats, 1))
        piano.add(S.felt_piano(S.midi_hz(n.midi), n.length * g.beat * 0.98, min(v, 0.62)), ctx.t(n.beat) + A.human(rng, 12),
                  pan=0.04)
    target = A.card_target(comp.key, 76)
    piano.add(S.felt_piano(S.midi_hz(comp.key.midi(target)), min(g.dur - g.card_at, 4.0) * 0.95, 0.5), g.card_at, pan=0.04)


def _card_fx(ctx: Ctx, fx: S.Bus) -> None:
    plan, g, rng, key = ctx.plan, ctx.g, ctx.rng, ctx.plan.key
    third = pitch_near((key.tonic + 4) % 12, 90, 84, 96)
    soft = (S.fm_bell(S.midi_hz(third), 1.0, 5.0, ratio=3.5, index=1.0, i_tau=0.3, a_tau=2.0)
            + S.fm_bell(S.midi_hz(third - 12), 0.35, 5.0, ratio=2.0, index=0.35, i_tau=0.5, a_tau=1.6))
    for sec in plan.sections:  # chapter-marker chimes
        if sec.kind in ("a", "b") and sec.nth == 0 and sec.beat0 > 0:
            fx.add(soft, ctx.t(sec.beat0), pan=0.12, gain=0.115)
    r5 = pitch_near(key.tonic, 74, 70, 81)
    parts = [(S.midi_hz(r5), 0.55, 1.0, 0.7), (S.midi_hz(r5 + 7), 0.45, 1.0, 0.6), (S.midi_hz(r5 + 12), 0.42, 3.5, 0.9),
             (S.midi_hz(pitch_near((key.tonic + 4) % 12, 88, 82, 94)), 0.3, 3.5, 0.7)]
    fx.add(S.chime(parts, spread=0.02, length=6.0, a_tau=2.6), g.card_at, gain=0.2 * 1.6)
    glint = pitch_near(key.tonic, 94, 90, 101)  # bright 'ting' on the downbeat so the arrival reads clearly
    fx.add(S.fm_bell(S.midi_hz(glint), 0.13, 3.0, ratio=3.5, index=1.8, i_tau=0.35, a_tau=1.1), g.card_at, pan=0.2)
    sweep = min(1.4, g.card_at * 0.5)
    fx.add(S.light_sweep(rng, pre=sweep, post=0.03), g.card_at - sweep, gain=0.12)
    for k, i in enumerate(A.sparkle_idx(key)):
        fx.add(S.fm_bell(S.midi_hz(key.midi(i)), 0.03, 1.4, index=0.5, i_tau=0.1, a_tau=0.4), g.card_at + 0.3 + k * 0.17,
               pan=-0.5 + 0.2 * k)


def render(ctx: Ctx) -> np.ndarray:
    """Render the whole piece (pre-master stereo mix) from the plan in `ctx`."""
    plan, g, rng, key = ctx.plan, ctx.g, ctx.rng, ctx.plan.key
    piano, strings, low, airb, fx = (S.Bus(g.dur) for _ in range(5))
    voic = A.voice_events(plan, 53, 69, lambda c: c.pcs[1:] if len(c.pcs) > 3 else c.pcs)
    _string_events(ctx, strings, voic)
    _piano(ctx, piano)
    _card_fx(ctx, fx)
    dyn = _dynamics(g, strings.st.shape[1])
    tonic = pitch_near(key.tonic, 44, 38, 49)
    ped = S.pedal(rng, g.dur + 1.0, [(S.midi_hz(tonic), 0.35), (S.midi_hz(tonic + 7), 0.3), (S.midi_hz(tonic + 12), 0.8),
                                      (S.midi_hz(tonic + 19), 0.18)])
    c = g.card_at
    low.add(ped * S.dyn_curve([(0, -3.0), (0.65 * c, -1.0), (c - 0.4, 1.5), (g.dur, -2.0)], ped.shape[1]), 0.0, gain=0.11)
    airb.add(S.air(rng, g.dur + 1.0), 0.0, gain=0.019)
    strings_st = S.soft_clip(strings.st * dyn, 0.8 * np.abs(strings.st * dyn).max())
    hall = S.make_ir(rng, 4.2, hf_rt60=1.8, predelay=0.035)
    piano_st = S.lp(piano.st, 6000.0)
    body = (piano_st + 0.4 * S.reverb(piano_st, hall) + strings_st + 0.5 * S.reverb(strings_st, hall)
            + low.st + 0.12 * S.reverb(low.st, hall) + airb.st + 0.3 * S.reverb(airb.st, hall))
    return A.breath(body + fx.st + 0.6 * S.reverb(fx.st, hall), g.card_at, 0.4, 0.07)
