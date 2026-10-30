"""Style "lofi" (信頼 tone, ~86 BPM): FM electric piano, swung soft drums, round bass, vinyl crackle, tape wow.

Rootless 9th / 13th voicings with smooth voice leading; swung 16ths (60 %); drums come in after the intro, drop to
hats in a break and stop for the last beat before the landing. The bed keeps 1-4 kHz calm (EP low-passed at 3 kHz,
snare low-passed) so a voice can sit on top. The landing is a soft kick, the tonic 9th chord and a light bell.
Ported from sns-ad-video's lofi composer, re-planned for any length (blocks, chunks) and card swells.
"""
from __future__ import annotations

import numpy as np

from cv.audio import arrange as A
from cv.audio import dsp
from cv.audio import instruments as S
from cv.audio.dsp import Bus
from cv.audio.form import Bar, Plan, StyleCfg
from cv.audio.melody import Composer, block_melody
from cv.audio.theory import pitch_near

CFG = StyleCfg(
    bpm=86.0,
    loops_a=((4, 3, 2, 5), (2, 5, 1, 6), (6, 2, 5, 1), (1, 6, 2, 5), (3, 6, 2, 5)),
    loops_b=((2, 5, 3, 6), (4, 3, 6, 5), (6, 5, 2, 1), (4, 5, 6, 3)),
    ext={1: "9", 2: "9", 3: "7", 4: "9", 5: "13", 6: "9", 7: "7"},
    card_ext="9", dom_ext="13", sus=False,
)
SWING = 0.6
HAT_VEL = (0.8, 0.0, 0.5, 0.22, 0.72, 0.0, 0.5, 0.3, 0.8, 0.0, 0.5, 0.22, 0.72, 0.18, 0.5, 0.34)
ANSWER = (((2, .5), (2.5, .5), (3, 1)), ((2, 1), (3, 1)), ((1, .5), (2, .5), (3, 1.5)))
KICKS = (((0, 1.0), (10, 0.78)), ((0, 1.0), (7, 0.5), (10, 0.72)))     # even / odd bars (16th position, velocity)
KEYS_GAIN, BASS_GAIN, DRUMS_GAIN, VINYL_GAIN = 0.4, 0.23, 1.05, 0.2


class Kit:
    """Soft lo-fi drum sounds (several variants of the snare / hat so repeats do not machine-gun)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.kick = S.lp(S.kick(rng, f0=150.0, f1=86.0, p_tau=0.03, a_tau=0.16, drive=1.6, click=0.04), 1800.0)
        self.snares = [S.lp(S.snare(rng, tone=180.0, n_tau=0.13, t_tau=0.05, lo=700.0, hi=5000.0, tone_mix=0.7), 2200.0)
                       for _ in range(3)]
        self.hats = [S.lp(S.hat(rng, tau=0.024, lo=6000.0, hi=11000.0), 10000.0) for _ in range(4)]


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    plan.extra["voic"] = A.voicings(plan, 55, 77, lambda c: c.pcs[1:] if len(c.pcs) > 3 else c.pcs)
    plan.extra["melody"] = block_melody(plan, rng, ANSWER, ANSWER, lo=67, hi=79, vel=0.34, kinds=("b", "a2"))


def make_ctx(plan: Plan, seed: int) -> dict:
    rng = np.random.default_rng([seed, 4, 99])
    return {"kit": Kit(rng), "plate": dsp.make_ir(rng, 1.6, hf_rt60=0.6, predelay=0.02)}


def tpos(plan: Plan, bar_beat: float, pos: int) -> float:
    """Time (s) of 16th position `pos` of the bar starting at beat `bar_beat`, off-16ths swung."""
    beat, sub = divmod(pos, 4)
    return plan.t(bar_beat + beat + (0.0, SWING * 0.5, 0.5, 0.5 + SWING * 0.5)[sub])


def _drums(plan: Plan, bar: Bar, kit: Kit, rng: np.random.Generator, tc: float, drums: Bus, room: Bus,
           kicks: list[float], stop_time: bool) -> None:
    n16 = 4 * bar.nbeats

    def live(pos: int) -> float | None:
        if pos >= n16 or (stop_time and pos >= n16 - 4):
            return None
        return tpos(plan, bar.beat0, pos) - tc

    if bar.kind in ("a", "a2", "b", "build") or bar.energy >= 0.8:
        for pos, v in KICKS[bar.i % 2]:
            if (t := live(pos)) is not None:
                drums.add(kit.kick, t + (rng.uniform(-0.004, 0.004) if pos else 0.0), gain=0.52 * v)
                kicks.append(t)
        for pos in (4, 12):
            if (t := live(pos)) is not None:
                s, ts = kit.snares[(bar.i * 2 + pos // 12) % 3], t + rng.uniform(0.0, 0.012)   # laid back
                drums.add(s, ts, pan=0.04, gain=0.34)
                room.add(s, ts, gain=0.34)
        if bar.i % 2 == 1 and (t := live(15)) is not None:
            drums.add(kit.snares[bar.i % 3], t, pan=0.04, gain=0.07)                           # ghost note
    if bar.i > 0 or bar.kind != "intro":
        for pos in range(16):
            hv = HAT_VEL[pos]
            if hv and (bar.kind != "brk" or pos % 4 == 0) and (t := live(pos)) is not None:
                drums.add(kit.hats[(bar.i + pos) % 4], t + rng.uniform(-0.005, 0.005), pan=0.28,
                          gain=0.085 * hv * rng.uniform(0.85, 1.1) * (0.6 if bar.kind in ("intro", "brk") else 1.0))


def _chords(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, keys: Bus, bass: Bus, voic: dict) -> None:
    """EP strums (rolled low to high) on every chord of the bar, pushes, V turnarounds and the bass line."""
    vel_k = 0.75 + 0.4 * bar.energy
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        beat = bar.beat0 + half * bar.nbeats / 2
        length = bar.nbeats / (2 if bar.chord2 is not None else 1)
        t0 = plan.t(beat) - tc
        notes = voic[(bar.i, half)]
        hold = min(plan.t(length), 2.6 * plan.beat)
        for i, m in enumerate(notes):
            vel = ((0.62 if i < 4 else 0.5) + rng.uniform(-0.05, 0.05)) * vel_k
            keys.add(S.ep(S.midi_hz(m), hold, vel), t0 + 0.009 * i + A.human(rng, 6), pan=-0.2 + 0.1 * i)
        if bar.kind != "brk" and bar.nbeats == 4 and half == 0:
            for i, m in enumerate(notes[1:]):                                # push on the 'and' of 2
                keys.add(S.ep(S.midi_hz(m), 1.3 * plan.beat, 0.38 * vel_k), tpos(plan, bar.beat0, 6) - tc + 0.01 * i,
                         pan=-0.05 + 0.1 * i)
            if ch.degree == 5 and bar.kind != "build":                      # turnaround on V: falling chord tones
                top = sorted(notes + [notes[0] + 12])[::-1][:4]
                for pos, m in zip((8, 10, 12, 14), top):
                    keys.add(S.ep(S.midi_hz(m + 12 if m < 72 else m), 0.4 * plan.beat, 0.4 * vel_k),
                             tpos(plan, bar.beat0, pos) - tc, pan=0.2)
        root = pitch_near(ch.root_pc, 46, 40, 52)
        bass.add(S.round_bass(S.midi_hz(root), 1.7 * plan.beat, 0.95 if bar.energy > 0.45 else 0.7), t0)
        if bar.nbeats == 4 and half == 0:
            bass.add(S.round_bass(S.midi_hz(root) * 1.5, 0.45 * plan.beat, 0.6), tpos(plan, bar.beat0, 10) - tc)
            nroot = pitch_near(plan.chord_at(bar.beat0 + 4.0).root_pc, root, 40, 52)
            approach = nroot + (-1 if rng.random() < 0.5 else 1) if nroot != root else root   # chromatic pick-up
            bass.add(S.round_bass(S.midi_hz(approach), 0.3 * plan.beat, 0.45), tpos(plan, bar.beat0, 14) - tc)


def _landing(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, kit: Kit, drums: Bus, keys: Bus, bass: Bus,
             fx: Bus, voic: dict) -> list[float]:
    t0 = bar.t0 - tc
    tail = bar.t1 - bar.t0
    drums.add(kit.kick, t0, gain=0.5)                                       # the groove's last act: one soft kick
    for i, m in enumerate(voic[(bar.i, 0)]):
        keys.add(S.ep(S.midi_hz(m), min(3.2, tail), 0.62 if i < 4 else 0.5), t0 + 0.009 * i, pan=-0.2 + 0.1 * i)
    root = pitch_near(bar.chord.root_pc, 46, 40, 52)
    bass.add(S.round_bass(S.midi_hz(root), min(2.2, tail), 0.95), t0)
    r = pitch_near(plan.key.tonic, 81, 76, 87)
    for i, (m, lv, ix) in enumerate(((r, 0.15, 1.1), (r + 7, 0.11, 0.9), (r + 12, 0.075, 0.7))):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 2.5, index=ix, i_tau=0.2, a_tau=0.9), t0 + 0.07 * i, pan=0.1 + 0.1 * i)
    return [t0]


def _build_lift(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, keys: Bus, fx: Bus) -> None:
    """Rising EP figure over beats 2-3.5 of the last build bar and a very soft, dark noise swell: the lo-fi arrival."""
    comp = Composer(plan, rng, 67, 79)
    end = comp.key.nearest_idx(79)
    end = min((c for c in range(end - 3, end + 3) if c % 7 in (1, 6)), key=lambda c: abs(c - end))   # 2nd or 7th
    b_end = bar.beat0 + bar.nbeats
    for n in comp.run(b_end - 2, 2.0, 0.5, end, vel0=0.26, vel1=0.5):
        keys.add(S.ep(S.midi_hz(n.midi), n.length * plan.beat, n.vel), plan.t(n.beat) - tc, pan=0.2)
    rl = min(plan.bar_len, 2.8)
    fx.add(dsp.lp(S.riser(rng, rl, 250.0, 1500.0, 2.4), 1800.0), plan.t(b_end) - tc - rl, gain=0.06)


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    """Stereo audio of `bars` (plus the ring-out of the last note), starting at bars[0].t0."""
    kit, plate, voic = ctx["kit"], ctx["plate"], plan.extra["voic"]
    tc, span = bars[0].t0, bars[-1].t1 - bars[0].t0
    drums, room, keys, bass, fx = (Bus(span) for _ in range(5))
    kicks: list[float] = []
    for bar in bars:
        if bar.kind == "land":
            kicks += _landing(plan, bar, rng, tc, kit, drums, keys, bass, fx, voic)
            continue
        last_pre = plan.bars[bar.i + 1].kind == "land"
        _drums(plan, bar, kit, rng, tc, drums, room, kicks, stop_time=last_pre and bar.kind == "build")
        _chords(plan, bar, rng, tc, keys, bass, voic)
        if last_pre and bar.kind == "build":
            _build_lift(plan, bar, rng, tc, keys, fx)
    lo, hi = bars[0].beat0, bars[-1].beat0 + bars[-1].nbeats
    for n in plan.extra["melody"]:
        if lo <= n.beat < hi:
            keys.add(S.ep(S.midi_hz(n.midi), n.length * plan.beat, n.vel), plan.t(n.beat) - tc, pan=0.2)
    n = keys.st.shape[1]
    keys_st = KEYS_GAIN * S.autopan(S.hp(S.lp(keys.st, 3000.0), 200.0), t0=tc) * dsp.duck_env(n, kicks, 0.12, 0.25)
    bass_st = BASS_GAIN * S.hp(S.lp(bass.st, 800.0), 55.0) * dsp.duck_env(n, kicks, 0.18, 0.22)
    music = S.hp(S.wow(np.tanh(1.2 * (keys_st + bass_st)) / 1.2, t0=tc), 40.0)

    def rv(x: np.ndarray, **kw) -> np.ndarray:                      # wet return, longer than x by the reverb tail
        return dsp.reverb(x, plate, keep_tail=True, **kw)

    dry = DRUMS_GAIN * S.hp(drums.st, 45.0, 4) + music + S.vinyl(rng, n / dsp.SR) * VINYL_GAIN + fx.st
    body = (A.padded(dry, plate.shape[1] - 1) + DRUMS_GAIN * 0.18 * rv(room.st, lp_fc=5000.0) + 0.25 * rv(keys_st, lp_fc=5000.0)
            + 0.4 * rv(fx.st))
    if bars[-1].kind == "land":
        body = A.breath(body, bars[-1].t0 - tc, 0.45, 0.07)
    return body
