"""Style "pop" (バラエティ tone, ~122 BPM): saw plucks + FM hook + light house groove, risers and soft impacts.

Per beat: kick on every beat (side-chains bass and chords), claps on 2 & 4, off-beat hats, 16th shaker; a break
(kick / clap / hat out) appears now and then. The build is a rising pluck run, snare roll, riser and reverse cymbal;
the landing is a big kick + impact + bell chime, then the groove thins out. Ported from buzz-video-ad's pop composer,
re-planned for any length (blocks, chunks) and card swells.
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
    bpm=122.0,
    loops_a=((1, 5, 6, 4), (6, 4, 1, 5), (1, 6, 4, 5), (4, 1, 5, 6), (1, 3, 4, 5)),
    loops_b=((4, 5, 6, 3), (6, 5, 4, 5), (2, 5, 1, 4), (4, 6, 1, 5)),
    ext={1: "add9", 2: "7", 3: "tri", 4: "add9", 5: "tri", 6: "7", 7: "tri"},
    card_ext="add9", dom_ext="tri", sus=False,
)
PATTERNS = (  # (16th position, velocity)
    ((0, 1.0), (3, .72), (6, .84), (8, .95), (11, .72), (14, .8)),                        # 3-3-2 bounce
    ((2, .8), (6, .9), (10, .8), (14, .9)),                                               # off-beat stabs
    ((0, 1.0), (2, .6), (4, .8), (6, .6), (8, .9), (10, .6), (12, .8), (14, .6)),         # driving 8ths
    ((0, 1.0), (3, .7), (6, .8), (10, .9), (13, .7)),                                     # syncopated
)
HOOK = (((0, .5), (.75, .5), (1.5, .5), (2, .75), (3, .75)),
        ((0, .75), (.75, .75), (1.5, .5), (2, .5), (2.75, .5), (3.5, .5)),
        ((0, .5), (.5, .5), (1, 1), (2, .75), (3, .75)), ((.5, .5), (1, .5), (1.5, .5), (2, 1), (3.5, .5)))
HOOK_CAD = (((0, .5), (.75, .5), (1.5, .5), (2, 2)), ((0, 1), (1, 1), (2, 2)), ((0, .75), (.75, .75), (1.5, 2.5)))
PAD_LVL = {"intro": .42, "a": .34, "a2": .36, "b": .34, "brk": .6, "build": .85, "land": .7}
ACCENT = (0.55, 0.38, 1.0, 0.45)       # shaker accents on the four 16ths of a beat
MUSIC_CEIL, DRUM_CEIL = 0.9, 0.9


class Kit:
    """The drum sounds of one piece (several variants each so repeats do not machine-gun)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.kick = S.kick(rng, f0=200.0, f1=84.0, p_tau=0.024, a_tau=0.12, drive=2.4)
        self.big = S.kick(rng, f0=210.0, f1=82.0, p_tau=0.026, a_tau=0.17, drive=2.4, click=0.14)
        self.claps = [S.clap(rng) for _ in range(3)]
        self.shakers = [S.shaker(rng) for _ in range(4)]
        self.hats = [S.hat(rng, tau=0.028) for _ in range(3)]
        self.rolls = [S.lp(S.snare(rng, tone=185.0 + 50.0 * x, n_tau=0.06, t_tau=0.035), 3500.0 + 7000.0 * x)
                      for x in np.linspace(0, 1, 8)]


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    plan.extra["stab"] = A.voicings(plan, 55, 79, lambda c: list(c.pcs[:3]) + [c.root_pc])
    plan.extra["pad"] = A.voicings(plan, 57, 81, lambda c: c.pcs)
    plan.extra["pat"] = {b.i: (PATTERNS[int(rng.integers(len(PATTERNS)))], bool(rng.integers(2))) for b in plan.blocks}
    hook = block_melody(plan, rng, HOOK, HOOK_CAD, lo=71, hi=86, vel=0.8)
    first = next((b for b in plan.blocks if b.kind in ("a", "a2")), None)
    if first is not None:                                         # the groove plays alone for the first two bars
        s = plan.bars[first.bar0].beat0
        hook = [n for n in hook if not s <= n.beat < s + 8]
    plan.extra["hook"] = hook


def make_ctx(plan: Plan, seed: int) -> dict:
    rng = np.random.default_rng([seed, 3, 99])
    return {"kit": Kit(rng), "room": dsp.make_ir(rng, 0.9, predelay=0.012), "hall": dsp.make_ir(rng, 2.3, predelay=0.025)}


def _drums(plan: Plan, bar: Bar, kit: Kit, rng: np.random.Generator, tc: float, drums: Bus, room: Bus,
           kicks: list[float]) -> None:
    kind = bar.kind
    groove = kind in ("a", "a2", "b") or (kind == "intro" and bar.i > 0) or (bar.energy >= 0.8 and kind != "build")
    opening = bar.i == 0
    for k in range(bar.nbeats):
        pos, b = k % 4, bar.beat0 + k
        t = plan.t(b) - tc
        if groove or opening:
            v = (1.0, .72, .85, .8)[pos] if opening and not groove else (.92 if pos % 2 == 0 else .82) * (.8 if kind == "intro" else 1.0)
            drums.add(kit.kick, t, gain=0.8 * v)
            kicks.append(t)
        if groove and pos in (1, 3):
            c = kit.claps[(b // 2) % 3]
            drums.add(c, t, pan=0.05 * (-1) ** (b // 2), gain=0.43)
            room.add(c, t, gain=0.43)
        if groove:
            drums.add(kit.hats[b % 3], t + 0.5 * plan.beat, pan=-0.3, gain=0.07 * (0.6 if kind == "intro" else 0.9))
        lvl = {"a": 1.0, "a2": 1.0, "b": 1.0, "brk": 0.45, "build": 0.0}.get(kind, 0.0)
        if kind == "intro":
            lvl = (0.5 if b >= 1 else 0.0) if opening else 0.75
        for j in range(4):
            if lvl and (kind != "brk" or j % 2 == 0):
                acc = ACCENT[j] if kind != "brk" else 0.6
                drums.add(kit.shakers[j], plan.t(b + j / 4) - tc, pan=0.3, gain=0.075 * lvl * acc * rng.uniform(0.9, 1.1))


def _fills(plan: Plan, bar: Bar, kit: Kit, tc: float, drums: Bus, room: Bus) -> None:
    """One beat of rising 16th snares before every b block and before the groove returns after a break."""
    nxt = plan.bars[bar.i + 1] if bar.i + 1 < len(plan.bars) else None
    if nxt is None or nxt.k != 0 or nxt.block == bar.block or nxt.beat0 < 4:
        return
    if nxt.kind == "b" or (nxt.kind in ("a", "a2") and bar.kind == "brk"):
        for i in range(4):
            s, tb = kit.rolls[2 + i], nxt.beat0 - 1 + i / 4
            drums.add(s, plan.t(tb) - tc, pan=0.1 * (-1) ** i, gain=0.3 * (0.4 + 0.2 * i))
            room.add(s, plan.t(tb) - tc, gain=0.2 * (0.4 + 0.2 * i))


def _roll(plan: Plan, bar: Bar, kit: Kit, tc: float, drums: Bus, room: Bus, fx: Bus,
          rng: np.random.Generator) -> None:
    """Snare roll over the last two beats before the landing (16ths, then 32nds), crescendo, with a reverse cymbal."""
    m = bar.beat0 + bar.nbeats
    times = [m - 2 + i / 4 for i in range(4)] + [m - 1 + i / 8 for i in range(8)]
    for i, tb in enumerate(times):
        x = i / (len(times) - 1)
        v, s = 0.22 + 0.6 * x ** 1.3, kit.rolls[min(7, int(x * 8))]
        drums.add(s, plan.t(tb) - tc, pan=0.1 * (-1) ** i, gain=0.42 * v)
        room.add(s, plan.t(tb) - tc, gain=0.3 * v)
    fx.add(S.rev_cymbal(rng, 0.62), plan.t(m) - tc - 0.62, gain=0.05)
    rl = min(plan.bar_len, 2.4)
    fx.add(S.riser(rng, rl, f_lo=300.0, f_hi=9000.0, curve=2.0), plan.t(m) - tc - rl, gain=0.12)


def _stabs(kind: str, opening: bool, pa: tuple, pb: tuple) -> list[tuple[int, float, float, float]]:
    """(16th position, velocity, length s, brightness) of the plucks in one bar of a block of `kind`."""
    if kind == "intro":
        if opening:
            return [(0, 1.0, 0.45, 1.0)] + [(p, 0.9 * v, 0.16, 1.0) for p, v in pa if p >= 8]
        return [(p, 0.95 * v, 0.16, 1.0) for p, v in pa]
    if kind in ("a", "a2", "b"):
        return [(p, v, 0.16, 1.0) for p, v in (pb if kind == "b" else pa)]
    if kind == "brk":
        return [(p, 0.55 * v, 0.16, 0.6) for p, v in pa]
    return [(2 * i, 0.55 + 0.05 * i, 0.16, 0.7 + 0.07 * i) for i in range(8)]       # build: rising 8th stabs


def _harmony(plan: Plan, bar: Bar, rng: np.random.Generator, tc: float, music: Bus, bass: Bus, pa: tuple, pb: tuple,
             bounce: bool, stab: dict, pad: dict, last_pre: bool) -> None:
    for half, ch in enumerate((bar.chord, bar.chord2)):
        if ch is None:
            continue
        beat = bar.beat0 + half * bar.nbeats / 2
        length = bar.nbeats / (2 if bar.chord2 is not None else 1)
        att = {"brk": 0.25, "build": 0.2}.get(bar.kind, 0.12)
        music.add(S.saw_pad(rng, [S.midi_hz(m) for m in pad[(bar.i, half)]], plan.t(length), attack=att),
                  plan.t(beat) - tc, gain=0.9 * PAD_LVL[bar.kind])
    if bar.kind != "build" or last_pre:
        for pos, v, dur, bright in _stabs(bar.kind, bar.i == 0, pa, pb):
            if pos < 4 * bar.nbeats:
                b = bar.beat0 + pos / 4
                for m in stab[(bar.i, 0 if bar.chord2 is None or pos < 2 * bar.nbeats else 1)]:
                    music.add(S.pluck(S.midi_hz(m), dur, bright), plan.t(b) - tc, gain=0.2 * v)
    root = S.midi_hz(A.bass_midi(bar.chord, 43, 54))
    t0 = plan.t(bar.beat0) - tc
    if bar.kind == "intro":
        bass.add(S.synth_bass(root, 0.35, 0.9), t0, gain=0.42)
    elif bar.kind in ("a", "a2", "b") and not bounce:
        for i in range(bar.nbeats):
            bass.add(S.synth_bass(root, 0.2, 0.8), t0 + (i + 0.5) * plan.beat, gain=0.4)
    elif bar.kind in ("a", "a2", "b"):
        for pos, v in PATTERNS[0]:
            if pos < 4 * bar.nbeats:
                bass.add(S.synth_bass(root, 0.25, 0.9 * v), t0 + pos / 4 * plan.beat, gain=0.42)
    elif bar.kind == "build":
        bass.add(S.synth_bass(root, 0.85, 0.5), t0, gain=0.42)


def _landing(plan: Plan, bar: Bar, kit: Kit, rng: np.random.Generator, tc: float, drums: Bus, room: Bus, music: Bus,
             bass: Bus, fx: Bus, pad: dict, stab: dict) -> list[float]:
    t0, tail = bar.t0 - tc, bar.t1 - bar.t0
    drums.add(kit.big, t0, gain=0.78)
    room.add(kit.big, t0, gain=0.3)
    kicks = [t0]
    for i, (kv, cv) in enumerate(((0.66, 0.5), (0.48, 0.42)), start=1):
        t = t0 + i * plan.beat
        if t < t0 + tail - 0.6:
            drums.add(kit.kick, t, gain=0.8 * kv)
            kicks.append(t)
            c = kit.claps[i]
            drums.add(c, t + 0.5 * plan.beat, gain=0.5 * cv)
            room.add(c, t + 0.5 * plan.beat, gain=0.5 * cv)
    music.add(S.saw_pad(rng, [S.midi_hz(m) for m in pad[(bar.i, 0)]], min(tail, 4.0), attack=0.05), t0, gain=0.9 * PAD_LVL["land"])
    for m in stab[(bar.i, 0)]:
        music.add(S.pluck(S.midi_hz(m), 0.6, 1.2), t0, gain=0.17)
    bass.add(S.synth_bass(S.midi_hz(A.bass_midi(bar.chord, 43, 54)), 0.9, 0.9), t0, gain=0.42)
    key = plan.key
    fx.add(S.impact(rng, 1.0, thump_level=0.3), t0, gain=0.5)
    for i, ((m, lv), ix) in enumerate(zip(A.card_bells(key), (2.0, 1.5, 1.1, 0.8))):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 3.2, index=ix, a_tau=1.2), t0 + i * 0.025, pan=-0.3 + 0.2 * i)
    for k, i in enumerate(A.sparkle_idx(key)):
        fx.add(S.fm_bell(S.midi_hz(key.midi(i)), 0.045 * (1 - 0.07 * k), 1.2, index=0.6, i_tau=0.1, a_tau=0.35),
               t0 + 0.3 + k * 0.04, pan=-0.6 + 0.24 * k)
    base = Composer(plan, rng, 71, 86).key.nearest_idx(81)
    base -= base % 7                                                       # a tonic
    for after, ln, idx in ((1.0, 0.5, base + 4), (1.75, 0.5, base + 2), (2.5, 0.5, base + 1), (3.25, 1.75, base)):
        if t0 + after * plan.beat < t0 + tail - 0.4:                       # closing figure 5 - 3 - 2 - 1
            music.add(S.fm_lead(S.midi_hz(key.midi(idx)), ln * plan.beat, 0.7), t0 + after * plan.beat, pan=0.05, gain=0.22)
    return kicks


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    kit, room_ir, hall = ctx["kit"], ctx["room"], ctx["hall"]
    tc, span = bars[0].t0, bars[-1].t1 - bars[0].t0
    drums, room, music, bass, fx = (Bus(span) for _ in range(5))
    kicks: list[float] = []
    for bar in bars:
        if bar.kind == "land":
            kicks += _landing(plan, bar, kit, rng, tc, drums, room, music, bass, fx, plan.extra["pad"], plan.extra["stab"])
            continue
        last_pre = plan.bars[bar.i + 1].kind == "land"
        pa, bounce = plan.extra["pat"][bar.block]
        pb = PATTERNS[(PATTERNS.index(pa) + 1) % len(PATTERNS)]
        _drums(plan, bar, kit, rng, tc, drums, room, kicks)
        _fills(plan, bar, kit, tc, drums, room)
        _harmony(plan, bar, rng, tc, music, bass, pa, pb, bounce, plan.extra["stab"], plan.extra["pad"], last_pre)
        if last_pre:
            _roll(plan, bar, kit, tc, drums, room, fx, rng)
        if bar.card and bar.k == 0:
            fx.add(S.impact(rng, 0.45, thump_level=0.3), bar.t0 - tc, gain=0.35)
    lo, hi = bars[0].beat0, bars[-1].beat0 + bars[-1].nbeats
    for n in plan.extra["hook"]:
        if lo <= n.beat < hi:
            music.add(S.fm_lead(S.midi_hz(n.midi), n.length * plan.beat, n.vel), plan.t(n.beat) - tc, pan=0.05, gain=0.22)
    n = drums.st.shape[1]
    music_st = dsp.soft_clip(music.st * dsp.duck_env(n, kicks, 0.22), MUSIC_CEIL)
    d_dry = S.hp(drums.st, 48.0, 4)
    tail = hall.shape[1] - 1

    def rv(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
        return dsp.reverb(x, ir, keep_tail=True)

    body = (A.padded(dsp.soft_clip(d_dry, DRUM_CEIL) + S.hp(bass.st * dsp.duck_env(n, kicks, 0.45), 60.0) + music_st
                     + S.lp(fx.st, 15000.0, 4), tail)
            + 0.22 * np.pad(rv(room.st, room_ir), ((0, 0), (0, tail - room_ir.shape[1] + 1)))
            + 0.26 * rv(music_st, hall) + 0.35 * S.lp(rv(fx.st, hall), 15000.0, 4))
    if bars[-1].kind == "land":
        body = A.breath(body, bars[-1].t0 - tc, 0.4, 0.06)
    return body
