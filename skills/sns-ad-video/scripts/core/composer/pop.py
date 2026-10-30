"""Style "pop" (~128 BPM): saw plucks + FM hook + light house groove, riser and soft impact into the card.

Per beat: kick on every beat (side-chains bass and chords), claps on 2 & 4, off-beat hats, 16th shaker;
a break (kick/clap/hat out) appears in longer pieces. The build is a rising pluck run, snare roll,
riser and reverse cymbal; the card lands on a big kick + impact + bell chime, then the groove thins out.
With `cuts`, a subtle swish peaks on every cut.
"""
from __future__ import annotations

import numpy as np

from core import synth as S
from core.composer import arrange as A
from core.composer.context import Ctx
from core.composer.melody import Composer
from core.composer.plan import StyleCfg
from core.composer.theory import pitch_near

CFG = StyleCfg(
    bpm=128.0,
    loops_a=((1, 5, 6, 4), (6, 4, 1, 5), (1, 6, 4, 5), (4, 1, 5, 6), (1, 3, 4, 5)),
    loops_b=((4, 5, 6, 3), (6, 5, 4, 5), (2, 5, 1, 4), (4, 6, 1, 5)),
    ext={1: "add9", 2: "7", 3: "tri", 4: "add9", 5: "tri", 6: "7", 7: "tri"},
    card_ext="add9", dom_ext="tri", sus=False,
)
MASTER = {"glue_db": 6.0, "glue_ratio": 1.8, "hp_fc": 45.0, "hp_order": 4}
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
PAD_LVL = {"intro": .42, "a": .34, "b": .34, "brk": .6, "build": .85, "card": .7}
ACCENT = (0.55, 0.38, 1.0, 0.45)  # shaker accents on the four 16ths of a beat


class Kit:
    """The drum sounds of one piece (several variants each so repeats do not machine-gun)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng  # the snare rolls are rendered on demand
        self.kick = S.kick(rng, f0=200.0, f1=84.0, p_tau=0.024, a_tau=0.12, drive=2.4)
        self.big = S.kick(rng, f0=210.0, f1=82.0, p_tau=0.026, a_tau=0.17, drive=2.4, click=0.14)
        self.claps = [S.clap(rng) for _ in range(3)]
        self.shakers = [S.shaker(rng) for _ in range(4)]
        self.hats = [S.hat(rng, tau=0.028) for _ in range(3)]

    def snare_roll(self, x: float) -> np.ndarray:
        """Snare hit that brightens as x goes 0 -> 1."""
        return S.lp(S.snare(self.rng, tone=185.0 + 50.0 * x, n_tau=0.06, t_tau=0.035), 3500.0 + 7000.0 * x)


# ------------------------------------------------------------------ drums
def _drums(ctx: Ctx, drums: S.Bus, room: S.Bus) -> list[float]:
    """All drum hits; returns the kick times (for the side-chain ducking)."""
    plan, g = ctx.plan, ctx.g
    kit, kicks = Kit(ctx.rng), []
    for b in range(g.n_beats):
        kind, pos, t = plan.section_at(b).kind, (b - g.pickup) % 4, ctx.t(b)
        first = kind == "intro" and b < g.pickup + 4      # the opening bar: kick + shaker only
        groove = kind in ("a", "b") or (kind == "intro" and not first)
        if first or groove:
            v = (1.0, .72, .85, .8)[pos] if first else (.92 if pos % 2 == 0 else .82) * (.8 if kind == "intro" else 1.0)
            drums.add(kit.kick, t, gain=0.8 * (1.0 if b == 0 else v))  # b == 0: the opening hit
            kicks.append(t)
        if groove and pos in (1, 3):
            c = kit.claps[(b // 2) % 3]
            drums.add(c, t, pan=0.05 * (-1) ** (b // 2), gain=0.43)
            room.add(c, t, gain=0.43)
        if groove:
            drums.add(kit.hats[b % 3], t + 0.5 * g.beat, pan=-0.3, gain=0.07 * (0.6 if kind == "intro" else 0.9))
        _shaker(ctx, drums, kit, b, kind, first)
    _roll(ctx, drums, room, kit)
    _fills(ctx, drums, room, kit)
    return kicks + _card_drums(ctx, drums, room, kit)


def _shaker(ctx: Ctx, drums: S.Bus, kit: Kit, b: int, kind: str, first: bool) -> None:
    lvl = {"a": 1.0, "b": 1.0, "brk": 0.45, "build": 0.0}.get(kind, 0.0)
    if kind == "intro":
        lvl = (0.5 if b >= 1 else 0.0) if first else 0.75
    for k in range(4):
        if lvl and (kind != "brk" or k % 2 == 0):  # the break plays 8ths only
            acc = ACCENT[k] if kind != "brk" else 0.6
            drums.add(kit.shakers[k], ctx.t(b + k / 4), pan=0.3, gain=0.075 * lvl * acc * ctx.rng.uniform(0.9, 1.1))


def _roll(ctx: Ctx, drums: S.Bus, room: S.Bus, kit: Kit) -> None:
    """Snare roll over the last two beats before the card (16ths, then 32nds), crescendo."""
    m = ctx.g.n_beats
    if not ctx.plan.of_kind("build"):
        return
    times = [m - 2 + i / 4 for i in range(4)] + [m - 1 + i / 8 for i in range(8)]
    for i, tb in enumerate(times):
        x = i / (len(times) - 1)
        v, s = 0.22 + 0.6 * x ** 1.3, kit.snare_roll(x)
        drums.add(s, ctx.t(tb), pan=0.1 * (-1) ** i, gain=0.42 * v)
        room.add(s, ctx.t(tb), gain=0.3 * v)


def _fills(ctx: Ctx, drums: S.Bus, room: S.Bus, kit: Kit) -> None:
    """One beat of rising 16th snares before every b section and before the groove returns after a break."""
    secs = ctx.plan.sections
    for prev, sec in zip(secs, secs[1:]):
        if sec.beat0 >= 4 and (sec.kind == "b" or (sec.kind == "a" and prev.kind == "brk")):
            for i in range(4):
                s, tb = kit.snare_roll(0.3 + 0.2 * i), sec.beat0 - 1 + i / 4
                drums.add(s, ctx.t(tb), pan=0.1 * (-1) ** i, gain=0.3 * (0.4 + 0.2 * i))
                room.add(s, ctx.t(tb), gain=0.2 * (0.4 + 0.2 * i))


def _card_drums(ctx: Ctx, drums: S.Bus, room: S.Bus, kit: Kit) -> list[float]:
    """Big kick on the card, then two fading kick + clap pairs while the tail lasts."""
    g = ctx.g
    drums.add(kit.big, g.card_at, gain=0.78)
    room.add(kit.big, g.card_at, gain=0.3)
    kicks = [g.card_at]
    for i, (kv, cv) in enumerate(((0.66, 0.5), (0.48, 0.42)), start=1):
        t = g.card_at + i * g.beat
        if t < g.dur - 0.6:
            drums.add(kit.kick, t, gain=0.8 * kv)
            kicks.append(t)
            c = kit.claps[i]
            drums.add(c, t + 0.5 * g.beat, gain=0.5 * cv)
            room.add(c, t + 0.5 * g.beat, gain=0.5 * cv)
    return kicks


# ------------------------------------------------------------------ plucks, bass, pads, hook
def _bar_stabs(kind: str, first: bool, pa: tuple, pb: tuple, pos0: int) -> list[tuple[int, float, float, float]]:
    """(16th position, velocity, length s, brightness) of the plucks in one bar of a section of `kind`."""
    if kind == "intro":
        if first:  # opening hit (pos0 = first audible 16th), then the second half of the pattern
            return [(pos0, 1.0, 0.45, 1.0)] + [(p, 0.9 * v, 0.16, 1.0) for p, v in pa if p >= max(8, pos0 + 1)]
        return [(p, 0.95 * v, 0.16, 1.0) for p, v in pa]
    if kind in ("a", "b"):
        return [(p, v, 0.16, 1.0) for p, v in (pb if kind == "b" else pa)]
    if kind == "brk":
        return [(p, 0.55 * v, 0.16, 0.6) for p, v in pa]
    return [(2 * i, 0.55 + 0.05 * i, 0.16, 0.7 + 0.07 * i) for i in range(8)]  # build: rising 8th stabs


def _plucks(ctx: Ctx, music: S.Bus, voic: list[list[int]], pa: tuple, pb: tuple) -> None:
    plan, g = ctx.plan, ctx.g

    def stab(beat: float, vel: float, dur: float, bright: float) -> None:
        for m in voic[plan.event_index_at(beat)]:
            music.add(S.pluck(S.midi_hz(m), dur, bright), ctx.t(beat), gain=0.2 * vel)

    last_build = plan.of_kind("build")[-1].end - 4 if plan.of_kind("build") else -1.0
    for bi, b0 in enumerate(A.all_bars(g)):
        kind = plan.section_at(max(b0, 0.0)).kind
        if kind == "build" and b0 != last_build:
            continue  # only the final bar of the build carries the rising stabs
        for pos, v, dur, bright in _bar_stabs(kind, bi == 0, pa, pb, int(round(max(0.0, -b0) * 4))):
            if b0 + pos / 4 >= -1e-9:
                stab(max(0.0, b0 + pos / 4), v, dur, bright)
    m = float(g.n_beats)
    stab(m, 0.85, 0.6, 1.2)
    for pos, v in pa[2:]:
        if ctx.t(m + pos / 4) < g.dur - 0.5:
            stab(m + pos / 4, 0.6 * v, 0.16, 1.0)


def _bass(ctx: Ctx, bass: S.Bus, bounce: bool) -> None:
    """Off-beat house bass (or a bass that follows the 3-3-2 pluck pattern when `bounce`)."""
    plan, g = ctx.plan, ctx.g
    for sec in plan.sections:
        for b0 in A.section_bars(sec, g.pickup):
            root = S.midi_hz(A.bass_midi(plan.chord_at(b0), 43, 54))
            if sec.kind == "intro":
                bass.add(S.synth_bass(root, 0.35, 0.9), ctx.t(b0), gain=0.42)
            elif sec.kind in ("a", "b") and not bounce:
                for i in range(4):
                    bass.add(S.synth_bass(root, 0.2, 0.8), ctx.t(b0 + i + 0.5), gain=0.4)
            elif sec.kind in ("a", "b"):
                for pos, v in PATTERNS[0]:
                    bass.add(S.synth_bass(root, 0.25, 0.9 * v), ctx.t(b0 + pos / 4), gain=0.42)
            elif sec.kind == "build":
                bass.add(S.synth_bass(root, 0.85, 0.5), ctx.t(b0), gain=0.42)
    root = S.midi_hz(A.bass_midi(plan.card.chord, 43, 54))
    bass.add(S.synth_bass(root, 0.9, 0.9), g.card_at, gain=0.42)
    if g.dur - g.card_at > 2.2:
        bass.add(S.synth_bass(root, 0.5, 0.6), g.card_at + 2 * g.beat, gain=0.42)


def _pads_and_hook(ctx: Ctx, music: S.Bus, pad_voic: list[list[int]]) -> None:
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    for ev, notes in zip(plan.events, pad_voic):
        att = {"brk": 0.25, "build": 0.2, "card": 0.05}.get(ev.kind, 0.12)
        music.add(S.saw_pad(rng, [S.midi_hz(m) for m in notes], ctx.t(ev.length), attack=att), ctx.t(ev.beat),
                  gain=0.9 * PAD_LVL[ev.kind])
    comp = Composer(plan, ctx.comp, 71, 86)
    hook = A.section_melody(comp, plan, HOOK, HOOK_CAD, vel=0.8)
    ctx.log["melody"] = hook
    quiet = [(s.beat0, s.beat0 + 8) for s in plan.of_kind("a") if s.nth == 0 and s.beats >= 16]  # groove alone first
    for n in hook:
        if not any(a <= n.beat < b for a, b in quiet):
            music.add(S.fm_lead(S.midi_hz(n.midi), n.length * g.beat, n.vel), ctx.t(n.beat), pan=0.05, gain=0.22)
    base = comp.key.nearest_idx(81)
    base -= base % 7  # a tonic
    for after, ln, idx in ((1.0, 0.5, base + 4), (1.75, 0.5, base + 2), (2.5, 0.5, base + 1), (3.25, 1.75, base)):
        t = g.card_at + after * g.beat  # closing figure 5 - 3 - 2 - 1
        if t < g.dur - 0.4:
            music.add(S.fm_lead(S.midi_hz(comp.key.midi(idx)), ln * g.beat, 0.7), t, pan=0.05, gain=0.22)


# ------------------------------------------------------------------ fx
def _fx(ctx: Ctx, fx: S.Bus) -> None:
    plan, g, rng, key = ctx.plan, ctx.g, ctx.rng, ctx.plan.key
    fx.add(S.impact(rng, 0.45, thump_level=0.4), 0.0, gain=0.55)
    pops = [pitch_near((key.tonic + o) % 12, 90, 81, 96) for o in (7, 0, 4)]
    for t, m in ((0.25, pops[0]), (0.75, pops[1])):
        fx.add(S.bubble_pop(rng, S.midi_hz(m)), t, pan=0.08, gain=0.34)
    for i, sec in enumerate(plan.sections):
        if sec.kind in ("a", "b") and sec.beat0 > 0:
            fx.add(S.bubble_pop(rng, S.midi_hz(pops[i % 3])), ctx.t(sec.beat0) + 0.1, pan=0.08, gain=0.3)
            fx.add(S.rev_cymbal(rng, 0.62), ctx.t(sec.beat0) - 0.62, gain=0.05)
            if plan.sections[i - 1].kind == "brk":  # the groove drops back in
                fx.add(S.impact(rng, 0.45, thump_level=0.3), ctx.t(sec.beat0), gain=0.35)
    for i, tc in enumerate(ctx.cuts):  # subtle swish peaking on every cut
        span = (-0.75, 0.75) if i % 2 == 0 else (0.75, -0.75)
        fx.add(S.swish(rng, pan=span), tc - 0.2, gain=0.14)
    rl = min(g.bar, 2.4)
    fx.add(S.riser(rng, rl, f_lo=300.0, f_hi=9000.0, curve=2.0), g.card_at - rl, gain=0.12)
    fx.add(S.rev_cymbal(rng, 0.62), g.card_at - 0.62, gain=0.05)
    fx.add(S.impact(rng, 1.0, thump_level=0.3), g.card_at, gain=0.5)
    for i, ((m, lv), ix) in enumerate(zip(A.card_bells(key), (2.0, 1.5, 1.1, 0.8))):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 3.2, index=ix, a_tau=1.2), g.card_at + i * 0.025, pan=-0.3 + 0.2 * i)
    for k, i in enumerate(A.sparkle_idx(key)):
        fx.add(S.fm_bell(S.midi_hz(key.midi(i)), 0.045 * (1 - 0.07 * k), 1.2, index=0.6, i_tau=0.1, a_tau=0.35),
               g.card_at + 0.3 + k * 0.04, pan=-0.6 + 0.24 * k)


def render(ctx: Ctx) -> np.ndarray:
    """Render the whole piece (pre-master stereo mix) from the plan in `ctx`."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    drums, room, music, bass, fx = (S.Bus(g.dur) for _ in range(5))
    pa = PATTERNS[int(ctx.comp.integers(len(PATTERNS)))]
    pb = PATTERNS[int(ctx.comp.integers(len(PATTERNS)))]
    bounce = bool(ctx.comp.integers(2))
    stab_voic = A.voice_events(plan, 55, 79, lambda c: list(c.pcs[:3]) + [c.root_pc])  # triad + octave root
    pad_voic = A.voice_events(plan, 57, 81, lambda c: c.pcs)
    kicks = _drums(ctx, drums, room)
    _plucks(ctx, music, stab_voic, pa, pb)
    _bass(ctx, bass, bounce)
    _pads_and_hook(ctx, music, pad_voic)
    _fx(ctx, fx)
    n = drums.st.shape[1]
    music_st = S.soft_clip(music.st * S.duck_env(n, kicks, depth=0.22), 0.85 * np.abs(music.st).max())
    room_ir, hall = S.make_ir(rng, 0.9, predelay=0.012), S.make_ir(rng, 2.3, predelay=0.025)
    d_dry = S.hp(drums.st, 48.0, 4)
    body = (S.soft_clip(d_dry, 0.8 * np.abs(d_dry).max()) + 0.22 * S.reverb(room.st, room_ir)
            + S.hp(bass.st * S.duck_env(n, kicks, depth=0.45), 60.0) + music_st + 0.26 * S.reverb(music_st, hall))
    return A.breath(body + S.lp(fx.st + 0.35 * S.reverb(fx.st, hall), 15000.0, 4), g.card_at, 0.4, 0.06)
