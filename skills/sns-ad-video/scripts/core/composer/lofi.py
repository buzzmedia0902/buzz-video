"""Style "lofi" (~86 BPM): FM electric piano, swung soft drums, round bass, vinyl crackle and tape wow.

Rootless 9th/13th voicings with smooth voice leading; swung 16ths (60 %); drums enter after the intro, drop
to hats in a break and stop for the last beat before the card. The bed keeps 1-4 kHz calm (EP low-passed at
3 kHz, snare low-passed) so a voice-over can sit on top. The card arrives on a soft kick, the tonic 9th chord
and a light two-note bell; the groove then falls away while the EP and crackle ring out.
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
    bpm=86.0,
    loops_a=((4, 3, 2, 5), (2, 5, 1, 6), (6, 2, 5, 1), (1, 6, 2, 5), (3, 6, 2, 5)),
    loops_b=((2, 5, 3, 6), (4, 3, 6, 5), (6, 5, 2, 1), (4, 5, 6, 3)),
    ext={1: "9", 2: "9", 3: "7", 4: "9", 5: "13", 6: "9", 7: "7"},
    card_ext="9", dom_ext="13", sus=False,
)
MASTER = {"glue_db": 6.0, "glue_ratio": 1.7, "hp_fc": 45.0, "hp_order": 4}
SWING = 0.6
HAT_VEL = (0.8, 0.0, 0.5, 0.22, 0.72, 0.0, 0.5, 0.3, 0.8, 0.0, 0.5, 0.22, 0.72, 0.18, 0.5, 0.34)
ANSWER = (((2, .5), (2.5, .5), (3, 1)), ((2, 1), (3, 1)), ((1, .5), (2, .5), (3, 1.5)))
KICKS = (((0, 1.0), (10, 0.78)), ((0, 1.0), (7, 0.5), (10, 0.72)))  # even / odd bars (16th position, velocity)
KEYS_GAIN, BASS_GAIN, DRUMS_GAIN, VINYL_GAIN = 0.4, 0.23, 1.05, 0.2


def tpos(g: Grid, bar_beat: float, pos: int) -> float:
    """Time (s) of 16th position `pos` (0-15) of the bar starting at beat `bar_beat`, swung off-16ths."""
    beat, sub = divmod(pos, 4)
    return g.t(bar_beat + beat + (0.0, SWING * 0.5, 0.5, 0.5 + SWING * 0.5)[sub])


class Kit:
    """Soft lo-fi drum sounds (several variants of the snare / hat so repeats do not machine-gun)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.kick = S.lp(S.kick(rng, f0=150.0, f1=86.0, p_tau=0.03, a_tau=0.16, drive=1.6, click=0.04), 1800.0)
        self.snares = [S.lp(S.snare(rng, tone=180.0, n_tau=0.13, t_tau=0.05, lo=700.0, hi=5000.0, tone_mix=0.7), 2200.0)
                       for _ in range(3)]
        self.hats = [S.lp(S.hat(rng, tau=0.024, lo=6000.0, hi=11000.0), 10000.0) for _ in range(4)]


def _hits(ctx: Ctx, drums: S.Bus, room: S.Bus, kit: Kit, bi: int, b0: float, kind: str) -> list[float]:
    """Drum hits of one bar. Kick + snare play in a / b / build, hats everywhere but the very first bar
    (quarters only in a break); the last beat before the card is stop-time. Returns the kick times."""
    g, rng = ctx.g, ctx.rng
    kicks: list[float] = []

    def live(pos: int) -> float | None:
        t = tpos(g, b0, pos)
        return t if 0 <= t < g.card_at - 1e-6 and not (kind == "build" and pos >= 12) else None

    if kind in ("a", "b", "build"):
        for pos, v in KICKS[bi % 2]:
            if (t := live(pos)) is not None:
                drums.add(kit.kick, t + (rng.uniform(-0.004, 0.004) if pos else 0.0), gain=0.52 * v)
                kicks.append(t)
        for pos in (4, 12):
            if (t := live(pos)) is not None:
                s, ts = kit.snares[(bi * 2 + pos // 12) % 3], t + rng.uniform(0.0, 0.012)  # laid back
                drums.add(s, ts, pan=0.04, gain=0.34)
                room.add(s, ts, gain=0.34)
        if bi % 2 == 1 and (t := live(15)) is not None:
            drums.add(kit.snares[bi % 3], t, pan=0.04, gain=0.07)  # ghost note
    if bi > 0 or kind != "intro":
        for pos in range(16):
            hv = HAT_VEL[pos]
            if hv and (kind != "brk" or pos % 4 == 0) and (t := live(pos)) is not None:
                drums.add(kit.hats[(bi + pos) % 4], t + rng.uniform(-0.005, 0.005), pan=0.28,
                          gain=0.085 * hv * rng.uniform(0.85, 1.1) * (0.6 if kind in ("intro", "brk") else 1.0))
    return kicks


def _drums(ctx: Ctx, drums: S.Bus, room: S.Bus) -> list[float]:
    kit, kicks = Kit(ctx.rng), []
    for bi, b0 in enumerate(A.all_bars(ctx.g)):
        kicks += _hits(ctx, drums, room, kit, bi, b0, ctx.plan.section_at(max(b0, 0.0)).kind)
    drums.add(kit.kick, ctx.g.card_at, gain=0.5)  # the groove's last act: one soft kick under the chord
    return kicks + [ctx.g.card_at]


def _keys(ctx: Ctx, keys: S.Bus, bass: S.Bus, voic: list[list[int]]) -> None:
    """EP chord strums (rolled low to high) on every chord event; the card chord also gets a held bass root."""
    g, rng = ctx.g, ctx.rng
    for ev, notes in zip(ctx.plan.events, voic):
        t0 = ctx.t(ev.beat)
        hold = min(ctx.t(ev.length), 3.2 if ev.kind == "card" else 2.6 * g.beat)
        for i, m in enumerate(notes):
            vel = (0.62 if i < 4 else 0.5) + rng.uniform(-0.05, 0.05)
            keys.add(S.ep(S.midi_hz(m), hold, vel), t0 + 0.009 * i + A.human(rng, 6), pan=-0.2 + 0.1 * i)
        if ev.kind == "card":
            root = pitch_near(ev.chord.root_pc, 46, 40, 52)
            bass.add(S.round_bass(S.midi_hz(root), min(2.2, g.dur - g.card_at), 0.95), t0)


def _bars(ctx: Ctx, keys: S.Bus, bass: S.Bus, voic: list[list[int]]) -> None:
    """Pushes, turnaround fills and the bass line, bar by bar."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    for b0 in A.all_bars(g):
        ei = plan.event_index_at(max(b0, 0.0))
        ev, notes = plan.events[ei], voic[ei]
        root = pitch_near(ev.chord.root_pc, 46, 40, 52)
        if b0 + 1.5 > 0:
            for i, m in enumerate(notes[1:]):  # push on the 'and' of 2
                keys.add(S.ep(S.midi_hz(m), 1.3 * g.beat, 0.38), tpos(g, b0, 6) + 0.01 * i, pan=-0.05 + 0.1 * i)
        if ev.chord.degree == 5 and b0 >= 0 and ev.kind != "build":  # turnaround on V: falling chord tones
            top = sorted(notes + [notes[0] + 12])[::-1][:4]
            for pos, m in zip((8, 10, 12, 14), top):
                keys.add(S.ep(S.midi_hz(m + 12 if m < 72 else m), 0.4 * g.beat, 0.4), tpos(g, b0, pos), pan=0.2)
        bass.add(S.round_bass(S.midi_hz(root), 1.7 * g.beat, 0.95), max(0.0, tpos(g, b0, 0)))
        bass.add(S.round_bass(S.midi_hz(root) * 1.5, 0.45 * g.beat, 0.6), tpos(g, b0, 10))
        nroot = pitch_near(plan.event_at(b0 + 4.0).chord.root_pc, root, 40, 52)
        approach = nroot + (-1 if rng.random() < 0.5 else 1) if nroot != root else root  # chromatic pick-up
        bass.add(S.round_bass(S.midi_hz(approach), 0.3 * g.beat, 0.45), tpos(g, b0, 14))


def _build_lift(ctx: Ctx, keys: S.Bus, fx: S.Bus, comp: Composer) -> None:
    """Rising EP figure over beats 2-3.5 of the last bar and a very soft, dark noise swell: the lo-fi 'arrival'."""
    g, key = ctx.g, ctx.plan.key
    if not ctx.plan.of_kind("build"):
        return
    end = comp.key.nearest_idx(79)
    end = min((c for c in range(end - 3, end + 3) if c % 7 in (1, 6)), key=lambda c: abs(c - end))  # 2nd or 7th
    for n in comp.run(g.n_beats - 2, 2.0, 0.5, end, vel0=0.26, vel1=0.5):
        keys.add(S.ep(S.midi_hz(n.midi), n.length * g.beat, n.vel), ctx.t(n.beat), pan=0.2)
    rl = min(g.bar, 2.8)
    fx.add(S.lp(S.riser(ctx.rng, rl, 250.0, 1500.0, 2.4), 1800.0), g.card_at - rl, gain=0.06)


def render(ctx: Ctx) -> np.ndarray:
    """Render the whole piece (pre-master stereo mix) from the plan in `ctx`."""
    plan, g, rng = ctx.plan, ctx.g, ctx.rng
    drums, room, keys, bass, fx = (S.Bus(g.dur) for _ in range(5))
    voic = A.voice_events(plan, 55, 77, lambda c: c.pcs[1:] if len(c.pcs) > 3 else c.pcs)
    comp = Composer(plan, ctx.comp, 67, 79)
    kinds = ("b",) if plan.of_kind("b") else ("a",)
    mel = A.section_melody(comp, plan, ANSWER, ANSWER, vel=0.34, kinds=kinds)  # sparse answers on top
    ctx.log["melody"] = mel
    for n in mel:
        keys.add(S.ep(S.midi_hz(n.midi), n.length * g.beat, n.vel), ctx.t(n.beat), pan=0.2)
    kicks = _drums(ctx, drums, room)
    _keys(ctx, keys, bass, voic)
    _bars(ctx, keys, bass, voic)
    _build_lift(ctx, keys, fx, comp)
    r = pitch_near(plan.key.tonic, 81, 76, 87)
    for i, (m, lv, ix) in enumerate(((r, 0.15, 1.1), (r + 7, 0.11, 0.9), (r + 12, 0.075, 0.7))):
        fx.add(S.fm_bell(S.midi_hz(m), lv, 2.5, index=ix, i_tau=0.2, a_tau=0.9), g.card_at + 0.07 * i, pan=0.1 + 0.1 * i)
    n = keys.st.shape[1]
    keys_st = KEYS_GAIN * S.autopan(S.hp(S.lp(keys.st, 3000.0), 200.0)) * S.duck_env(n, kicks, depth=0.12, recover=0.25)
    bass_st = BASS_GAIN * S.hp(S.lp(bass.st, 800.0), 55.0) * S.duck_env(n, kicks, depth=0.18, recover=0.22)
    music = S.hp(S.wow(np.tanh(1.2 * (keys_st + bass_st)) / 1.2), 40.0)
    plate = S.make_ir(rng, 1.6, hf_rt60=0.6, predelay=0.02)
    body = (DRUMS_GAIN * (S.hp(drums.st, 45.0, 4) + 0.18 * S.reverb(room.st, plate, lp_fc=5000.0))
            + music + 0.25 * S.reverb(keys_st, plate, lp_fc=5000.0) + S.vinyl(rng, n / S.SR) * VINYL_GAIN)
    return A.breath(body + fx.st + 0.4 * S.reverb(fx.st, plate), g.card_at, 0.45, 0.07)
