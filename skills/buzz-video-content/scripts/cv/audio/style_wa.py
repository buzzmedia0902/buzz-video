"""Style "wa" (和・重厚 tone, ~72 BPM): original 和楽器 music - koto, 17-string bass koto, shamisen, shakuhachi, sho, taiko.

No sample, no quoted tune: every melody is generated from the seed (wa_score / wa_arrange) in 都節 (miyako-bushi, calm)
or 陽音階 (bright) with koto techniques (押し手 bends, 揺り, 突き色), every sound from DSP (wa_inst). The piece is told in
movements with their own themes, so a ten-minute track keeps changing. The taiko (weight at 78-150 Hz, so it reads on a
phone) rolls under a title card, lifts the B phrases and lands the piece on the end card with a roll and a don.
The score keeps the style's own beat map (a ritardando into the final stroke), so `beat_times` comes from it.
Rendered chunk by chunk: stems are balanced to fixed loudness targets (measured the first time a stem sounds), shaped
by the stem EQ, sent to one shared hall, and passed through the narration-friendly master EQ (1-4 kHz dipped).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from cv.audio import dsp
from cv.audio import wa_inst as wi
from cv.audio.dsp import Bus, convolve_st, eq, high_shelf, hp, lp, n_of, peaking
from cv.audio.form import Plan, StyleCfg
from cv.audio.wa_arrange import Score, compose
from cv.audio.wa_score import MOODS

CFG = StyleCfg(bpm=72.0, loops_a=((1, 5, 6, 4),), loops_b=((4, 5, 6, 3),),
               ext={d: "tri" for d in range(1, 8)}, card_ext="tri", dom_ext="tri")     # the generic grid only times chunks
MOOD = "calm"
IR_SEED = 4242
STEMS = ("koto", "koto2", "bass", "pad", "shaku", "shami", "taiko")
STEM_LUFS = {"koto": -20.0, "koto2": -27.0, "bass": -31.0, "pad": -33.5, "shaku": -24.5, "shami": -27.5, "taiko": -27.0}
SEND = {"koto": 0.30, "koto2": 0.34, "bass": 0.16, "pad": 0.42, "shaku": 0.48, "shami": 0.24, "taiko": 0.38}
KOTO_CFG = {"koto": {"bright": 0.62, "t60": 1.0, "pan": (-0.28, 0.06), "click": 0.07},
            "koto2": {"bright": 0.42, "t60": 0.85, "pan": (0.22, 0.45), "click": 0.04},
            "bass": {"bright": 0.22, "t60": 0.9, "pan": (-0.1, -0.1), "click": 0.0}}


def mtof(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def prepare(plan: Plan, rng: np.random.Generator) -> None:
    seed = int(rng.integers(1 << 30))
    mood = MOODS[plan.extra.get("mood") or MOOD]
    score = compose(plan.t_land, seed, plan.bpm if plan.exact else CFG.bpm, mood, exact=plan.exact,
                    title=plan.bars[0].card == "title", end_card=plan.bars[-1].card == "end")
    plan.extra.update(score=score, seed=seed)
    plan.t_land = score.t_final
    plan.bpm = 60.0 / score.tmap.spb
    plan.beat = score.tmap.spb


def key_name(plan: Plan) -> str:
    return "D 都節 (miyako-bushi)" if plan.extra["score"].mood.name == "calm" else "D 陽音階 (yo)"


def beat_times(plan: Plan) -> list[float]:
    tm = plan.extra["score"].tmap
    out, b = [], 0
    while True:
        t = tm.sec(b)
        if t > plan.duration:
            return out
        out.append(round(t, 4))
        b += 1


def _koto_lengths(score: Score, t_end: float) -> dict[int, float]:
    """Per koto-family event: how long its string rings (a new pluck on the same string stops the old one)."""
    strings: dict = defaultdict(list)
    for i, ev in enumerate(score.events):
        if ev.inst in KOTO_CFG:
            strings[(ev.inst, ev.open_pitch if ev.open_pitch is not None else ev.pitch)].append(i)
    out = {}
    for (inst, open_p), idxs in strings.items():
        idxs.sort(key=lambda i: score.events[i].t)
        natural = wi.koto_t60(mtof(open_p)) * KOTO_CFG[inst]["t60"] * 1.1 + 0.1
        for k, i in enumerate(idxs):
            ev = score.events[i]
            length = natural if k + 1 == len(idxs) else min(natural, score.events[idxs[k + 1]].t - ev.t + 0.02)
            out[i] = min(length, t_end - ev.t)
    return out


def make_ctx(plan: Plan, seed: int) -> dict:
    score: Score = plan.extra["score"]
    return {"klen": _koto_lengths(score, plan.duration), "gain": {}, "times": np.array([e.t for e in score.events]),
            "ir": dsp.make_ir(np.random.default_rng(IR_SEED), 2.0, 0.75, 0.018),
            "body": {"koto": wi.body_ir(wi.KOTO_BODY, 101, 0.15), "bass": wi.body_ir(wi.BASS_BODY, 102, 0.2),
                     "shami": wi.body_ir(wi.SHAMISEN_BODY, 103, 0.35, 0.08)}, "seed": plan.extra["seed"]}


def _bend(rng: np.random.Generator, ev, open_p: int, press: int, n: int) -> np.ndarray | None:
    if not ev.orn:
        return None
    press_c = 100.0 * (ev.pitch - open_p) if ev.orn in ("ato", "press") else 100.0 * press
    return wi.bend_cents(ev.orn, n, press_c, ev.hold, rng)


def _event(ctx: dict, score: Score, idx: int, t_end: float, buses: dict, w0: float) -> None:
    ev = score.events[idx]
    rng = np.random.default_rng([ctx["seed"], 21, idx])
    t = ev.t - w0
    room = t_end - ev.t
    if ev.inst in KOTO_CFG:
        cfg = KOTO_CFG[ev.inst]
        open_p = ev.open_pitch if ev.open_pitch is not None else ev.pitch
        f_open, length = mtof(open_p), ctx["klen"][idx]
        if length < 0.03:
            return
        lo, hi = cfg["pan"]
        pan = lo + (hi - lo) * float(np.clip((open_p - 50) / 31.0, 0.0, 1.0))
        gain = 1.0 if ev.inst == "bass" else min(1.0, (f_open / 294.0) ** 0.45)
        cents = _bend(rng, ev, open_p, score.mood.press, max(n_of(length), 256))
        buses[ev.inst].add(wi.koto_pluck(f_open, ev.vel, rng, length, cents, cfg["bright"], cfg["t60"], cfg["click"]), t, pan, gain)
    elif ev.inst == "pad":
        start = ev.t if ev.t < 0.05 else ev.t - 0.25
        att = 0.25 if ev.t < 0.05 else 0.6
        rel = float(np.clip(0.4 * (t_end - start), 0.5, 1.2))
        dur = min(ev.dur + (ev.t - start), t_end - start - rel - 0.1)
        if dur > 0.2:
            buses["pad"].add(wi.sho_pad(tuple(mtof(p) for p in ev.extra), dur, rng, att, rel), start - w0)
    elif ev.inst == "shaku":
        rel = 0.6 if ev.dur > 4.0 else 0.3
        dur = min(ev.dur, room - rel - 0.05)
        if dur > 0.3:
            scoop = rng.uniform(-80.0, -40.0) if score.mood.name == "calm" else rng.uniform(-50.0, -25.0)
            buses["shaku"].add(wi.shakuhachi(mtof(ev.pitch), dur, ev.vel, rng, scoop, rng.uniform(14.0, 24.0), 0.3, rel), t, 0.22)
    elif ev.inst == "shami":
        buses["shami"].add(wi.shamisen_pluck(mtof(ev.pitch), ev.vel, rng, 0.8), t, -0.38)
    elif ev.inst == "roll":
        length = min(ev.dur, room - 0.2)
        if length > 0.3:
            buses["taiko"].add(wi.taiko_roll(length, ev.vel, rng, ev.extra[0]), t, 0.0)
    elif ev.inst == "taiko":
        buses["taiko"].add(wi.taiko_hit(ev.vel, rng), t, 0.0)


def _extent(ctx: dict, ev, idx: int) -> float:
    """Seconds an event sounds after its start (generous), so the chunk buffer holds it to the end."""
    if ev.inst in KOTO_CFG:
        return ctx["klen"][idx] + 0.1
    if ev.inst in ("pad", "shaku"):
        return ev.dur + 1.6
    if ev.inst == "roll":
        return ev.dur + 1.2
    return 1.5


def _stem_eq(name: str, x: np.ndarray) -> np.ndarray:
    if name == "koto":
        return hp(x, 90.0, 2)
    if name == "koto2":
        return hp(x, 110.0, 2)
    if name == "bass":
        return lp(hp(x, 60.0, 4), 2500.0, 2)
    if name == "pad":
        return lp(hp(x, 140.0, 2), 1400.0, 2)
    if name == "shaku":
        return lp(hp(x, 220.0, 2), 5500.0, 2)
    if name == "shami":
        return lp(hp(x, 160.0, 2), 6000.0, 2)
    return lp(hp(x, 55.0, 4), 2500.0, 2)        # taiko


def render(plan: Plan, bars: list, ctx: dict, rng: np.random.Generator) -> np.ndarray:
    score: Score = plan.extra["score"]
    w0 = bars[0].t0
    w1 = float("inf") if bars[-1].kind == "land" else bars[-1].t1
    span = (plan.duration if w1 == float("inf") else w1) - w0
    idxs = [int(i) for i in np.nonzero((ctx["times"] >= w0) & (ctx["times"] < w1))[0]]
    reach = max([score.events[i].t + _extent(ctx, score.events[i], i) - w0 for i in idxs] + [span])
    buses = {name: Bus(reach, tail=0.5) for name in STEMS}
    for idx in idxs:
        _event(ctx, score, idx, plan.duration, buses, w0)
    ir = ctx["ir"]
    n = buses["koto"].st.shape[1]
    dry, send = np.zeros((2, n)), np.zeros((2, n))
    for name in STEMS:
        x = buses[name].st
        if name in ctx["body"]:
            x = convolve_st(x, ctx["body"][name])
        x = _stem_eq(name, x)
        if name not in ctx["gain"]:
            level = dsp.lufs(x)
            if level <= -69.0:
                continue
            ctx["gain"][name] = STEM_LUFS[name] - level
        x = x * dsp.from_db(ctx["gain"][name])
        dry += x
        send += SEND[name] * x
    wet = lp(hp(dsp.reverb(send, ir, hp_fc=20.0, keep_tail=True), 180.0, 2), 6500.0, 2)
    mix = np.pad(dry, ((0, 0), (0, wet.shape[1] - n))) + wet
    x = hp(mix, 52.0, 6)
    x = eq(x, peaking(2300.0, -5.0, 0.7), high_shelf(7500.0, -3.0), high_shelf(220.0, 2.0, 0.7))
    return lp(x, 13000.0, 2)

