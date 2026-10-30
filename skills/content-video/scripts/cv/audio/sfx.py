"""Sound effects: named one-shots in five sets, placed on the events of cues_out.json.

  render(set, events, duration, sfx_db=0, seed=0) -> float32 (2, round(duration * 48000))
  events   [[t, name, gain_db], ...]   t = when the sound is HEARD (sweeps start earlier so their peak lands there)
  sets     soft (trust, minimal: rounded, short)   cute (friendly: marimba, glock, bubbles)   variety (punchy, cartoon)
           wa (hyoshigi, taiko, tsuzumi, rin, washi paper)   none (nothing)
  names    whoosh  pop  impact  tick  click  chime  sparkle  boing  taiko  hyoshigi
           (aliases: swish, scroll, brush, thud, hit, ding, bell, rin, blip, tsuzumi, key, type, kirakira, spring, don, clap)

Levels are absolute: every one-shot is normalised to the peak in PEAK_DB (dBFS, voice reference -16 LUFS) and then
moved by the event's gain and `sfx_db`, so "gain 0" is a normal-sized effect and a tone only adds offsets.
A name a set does not define falls back to the soft set. Every set is deterministic for a seed.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import resample_poly

from cv.audio import dsp
from cv.audio import instruments as S
from cv.audio import wa_inst as wi
from cv.audio.dsp import SR, bp, fade_end, hp, lp, n_of, tvec

SETS = ("soft", "cute", "variety", "wa", "none")
ALIAS = {"swish": "whoosh", "scroll": "whoosh", "brush": "whoosh", "thud": "impact", "hit": "impact", "slam": "impact",
         "ding": "chime", "bell": "chime", "rin": "chime", "blip": "pop", "bubble": "pop", "tsuzumi": "pop", "key": "click",
         "type": "click", "kirakira": "sparkle", "shine": "sparkle", "spring": "boing", "don": "taiko", "clap": "hyoshigi",
         "ki": "hyoshigi"}
# nominal peak (dBFS, pre-master) of each name at event gain 0; SET_OFFSET_DB moves a whole set
PEAK_DB = {"whoosh": -5.0, "pop": -5.0, "impact": -2.0, "tick": -9.0, "click": -7.0, "chime": -7.0, "sparkle": -10.0,
           "boing": -5.0, "taiko": -2.0, "hyoshigi": -4.0}
SET_OFFSET_DB = {"soft": -1.5, "cute": 0.0, "variety": 1.0, "wa": 0.0}
# seconds a sweep runs before it is "heard" (its peak): the sound starts that much earlier than the event time
PRE = {("soft", "whoosh"): 0.28, ("cute", "whoosh"): 0.23, ("variety", "whoosh"): 0.23, ("wa", "whoosh"): 0.28}
ROOM_SEED = 5151


def _mono2(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x]) if x.ndim == 1 else x


def _trim(x: np.ndarray, sec: float) -> np.ndarray:
    n = n_of(sec)
    y = x[..., :n] if x.shape[-1] >= n else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])
    return y * fade_end(n, n_of(0.04))


def _sum_at(length: float, parts) -> np.ndarray:
    """Mono mix of (start s, signal) parts in a buffer of `length` s."""
    out = np.zeros(n_of(length))
    for at, sig in parts:
        s = n_of(at)
        out[s:s + len(sig)] += sig[: len(out) - s]
    return out


def _room(mono: np.ndarray, wet: float, length: float, rt60: float = 0.5) -> np.ndarray:
    """Place a mono sound in a small room (stereo, fixed length)."""
    n = n_of(length)
    x = np.zeros(n)
    x[: min(n, len(mono))] = mono[:n]
    ir = dsp.make_ir(np.random.default_rng(ROOM_SEED), rt60, rt60 * 0.4, 0.006)
    return np.stack([x, x]) + wet * hp(dsp.reverb(np.stack([x, x]), ir, hp_fc=220.0), 220.0)


# =================================================================== soft
def soft_whoosh(rng):
    return lp(S.swish(rng, pre=0.40, post=0.30, f_lo=260.0, f_hi=3200.0, pan=(-0.5, 0.5), bw_oct=1.7, curve=2.0), 3800.0)


def soft_pop(rng):
    return _mono2(lp(S.bubble_pop(rng, 830.0), 4500.0))


def soft_impact(rng):
    return _mono2(lp(_trim(S.impact(rng, 0.7, thump_level=0.55), 1.2), 3500.0))


def soft_tick(rng):
    return _mono2(S.tick(0.85))


def soft_click(rng):
    return _mono2(S.key_tap(rng, pitch=1.0))


def soft_chime(rng):
    return _mono2(S.chime([(S.hz("D6"), 0.10, 3.5, 2.0), (S.hz("A6"), 0.08, 3.5, 1.5), (S.hz("F#7"), 0.05, 3.5, 1.1)],
                          spread=0.04, length=2.2, a_tau=1.0))


def soft_sparkle(rng):
    return _mono2(_sum_at(1.4, [(0.07 * i, S.fm_bell(f, 0.5 * (1 - 0.1 * i), 0.9, index=0.8, i_tau=0.1, a_tau=0.3))
                                for i, f in enumerate((1568.0, 1976.0, 2349.0, 3136.0, 3951.0))]))


def soft_boing(rng):
    return _mono2(lp(S.sticker_wobble(), 3000.0))


def soft_taiko(rng):
    return _mono2(lp(wi.taiko_hit(1.0, rng), 900.0))


def soft_hyoshigi(rng):
    return _mono2(_sum_at(0.6, [(0.0, S.key_tap(rng, pitch=1.6)), (0.22, S.key_tap(rng, pitch=1.6))]))


# =================================================================== cute
def cute_whoosh(rng):
    t = tvec(0.55)
    f = 500.0 * (1800.0 / 500.0) ** np.clip(t / 0.35, 0, 1)
    shape = np.sin(np.pi * np.clip(t / 0.5, 0, 1)) ** 1.5
    return _mono2(lp(0.8 * np.sin(2 * np.pi * np.cumsum(f) / SR) * shape
                     + 0.15 * S.shaped_noise(rng, len(t), 3000.0, 1.5) * shape, 6000.0))


def cute_pop(rng):
    return _mono2(S.bubble_pop(rng, 1320.0))


def cute_impact(rng):
    return _mono2(_sum_at(0.9, [(0.0, S.marimba(196.0, 0.4, 1.0)), (0.0, S.fm_bell(1568.0, 0.35, 0.8, index=0.8, i_tau=0.1, a_tau=0.25))]))


def cute_tick(rng):
    return _mono2(S.marimba(1568.0, 0.08, 0.9))


def cute_click(rng):
    t = tvec(0.06)
    return _mono2(np.sin(2 * np.pi * 1900.0 * t) * np.exp(-t / 0.006)
                  + 0.3 * lp(rng.normal(0, 1, len(t)), 5000.0) * np.exp(-t / 0.002))


def cute_chime(rng):
    parts = []
    for i, f in enumerate((S.hz("C6"), S.hz("E6"), S.hz("G6"))):
        parts += [(0.09 * i, S.marimba(f, 0.3, 0.8)), (0.09 * i, S.fm_bell(f * 2, 0.15, 0.9, index=0.6, i_tau=0.1, a_tau=0.3))]
    return _mono2(_sum_at(1.5, parts))


def cute_sparkle(rng):
    return _mono2(_sum_at(1.2, [(0.05 * i, S.fm_bell(f, 0.4, 0.7, index=0.6, i_tau=0.08, a_tau=0.2))
                                for i, f in enumerate((2093.0, 2637.0, 3136.0, 3520.0, 4186.0, 5274.0))]))


def cute_boing(rng):
    return _mono2(S.sticker_wobble())


def cute_taiko(rng):
    return _mono2(S.marimba(130.0, 0.5, 1.0))


def cute_hyoshigi(rng):
    return _mono2(_sum_at(0.5, [(0.0, cute_click(rng)[0]), (0.15, cute_click(rng)[0])]))


# =================================================================== variety
def var_whoosh(rng):
    return lp(S.swish(rng, pre=0.35, post=0.30, f_lo=300.0, f_hi=6500.0, pan=(-0.7, 0.7), bw_oct=1.4, curve=2.4), 9000.0)


def var_pop(rng):
    return _mono2(_sum_at(0.25, [(0.0, S.bubble_pop(rng, 1040.0)), (0.05, 0.6 * S.bubble_pop(rng, 1560.0))]))


def var_impact(rng):
    return _mono2(_trim(S.impact(rng, 1.0, thump_level=0.9), 1.4))


def var_tick(rng):
    return _mono2(S.glide_tone(1500.0, 2200.0, 0.07, 0.02))


def var_click(rng):
    return _mono2(S.key_tap(rng, pitch=1.3))


def var_chime(rng):
    return _mono2(_sum_at(1.6, [(0.0, S.chime([(S.hz("E6"), 0.12, 3.5, 2.2)], length=1.2, a_tau=0.7)),
                                (0.2, S.chime([(S.hz("C6"), 0.12, 3.5, 2.2)], length=1.2, a_tau=0.7))]))


def var_sparkle(rng):
    return _mono2(_sum_at(1.3, [(0.045 * i, S.fm_bell(f, 0.45, 0.8, index=0.8, i_tau=0.1, a_tau=0.25))
                                for i, f in enumerate((2093.0, 2637.0, 3136.0, 4186.0, 3136.0, 4186.0, 5274.0))]))


def var_boing(rng):
    return _mono2(resample_poly(S.sticker_wobble(), 3, 4))


def var_taiko(rng):
    return _mono2(lp(wi.taiko_hit(1.0, rng), 4000.0))


def var_hyoshigi(rng):
    return _mono2(_sum_at(0.5, [(0.0, S.snap(rng)), (0.12, S.snap(rng))]))


# =================================================================== wa
def _clack(rng, vel: float = 1.0) -> np.ndarray:
    """One hyoshigi strike: two hardwood bars (free-free bending modes 1 : 2.76 : 5.4), a knock and a sharp contact."""
    t = tvec(0.3)
    modes = ((1960.0, 0.050, 1.0), (2230.0, 0.042, 0.8), (5400.0, 0.016, 0.32), (6130.0, 0.013, 0.26),
             (820.0, 0.022, 0.42), (1180.0, 0.016, 0.33), (1530.0, 0.012, 0.24))
    y = np.zeros(len(t))
    for f, tau, a in modes:
        f *= 1.0 + rng.uniform(-0.01, 0.01)
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / (tau * rng.uniform(0.9, 1.1)))
    y += 0.9 * dsp.unit(bp(rng.normal(0.0, 1.0, len(t)), 1200.0, 12000.0)) * np.exp(-t / 0.0007)
    return vel * dsp.unit(y * dsp.ramp_in(len(t), 8) * fade_end(len(t), n_of(0.05)))


def wa_whoosh(rng):
    """Washi paper swish: swept noise with a crackle of fibres."""
    n = n_of(0.9)
    t = np.arange(n) / SR
    sw = S.swish(rng, pre=0.40, post=0.45, f_lo=900.0, f_hi=5200.0, pan=(-0.6, 0.6), bw_oct=1.6, curve=2.0)
    out = np.zeros((2, n))
    out[:, : sw.shape[1]] = sw
    fib = np.zeros(n)
    for pos, amp in zip(rng.integers(0, n - 200, rng.poisson(100)), rng.uniform(0.1, 1.0, 100) ** 2):
        g = n_of(rng.uniform(0.0008, 0.003))
        fib[pos:pos + g] += amp * rng.normal(0.0, 1.0, g) * np.exp(-np.arange(g) / (0.25 * g))
    env = dsp.smoothstep(t / 0.07) * (1.0 - dsp.smoothstep((t - 0.5) / 0.35))
    return lp(out + 0.25 * env * bp(fib, 2500.0, 8000.0)[None, :], 9000.0)


def wa_pop(rng):
    """Soft hollow wood 'pokotto': short upward-gliding resonance + gentle knock."""
    t = tvec(0.2)
    ph = 2 * np.pi * np.cumsum(700.0 * (1.0 - 0.25 * np.exp(-t / 0.01))) / SR
    y = (np.sin(ph) * np.exp(-t / 0.04) + 0.35 * np.sin(2.31 * ph + 0.5) * np.exp(-t / 0.018)
         + 0.12 * np.sin(4.1 * ph) * np.exp(-t / 0.008))
    y += 0.2 * dsp.unit(bp(rng.normal(0.0, 1.0, len(t)), 800.0, 3500.0)) * np.exp(-t / 0.0012)
    return _mono2(lp(y * dsp.ramp_in(len(t), n_of(0.0008)) * fade_end(len(t), n_of(0.05)), 6000.0))


def wa_impact(rng):
    out = _mono2(wi.taiko_hit(1.0, rng))
    cl = _clack(rng, 0.5)
    out[:, : len(cl)] += 0.4 * cl[None, :]
    return out


def wa_tick(rng):
    return _room(_clack(rng, 0.7), 0.15, 0.5)


def wa_click(rng):
    return _room(_clack(rng, 0.5)[: n_of(0.12)], 0.1, 0.3)


def wa_chime(rng):
    """Small bell (rin): partials 1 : 2.74 : 5.18 : 8.3, each a slowly beating doublet."""
    t = tvec(2.5)
    out = np.zeros((2, len(t)))
    for ratio, amp, tau, beat in ((1.0, 1.0, 0.75, 1.6), (2.74, 0.5, 0.42, 2.4), (5.18, 0.18, 0.2, 3.3), (8.3, 0.06, 0.09, 4.4)):
        f, p0 = 1180.0 * ratio, rng.uniform(0, 2 * np.pi)
        for ch in range(2):
            pair = np.sin(2 * np.pi * (f - beat / 2) * t + p0) + np.sin(2 * np.pi * (f + beat / 2) * t + p0 + 0.9 * ch)
            out[ch] += 0.5 * amp * pair * np.exp(-t / tau)
    return out * dsp.ramp_in(len(t), n_of(0.0015)) * fade_end(len(t), n_of(0.7))


def wa_sparkle(rng):
    """A koto glissando up the 陽音階 (D E G A B) in the high register."""
    return _mono2(_sum_at(2.0, [(0.07 * i, wi.koto_pluck(440.0 * 2 ** ((m - 69) / 12), 0.7 - 0.04 * i, rng, 1.2))
                                for i, m in enumerate((74, 76, 79, 81, 83, 86))]))


def wa_boing(rng):
    """Kotsuzumi 'pon': grip released after the stroke -> falling pitch, hollow ring."""
    t = tvec(0.9)
    ph = 2 * np.pi * np.cumsum(290.0 + 200.0 * np.exp(-t / 0.06)) / SR
    y = np.sin(ph) * np.exp(-t / 0.2)
    for ratio, amp, tau in ((1.59, 0.35, 0.08), (2.14, 0.22, 0.05), (2.65, 0.12, 0.035), (3.16, 0.07, 0.025)):
        y += amp * np.sin(ratio * ph + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    y += 0.25 * dsp.unit(bp(rng.normal(0.0, 1.0, len(t)), 520.0, 820.0)) * np.exp(-t / 0.05)
    y += 0.35 * dsp.unit(bp(rng.normal(0.0, 1.0, len(t)), 1500.0, 6000.0)) * np.exp(-t / 0.0015)
    return _room(lp(y * dsp.ramp_in(len(t), n_of(0.001)) * fade_end(len(t), n_of(0.25)), 7000.0), 0.22, 0.9)


def wa_taiko(rng):
    return _mono2(wi.taiko_hit(1.0, rng))


def wa_hyoshigi(rng):
    return _room(_sum_at(1.2, [(0.0, _clack(rng, 1.0)), (0.38, _clack(rng, 0.93))]), 0.3, 1.2)


RECIPES = {
    "soft": {"whoosh": soft_whoosh, "pop": soft_pop, "impact": soft_impact, "tick": soft_tick, "click": soft_click,
             "chime": soft_chime, "sparkle": soft_sparkle, "boing": soft_boing, "taiko": soft_taiko,
             "hyoshigi": soft_hyoshigi},
    "cute": {"whoosh": cute_whoosh, "pop": cute_pop, "impact": cute_impact, "tick": cute_tick, "click": cute_click,
             "chime": cute_chime, "sparkle": cute_sparkle, "boing": cute_boing, "taiko": cute_taiko,
             "hyoshigi": cute_hyoshigi},
    "variety": {"whoosh": var_whoosh, "pop": var_pop, "impact": var_impact, "tick": var_tick, "click": var_click,
                "chime": var_chime, "sparkle": var_sparkle, "boing": var_boing, "taiko": var_taiko,
                "hyoshigi": var_hyoshigi},
    "wa": {"whoosh": wa_whoosh, "pop": wa_pop, "impact": wa_impact, "tick": wa_tick, "click": wa_click,
           "chime": wa_chime, "sparkle": wa_sparkle, "boing": wa_boing, "taiko": wa_taiko, "hyoshigi": wa_hyoshigi},
}


def canonical(name: str) -> str:
    n = ALIAS.get(str(name).lower(), str(name).lower())
    if n not in PEAK_DB:
        raise ValueError(f"unknown sfx {name!r}; choose from {', '.join(sorted(PEAK_DB))} "
                         f"(aliases: {', '.join(sorted(ALIAS))})")
    return n


def one_shot(set_name: str, name: str, seed: int = 0) -> np.ndarray:
    """One sound of a set as stereo float at the nominal peak of its name (event gain 0), lead silence trimmed."""
    n = canonical(name)
    fn = RECIPES.get(set_name, RECIPES["soft"]).get(n) or RECIPES["soft"][n]
    y = hp(_mono2(np.asarray(fn(np.random.default_rng([seed, zlib.crc32(n.encode()) % 9973])), dtype=float)), 60.0, 2)
    peak = float(np.abs(y).max())
    first = int(np.argmax(np.abs(y).max(axis=0) > peak * 10 ** (-50 / 20))) if peak > 0 else 0
    return y[:, max(0, first - 2):] / (peak + 1e-12) * dsp.from_db(PEAK_DB[n] + SET_OFFSET_DB.get(set_name, 0.0))


def unknown_names(events) -> list[str]:
    """Event names no set knows (render skips them; the caller can warn)."""
    bad = []
    for ev in events:
        try:
            canonical(ev[1])
        except ValueError:
            bad.append(str(ev[1]))
    return sorted(set(bad))


def render(set_name: str, events, duration: float, sfx_db: float = 0.0, seed: int = 0) -> np.ndarray:
    """Place the events: stereo float32 of exactly round(duration * SR) samples (silence for set 'none'); events with an
    unknown name are skipped."""
    if set_name not in SETS:
        raise ValueError(f"unknown sfx set {set_name!r}; choose from {', '.join(SETS)}")
    n = n_of(duration)
    bus = dsp.Bus(duration, tail=3.0)
    if set_name != "none":
        cache: dict[str, np.ndarray] = {}
        for ev in sorted((e for e in events if str(e[1]) not in unknown_names([e])), key=lambda e: float(e[0])):
            t, name = float(ev[0]), canonical(ev[1])
            gain = float(ev[2]) if len(ev) > 2 and ev[2] is not None else 0.0
            if name not in cache:
                cache[name] = one_shot(set_name, name, seed)
            bus.add(cache[name], max(0.0, t - PRE.get((set_name, name), 0.0)), gain=float(dsp.from_db(gain + sfx_db)))
    return (bus.st[:, :n] * fade_end(n, n_of(0.005))).astype(np.float32)
