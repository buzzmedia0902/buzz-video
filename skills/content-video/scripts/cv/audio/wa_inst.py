"""和楽器 voices (no samples): koto, 17-string bass koto, shamisen, shakuhachi, sho drone, taiko.

koto       Karplus-Strong string (instruments.ks_string): pick-position pulse excitation (tsume near the bridge ->
           bright), tsume click, tension glide, 押し手 bends (ato-oshi / oshi-hanashi / yuri / tsuki-iro) via a cubic
           time-warp. The body resonance is a short modal IR convolved on the bus (`body_ir`).
shamisen   bright short KS + amplitude-normalised asymmetric "sawari" buzz + bachi / skin hit.
shakuhachi additive tone, meri scoop, delayed vibrato, breath + pitched noise.
sho        soft free-reed drone (detuned pairs, slow breathing), low-passed.
taiko      modal membrane with a pitch drop; its weight sits at 78-150 Hz so it still reads on a phone speaker.
"""
from __future__ import annotations

import numpy as np

from cv.audio.dsp import SR, bp, fade_end, lp, n_of, ramp_in, smooth_noise, smoothstep, tvec, unit
from cv.audio.instruments import ks_string, pluck_excitation

KOTO_BODY = ((112, 11, 0.55), (186, 13, 0.85), (262, 15, 1.0), (345, 16, 0.85), (452, 18, 0.7),
             (590, 19, 0.6), (770, 21, 0.45), (985, 22, 0.32), (1260, 24, 0.22), (1620, 26, 0.13),
             (2120, 28, 0.07), (2900, 30, 0.04))
BASS_BODY = ((92, 10, 0.5), (140, 12, 0.9), (205, 14, 1.0), (290, 15, 0.8), (400, 17, 0.6),
             (540, 18, 0.45), (720, 20, 0.3), (980, 22, 0.18), (1350, 24, 0.1))
SHAMISEN_BODY = ((300, 7, 0.8), (470, 7, 1.0), (690, 8, 0.8), (980, 8, 0.6), (1380, 8, 0.4),
                 (1900, 9, 0.22), (2600, 9, 0.1))



def warp(src: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Variable-rate read of `src` (cubic Hermite): out[m] = src(sum(ratio[:m])). Scales every partial together, which is
    what pressing a koto string behind its bridge does."""
    pos = np.concatenate(([0.0], np.cumsum(ratio[:-1])))
    pos = np.minimum(pos, len(src) - 1.001)
    i = pos.astype(np.int64)
    f = pos - i
    s = np.concatenate(([src[0]], src, [0.0, 0.0]))
    xm1, x0, x1, x2 = s[i], s[i + 1], s[i + 2], s[i + 3]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * f + c2) * f + c1) * f + x0


def envelope(x: np.ndarray, win: int, smooth_hz: float = 30.0) -> np.ndarray:
    """Peak envelope: running max of |x| over `win` samples, then low-passed."""
    from scipy.ndimage import maximum_filter1d
    return np.maximum(lp(maximum_filter1d(np.abs(x), size=max(3, int(win)), mode="nearest"), smooth_hz, 1), 0.0)


# =================================================================== koto
def koto_t60(f0: float) -> float:
    """Fundamental T60 of a koto string: ~4.3 s at G3, ~1.9 s at A5."""
    return float(np.clip(3.4 * (294.0 / f0) ** 0.55, 1.2, 5.5))


def bend_cents(kind: str, n: int, press_c: float, hold: float,
               rng: np.random.Generator) -> np.ndarray:
    """Left-hand technique as a pitch-offset curve (cents) over n samples."""
    t = np.arange(n) / SR
    if kind == "ato":        # 後押し: pluck open, press up afterwards
        t0 = hold if hold > 0 else rng.uniform(0.16, 0.3)
        return press_c * smoothstep((t - t0) / rng.uniform(0.09, 0.15))
    if kind == "hanashi":    # 押し放し: pluck while pressed, release down at `hold`
        return press_c * (1.0 - smoothstep((t - hold) / rng.uniform(0.07, 0.12)))
    if kind == "press":      # 強押し: pressed for the whole note
        return np.full(n, float(press_c))
    if kind == "yuri":       # 揺り: pressing vibrato (pitch only goes up)
        t0, rate = rng.uniform(0.25, 0.4), rng.uniform(4.2, 5.4)
        on = smoothstep((t - t0) / 0.35)
        wob = 0.5 - 0.5 * np.cos(2 * np.pi * rate * np.maximum(t - t0, 0.0))
        return rng.uniform(14.0, 26.0) * on * wob
    if kind == "tsuki":      # 突き色: quick push and return
        t0, w = rng.uniform(0.08, 0.16), rng.uniform(0.1, 0.16)
        return rng.uniform(40.0, 80.0) * np.sin(np.pi * np.clip((t - t0) / w, 0.0, 1.0)) ** 2
    if kind == "hiki":       # 引き色: slight pull below pitch
        return -rng.uniform(18.0, 35.0) * smoothstep((t - rng.uniform(0.15, 0.3)) / 0.2)
    return np.zeros(n)


def koto_pluck(f_open: float, vel: float, rng: np.random.Generator, length: float,
               cents: np.ndarray | None = None, bright: float = 0.6, t60_scale: float = 1.0,
               click: float = 0.07) -> np.ndarray:
    """One koto string pluck (mono, peak ~ vel**1.5). `cents` (len >= n) adds a bend."""
    n = max(n_of(length), 256)
    t = np.arange(n) / SR
    c = (2.5 + 8.0 * vel) * np.exp(-t / 0.045)          # tension glide (starts sharp)
    if cents is not None:
        c = c + cents[:n]
    ratio = 2.0 ** (c / 1200.0)
    hard = float(np.clip(0.2 + 0.8 * vel * bright, 0.05, 1.0))
    exc = pluck_excitation(f_open, rng, rng.uniform(0.07, 0.13), hard)
    t60 = koto_t60(f_open) * t60_scale * rng.uniform(0.9, 1.1)
    loss = float(np.clip(0.2 - 0.12 * bright - 0.04 * vel, 0.05, 0.24))
    y = warp(ks_string(f_open, int(np.ceil(ratio.sum())) + 8, exc, t60, loss), ratio)
    y /= np.abs(y[: n_of(0.06)]).max() + 1e-12
    nc = n_of(0.005)
    clk = bp(rng.normal(0.0, 1.0, nc), 2200.0, 7000.0) * np.exp(-np.arange(nc) / (0.0009 * SR))
    y[:nc] += click * vel * unit(clk)
    return y * ramp_in(n, 10) * fade_end(n, n_of(0.03)) * vel ** 1.5


def body_ir(modes: tuple, seed: int, wet: float, length: float = 0.14) -> np.ndarray:
    """Direct impulse + short modal body response (energy `wet` relative to direct)."""
    rng = np.random.default_rng(seed)
    t = tvec(length)
    res = np.zeros(len(t))
    for f, q, a in modes:
        f *= 1.0 + rng.uniform(-0.03, 0.03)
        res += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t * np.pi * f / q)
    res += 0.25 * unit(lp(rng.normal(0.0, 1.0, len(t)), 2500.0)) * np.exp(-t / 0.01)
    res *= ramp_in(len(t), 24) * fade_end(len(t), n_of(0.03))
    res *= np.sqrt(wet / np.sum(res ** 2))
    res[0] += 1.0
    return res


# =================================================================== shamisen
def shamisen_pluck(f0: float, vel: float, rng: np.random.Generator, length: float = 0.8) -> np.ndarray:
    """Short bachi stroke: bright string, sawari buzz, skin 'pan'. Mono, peak ~ vel**1.4."""
    n = n_of(length)
    t = np.arange(n) / SR
    ratio = 2.0 ** ((5.0 + 10.0 * vel) * np.exp(-t / 0.03) / 1200.0)
    exc = pluck_excitation(f0, rng, rng.uniform(0.04, 0.07), 0.95, 0.35)
    y = warp(ks_string(f0, int(np.ceil(ratio.sum())) + 8, exc, rng.uniform(0.9, 1.25), 0.05), ratio)
    y /= np.abs(y[: n_of(0.05)]).max() + 1e-12
    env = envelope(y, int(SR / f0) + 2, 40.0) + 1e-3
    u = np.clip(y / env, -1.5, 1.5)              # amplitude-normalised -> buzz persists
    drive, bias = 2.4, 0.35
    lin = drive * (1.0 - np.tanh(drive * bias) ** 2)
    buzz = np.tanh(drive * (u + bias)) - np.tanh(drive * bias) - lin * u
    buzz = bp(buzz * env, 650.0, 4200.0) * np.exp(-t / 0.5)
    hn = n_of(0.03)
    th = np.arange(hn) / SR
    hit = unit(bp(rng.normal(0.0, 1.0, hn), 220.0, 1600.0) * np.exp(-th / 0.005))
    hit += 0.8 * np.sin(2 * np.pi * 410.0 * th) * np.exp(-th / 0.02)
    out = y + 0.32 * buzz
    out[:hn] += 0.45 * vel * hit
    return out * ramp_in(n, 8) * fade_end(n, n_of(0.04)) * vel ** 1.4


# =================================================================== shakuhachi
def shakuhachi(f0: float, dur: float, vel: float, rng: np.random.Generator, scoop: float = -60.0,
               vib_depth: float = 20.0, breath: float = 0.3, release: float = 0.3) -> np.ndarray:
    """Breathy end-blown flute tone (mono, peak ~ vel)."""
    n = n_of(dur + release)
    t = np.arange(n) / SR
    rate = rng.uniform(4.6, 5.4) * (1.0 + 0.04 * smooth_noise(rng, n, 1.5))
    vib_on = smoothstep((t - min(0.45, 0.35 * dur)) / 0.6)
    vph = 2 * np.pi * np.cumsum(rate) / SR + rng.uniform(0, 2 * np.pi)
    cents = scoop * np.exp(-t / 0.09) + vib_depth * vib_on * np.sin(vph)
    ph = 2 * np.pi * np.cumsum(f0 * 2.0 ** (cents / 1200.0)) / SR
    harm = (1.0, 0.30 + 0.25 * vel, 0.12 + 0.12 * vel, 0.05 + 0.05 * vel, 0.025)
    tone = sum(a * np.sin((k + 1) * ph + rng.uniform(0, 2 * np.pi)) for k, a in enumerate(harm))
    pitched = smooth_noise(rng, n, 150.0) * np.sin(ph + 0.3)
    air = bp(rng.normal(0.0, 1.0, n), 700.0, 3000.0)
    air /= np.std(air) + 1e-12
    att = 0.08 + 0.06 * (1.0 - vel)
    env = smoothstep(t / att) * (1.0 + 0.1 * np.exp(-((t - att) / 0.07) ** 2))
    env *= (1.0 + 0.12 * np.sin(np.pi * np.clip(t / max(dur, 0.1), 0.0, 1.0)))
    env *= 1.0 - smoothstep((t - dur) / release)
    am = 1.0 + 0.05 * vib_on * np.sin(vph + 0.6) + 0.03 * smooth_noise(rng, n, 25.0)
    chiff = np.exp(-t / 0.06)
    out = env * am * (tone + breath * 0.35 * pitched) + breath * (0.05 + 0.22 * chiff) * air * env
    out = lp(out, 5000.0, 2) * ramp_in(n, n_of(0.004)) * fade_end(n, n_of(0.02))
    return vel * unit(out)


# =================================================================== sho-like drone
def sho_pad(freqs: tuple, dur: float, rng: np.random.Generator, attack: float = 0.6,
            release: float = 1.0) -> np.ndarray:
    """Soft reed drone (stereo): detuned L/R pairs, slow breathing, <1.4 kHz."""
    n = n_of(dur + release)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for f in freqs:
        for ch, det in enumerate((-2.0, 2.0)):
            ff = f * 2.0 ** ((det + rng.uniform(-0.8, 0.8)) / 1200.0)
            for k in range(1, 8):
                if k * ff > 2200.0:
                    break
                out[ch] += np.sin(2 * np.pi * k * ff * t + rng.uniform(0, 2 * np.pi)) / k ** 1.25
    breath = 1.0 + 0.07 * np.sin(2 * np.pi * rng.uniform(0.15, 0.25) * t + rng.uniform(0, 6.28))
    env = smoothstep(t / attack) * (1.0 - smoothstep((t - dur) / release))
    return lp(out * breath * env, 1400.0, 2) / (2.0 * max(1, len(freqs)))


# =================================================================== taiko
def taiko_hit(vel: float, rng: np.random.Generator, f0: float = 100.0, stick: float = 0.04,
              length: float = 1.3) -> np.ndarray:
    """Soft-mallet taiko 'don' (mono, peak ~ vel): membrane modes with a pitch drop."""
    t = tvec(length)
    ph = 2 * np.pi * np.cumsum(f0 * (1.0 + 0.35 * np.exp(-t / 0.035))) / SR
    y = np.sin(ph) * np.exp(-t / 0.42)
    for ratio, amp, tau in ((1.52, 0.45, 0.2), (1.98, 0.25, 0.13), (2.44, 0.14, 0.08)):
        y += amp * np.sin(ratio * ph + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
    y += 0.5 * unit(lp(rng.normal(0.0, 1.0, len(t)), 280.0)) * np.exp(-t / 0.014)
    y += stick * unit(bp(rng.normal(0.0, 1.0, len(t)), 500.0, 2200.0)) * np.exp(-t / 0.004)
    y *= ramp_in(len(t), n_of(0.0015)) * fade_end(len(t), n_of(0.08))
    return vel * unit(y)


def taiko_roll(length: float, vel: float, rng: np.random.Generator, peak_pos: float = 0.5,
               rate: float = 11.0, f0: float = 96.0) -> np.ndarray:
    """Soft 'dorodoro' roll: alternating-hand hits with a swell peaking at peak_pos."""
    variants = [taiko_hit(1.0, rng, f0 * (1.0 + 0.01 * k), stick=0.0, length=1.0) for k in range(4)]
    out = np.zeros(n_of(length + 1.0))
    tt = 0.0
    k = 0
    while tt < length:
        x = tt / length
        g = (np.sin(0.5 * np.pi * x / peak_pos) ** 2 if x < peak_pos
             else np.cos(0.5 * np.pi * (x - peak_pos) / max(1e-3, 1.0 - peak_pos)) ** 2)
        s = n_of(tt)
        h = variants[k % 4]
        e = min(len(out), s + len(h))
        out[s:e] += (0.15 + 0.85 * g) * (1.0 + rng.normal(0.0, 0.08)) * h[: e - s]
        tt += (1.0 / rate) * (1.0 + rng.normal(0.0, 0.07))
        k += 1
    return vel * unit(out) * fade_end(len(out), n_of(0.1))
