"""Instrument and effect voices (48 kHz, float64, numpy / scipy only, no samples): the tone generators every BGM
style and the SFX sets are built from.

  notes      note_midi / midi_hz / hz
  keys       felt_piano, ep (FM Rhodes), fm_bell, chime, fm_lead, pluck (detuned saws)
  pads       warm_pad, saw_pad, strings, air, sub_sine
  basses     round_bass (lo-fi), synth_bass (pop)
  strings    nylon_guitar (Karplus-Strong), marimba
  drums      kick, snare, clap, hat, shaker, snap, brush_hit
  fx         swish, riser, impact, rev_cymbal, bubble_pop, tick, glide_tone, sticker_wobble, key_tap, vinyl, wow, autopan
Ported from buzz-video-ad's core/synth.py (DSP unchanged) plus the guitar / marimba / snap voices. Cached instrument
functions return read-only arrays: copy before modifying in place.
"""
from __future__ import annotations

from functools import lru_cache, wraps
from typing import Sequence

import numpy as np
from scipy.signal import lfilter, resample_poly

from cv.audio.dsp import SR, bp, env_ar, eq, fade_end, hp, lp, pan_st, peaking, shaped_noise, tvec

# =============== basics
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def note_midi(name: str) -> int:
    """'F#4' / 'Bb3' -> MIDI number (C4 = 60)."""
    pc, i = _PC[name[0].upper()], 1
    while i < len(name) and name[i] in "#b":
        pc += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + pc


def midi_hz(m: float) -> float:
    """MIDI note number (may be fractional) -> Hz."""
    return float(440.0 * 2.0 ** ((m - 69.0) / 12.0))


def hz(name: str) -> float:
    """Note name ('F#4', 'Bb3') -> Hz."""
    return midi_hz(note_midi(name))


def cached(maxsize: int = 600):
    """lru_cache that rounds float arguments (pitch, length, velocity) so near-equal notes share one rendering."""
    def deco(fn):
        c = lru_cache(maxsize)(lambda *a: _frozen(fn(*a)))

        @wraps(fn)
        def wrapper(*args):
            return c(*[round(a, 2) if isinstance(a, float) else a for a in args])
        wrapper.cache_clear = c.cache_clear
        return wrapper
    return deco


def _frozen(x: np.ndarray) -> np.ndarray:
    x.setflags(write=False)
    return x


def clear_caches() -> None:
    """Free the note caches (a 20-minute render holds hundreds of MB of them)."""
    for fn in list(globals().values()):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


def saw(freq: float | np.ndarray, n: int, phase0: float = 0.0) -> np.ndarray:
    """Band-limited (polyBLEP) sawtooth; freq may be a per-sample array (vibrato)."""
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    dt = f / SR
    ph = (phase0 + np.concatenate([[0.0], np.cumsum(dt[:-1])])) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= 2.0 * x - x * x - 1.0
    m = ph > 1.0 - dt
    x = (ph[m] - 1.0) / dt[m]
    y[m] -= x * x + 2.0 * x + 1.0
    return y


# =============== keys
def fm_bell(freq: float, level: float, length: float = 3.0, ratio: float = 3.5,
            index: float = 2.2, i_tau: float = 0.45, a_tau: float = 1.4,
            attack: float = 0.002, os: int = 4) -> np.ndarray:
    """FM bell rendered 4x oversampled so bright high notes do not alias."""
    sr = SR * os
    t = tvec(length, sr)
    idx = index * np.exp(-t / i_tau)
    y = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * ratio * t))
    y *= np.minimum(1.0, t / attack) * np.exp(-t / a_tau) * fade_end(len(t), int(0.03 * sr))
    return level * resample_poly(y, 1, os)


def chime(notes: Sequence[tuple[float, float, float, float]], spread: float = 0.02,
          length: float = 6.0, a_tau: float = 2.4) -> np.ndarray:
    """Rolled FM bell chord. notes = (freq Hz, level, ratio, index) per voice; voice i starts
    at i*spread s (voice 0 sits exactly on the onset). Mono."""
    out = np.zeros(int((length + 0.5) * SR))
    for i, (f, lv, ratio, idx) in enumerate(notes):
        b = fm_bell(f, lv, length, ratio=ratio, index=idx, i_tau=0.5, a_tau=a_tau)
        s = int(round(spread * i * SR))
        out[s:s + len(b)] += b
    return out


@lru_cache(maxsize=300)
def _felt_piano(freq: float, dur: float, vel: float) -> np.ndarray:
    rng = np.random.default_rng(int(freq * 10) * 1009 + int(dur * 100) * 31 + int(vel * 100))
    rel = 0.35
    t = tvec(dur + rel)
    out = np.zeros(len(t))
    tau0 = 2.6 * (220.0 / freq) ** 0.35
    bright = 0.55 + 0.6 * vel
    for k in range(1, 11):
        fk = k * freq * np.sqrt(1 + 0.00035 * k * k)
        if fk > 9000:
            break
        amp = (1.0 / k ** 1.35) * np.exp(-(k - 1) * (0.42 - 0.22 * bright))
        tau = tau0 / (1 + 0.45 * (k - 1))
        env = 0.55 * np.exp(-t / (tau * 0.22)) + 0.45 * np.exp(-t / tau)
        for det in (-0.7, 0.7):  # two slightly detuned strings
            out += 0.5 * amp * env * np.sin(2 * np.pi * fk * 2 ** (det / 1200) * t + rng.uniform(0, 2 * np.pi))
    out *= np.minimum(1, t / 0.004) * np.where(t > dur, np.clip(1 - (t - dur) / rel, 0, 1) ** 2, 1)
    nth = int(0.012 * SR)  # hammer thump
    thump = rng.normal(0, 1, nth) * np.exp(-np.arange(nth) / (0.003 * SR))
    out[:nth] += 0.05 * bp(thump, 150.0, 900.0)
    out *= vel
    out.setflags(write=False)
    return out


def felt_piano(freq: float, dur: float, vel: float) -> np.ndarray:
    """Soft felt-piano tone: inharmonic partials, two-stage decay, hammer thump (cached, read-only)."""
    return _felt_piano(round(freq, 3), round(dur, 2), round(vel, 2))


@cached()
def ep(freq: float, dur: float, vel: float) -> np.ndarray:
    """FM Rhodes-style electric piano: 1:1 FM body with velocity 'bark' + 14:1 tine strike."""
    t = tvec(dur + 0.45)
    w = 2 * np.pi * freq * t
    i_body = (0.5 + 1.0 * vel) * np.exp(-t / 0.3) + 0.45 * vel
    body = np.sin(w + i_body * np.sin(w))  # 1:1 FM, zero phase -> no DC sideband
    tine = np.sin(w + 0.9 * vel * np.exp(-t / 0.018) * np.sin(14 * w)) * np.exp(-t / 0.25)
    amp = 0.35 * np.exp(-t / 0.22) + 0.65 * np.exp(-t / 2.4)
    return vel * (body + 0.25 * tine) * amp * env_ar(t, 0.002, dur, 0.3)


@cached()
def fm_lead(freq: float, dur: float, vel: float) -> np.ndarray:
    """Glassy FM pluck (1:1) for hooks; bright strike that settles to a near-sine."""
    t = tvec(dur + 0.4)
    idx = (1.3 + 1.2 * vel) * np.exp(-t / 0.09) + 0.35
    y = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * t))
    y += 0.22 * np.sin(4 * np.pi * freq * t) * np.exp(-t / 0.25)
    return vel * y * env_ar(t, 0.0012, dur, 0.18) * (0.35 + 0.65 * np.exp(-t / 0.35))


@lru_cache(maxsize=300)
def _pluck(freq: float, dur: float, bright: float, tau: float) -> np.ndarray:
    rng = np.random.default_rng(int(freq * 100) + int(bright * 10))
    t = tvec(dur + 0.25)
    out = np.zeros((2, len(t)))
    for ch, det in enumerate((-7.0, 7.0)):
        f = freq * 2 ** (det / 1200)
        for k in range(1, int(min(40, 10000 / f)) + 1):
            fk = k * f
            amp = (1.0 / k) / np.sqrt(1.0 + (fk / 5000.0) ** 4)
            rate = 1.0 / tau + 2.6 / bright * fk / 1000.0
            out[ch] += amp * np.exp(-rate * t) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    out = out * env_ar(t, 0.0015, dur, 0.08) * 0.5
    out.setflags(write=False)
    return out


def pluck(freq: float, dur: float, bright: float = 1.0, tau: float = 0.3) -> np.ndarray:
    """Synth pluck: two detuned saws (L/R) whose upper harmonics die away first. Stereo, cached."""
    return _pluck(round(freq, 3), round(dur, 3), round(bright, 2), round(tau, 3))


# =============== pads and textures
def warm_pad(rng: np.random.Generator, freqs: Sequence[float], dur: float, level: float) -> np.ndarray:
    """Warm detuned string-like additive pad with slow attack/release (mono)."""
    att, rel = 0.9, 1.4
    t = tvec(dur + rel)
    out = np.zeros(len(t))
    for f in freqs:
        for det in (-7, 0, 7):
            ff = f * 2 ** (det / 1200)
            vib = 1 + 0.0015 * np.sin(2 * np.pi * (4.7 + det * 0.05) * t)
            for h, a in ((1, 1.0), (2, 0.42), (3, 0.2), (4, 0.09), (5, 0.05)):
                if ff * h < 7000:
                    out += a * np.sin(2 * np.pi * ff * h * vib * t + rng.uniform(0, 6.28))
    out /= max(1, len(freqs) * 3)
    env = np.minimum(1, t / att) ** 2 * np.where(t > dur, np.clip(1 - (t - dur) / rel, 0, 1) ** 1.5, 1)
    out = lp(out * env * (1 + 0.12 * np.sin(2 * np.pi * 0.21 * t)), 2400.0)
    return out * level


@cached()
def sub_sine(freq: float, dur: float, level: float) -> np.ndarray:
    """Soft sine sub with a touch of 2nd harmonic. Keep freq >= ~80 Hz so 40-80 Hz stays calm."""
    rel = 0.6
    t = tvec(dur + rel)
    env = np.minimum(1, t / 0.06) * np.where(t > dur, np.clip(1 - (t - dur) / rel, 0, 1), 1)
    return level * env * (np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t))


def saw_pad(rng: np.random.Generator, freqs: Sequence[float], dur: float, attack: float = 0.2,
            release: float = 0.6, cutoff: float = 2600.0) -> np.ndarray:
    """Detuned polyBLEP saw pad, spread over L/R. Stereo."""
    t = tvec(dur + release)
    out = np.zeros((2, len(t)))
    for i, f in enumerate(freqs):
        for j, det in enumerate((-11.0, -4.0, 4.0, 11.0)):
            out[(i + j) % 2] += saw(f * 2 ** (det / 1200), len(t), rng.uniform())
    out = lp(hp(out, 160.0), cutoff)
    return out * env_ar(t, attack, dur, release) / (2 * len(freqs))


def strings(rng: np.random.Generator, freqs: Sequence[float], dur: float, attack: float = 0.7,
            release: float = 0.9, cutoff: float = 2100.0, voices: int = 5, spread: float = 9.0) -> np.ndarray:
    """Warm string ensemble: detuned polyBLEP saws with delayed vibrato, spread in stereo."""
    t = tvec(dur + release)
    out = np.zeros((2, len(t)))
    vib_on = np.clip((t - 0.5) / 1.2, 0.0, 1.0)
    for f in freqs:
        for v in range(voices):
            x = 2 * v / (voices - 1) - 1
            vib = 1 + rng.uniform(0.0011, 0.002) * vib_on * np.sin(2 * np.pi * rng.uniform(4.6, 5.6) * t
                                                                  + rng.uniform(0, 2 * np.pi))
            y = saw(f * 2 ** ((spread * x + rng.uniform(-1.5, 1.5)) / 1200) * vib, len(t), rng.uniform())
            out += pan_st(y, 0.65 * x)
    out = lp(lp(hp(out, 90.0), cutoff), 5000.0)
    swell = 0.78 + 0.22 * np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.5  # each chord breathes
    return out * env_ar(t, attack, dur, release) * swell / (len(freqs) * voices)


def air(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Very quiet breathy air: band-passed noise drifting 3.5-7 kHz, slow random amplitude. Stereo."""
    n = len(tvec(seconds))
    t = np.arange(n) / SR
    out = []
    for ch in range(2):
        fc = 5000 * 2 ** (0.5 * np.sin(2 * np.pi * 0.045 * t + ch * 1.7))
        amp = 0.6 + 0.4 * np.sin(2 * np.pi * 0.09 * t + rng.uniform(0, 6.28)) ** 2
        out.append(shaped_noise(rng, n, fc, bw_oct=1.6) * amp)
    return np.stack(out)


# =============== basses
@cached()
def synth_bass(freq: float, dur: float, vel: float) -> np.ndarray:
    """Round pop bass (fundamental >= ~98 Hz) with phone-audible harmonics."""
    t = tvec(dur + 0.08)
    w = 2 * np.pi * freq * t
    y = (np.sin(w) + 0.5 * np.exp(-t / 0.10) * np.sin(2 * w) + 0.28 * np.exp(-t / 0.07) * np.sin(3 * w)
         + 0.12 * np.exp(-t / 0.05) * np.sin(4 * w))
    y = np.tanh(1.4 * y) / np.tanh(1.4)
    return vel * y * env_ar(t, 0.003, dur, 0.05) * (0.75 + 0.25 * np.exp(-t / 0.15))


@cached()
def round_bass(freq: float, dur: float, vel: float) -> np.ndarray:
    """Soft sine-ish bass for lo-fi (fundamental + faint 2nd/3rd)."""
    t = tvec(dur + 0.12)
    w = 2 * np.pi * freq * t
    y = np.sin(w) + 0.22 * np.sin(2 * w) * np.exp(-t / 0.3) + 0.08 * np.sin(3 * w) * np.exp(-t / 0.15)
    y = np.tanh(1.3 * y) / np.tanh(1.3)
    return vel * y * env_ar(t, 0.008, dur, 0.09) * (0.7 + 0.3 * np.exp(-t / 0.4))


# =============== plucked strings (Karplus-Strong) and mallets
def ks_string(f0: float, n: int, exc: np.ndarray, t60: float, loss_a: float) -> np.ndarray:
    """Karplus-Strong loop solved as one IIR (lfilter): y = x + g * L(z) * A(z) * z^-M * y, with the loss filter
    L = [a, 1-2a, a] (upper partials die faster, like string damping) and a first-order allpass A for the fractional
    part of the period. g sets the fundamental's T60."""
    period = SR / f0
    m = int(np.floor(period - 1.1))
    d = period - 1.0 - m
    c = (1.0 - d) / (1.0 + d)
    g = 10.0 ** (-3.0 / (f0 * t60))
    den = np.zeros(m + 4)
    den[0], den[1] = 1.0, c
    den[m:m + 4] -= g * np.convolve([loss_a, 1.0 - 2.0 * loss_a, loss_a], [c, 1.0])
    x = np.zeros(n)
    k = min(n, len(exc))
    x[:k] = exc[:k]
    return lfilter([1.0, c], den, x)


def pluck_excitation(f0: float, rng: np.random.Generator, pick: float, hardness: float,
                     noise_mix: float = 0.3) -> np.ndarray:
    """One period of bridge force for a string plucked at relative position `pick` (two-level pulse ->
    sin(n*pi*pick)/n partials) plus a little noise, low-passed by pick hardness. Zero mean: no DC enters the loop."""
    m = max(8, int(round(SR / f0)))
    k = int(np.clip(round(pick * m), 1, m - 1))
    x = np.concatenate([np.full(k, 1.0 / k), np.full(m - k, -1.0 / (m - k))])
    x = x / np.abs(x).max()
    nz = rng.normal(0.0, 1.0, m)
    nz -= nz.mean()
    x = x + noise_mix * nz * (np.std(x) / (np.std(nz) + 1e-12))
    return lp(np.concatenate([x, np.zeros(m)]), 900.0 + 7500.0 * hardness, 2)


@lru_cache(maxsize=400)
def _nylon(freq: float, dur: float, vel: float, bright: float) -> np.ndarray:
    rng = np.random.default_rng(int(freq * 10))
    n = int((dur + 0.4) * SR)
    exc = pluck_excitation(freq, rng, 0.16, 0.2 + 0.6 * bright, 0.25)
    y = ks_string(freq, n, exc, float(np.clip(1.7 * (220.0 / freq) ** 0.4, 0.5, 3.2)), 0.15 - 0.07 * bright)
    y = y / (np.abs(y[: int(0.06 * SR)]).max() + 1e-9)
    y = eq(y, peaking(105.0, 4.0, 2.5), peaking(210.0, 2.5, 2.5), peaking(2300.0, -2.0, 1.0))   # wooden body
    t = np.arange(n) / SR
    y = y * np.minimum(1.0, t / 0.002) * np.where(t > dur, np.clip(1 - (t - dur) / 0.4, 0, 1) ** 2, 1.0)
    y.setflags(write=False)
    return y * vel


def nylon_guitar(freq: float, dur: float, vel: float, bright: float = 0.5) -> np.ndarray:
    """Nylon-string guitar note (mono): KS string, wooden body resonances, damped when the note ends."""
    return _nylon(round(freq, 2), round(dur, 2), round(vel, 2), round(bright, 1))


@lru_cache(maxsize=300)
def _marimba(freq: float, dur: float, vel: float) -> np.ndarray:
    rng = np.random.default_rng(int(freq * 7))
    t = tvec(dur + 0.25)
    tau = 0.8 * (262.0 / freq) ** 0.45
    y = (np.sin(2 * np.pi * freq * t) * np.exp(-t / tau) + 0.4 * np.sin(2 * np.pi * 4.0 * freq * t) * np.exp(-t / 0.05)
         + 0.1 * np.sin(2 * np.pi * 9.2 * freq * t) * np.exp(-t / 0.015))
    nc = int(0.006 * SR)
    y[:nc] += 0.25 * lp(rng.normal(0, 1, nc), 3500.0) * np.exp(-np.arange(nc) / (0.0012 * SR))      # mallet
    y *= np.minimum(1.0, t / 0.001) * np.where(t > dur, np.clip(1 - (t - dur) / 0.25, 0, 1), 1.0)
    y.setflags(write=False)
    return y * vel


def marimba(freq: float, dur: float, vel: float) -> np.ndarray:
    """Marimba bar (mono): tuned 1 : 4 : 9.2 partials, short wooden decay, soft mallet click."""
    return _marimba(round(freq, 2), round(dur, 2), round(vel, 2))


# =============== drums
def kick(rng: np.random.Generator, f0: float = 150.0, f1: float = 62.0, p_tau: float = 0.03,
         a_tau: float = 0.15, drive: float = 1.6, click: float = 0.12) -> np.ndarray:
    """Sine kick with pitch sweep; tanh drive adds upper harmonics so phones hear it."""
    t = tvec(a_tau * 6 + 0.02)
    f = f1 + (f0 - f1) * np.exp(-t / p_tau)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / a_tau)
    body = np.tanh(drive * body) / np.tanh(drive)
    nc = int(0.006 * SR)
    c = bp(rng.normal(0.0, 1.0, nc), 1200.0, 6000.0) * np.exp(-np.arange(nc) / (0.0012 * SR))
    body[:nc] += click * c / (np.abs(c).max() + 1e-9)
    return body * np.minimum(1.0, t / 0.0006) * fade_end(len(t), int(0.01 * SR))


def clap(rng: np.random.Generator, tail: float = 0.075, lo: float = 900.0, hi: float = 4800.0) -> np.ndarray:
    """Four quick noise bursts + a short tail, band-passed."""
    t = tvec(0.45)
    env = np.zeros_like(t)
    offsets = (0.0, 0.0085, 0.0175, 0.0255)
    for i, o in enumerate(offsets):
        o += rng.uniform(-0.0008, 0.0008) if i else 0.0
        last = i == len(offsets) - 1
        on = t >= o
        dt = t[on] - o
        env[on] += (1.0 if last else 0.75) * np.exp(-dt / (tail if last else 0.0032)) * np.minimum(1, dt / 0.0004)
    y = bp(rng.normal(0.0, 1.0, len(t)), lo, hi) * env
    y += 0.4 * bp(rng.normal(0.0, 1.0, len(t)), 1050.0, 1500.0) * env
    y *= fade_end(len(t), int(0.02 * SR))
    return y / (np.abs(y).max() + 1e-9)


def snare(rng: np.random.Generator, tone: float = 190.0, n_tau: float = 0.11, t_tau: float = 0.045,
          lo: float = 1200.0, hi: float = 8000.0, tone_mix: float = 0.6) -> np.ndarray:
    """Snare: pitched body (with a 1.47x overtone) plus band-passed noise; peak-normalised."""
    t = tvec(max(n_tau, t_tau) * 6 + 0.02)
    f = tone * (1.0 + 0.25 * np.exp(-t / 0.01))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = (np.sin(ph) + 0.45 * np.sin(1.47 * ph)) * np.exp(-t / t_tau)
    nz = bp(rng.normal(0.0, 1.0, len(t)), lo, hi)
    nz = nz / (np.sqrt(np.mean(nz[: int(0.02 * SR)] ** 2)) + 1e-9) * 0.35 * np.exp(-t / n_tau)
    y = (tone_mix * body + nz) * np.minimum(1.0, t / 0.0005) * fade_end(len(t), int(0.01 * SR))
    return y / (np.abs(y).max() + 1e-9)


def hat(rng: np.random.Generator, tau: float = 0.03, lo: float = 7000.0, hi: float = 13000.0) -> np.ndarray:
    """Closed hi-hat: short band-passed noise burst; peak-normalised."""
    t = tvec(tau * 7 + 0.005)
    y = bp(rng.normal(0.0, 1.0, len(t)), lo, hi) * np.exp(-t / tau)
    y *= np.minimum(1.0, t / 0.0004) * fade_end(len(t), int(0.004 * SR))
    return y / (np.abs(y).max() + 1e-9)


def shaker(rng: np.random.Generator, tau: float = 0.04, attack: float = 0.006,
           lo: float = 3800.0, hi: float = 9500.0) -> np.ndarray:
    """Shaker: noise burst with a slightly soft attack; peak-normalised."""
    t = tvec(attack + tau * 6)
    env = np.where(t < attack, (t / attack) ** 2, np.exp(-(t - attack) / tau))
    y = bp(rng.normal(0.0, 1.0, len(t)), lo, hi) * env * fade_end(len(t), int(0.004 * SR))
    return y / (np.abs(y).max() + 1e-9)


def snap(rng: np.random.Generator) -> np.ndarray:
    """Finger snap: a band-passed noise crack with a short pitched pop, peak-normalised."""
    t = tvec(0.14)
    y = bp(rng.normal(0.0, 1.0, len(t)), 1300.0, 4200.0) * np.exp(-t / 0.018) + 0.5 * np.sin(2 * np.pi * 1700 * t) * np.exp(-t / 0.006)
    y *= np.minimum(1.0, t / 0.0004) * fade_end(len(t), int(0.01 * SR))
    return y / (np.abs(y).max() + 1e-9)


def brush_hit(rng: np.random.Generator, tau: float = 0.1) -> np.ndarray:
    """Brush swish on a snare head: soft-attack band noise (2-9 kHz), peak-normalised."""
    t = tvec(tau * 5 + 0.02)
    y = bp(rng.normal(0.0, 1.0, len(t)), 2000.0, 9000.0) * np.minimum(1.0, t / 0.012) * np.exp(-t / tau)
    y *= fade_end(len(t), int(0.01 * SR))
    return y / (np.abs(y).max() + 1e-9)


# =============== fx
def bubble_pop(rng: np.random.Generator, freq: float) -> np.ndarray:
    """Tuned bubble 'pop': quick upward glide onto the note, short decay, tiny click."""
    t = tvec(0.18)
    ph = 2 * np.pi * np.cumsum(freq * (1.0 - 0.42 * np.exp(-t / 0.011))) / SR
    y = np.sin(ph) + 0.16 * np.sin(2 * ph) * np.exp(-t / 0.03)
    y *= (1.0 - np.exp(-t / 0.0012)) * np.exp(-t / 0.05)
    nc = int(0.003 * SR)
    y[:nc] += 0.12 * bp(rng.normal(0, 1, nc), 2500.0, 7000.0) * np.exp(-np.arange(nc) / (0.0008 * SR))
    return y * fade_end(len(t), int(0.01 * SR))


def swish(rng: np.random.Generator, pre: float = 0.2, post: float = 0.14, f_lo: float = 700.0,
          f_hi: float = 7500.0, pan: tuple[float, float] = (-0.75, 0.75), bw_oct: float = 1.2,
          curve: float = 2.6) -> np.ndarray:
    """Noise sweep that peaks exactly `pre` s after its start (place it at cut - pre). Stereo.
    Short defaults = swish; pre~0.5, f_lo~220, bw_oct~1.8 = whoosh."""
    t = tvec(pre + post)
    x = np.clip(t / pre, 0.0, 1.0)
    fc = np.where(t <= pre, f_lo * (f_hi / f_lo) ** (x ** 1.4), f_hi * (1 + 0.3 * (t - pre) / post))
    y = lp(shaped_noise(rng, len(t), fc, bw_oct=bw_oct), 11000.0, 4)
    env = np.where(t < pre, x ** curve, np.exp(-(t - pre) / (0.05 * (post / 0.14) ** 0.6))) * fade_end(len(t), int(0.01 * SR))
    p = pan[0] + (pan[1] - pan[0]) * (0.5 - 0.5 * np.cos(np.pi * np.clip(t / (pre + post), 0, 1)))
    return pan_st(y * env, p)


def riser(rng: np.random.Generator, length: float, f_lo: float = 350.0, f_hi: float = 8000.0,
          curve: float = 2.2, bw_oct: float = 1.1) -> np.ndarray:
    """Stereo filtered-noise riser that ends (fades to zero) exactly at `length`."""
    t = tvec(length)
    x = t / length
    fc = f_lo * (f_hi / f_lo) ** (x ** 1.2)
    env = x ** curve * fade_end(len(t), int(0.02 * SR))
    return np.stack([shaped_noise(rng, len(t), fc, bw_oct) * env, shaped_noise(rng, len(t), fc, bw_oct) * env])


def impact(rng: np.random.Generator, size: float = 1.0, thump_level: float = 0.8) -> np.ndarray:
    """Soft, mid-focused impact: tuned thump (ends ~85 Hz), air burst, optional crash."""
    t = tvec(2.0)
    body = np.sin(2 * np.pi * np.cumsum(85.0 + 95.0 * np.exp(-t / 0.05)) / SR) * np.exp(-t / (0.16 + 0.12 * size))
    thump = np.tanh(1.8 * body) / np.tanh(1.8)
    air_ = lp(rng.normal(0, 1, len(t)), 1800.0)
    air_ *= np.exp(-t / 0.07) / (np.sqrt(np.mean(air_ ** 2)) * 4)
    crash = lp(hp(rng.normal(0, 1, len(t)), 4500.0), 10000.0, 4)
    crash *= np.exp(-t / (0.2 + 0.7 * size)) / (np.sqrt(np.mean(crash ** 2)) * 6) * size
    return (thump_level * thump + air_ + crash) * np.minimum(1.0, t / 0.0008) * fade_end(len(t), int(0.05 * SR))


def rev_cymbal(rng: np.random.Generator, length: float) -> np.ndarray:
    """Reverse-cymbal swell that peaks at `length`. Stereo, peak-normalised."""
    t = tvec(length)
    env = np.exp((t - length) / (0.3 * length)) * np.minimum(1.0, t / 0.05) * fade_end(len(t), int(0.006 * SR))
    st = lp(hp(rng.normal(0, 1, (2, len(t))), 3500.0), 12000.0) * env
    return st / np.abs(st).max()


def tick(ratio: float = 1.0) -> np.ndarray:
    """Soft UI tick tuned to D (1175 / 2350 / 4700 Hz); `ratio` transposes it."""
    t = tvec(0.06)
    y = (0.45 * np.sin(2 * np.pi * 1175 * ratio * t) * np.exp(-t / 0.012)
         + 0.6 * np.sin(2 * np.pi * 2350 * ratio * t) * np.exp(-t / 0.006)
         + 0.3 * np.sin(2 * np.pi * min(4700 * ratio, 9000) * t) * np.exp(-t / 0.003))
    return y * np.minimum(1.0, t / 0.0004) * fade_end(len(t), int(0.01 * SR))


def glide_tone(f0: float, f1: float, dur: float, shape: float = 0.012) -> np.ndarray:
    """Exponential pitch glide f0 -> f1 with a fast decay (chat send / receive blips). Mono, unit-ish."""
    t = tvec(dur)
    ph = 2 * np.pi * np.cumsum(f0 * (f1 / f0) ** (t / dur)) / SR
    return np.minimum(1, t / 0.003) * np.exp(-t / shape) * (np.sin(ph) + 0.25 * np.sin(2 * ph))


def sticker_wobble() -> np.ndarray:
    """Wobbly rising 'boing' used for chat stickers. Mono."""
    t = tvec(0.35)
    f = 420 + 260 * np.sin(2 * np.pi * 9 * t) * np.exp(-t / 0.12) + 300 * t
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * fade_end(len(t), int(0.02 * SR))


def key_tap(rng: np.random.Generator, pitch: float = 1.0) -> np.ndarray:
    """Soft keyboard tap: a dull 'thock' plus a short band-limited click. Mono."""
    t = tvec(0.09)
    f = 190.0 * pitch * (1.0 + 0.35 * np.exp(-t / 0.012))
    thock = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.018)
    nc = int(0.012 * SR)
    click = bp(rng.normal(0, 1, nc), 1800.0 * pitch, 5200.0) * np.exp(-np.arange(nc) / (0.0025 * SR))
    y = 0.55 * thock
    y[:nc] += 0.5 * click / (np.abs(click).max() + 1e-9)
    return y * np.minimum(1.0, t / 0.0005) * fade_end(len(t), int(0.01 * SR))


def vinyl(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Very light crackle (sparse band-passed clicks + rare dull pops) over a faint hiss. Stereo."""
    n = int(round(seconds * SR))
    out = np.zeros((2, n))
    for ch in range(2):
        imp = np.zeros(n)
        k = rng.poisson(8 * seconds)
        np.add.at(imp, rng.integers(0, n, k), rng.lognormal(-1.2, 0.6, k) * rng.choice([-1.0, 1.0], k))
        pops = np.zeros(n)
        k2 = rng.poisson(0.6 * seconds)
        np.add.at(pops, rng.integers(0, n, k2), rng.uniform(0.6, 1.2, k2) * rng.choice([-1.0, 1.0], k2))
        out[ch] = lp(bp(imp, 2500.0, 8000.0), 6000.0, 4) + 0.35 * bp(pops, 900.0, 3000.0)
    return out + 0.012 * lp(bp(rng.normal(0.0, 1.0, (2, n)), 4500.0, 11000.0), 9000.0, 4)


def wow(st: np.ndarray, depth_ms: float = 1.2, rate: float = 0.5, flutter_ms: float = 0.04,
        f_rate: float = 6.5, t0: float = 0.0) -> np.ndarray:
    """Tape wow & flutter via a slowly modulated delay (a few cents of drift); `t0` keeps the LFO phase continuous
    when a long track is rendered in chunks."""
    n = st.shape[1]
    t = np.arange(n) / SR + t0
    d = (depth_ms * 1e-3 * SR) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)) \
        + (flutter_ms * 1e-3 * SR) * np.sin(2 * np.pi * f_rate * t)
    idx = np.arange(n) - d - 2.0
    return np.stack([np.interp(idx, np.arange(n), st[c], left=0.0) for c in range(2)])


def autopan(st: np.ndarray, rate: float = 2.6, depth: float = 0.2, t0: float = 0.0) -> np.ndarray:
    """Slow opposite-phase L/R amplitude modulation of a stereo signal."""
    t = np.arange(st.shape[1]) / SR + t0
    lfo = np.sin(2 * np.pi * rate * t)
    return np.stack([st[0] * (1 - depth * (0.5 + 0.5 * lfo)), st[1] * (1 - depth * (0.5 - 0.5 * lfo))])
