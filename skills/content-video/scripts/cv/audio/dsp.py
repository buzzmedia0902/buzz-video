"""DSP kit of the audio chain (48 kHz, numpy + scipy only): filters, envelopes, reverb, loudness, true peak.

Loudness is BS.1770-4 (K-weighting, 400 ms blocks, 75 % overlap, absolute -70 LUFS and relative -10 LU gates)
and is computed from 100 ms sub-block energies, so it runs in constant memory on a 20-minute mix and can be
restricted to a selection of blocks (e.g. "while the speaker talks"). True peak is measured on a 4x oversampled
copy exactly like ffmpeg's ebur128; `limit_true_peak` limits on that oversampled envelope, because a plain
sample-peak limiter leaves inter-sample overs (+0.2 dBTP was measured).
"""
from __future__ import annotations

import functools
import re
import subprocess

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, istft, lfilter, resample_poly, sosfilt, stft
from scipy.io import wavfile

SR = 48000
HOP = SR // 10                      # loudness sub-block: 100 ms


# =================================================================== basics
def n_of(seconds: float) -> int:
    return int(round(seconds * SR))


def tvec(seconds: float, sr: int = SR) -> np.ndarray:
    """Time axis (s) for `seconds` of audio (at `sr`, for oversampled synthesis)."""
    return np.arange(int(round(seconds * sr))) / sr


def from_db(x):
    """dB -> linear gain (float or array)."""
    return 10.0 ** (x / 20.0)


def to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def smoothstep(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def fade_end(n: int, m: int) -> np.ndarray:
    """Ones with a raised-cosine fade to exactly zero over the last m samples (de-clicks tails)."""
    g = np.ones(n)
    m = min(max(int(m), 0), n)
    if m:
        g[n - m:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(1, m + 1) / m)
    return g


def ramp_in(n: int, m: int) -> np.ndarray:
    """Ones with a raised-cosine rise from zero over the first m samples."""
    g = np.ones(n)
    m = min(max(int(m), 0), n)
    if m:
        g[:m] = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    return g


def env_ar(t: np.ndarray, attack: float, hold: float, release: float) -> np.ndarray:
    """Raised-cosine attack to 1, hold until `hold` s, raised-cosine release over `release` s."""
    a = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    r = np.clip((t - hold) / max(release, 1e-6), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * a)) * (0.5 + 0.5 * np.cos(np.pi * r))


def unit(x: np.ndarray) -> np.ndarray:
    """Scale to peak 1 (no-op on silence)."""
    m = np.abs(x).max()
    return x / m if m > 0 else x


# =================================================================== filters
@functools.lru_cache(maxsize=256)
def _sos(kind: str, fc, order: int) -> np.ndarray:
    fc = tuple(min(f, 0.49 * SR) for f in fc) if isinstance(fc, tuple) else min(fc, 0.49 * SR)
    return butter(order, list(fc) if isinstance(fc, tuple) else fc, kind, fs=SR, output="sos")


def lp(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return sosfilt(_sos("low", float(fc), order), x, axis=-1)


def hp(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return sosfilt(_sos("high", float(fc), order), x, axis=-1)


def bp(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    return sosfilt(_sos("band", (float(lo), float(hi)), order), x, axis=-1)


def peaking(f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ peaking-EQ biquad as one SOS section."""
    a_ = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a_, -2 * np.cos(w), 1 - alpha * a_])
    a = np.array([1 + alpha / a_, -2 * np.cos(w), 1 - alpha / a_])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def high_shelf(f0: float, gain_db: float, slope: float = 0.8) -> np.ndarray:
    """RBJ high-shelf biquad as one SOS section."""
    a_ = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    cw = np.cos(w)
    alpha = np.sin(w) / 2 * np.sqrt((a_ + 1 / a_) * (1 / slope - 1) + 2)
    sq = 2 * np.sqrt(a_) * alpha
    b = np.array([a_ * ((a_ + 1) + (a_ - 1) * cw + sq), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw),
                  a_ * ((a_ + 1) + (a_ - 1) * cw - sq)])
    a = np.array([(a_ + 1) - (a_ - 1) * cw + sq, 2 * ((a_ - 1) - (a_ + 1) * cw), (a_ + 1) - (a_ - 1) * cw - sq])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def eq(x: np.ndarray, *sections: np.ndarray) -> np.ndarray:
    return sosfilt(np.concatenate(sections, axis=0), x, axis=-1)


# =================================================================== noise sources
def shaped_noise(rng: np.random.Generator, n: int, fc: float | np.ndarray, bw_oct: float = 1.0,
                 nfft: int = 1024, hop: int = 256) -> np.ndarray:
    """White noise band-passed (Gaussian in log-frequency, FWHM bw_oct) around a moving centre fc; unit RMS."""
    x = rng.normal(0.0, 1.0, n + nfft)
    f, tt, z = stft(x, fs=SR, nperseg=nfft, noverlap=nfft - hop)
    fcs = np.interp(tt, np.arange(n) / SR, np.broadcast_to(np.asarray(fc, dtype=float), (n,)))
    sig = bw_oct / 2.355
    g = np.exp(-0.5 * ((np.log2(np.maximum(f, 1.0))[:, None] - np.log2(fcs)[None, :]) / sig) ** 2)
    g /= np.sqrt((g ** 2).sum(axis=0, keepdims=True)) + 1e-12
    _, y = istft(z * g, fs=SR, nperseg=nfft, noverlap=nfft - hop)
    y = y[:n]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def smooth_noise(rng: np.random.Generator, n: int, fc: float) -> np.ndarray:
    """Unit-std low-passed Gaussian noise (slow random modulation source)."""
    pre = n_of(0.2)
    x = lp(rng.normal(0.0, 1.0, n + pre), fc, 2)[pre:]
    return x / (np.std(x) + 1e-12)


# =================================================================== placement / space
def pan_st(sig: np.ndarray, pan: float | np.ndarray = 0.0) -> np.ndarray:
    """Mono -> stereo with a constant-power pan law (pan -1..1, may be per-sample)."""
    ang = (np.clip(np.asarray(pan, dtype=float), -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.stack([sig * np.cos(ang), sig * np.sin(ang)])


class Bus:
    """Stereo accumulator: mono sources are panned, (2, n) sources are added as they are."""

    def __init__(self, seconds: float, tail: float = 3.0) -> None:
        self.st = np.zeros((2, n_of(seconds + tail)))

    def add(self, sig: np.ndarray, at: float, pan: float = 0.0, gain: float = 1.0) -> None:
        """Mix `sig` in at `at` s; a negative `at` trims the head, anything past the end is dropped."""
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 1:
            sig = pan_st(sig, pan)
        s = n_of(at)
        if s < 0:
            sig, s = sig[:, -s:], 0
        e = min(self.st.shape[1], s + sig.shape[1])
        if e > s:
            self.st[:, s:e] += gain * sig[:, : e - s]


def make_ir(rng: np.random.Generator, rt60: float, hf_rt60: float | None = None, predelay: float = 0.02,
            hi_cut: float = 10000.0, nfft: int = 1024, hop: int = 256) -> np.ndarray:
    """Stereo noise-reverb impulse response, decay time falling from rt60 (<400 Hz) to hf_rt60 (>8 kHz)."""
    hf_rt60 = 0.45 * rt60 if hf_rt60 is None else hf_rt60
    n = int((rt60 * 1.05 + 0.1) * SR)
    t = np.arange(n) / SR
    chans = []
    for _ in range(2):
        x = rng.normal(0.0, 1.0, n + nfft)
        f, tt, z = stft(x, fs=SR, nperseg=nfft, noverlap=nfft - hop)
        w = np.clip(np.log2(np.maximum(f, 1.0) / 400.0) / np.log2(8000.0 / 400.0), 0.0, 1.0)
        rt = (rt60 * (1 - w) + hf_rt60 * w)[:, None]
        _, y = istft(z * np.exp(-6.9078 * tt[None, :] / rt), fs=SR, nperseg=nfft, noverlap=nfft - hop)
        y = lp(y[:n], hi_cut) * np.minimum(1.0, t / 0.008) * fade_end(n, int(0.15 * n))
        y = np.concatenate([np.zeros(int(predelay * SR)), y])
        chans.append(y / np.sqrt(np.sum(y ** 2)))
    return np.stack(chans)


def reverb(st: np.ndarray, ir: np.ndarray, hp_fc: float = 140.0, lp_fc: float | None = None,
           keep_tail: bool = False) -> np.ndarray:
    """Wet-only convolution return, high-passed so the tail never adds low-end mud. `keep_tail` keeps the ring-out
    past the input (length n + len(ir) - 1) for chunked rendering: the caller overlap-adds it."""
    n = st.shape[1] + (ir.shape[1] - 1 if keep_tail else 0)
    wet = np.stack([fftconvolve(st[c], ir[c])[:n] for c in range(2)])
    wet = hp(wet, hp_fc)
    return lp(wet, lp_fc) if lp_fc else wet


def convolve_st(st: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Channel-wise convolution, truncated to the input length (instrument bodies: a short modal impulse response)."""
    ir = np.atleast_2d(ir)
    if ir.shape[0] == 1:
        ir = np.repeat(ir, 2, axis=0)
    return np.stack([fftconvolve(st[c], ir[c])[: st.shape[1]] for c in range(2)])


# =================================================================== dynamics
def soft_clip(st: np.ndarray, ceiling: float, knee: float = 0.7) -> np.ndarray:
    """Bus peak shaver: linear below knee*ceiling, tanh-rounded above (2x oversampled)."""
    up = resample_poly(st, 2, 1, axis=-1)
    a = knee * ceiling
    mag = np.abs(up)
    over = mag > a
    up[over] = np.sign(up[over]) * (a + (ceiling - a) * np.tanh((mag[over] - a) / (ceiling - a)))
    return resample_poly(up, 1, 2, axis=-1)[..., : st.shape[-1]]


def duck_env(n: int, times, depth: float = 0.3, recover: float = 0.2, lead: float = 0.003) -> np.ndarray:
    """Side-chain style gain dips (to 1 - depth) at each trigger time, smooth recovery."""
    g = np.ones(n)
    tt = np.arange(int((recover + lead) * SR)) / SR - lead
    x = np.clip(tt / recover, 0.0, 1.0)
    curve = np.where(tt < 0, 1.0 - depth * (1.0 + tt / lead), 1.0 - depth * (1.0 - x * x * (3 - 2 * x)))
    for tk in times:
        s = int(round((tk - lead) * SR))
        a, b = max(0, s), min(n, s + len(curve))
        if b > a:
            g[a:b] = np.minimum(g[a:b], curve[a - s: b - s])
    return g


# =================================================================== loudness (BS.1770-4)
_KW = (((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585)),
       ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621)))


def k_energy(x: np.ndarray) -> np.ndarray:
    """Mean square of the K-weighted signal per 100 ms hop, summed over channels (constant memory)."""
    y = np.atleast_2d(x)
    ch = y.shape[0]
    n_sub = y.shape[1] // HOP
    out = np.zeros(n_sub)
    zi = [np.zeros((ch, 2)), np.zeros((ch, 2))]
    step = HOP * 100
    for s in range(0, n_sub * HOP, step):
        e = min(s + step, n_sub * HOP)
        seg = y[:, s:e].astype(np.float64)
        for k, (b, a) in enumerate(_KW):
            seg, zi[k] = lfilter(b, a, seg, axis=1, zi=zi[k])
        out[s // HOP: e // HOP] = (seg ** 2).reshape(ch, -1, HOP).mean(axis=2).sum(axis=0)
    return out


def gated_lufs(e: np.ndarray, select: np.ndarray | None = None) -> float:
    """Integrated loudness from sub-block energies; `select` (bool per 400 ms block) restricts the blocks."""
    if len(e) < 4:
        return -70.0
    z = np.convolve(e, np.ones(4) / 4, mode="valid")
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    ok = lk > -70.0
    if select is not None:
        ok = ok & select[: len(z)]
    if not ok.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(z[ok].mean()) - 10.0
    sel = ok & (lk > rel)
    return float(-0.691 + 10 * np.log10(z[sel].mean())) if sel.any() else -70.0


def lufs(x: np.ndarray, select: np.ndarray | None = None) -> float:
    return gated_lufs(k_energy(x), select)


def block_mask(frame_mask: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Per 100 ms-hop 400 ms block: True where at least `frac` of the (100 ms) frames of `frame_mask` are set."""
    m = np.asarray(frame_mask, dtype=float)
    if len(m) < 4:
        return np.zeros(0, dtype=bool)
    return np.convolve(m, np.ones(4) / 4, mode="valid") >= frac


# =================================================================== true peak and limiter
SEG = SR * 20                       # the limiter works in 20 s segments (with margins) to stay light on a 20-minute mix


def _tp_segment(y: np.ndarray) -> np.ndarray:
    """Per-sample peak of the 4x oversampled signal (max over channels and the four phases) of one segment."""
    up = np.abs(resample_poly(y.astype(np.float64), 4, 1, axis=1)).max(axis=0)
    return np.maximum(up.reshape(-1, 4).max(axis=1), np.abs(y).max(axis=0)).astype(np.float32)


def true_peak_db(x: np.ndarray) -> float:
    """True peak (dBTP) like ffmpeg's ebur128: 4x oversampled, without keeping the envelope in memory."""
    y = np.atleast_2d(x)
    n, step, pad, peak = y.shape[1], SR * 10, 256, 0.0
    for s0 in range(0, n, step):
        a, z = max(0, s0 - pad), min(n, s0 + step + pad)
        k = min(step, n - s0)
        peak = max(peak, float(_tp_segment(y[:, a:z])[s0 - a: s0 - a + k].max()))
    return to_db(peak)


def _gain_curve(pk: np.ndarray, ceil: float, look: int, release_ms: float) -> np.ndarray:
    """Limiter gain per sample for a true-peak envelope: dip starts `look` samples early and is held `look` late, then
    releases exponentially; never above what each oversampled peak needs."""
    n = len(pk)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9)).astype(np.float32)
    g = minimum_filter1d(g, size=2 * look + 1, mode="nearest")
    blk = 48
    nb = int(np.ceil(n / blk))
    gb = np.pad(g, (0, nb * blk - n), constant_values=1.0).reshape(nb, blk).min(axis=1)
    a = np.exp(-blk / (release_ms * 1e-3 * SR))
    rel = np.empty(nb, np.float32)
    prev = 1.0
    for i, v in enumerate(gb.tolist()):
        prev = min(v, 1.0 - (1.0 - prev) * a)
        rel[i] = prev
    return uniform_filter1d(np.repeat(rel, blk)[:n], size=look + 1, mode="nearest")


def _limit_once(y: np.ndarray, ceil: float, look: int, release_ms: float) -> tuple[np.ndarray, float]:
    """One limiting pass in overlapping segments (1 s margins make the gain curve seamless)."""
    n = y.shape[1]
    out = np.empty_like(y)
    worst = 1.0
    for s in range(0, n, SEG):
        a, b = max(0, s - SR), min(n, s + SEG + SR)
        gs = _gain_curve(_tp_segment(y[:, a:b]), ceil, look, release_ms)
        e = min(n, s + SEG)
        g = gs[s - a: e - a]
        out[:, s:e] = y[:, s:e] * g[None, :]
        worst = min(worst, float(g.min()))
    return out, to_db(worst)


def limit_true_peak(x: np.ndarray, ceiling_db: float = -1.0, look_ms: float = 3.0, release_ms: float = 120.0,
                    tries: int = 4) -> tuple[np.ndarray, float]:
    """Look-ahead limiter driven by the oversampled peak envelope; re-measures the result and tightens the target until
    the true peak is at or below `ceiling_db`. Returns (limited float32 (c, n), max gain reduction in dB)."""
    y = np.atleast_2d(x)
    y = y if y.dtype == np.float32 else y.astype(np.float32)
    look = max(1, int(look_ms * 1e-3 * SR))
    target, worst = ceiling_db - 0.03, 0.0
    for k in range(tries + 1):
        tp = true_peak_db(y)
        if tp <= ceiling_db or k == tries:
            break
        if k:
            target -= max(0.02, tp - ceiling_db + 0.02)
        y, gr = _limit_once(y, from_db(target), look, release_ms)
        worst = min(worst, gr)
    return y, worst


# =================================================================== files
def write_wav(path: str, x: np.ndarray) -> None:
    """(n,) or (c, n) float -> 32-bit float WAV at 48 kHz."""
    a = x if x.ndim == 1 else x.T
    wavfile.write(path, SR, np.ascontiguousarray(a, dtype=np.float32))


def measure_file(path: str) -> dict[str, float]:
    """Integrated loudness (I), loudness range (LRA) and true peak (TP, dBTP) of a file's audio via ffmpeg
    ebur128 - what a platform or a QA tool will read, AAC decoder included."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    s = err[err.rfind("Summary:"):]

    def num(pat: str) -> float:
        m = re.search(pat + r"\s*(-?[\d.]+|-inf)", s)
        return float(m.group(1)) if m else float("nan")

    return {"I": num(r"I:"), "LRA": num(r"LRA:"), "TP": num(r"Peak:")}

