"""Measurement and mastering for the music generator (BS.1770 loudness, true peak, glue, limiter, WAV I/O).

``master()`` is the last stage of ``render_bgm``: high-pass, gentle bus compression, de-click fade in,
gentle fade out, loudness to the target (default -14 LUFS integrated) and a look-ahead true-peak limiter,
all in numpy (no ffmpeg needed). ``ffmpeg_loudness()`` re-measures a written file exactly as the
delivery spec does (``ebur128=peak=true``) when ffmpeg is installed.
"""
from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter, resample_poly

from core.synth import SR, hp

# BS.1770-4 K-weighting (48 kHz): high-shelf then RLB high-pass
_KW = ((np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
        np.array([1.0, -1.69065929318241, 0.73248077421585])),
       (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621])))
BANDS = (("<60", 0.0, 60.0), ("60-200", 60.0, 200.0), ("200-2k", 200.0, 2000.0),
         ("2k-8k", 2000.0, 8000.0), (">8k", 8000.0, 24000.0))


def lufs(st: np.ndarray) -> float:
    """BS.1770-4 integrated loudness (400 ms blocks, 75 % overlap, absolute + relative gate)."""
    y = st
    for b, a in _KW:
        y = lfilter(b, a, y, axis=1)
    cs = np.concatenate([np.zeros((2, 1)), np.cumsum(y ** 2, axis=1)], axis=1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    starts = np.arange(0, y.shape[1] - blk + 1, hop)
    if len(starts) == 0:
        return -70.0
    z = ((cs[:, starts + blk] - cs[:, starts]) / blk).sum(axis=0)
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    z1 = z[lk > -70]
    if len(z1) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(z1.mean()) - 10
    keep = (lk > -70) & (lk > rel)
    return float(-0.691 + 10 * np.log10(z[keep].mean())) if keep.any() else -70.0


def true_peak_db(st: np.ndarray) -> float:
    """True peak (dBTP) from a 4x oversampled copy."""
    return float(20 * np.log10(np.abs(resample_poly(st, 4, 1, axis=1)).max() + 1e-12))


def band_levels(st: np.ndarray) -> dict[str, float]:
    """Energy per band in dB relative to the total (a boomy mix shows '<60' above about -25 dB)."""
    spec = (np.abs(np.fft.rfft(st, axis=1)) ** 2).sum(axis=0)
    f = np.fft.rfftfreq(st.shape[1], 1 / SR)
    tot = spec.sum() + 1e-30
    return {n: round(float(10 * np.log10(spec[(f >= lo) & (f < hi)].sum() / tot + 1e-30)), 1)
            for n, lo, hi in BANDS}


def _block_follow(v: np.ndarray, block: int, att: float, rel: float) -> np.ndarray:
    """Attack/release one-pole follower on block means of v (fast enough in pure Python)."""
    nb = int(np.ceil(len(v) / block))
    blk = np.pad(v, (0, nb * block - len(v)), mode="edge").reshape(nb, block).mean(axis=1)
    ca, cr = np.exp(-block / (att * SR)), np.exp(-block / (rel * SR))
    out = np.empty(nb)
    e = blk[0]
    for i, x in enumerate(blk.tolist()):
        c = ca if x > e else cr
        e = c * e + (1.0 - c) * x
        out[i] = e
    return np.interp(np.arange(len(v)), (np.arange(nb) + 0.5) * block, out)


def glue(st: np.ndarray, below_max_db: float = 6.0, ratio: float = 1.8, att: float = 0.015,
         rel: float = 0.2, knee: float = 6.0) -> tuple[np.ndarray, float]:
    """Gentle RMS bus compressor; the threshold sits `below_max_db` under the loudest moment."""
    lvl = 10 * np.log10(_block_follow(np.mean(st ** 2, axis=0), 32, att, rel) + 1e-12)
    over = lvl - (lvl.max() - below_max_db)
    soft = (over + knee / 2) ** 2 / (2 * knee)  # quadratic soft knee
    over = np.where(over > knee / 2, over, np.where(over < -knee / 2, 0.0, soft))
    gr = -over * (1 - 1 / ratio)
    return st * 10 ** (gr / 20), float(gr.min())


def limiter(st: np.ndarray, ceiling_db: float, look_ms: float = 3.0,
            release_ms: float = 90.0) -> tuple[np.ndarray, float]:
    """Look-ahead true-peak limiter (4x oversampled detection, smooth gain curve)."""
    pk = np.abs(resample_poly(st, 4, 1, axis=1)).max(axis=0).reshape(-1, 4).max(axis=1)
    g = np.minimum(1.0, 10 ** (ceiling_db / 20) / np.maximum(pk, 1e-9))
    look = int(look_ms * 1e-3 * SR)
    g = minimum_filter1d(g, size=2 * look + 1, mode="nearest")
    blk = 48
    nb = int(np.ceil(len(g) / blk))
    gb = np.pad(g, (0, nb * blk - len(g)), constant_values=1.0).reshape(nb, blk).min(axis=1)
    a = np.exp(-blk / (release_ms * 1e-3 * SR))
    y = np.empty(nb)
    prev = 1.0
    for i, v in enumerate(gb.tolist()):
        prev = min(v, 1.0 - (1.0 - prev) * a)
        y[i] = prev
    gs = uniform_filter1d(np.repeat(y, blk)[: len(g)], size=look + 1, mode="nearest")
    return st * gs, float(20 * np.log10(gs.min()))


def fade_curve(n: int, fade_in: float, fade_out: float) -> np.ndarray:
    """1.5 ms de-click + sin^2 fade in over `fade_in` s and sin^2 fade out over the last `fade_out` s
    (reaching exactly 0 at the final sample)."""
    t = np.arange(n) / SR
    g = np.minimum(1.0, t / 0.0015)
    if fade_in > 0:
        g = g * np.sin(0.5 * np.pi * np.clip(t / fade_in, 0.0, 1.0)) ** 2
    if fade_out > 0:
        te = (n - 1) / SR
        g = g * np.sin(0.5 * np.pi * np.clip((te - t) / fade_out, 0.0, 1.0)) ** 2
    return g


def master(mix: np.ndarray, dur: float, *, fade_in: float = 0.008, fade_out: float = 0.6,
           hp_fc: float = 40.0, hp_order: int = 4, glue_db: float | None = 6.0, glue_ratio: float = 1.7,
           target: float = -14.0, tp_ceiling: float = -1.3) -> tuple[np.ndarray, dict]:
    """High-pass, glue, fades, loudness to `target`, true-peak safety. Returns (2, round(dur*SR)) and metrics."""
    n = int(round(dur * SR))
    st = np.zeros((2, n))
    k = min(n, mix.shape[1])
    st[:, :k] = hp(mix[:, :k], hp_fc, hp_order)
    glue_gr = 0.0
    if glue_db is not None:
        st, glue_gr = glue(st, glue_db, glue_ratio)
    st = st * fade_curve(n, fade_in, fade_out)
    lim_gr = 0.0
    for _ in range(10):
        st = st * 10 ** ((target - lufs(st)) / 20)
        if true_peak_db(st) <= tp_ceiling:
            break
        st, g = limiter(st, tp_ceiling - 0.1)
        lim_gr = min(lim_gr, g)
    tp = true_peak_db(st)
    if tp > tp_ceiling:  # last resort: plain trim (loudness stays within a few hundredths of a dB)
        st = st * 10 ** ((tp_ceiling - tp) / 20)
    return st, {"lufs": round(lufs(st), 2), "true_peak": round(true_peak_db(st), 2),
                "glue_gr_db": round(glue_gr, 2), "limiter_gr_db": round(lim_gr, 2), "bands": band_levels(st)}


def write_wav24(path: str | Path, st: np.ndarray) -> None:
    """(2, n) float in [-1, 1] -> 24-bit PCM stereo WAV at 48 kHz."""
    x = np.clip(np.round(st.T * 8388607.0), -8388608, 8388607).astype("<i4")
    raw = np.ascontiguousarray(x).reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(raw)


def write_wav_f32(path: str | Path, st: np.ndarray) -> None:
    """(2, n) float -> 32-bit float stereo WAV at 48 kHz."""
    wavfile.write(str(path), SR, np.ascontiguousarray(st.T.astype(np.float32)))


def ffmpeg_loudness(path: str | Path) -> dict[str, float] | None:
    """Integrated loudness / true peak as ffmpeg's ebur128 reports them (None if ffmpeg is missing)."""
    try:
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                              "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    s = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", s)
    tp = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", s)
    return {"I": float(i.group(1)), "TP": float(tp.group(1))} if i and tp else None
