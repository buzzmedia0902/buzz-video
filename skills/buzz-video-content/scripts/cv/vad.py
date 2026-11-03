"""Speech activity of a source and the jet cut (pauses shortened).

Speech = 150-4000 Hz band energy above a threshold on 10 ms frames (closing 150 ms, opening 50 ms). The threshold
adapts to the recording: 12 dB over the quiet floor (15th percentile), kept within -55 .. -34 dB.
"""
from __future__ import annotations

import functools
import os

import numpy as np

HOP = 0.01


class Speech:
    def __init__(self, wav: str, cache: str, thr: float | None = None):
        self.db = self._band_db(wav, cache)
        floor = float(np.percentile(self.db, 15))
        self.thr = thr if thr is not None else float(np.clip(floor + 12.0, -55.0, -34.0))
        from scipy.ndimage import binary_closing, binary_opening
        sp = self.db > self.thr
        sp = binary_closing(sp, structure=np.ones(15))
        self.mask = binary_opening(sp, structure=np.ones(5))
        self.duration = len(self.db) * HOP

    @staticmethod
    def _band_db(wav: str, cache: str) -> np.ndarray:
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(wav):
            return np.load(cache)
        from scipy.io import wavfile
        from scipy.signal import butter, sosfilt
        sr, x = wavfile.read(wav)
        m = x.astype(np.float64)
        if m.ndim == 2:
            m = m.mean(1)
        if x.dtype == np.int16:
            m /= 32768
        b = sosfilt(butter(4, [150, 4000], btype="band", fs=sr, output="sos"), m)
        hop = int(sr * HOP)
        n = len(b) // hop
        db = 20 * np.log10(np.sqrt((b[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, db)
        return db

    def bursts(self, t0: float = 0.0, t1: float | None = None, min_len: float = 0.0) -> list[tuple[float, float]]:
        """Speech runs (start, end) in seconds inside [t0, t1]."""
        t1 = self.duration if t1 is None else t1
        a, b = int(t0 / HOP), int(t1 / HOP)
        seg = np.concatenate([[0], self.mask[a:b].astype(int), [0]])
        d = np.diff(seg)
        on, off = np.where(d == 1)[0], np.where(d == -1)[0]
        return [(round((a + s) * HOP, 3), round((a + e) * HOP, 3)) for s, e in zip(on, off) if (e - s) * HOP >= min_len]

    def onset_near(self, t: float, win: float = 0.35) -> float:
        """Closest speech onset to t within +-win (else t)."""
        cands = [s for s, _ in self.bursts(max(0.0, t - win), t + win) if abs(s - t) <= win]
        return min(cands, key=lambda s: abs(s - t)) if cands else t

    def offset_near(self, t: float, win: float = 0.35) -> float:
        cands = [e for _, e in self.bursts(max(0.0, t - win), t + win) if abs(e - t) <= win]
        return min(cands, key=lambda e: abs(e - t)) if cands else t

    def tighten(self, t0: float, t1: float, pause: float, tail: float, lead: float) -> list[tuple[float, float]]:
        """[t0, t1] minus long pauses: every silence of `pause` s or more keeps only `tail` after the last word
        and `lead` before the next one (the jet cut). pause <= 0 keeps everything."""
        if pause <= 0:
            return [(t0, t1)]
        a, b = int(t0 / HOP), int(t1 / HOP)
        seg = np.concatenate([[1], self.mask[a:b].astype(int), [1]])
        d = np.diff(seg)
        gs, ge = np.where(d == -1)[0], np.where(d == 1)[0]
        keep, cur = [], t0
        for s, e in zip(gs, ge):
            g0, g1 = (a + s) * HOP, (a + e) * HOP
            if g1 - g0 < pause or g0 <= t0 + 1e-6 or g1 >= t1 - 1e-6:
                continue
            keep.append((cur, g0 + tail))
            cur = g1 - lead
        keep.append((cur, t1))
        return [(x0, x1) for x0, x1 in keep if x1 - x0 >= 0.1]


@functools.lru_cache(maxsize=None)
def for_source(path: str, work_dir: str, sid: str) -> Speech:
    from cv import media
    wav = media.extract_audio(path, os.path.join(work_dir, "audio", f"{sid}_48k.wav"))
    return Speech(wav, os.path.join(work_dir, "audio", f"{sid}_band.npy"))
