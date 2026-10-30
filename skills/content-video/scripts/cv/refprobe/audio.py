"""Sound of a reference video: speech share, remaining pauses, loudness, music under the voice, SFX, speech rate."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import numpy as np

WHISPER_MODEL = os.path.expanduser("~/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin")


def extract_wav(path: str, secs: float, out: str, channels: int = 1, sr: int = 48000) -> str:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", f"{secs:.2f}", "-i", path, "-map", "0:a:0", "-vn", "-ac",
                    str(channels), "-ar", str(sr), "-c:a", "pcm_f32le", out], check=True)
    return out


def loudness(wav: str) -> dict:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    s = r[r.rfind("Summary:"):]
    pick = lambda key, unit: float(s.split(key)[1].split(unit)[0])
    return {"lufs": pick("I:", "LUFS"), "lra": pick("LRA:", "LU"), "true_peak_db": pick("Peak:", "dBFS")}


def frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


def audio_stats(wav: str, work: str, stereo: str | None = None) -> tuple[dict, np.ndarray]:
    """Speech share, remaining pauses (jet cut), music under the voice, sound effects in the pauses."""
    from scipy.io import wavfile
    from cv.vad import Speech
    cache = os.path.join(work, "band.npy")
    sp = Speech(wav, cache)                       # the engine's own speech mask: the pauses mean what the jet cut means
    if sp.mask.mean() > 0.97:                     # music under the voice keeps every frame over its threshold:
        floor, top = np.percentile(sp.db, 10), np.percentile(sp.db, 90)   # put the threshold between music and voice
        sp = Speech(wav, cache, thr=float(floor + np.clip(0.45 * (top - floor), 5.0, 12.0)))
    sr, x = wavfile.read(wav)
    x = x.astype(np.float64)
    dur = len(x) / sr
    bursts = sp.bursts()
    gaps = np.array([b[0] - a[1] for a, b in zip(bursts, bursts[1:])])
    gaps = gaps[gaps >= 0.15]
    out: dict = {"speech_share": round(float(sp.mask.mean()), 2), "bursts_per_min": round(len(bursts) / dur * 60, 1)}
    if len(gaps) >= 6:
        hist, edges = np.histogram(gaps, bins=np.arange(0.15, 0.81, 0.02))
        k = int(np.argmax(hist))
        peak = float((edges[k] + edges[k + 1]) / 2)
        share = float(((gaps > peak - 0.05) & (gaps < peak + 0.05)).mean())
        out["pauses"] = {"count": len(gaps), "per_min": round(len(gaps) / dur * 60, 1),
                         "median_s": round(float(np.median(gaps)), 2), "p90_s": round(float(np.percentile(gaps, 90)), 2),
                         "p97_s": round(float(np.percentile(gaps, 97)), 2), "max_s": round(float(gaps.max()), 2),
                         "over_0.5s_per_min": round(float((gaps >= 0.5).sum()) / dur * 60, 1),
                         "peak_s": round(peak, 2), "peak_share": round(share, 2),
                         "jet_cut": bool(share >= 0.30 and peak <= 0.40 and np.percentile(gaps, 97) <= 0.8)}
    else:
        out["pauses"] = None
    out.update(loudness(wav))
    out["music"] = music_stats(x, sr, sp.mask, stereo)
    out["sfx"] = sfx_rate(x, sr, sp.mask, dur)
    return out, sp.mask


def music_stats(x: np.ndarray, sr: int, mask: np.ndarray, stereo: str | None) -> dict | None:
    """Music under the voice: the level of the pauses (their middle, 60 ms away from speech) against the voice, the
    stereo width of the pauses (a produced track is wide, a phone's room noise is not) and ducking (the low band
    < 90 Hz is louder in the pauses than under speech when the music is lowered for the voice)."""
    from scipy.io import wavfile
    from scipy.ndimage import binary_dilation
    from scipy.signal import butter, sosfilt
    hop = int(sr * 0.01)
    db = frame_db(x, hop)
    n = min(len(db), len(mask))
    speech, quiet = mask[:n], ~binary_dilation(mask[:n], iterations=6)
    if speech.sum() < 100 or quiet.sum() < 20:
        return None
    voice = float(np.median(db[:n][speech]))
    bed = float(np.percentile(db[:n][quiet], 30))
    low = frame_db(sosfilt(butter(4, [40, 90], btype="band", fs=sr, output="sos"), x), hop)[:n]
    duck = float(np.clip(np.median(low[quiet]) - np.median(low[speech]), 0, 10)) if bed - voice > -30 else 0.0
    side = None
    if stereo:
        _, st = wavfile.read(stereo)
        if st.ndim == 2 and st.shape[1] == 2:
            bp = butter(4, [200, 6000], btype="band", fs=sr, output="sos")
            mid, sid = sosfilt(bp, (st[:, 0] + st[:, 1]) / 2.0), sosfilt(bp, (st[:, 0] - st[:, 1]) / 2.0)
            e = lambda y: 10 * np.log10(np.median((y[: n * hop].reshape(n, hop) ** 2).mean(1)[quiet]) + 1e-18)
            side = round(float(e(sid) - e(mid)), 1)
    level = bed - voice
    present = "no" if level < -30 else "yes" if (side is not None and side >= -18) else "maybe"
    return {"bed_vs_voice_db": round(level, 1), "stereo_side_db": side, "ducking_db": round(duck, 1),
            "under_voice_db": round(level - duck, 1), "present": present,
            "note": "yes = a level and a stereo width of produced music; maybe = something steady in the pauses "
                    "(mono music or room noise): ask the user"}


def sfx_rate(x: np.ndarray, sr: int, mask: np.ndarray, dur: float) -> dict:
    """Sharp high-frequency hits inside the pauses (whoosh / pop / impact). Sound effects under speech are not
    counted, and a regular beat is not a sound effect."""
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.signal import butter, sosfilt
    hop = int(sr * 0.01)
    hp = frame_db(sosfilt(butter(4, 2500, btype="high", fs=sr, output="sos"), x), hop)
    base = median_filter(hp, size=51)
    near = uniform_filter1d(np.pad(mask, (0, max(0, len(hp) - len(mask))))[: len(hp)].astype(float), 11) > 0.01
    hits = [i for i in range(2, len(hp) - 2) if not near[i] and hp[i] - base[i] >= 14 and hp[i] > -60
            and hp[i] == hp[i - 2:i + 3].max()]
    merged = [h for j, h in enumerate(hits) if j == 0 or h - hits[j - 1] > 20]
    ioi = np.diff(merged) if len(merged) > 1 else np.array([])
    beat = len(ioi) >= 8 and float(ioi.std() / ioi.mean()) < 0.25
    n = 0 if beat else len(merged)
    return {"per_min_in_pauses": round(n / dur * 60, 1), "level": "none" if n / dur * 60 < 1 else "light" if n / dur * 60 < 4
            else "medium" if n / dur * 60 < 9 else "heavy", "note": "sounds under speech are not counted"}


def speech_rate(wav: str, mask: np.ndarray, work: str, lang: str) -> dict | None:
    """Characters per second of speech: Whisper on the densest 60 s window."""
    from cv.vad import HOP
    if not shutil.which("whisper-cli") or not os.path.exists(WHISPER_MODEL):
        return None
    dur = len(mask) * HOP
    win = min(60.0, dur)
    step = int(10 / HOP)
    w = int(win / HOP)
    starts = range(0, max(1, len(mask) - w + 1), step)
    best = max(starts, key=lambda s: mask[s:s + w].mean())
    t0 = best * HOP
    sample = os.path.join(work, "rate_16k.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-t", f"{win:.2f}", "-i", wav, "-ar", "16000",
                    "-ac", "1", sample], check=True)
    prefix = os.path.join(work, "rate")
    subprocess.run(["whisper-cli", "-m", WHISPER_MODEL, "-f", sample, "-l", lang, "-nt", "-np", "-otxt", "-of", prefix],
                   capture_output=True, text=True, check=True)
    text = " ".join(open(prefix + ".txt", encoding="utf-8").read().split())
    chars = len(re.sub(r"[\s、。，．,.!！?？…・「」『』（）()\"“”'’‘~〜\-－―]", "", text))
    speech_s = float(mask[best:best + w].sum() * HOP)
    clock, spoken = chars / win, chars / max(speech_s, 1.0)
    return {"lang": lang, "window_s": [round(t0, 1), round(t0 + win, 1)], "chars": chars, "speech_s": round(speech_s, 1),
            # a tight edit leaves almost no pauses: the clock rate is then the steadier number (the speech mask
            # drops soft phonemes), otherwise characters per second of speech
            "chars_per_sec": round(clock if speech_s / win >= 0.75 else spoken, 1),
            "chars_per_sec_speech": round(spoken, 1), "chars_per_sec_clock": round(clock, 1),
            "words_per_sec": round(len(text.split()) / win, 1), "text": text[:300]}


# ---------------------------------------------------------------- telops
