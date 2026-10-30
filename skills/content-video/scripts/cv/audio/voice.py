"""Voice track: the source audio cut exactly like the picture, then cleaned up.

Every non-card shot [a, b) of cues_out.json becomes output samples [t0, t1) of the same length as its frames
(sample positions come from the rounded shot times, so nothing accumulates): 12 ms fades at both ends, Rubber Band
(R3 `--fine`, pitch-preserving `-T`) when the shot plays faster or slower, padded / trimmed to the exact length.
Shots with audio "keep" form the dialogue (high-pass + EQ + gentle compression, a denoiser only when the
recording is noisy); "duck" shots keep their ambience (the mix puts it 12 dB down); "mute" shots are dropped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile

from cv import media
from cv.audio import dsp
from cv.audio.dsp import SR

FADE = int(0.012 * SR)
PRE_LUFS = -24.0            # per-source level before the chain, so interleaved sources match
CHAIN = ("highpass=f=90,equalizer=f=280:t=q:w=1.0:g=-2.5,equalizer=f=3200:t=q:w=1.2:g=2.0,"
         "acompressor=threshold=-26dB:ratio=2.5:attack=8:release=160:makeup=2dB")
AMBIENCE_CHAIN = "highpass=f=110"
NOISY_SNR_DB = 26.0         # speech (90th percentile) over the quiet floor (5th) below this -> denoise
AFFTDN_DELAY = 1200         # samples afftdn delays its output (measured); removed again so picture and voice stay in sync


@dataclass
class Voice:
    dialogue: np.ndarray    # (n,) float32, chain applied, level not normalised yet (mix.py does that)
    ambience: np.ndarray    # (n,) float32, un-ducked ambience of "duck" shots (high-passed)
    info: dict = field(default_factory=dict)


def _source(path: str, sid: str, work_dir: str) -> np.ndarray | None:
    """Memory-mapped 48 kHz mono float WAV of a source (extracted once, shared with the VAD); None if it has no audio."""
    if not media.probe(path)["audio"]:
        return None
    wav = media.extract_audio(path, os.path.join(work_dir, "audio", f"{sid}_48k.wav"))
    return wavfile.read(wav, mmap=True)[1]


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _edges(x: np.ndarray) -> np.ndarray:
    k = min(FADE, len(x) // 2)
    g = np.ones(len(x), np.float32)
    if k:
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)).astype(np.float32)
        g[:k], g[len(x) - k:] = ramp, ramp[::-1]
    return x * g


def _stretch(x: np.ndarray, speed: float, tmp: str, tag: str) -> np.ndarray:
    """Pitch-preserving tempo change with Rubber Band's R3 engine (`-T` = speed-up factor); ffmpeg's atempo when
    rubberband is not installed."""
    a, b = os.path.join(tmp, f"{tag}_in.wav"), os.path.join(tmp, f"{tag}_out.wav")
    wavfile.write(a, SR, x.astype(np.float32))
    if shutil.which("rubberband"):
        subprocess.run(["rubberband", "-q", "--fine", "-T", f"{speed:.6f}", a, b], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", f"atempo={speed:.6f}", "-c:a", "pcm_f32le", b],
                       check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)     # libsndfile's PEAK chunk
        return wavfile.read(b)[1].astype(np.float32)


def _noise(src: np.ndarray, a0: int, a1: int) -> dict:
    """Quiet floor (5th percentile of 20 ms frames) and speech level (90th) over the span of a source that is used,
    pauses included - the jet-cut result would hide the floor."""
    parts = []
    for s in range(a0, min(a1, len(src)), SR * 60):
        x = np.asarray(src[s: min(s + SR * 60, a1)], np.float64)
        fr = x[: len(x) // 960 * 960].reshape(-1, 960)
        parts.append(10 * np.log10((fr ** 2).mean(axis=1) + 1e-12))
    db = np.concatenate(parts) if parts else np.zeros(0)
    db = db[db > -90.0]
    if len(db) < 50:
        return {"floor_db": None, "snr_db": None, "denoise": False}
    floor, loud = float(np.percentile(db, 5)), float(np.percentile(db, 90))
    return {"floor_db": round(floor, 1), "snr_db": round(loud - floor, 1), "denoise": loud - floor < NOISY_SNR_DB}


def _run_chain(x: np.ndarray, chain: str, tmp: str, tag: str, delay: int = 0) -> np.ndarray:
    """ffmpeg audio chain over a whole track; `delay` = samples the chain adds (removed so timing is unchanged)."""
    a, b = os.path.join(tmp, f"{tag}_raw.wav"), os.path.join(tmp, f"{tag}_chain.wav")
    wavfile.write(a, SR, x.astype(np.float32))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", chain, "-ac", "1", "-c:a", "pcm_f32le", b],
                   check=True)
    return _fit(wavfile.read(b)[1].astype(np.float32)[delay:], len(x))


def build(cues: dict, work_dir: str, workers: int = 8) -> Voice:
    """Dialogue and ambience tracks (output timeline, exactly round(duration * SR) samples)."""
    n_total = dsp.n_of(cues["duration"])
    sources: dict[str, np.ndarray | None] = {}
    jobs = []
    for s in cues["shots"]:
        if s["kind"] == "card" or not s.get("src") or s.get("audio", "keep") == "mute":
            continue
        sid = s["src"]
        if sid not in sources:
            sources[sid] = _source(s["path"], sid, work_dir)
        if sources[sid] is None:
            continue
        s0, s1 = dsp.n_of(s["t0"]), dsp.n_of(s["t1"])
        a0 = dsp.n_of(s["a"])
        raw = np.array(sources[sid][a0: a0 + int(round((s1 - s0) * s["speed"]))], np.float32)
        jobs.append({"sid": sid, "s0": s0, "m": s1 - s0, "a": a0, "speed": float(s["speed"]), "raw": raw,
                     "role": "dialogue" if s.get("audio", "keep") == "keep" else "ambience"})
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_stretch, j["raw"], j["speed"], tmp, f"s{i}")
                if abs(j["speed"] - 1.0) > 1e-4 and len(j["raw"]) else None for i, j in enumerate(jobs)]
        for j, f in zip(jobs, futs):
            j["raw"] = f.result() if f else j["raw"]
            j["seg"] = _edges(_fit(j["raw"], j["m"]))
        gains = _source_gains(jobs)
        amb = np.zeros(n_total, np.float32)
        dia: dict[str, np.ndarray] = {}
        for j in jobs:
            track = amb if j["role"] == "ambience" else dia.setdefault(j["sid"], np.zeros(n_total, np.float32))
            e = min(n_total, j["s0"] + j["m"])
            track[j["s0"]:e] += j["seg"][: e - j["s0"]] * np.float32(dsp.from_db(gains[j["sid"]]))
        noise = {}
        for sid, track in dia.items():
            used = [j for j in jobs if j["sid"] == sid and j["role"] == "dialogue"]
            lo, hi = min(j["a"] for j in used), max(j["a"] + j["m"] * j["speed"] for j in used)
            noise[sid] = _noise(sources[sid], int(lo), int(hi))
            if noise[sid]["denoise"]:
                nf = int(np.clip(noise[sid]["floor_db"] + 2, -80, -25))
                dia[sid] = _run_chain(track, f"afftdn=nr=10:nf={nf}:tn=1", tmp, f"dn_{sid}", AFFTDN_DELAY)
        talk = sum(dia.values()) if dia else np.zeros(n_total, np.float32)
        talk = _run_chain(talk, CHAIN, tmp, "dia") if np.abs(talk).max() > 0 else talk
        amb = _run_chain(amb, AMBIENCE_CHAIN, tmp, "amb") if np.abs(amb).max() > 0 else amb
    info = {"shots": len(jobs), "dialogue_shots": sum(j["role"] == "dialogue" for j in jobs),
            "ambience_shots": sum(j["role"] == "ambience" for j in jobs),
            "stretched": sum(abs(j["speed"] - 1.0) > 1e-4 for j in jobs), "source_gain_db": gains, "noise": noise}
    return Voice(talk, amb, info)


def _source_gains(jobs: list[dict]) -> dict[str, float]:
    """dB per source that brings its used material to PRE_LUFS (a source with almost no signal is left alone)."""
    out = {}
    for sid in {j["sid"] for j in jobs}:
        segs = [j["seg"] for j in jobs if j["sid"] == sid]
        loud = dsp.lufs(np.concatenate(segs)) if sum(map(len, segs)) > dsp.n_of(0.8) else -70.0
        out[sid] = round(float(np.clip(PRE_LUFS - loud, -18.0, 18.0)), 2) if loud > -60.0 else 0.0
    return out


def main() -> None:
    """Debug: write the dialogue / ambience tracks of a version.  python -m cv.audio.voice cues_out.json work_dir"""
    import json
    cues = json.load(open(sys.argv[1], encoding="utf-8"))
    v = build(cues, sys.argv[2])
    out = os.path.join(sys.argv[2], "voice_debug.wav")
    dsp.write_wav(out, v.dialogue)
    print(out, v.info)


if __name__ == "__main__":
    main()
