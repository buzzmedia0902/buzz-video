#!/usr/bin/env python3
"""Finish one version at one aspect: music + SFX + narration mix, -14 LUFS, mux, light copy, cover, QA.

  python finish.py project.json --version vlog --aspect 4:5 [--rebuild-bgm] [--no-light]

Needs the silent master (render.py --video) and work/<version>/cues.json. The BGM is composed once per
version (bgm.py, style from version "bgm" or the taste default) and reused for every aspect ratio.
Outputs in out_dir:  <Project>_<label>_<9x16>_<N>秒.mp4, ..._軽量版.mp4 (phone/chat sharing, < 30 MB), ..._カバー.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import media  # noqa: E402

MUSIC_GAIN_DB = {"chat": -5.0, "vlog": -3.0}  # music under SFX / narration
BT709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"  # tags for masters made before tagging
LIGHT_MAX_MB = 28.0


def bgm_track(cues: dict, path: str, rebuild: bool) -> tuple[np.ndarray, dict]:
    """Compose (or reuse) the version's BGM; regenerate when style/length/card/tempo changed.
    Returns (audio, info) where info has the key the SFX are tuned to."""
    import bgm
    b = cues.get("bgm", {})
    key = {"style": b.get("style", "piano"), "dur": cues["duration"], "card": cues["card_at"], "bpm": b.get("bpm"),
           "beats": b.get("beats"), "seed": b.get("seed", 7), "cuts": cues.get("cuts", [])}
    meta, info_path = path + ".key.json", path + ".json"
    fresh = os.path.exists(path) and os.path.exists(meta) and os.path.exists(info_path)
    if rebuild or not fresh or json.load(open(meta)) != key:
        audio, info = bgm.render_bgm(key["style"], key["dur"], key["card"], bpm=key["bpm"], seed=key["seed"],
                                     cuts=tuple(key["cuts"]), beats=key["beats"])
        media.write_wav(path, audio)
        json.dump(key, open(meta, "w"))
        json.dump(info, open(info_path, "w"), ensure_ascii=False, default=str)
        print(f"[finish] BGM {key['style']} {info.get('bpm', '?')} BPM key {info.get('key', '?')} -> {path}",
              file=sys.stderr)
    return media.read_audio(path), json.load(open(info_path))


def fit_len(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))))[:, :n]


def voice_track(cues: dict, n: int) -> np.ndarray:
    v = np.zeros(n, np.float32)
    for t, path in cues.get("voice", []):
        w = media.read_audio(path, 1)[0]
        s = int(t * media.SR)
        e = min(n, s + len(w))
        if s < n:
            v[s:e] += w[:e - s]
    return v


def duck_gain(voice: np.ndarray, card_at: float) -> np.ndarray:
    """Music gain under narration: 20 ms RMS voice detection, 60 ms look-ahead, 400 ms hold, ~-15 dB duck.
    (Sample-threshold detection under-ducks and pumps; RMS + hold is what sounded right.)"""
    n, hop = len(voice), 960
    frames = n // hop
    rms = np.sqrt(np.mean(voice[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    act = (20 * np.log10(rms) > -42).astype(np.float32)
    act = np.maximum(act, np.concatenate([act[3:], np.zeros(3, np.float32)]))
    held = act.copy()
    for k in range(1, 21):
        held[k:] = np.maximum(held[k:], act[:-k])
    env_f, prev = np.zeros(frames, np.float32), 0.0
    for k in range(frames):
        coef = 0.45 if held[k] > prev else 0.93
        prev = held[k] + (prev - held[k]) * coef
        env_f[k] = prev
    env = np.interp(np.arange(n), (np.arange(frames) + 0.5) * hop, env_f).astype(np.float32)
    gain = 1 - 0.83 * np.clip(env, 0, 1)
    tail = np.clip((np.arange(n) / media.SR - (card_at + 1.9)) / 0.35, 0, 1)
    return gain * 10 ** (-4 * tail / 20)  # tame the final music hit after the spoken CTA


def build_audio(project, version: str, v: dict, cues: dict, rebuild: bool) -> str:
    wd = os.path.join(project.work_dir, version)
    n = int(round(cues["duration"] * media.SR))
    music, info = bgm_track(cues, os.path.join(wd, "bgm.wav"), rebuild)
    music = fit_len(music, n)
    music *= 10 ** (float(v.get("bgm", {}).get("gain_db", MUSIC_GAIN_DB.get(v["taste"], 0.0))) / 20)
    voice = voice_track(cues, n)
    if np.abs(voice).max() > 0:
        music *= duck_gain(voice, cues["card_at"])[None, :]
    mix = music + voice[None, :]
    if cues.get("sfx"):
        import sfx
        mix += fit_len(sfx.render_sfx(cues["sfx"], cues["duration"], key=info.get("key")), n)  # tonal SFX in key
    raw = os.path.join(wd, "mix_raw.wav")
    media.write_wav(raw, mix)
    return raw


def encode_aac(wav: str, out: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "256k", "-ar", str(media.SR), out],
                   check=True)


def mux(master: str, aac: str, out: str, taste: str) -> None:
    """Master video + the already encoded AAC (copied, so the measured true peak is what ships)."""
    vcodec = (["-c:v", "libx264", "-preset", "slow", "-crf", "21", "-tune", "grain", "-pix_fmt", "yuv420p",
               "-vf", BT709] if taste == "cinematic" else ["-c:v", "copy"])  # grain makes CRF16 huge: tune grain
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", master, "-i", aac, "-map", "0:v", "-map", "1:a", *vcodec,
                    "-c:a", "copy", "-movflags", "+faststart", "-shortest", out], check=True)


def light_copy(final: str, out: str, dur: float, log: str) -> None:
    kbps = int(min(5500, LIGHT_MAX_MB * 8e3 / dur - 200))
    common = ["-vf", BT709, "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-maxrate",
              f"{int(kbps * 1.45)}k", "-bufsize", f"{kbps * 2}k"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, *common, "-pass", "1", "-passlogfile", log, "-an",
                    "-f", "null", "-"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, *common, "-pass", "2", "-passlogfile", log,
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
    for f in os.listdir(os.path.dirname(log)):
        if f.startswith(os.path.basename(log)):
            os.remove(os.path.join(os.path.dirname(log), f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--version", required=True)
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "4:5", "1:1", "16:9"])
    ap.add_argument("--rebuild-bgm", action="store_true")
    ap.add_argument("--no-light", action="store_true")
    a = ap.parse_args()
    from core import spec
    import qa
    project = spec.load(a.project)
    v = project.version(a.version)
    tag = a.aspect.replace(":", "x")
    cues_path = os.path.join(project.work_dir, a.version, "cues.json")
    master = os.path.join(project.work_dir, "render", f"{a.version}_{tag}.mp4")
    for p, hint in ((cues_path, "render.py --info"), (master, f"render.py --version {a.version} --aspect {a.aspect} --video")):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run {hint} first")
    cues = json.load(open(cues_path, encoding="utf-8"))
    if abs(media.duration(master) - cues["duration"]) > 0.1:
        raise SystemExit("master length differs from cues.json: re-render the master after spec changes")
    raw = build_audio(project, a.version, v, cues, a.rebuild_bgm)
    os.makedirs(project.out_dir, exist_ok=True)
    label = v.get("label", spec.TASTE_JA[v["taste"]])
    stem = os.path.join(project.out_dir, f"{project.name}_{label}_{tag}_{round(cues['duration'])}秒")
    wd = os.path.join(project.work_dir, a.version)
    final, norm, aac, limit = stem + ".mp4", os.path.join(wd, "mix_norm.wav"), os.path.join(wd, "mix.m4a"), 0.79
    for _ in range(4):  # AAC overshoots the PCM peak: lower the limiter until the encoded true peak <= -1 dBTP
        media.normalize(raw, norm, -14.0, limit)
        encode_aac(norm, aac)
        lufs, tp = media.loudness(aac)
        if tp <= -1.0:
            break
        limit *= 10 ** (-1.5 / 20)
    mux(master, aac, final, v["taste"])
    if not a.no_light:
        light_copy(final, stem + "_軽量版.mp4", cues["duration"], os.path.join(project.work_dir, a.version, "2pass"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{cues['cover_at']:.3f}", "-i", final, "-frames:v", "1",
                    stem + "_カバー.png"], check=True)
    r = qa.report(final, cues["card_at"], loud=(lufs, tp))
    if not a.no_light:
        r["light_mb"] = round(os.path.getsize(stem + "_軽量版.mp4") / 1e6, 1)
    with open(os.path.join(project.work_dir, a.version, f"report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
