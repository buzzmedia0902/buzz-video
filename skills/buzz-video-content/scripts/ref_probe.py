#!/usr/bin/env python3
"""Measure a reference video the user likes, so the edit can match its トンマナ (tone and manner).

  python ref_probe.py <URL or file> [--minutes 3] [--out work/ref] [--lang ja]

Only the first N minutes are looked at (a URL is downloaded with yt-dlp, <= 720p, that section only). It measures
  cuts            shot changes per minute (spikes in the frame difference: jump cuts, zoom cuts, slide changes)
  camera          zoom switches and slow push-ins per minute, how big the wide <-> medium step is (needs a face)
  speech / pauses speech share, the pauses that remain between phrases (how tight the jet cut is), characters per second
                  (Whisper on the densest 60 s)
  sound           loudness (LUFS), music under the voice (level of the pauses vs the voice), sound effects in the pauses
  telops          how often text is on screen, where (Apple Vision text boxes), how big, fill / outline colours, plate
  look            saturation, contrast, brightness
and writes  <out>/ref.json  (metrics + the nearest tone + suggested edit.json "look" overrides),
            <out>/contact.jpg (a frame every ~5 s) and <out>/telops.jpg (crops of the telops) for Claude to look at.
Look at both sheets before trusting the numbers: fonts and graphics are not measured.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BIN = os.path.expanduser("~/.local/share/content-video/bin")
DETECT_FPS = 30


# ---------------------------------------------------------------- fetching
def fetch(url: str, out: str, minutes: float) -> str:
    """First `minutes` of a URL, <= 720p (only measured, never used in the edit). Falls back to the 1080p HLS
    stream when the first format gives nothing; every try downloads only that section."""
    target = os.path.join(out, "ref_src")
    for old in glob.glob(target + ".*"):
        os.remove(old)
    section = ["--download-sections", f"*0-{int(minutes * 60)}"]
    tries = [["-f", "bv*[height<=720]+ba/b[height<=720]/b"],
             ["-f", "270+ba/270/bv*+ba/b"]]
    err = ""
    for extra in tries:
        r = subprocess.run(["yt-dlp", "--no-playlist", "--merge-output-format", "mp4", "-o", target + ".%(ext)s",
                            *section, *extra, url], capture_output=True, text=True)
        got = glob.glob(target + ".*")
        if r.returncode == 0 and got:
            return got[0]
        err = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
        print(f"[ref] yt-dlp {' '.join(extra[:2])}: {err[0][:140]} -> next try", file=sys.stderr)
    raise SystemExit(f"yt-dlp could not fetch {url} ({err[0][:200]}). Download it yourself and pass the file.")


def probe(path: str) -> dict:
    from cv import media
    return media.probe(path)


def ffmpeg_raw(path: str, secs: float, vf: str) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-t", f"{secs:.2f}", "-i", path,
                           "-vf", vf, "-f", "rawvideo", "-"], capture_output=True, check=True).stdout


def sample_frames(path: str, secs: float, step: float, out_dir: str, tag: str, width: int) -> list[tuple[float, str]]:
    """Frames every `step` s as JPEGs: [(time, file)]; the times come from showinfo so they are exact."""
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, f"{tag}*.jpg")):
        os.remove(old)
    vf = (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})',showinfo,"
          f"scale='min({width},iw)':-2:flags=lanczos")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "info", "-hwaccel", "videotoolbox", "-t", f"{secs:.2f}", "-i",
                        path, "-vf", vf, "-fps_mode", "vfr", "-q:v", "3", os.path.join(out_dir, f"{tag}%04d.jpg")],
                       capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", r.stderr)]
    files = sorted(glob.glob(os.path.join(out_dir, f"{tag}*.jpg")))
    return list(zip(times, files))


def helper(name: str) -> str:
    """Compile tools/<name>.swift once into ~/.local/share/content-video/bin."""
    src, out = os.path.join(HERE, "tools", f"{name}.swift"), os.path.join(BIN, name)
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        if not shutil.which("swiftc"):
            raise SystemExit("swiftc not found (Xcode command line tools): faces / telop boxes need Apple Vision")
        os.makedirs(BIN, exist_ok=True)
        subprocess.run(["swiftc", "-O", src, "-o", out], check=True)
    return out


def jsonl(binary: str, files: list[str]) -> list[dict]:
    """Run a folder tool (faces / textboxes) on `files` (symlinked as 000001.jpg ...); one dict per file."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(files):
            os.symlink(os.path.abspath(f), os.path.join(d, f"{i + 1:06d}.jpg"))
        rows = {}
        for line in subprocess.run([binary, d], capture_output=True, text=True, check=True).stdout.splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                rows[r["f"]] = r
    return [rows.get(f"{i + 1:06d}.jpg", {}) for i in range(len(files))]


# ---------------------------------------------------------------- cuts and camera


def cut_events(path: str, secs: float, fps: float) -> tuple[list[float], np.ndarray]:
    """Cut times: a one-frame spike in the picture difference (jump cuts, zoom cuts; checked against a rendered short
    with known shots: 28 of 32 found, none invented) or a run of frames where most of the picture changes (slide
    changes, wipes)."""
    raw = ffmpeg_raw(path, secs, f"fps={fps:g},scale=96:54:flags=area,format=gray")
    g = np.frombuffer(raw, np.uint8)[: len(raw) // (96 * 54) * 96 * 54].reshape(-1, 54, 96).astype(np.int16)
    diff = np.abs(np.diff(g, axis=0))
    n = len(diff)
    d = diff.mean(axis=(1, 2))
    frac = (diff.reshape(n, 6, 9, 8, 12).mean(axis=(2, 4)) > 6).mean(axis=(1, 2))
    cuts = [i for i in range(2, n - 2) if d[i] >= 2.5 and d[i] >= 2.0 * max(d[i - 2], d[i - 1], d[i + 1], d[i + 2])
            and d[i] == max(d[i - 2:i + 3])]
    run = []
    for i in range(n + 1):
        if i < n and frac[i] >= 0.6 and d[i] >= 8:
            run.append(i)
        elif run:
            if all(abs(run[0] - c) > 4 for c in cuts):
                cuts.append(run[0])
            run = []
    return [round((c + 1) / fps, 3) for c in sorted(cuts)], d


def shot_stats(cuts: list[float], dur: float) -> dict:
    edges = [0.0] + cuts + [dur]
    lens = np.diff(edges)
    return {"cuts": len(cuts), "cuts_per_min": round(len(cuts) / dur * 60, 1),
            "median_shot_s": round(float(np.median(lens)), 2),
            "p25_shot_s": round(float(np.percentile(lens, 25)), 2), "p75_shot_s": round(float(np.percentile(lens, 75)), 2),
            "short_shots_pct": round(100 * float((lens < 1.0).mean()), 0),
            "long_shots_pct": round(100 * float((lens > 6.0).mean()), 0), "times": cuts[:200]}


def camera_stats(path: str, secs: float, cuts: list[float], work: str) -> dict | None:
    """Zoom switches (face size jumps at a cut) and slow push-ins (face grows inside a shot), from the largest face
    every 0.5 s. None when there is no face to follow."""
    frames = sample_frames(path, secs, 0.5, os.path.join(work, "faces"), "p", 480)
    rows = jsonl(helper("faces"), [f for _, f in frames])
    t = np.array([x for x, _ in frames])
    fh = np.array([max((f[3] for f in r.get("faces", []) if f[4] > 0.5), default=np.nan) for r in rows])
    if len(t) < 8 or np.isnan(fh).mean() > 0.5:
        return None
    ratios, at = [], []
    for c in cuts:
        a = fh[(t >= c - 1.5) & (t < c) & ~np.isnan(fh)]
        b = fh[(t > c) & (t <= c + 1.5) & ~np.isnan(fh)]
        if len(a) >= 2 and len(b) >= 2:
            r = float(np.median(b) / np.median(a))
            if abs(np.log(r)) >= np.log(1.08):
                ratios.append(max(r, 1 / r))
                at.append(c)
    pushes = []
    edges = [0.0] + cuts + [float(t[-1])]
    for a, b in zip(edges, edges[1:]):
        sel = (t >= a) & (t <= b) & ~np.isnan(fh)
        if b - a >= 2.5 and sel.sum() >= 5:
            x, y = t[sel], np.log(fh[sel])
            slope = np.polyfit(x, y, 1)[0]
            corr = np.corrcoef(x, y)[0, 1] if y.std() > 0 else 0
            if np.exp(slope * (b - a)) - 1 >= 0.035 and corr > 0.8:
                pushes.append(float(np.exp(slope * (b - a))))
    mins = secs / 60
    hold = float(np.median(np.diff([0.0] + at))) if len(at) >= 2 else None
    return {"zoom_switch_per_min": round(len(ratios) / mins, 1),
            "med_ratio": round(float(np.median(ratios)), 2) if ratios else None,
            "hold_s": round(hold, 1) if hold else None, "push_per_min": round(len(pushes) / mins, 1),
            "push_zoom": round(float(np.median(pushes)), 2) if pushes else None,
            "face_size_pct": round(100 * float(np.nanmedian(fh)), 0)}


# ---------------------------------------------------------------- sound


def main() -> None:
    from cv.refprobe import audio as A
    from cv.refprobe import telops as T
    from cv.refprobe import tone as N
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="URL or video file")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--out", default="work/ref")
    ap.add_argument("--lang", default="ja", help="spoken language for the speech rate (Whisper code)")
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    src = a.source
    if re.match(r"^https?://", src):
        if not shutil.which("yt-dlp"):
            raise SystemExit("yt-dlp not found (brew install yt-dlp), or pass a video file")
        src = fetch(src, out, a.minutes)
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        raise SystemExit(f"{src}: not found")
    info = probe(src)
    secs = min(a.minutes * 60, info["duration"])
    layout = "tall" if info["h"] > info["w"] else "wide"
    fps = min(info["fps"] or 30.0, DETECT_FPS)
    print(f"[ref] {os.path.basename(src)}: {info['w']}x{info['h']} {info['fps']:.0f} fps, measuring {secs:.0f} s",
          file=sys.stderr)

    cuts, _ = cut_events(src, secs, fps)
    shots = shot_stats(cuts, secs)
    camera = camera_stats(src, secs, cuts, out)
    audio: dict = {"speech_share": None, "pauses": None, "music": None, "sfx": None}
    speech, mask = None, None
    if info["audio"]:
        wav = A.extract_wav(src, secs, os.path.join(out, "ref_audio.wav"))
        audio, mask = A.audio_stats(wav, out, A.extract_wav(src, secs, os.path.join(out, "ref_audio_st.wav"), 2))
        speech = A.speech_rate(wav, mask, out, a.lang)
    width = 720 if layout == "tall" else 1280
    frames = sample_frames(src, secs, 1.5, os.path.join(out, "frames"), "t", width)
    rows = jsonl(helper("textboxes"), [f for _, f in frames])
    m = {"source": a.source, "analysed_s": round(secs, 1), "size": [info["w"], info["h"]], "layout": layout,
         "fps": round(info["fps"], 2),
         "shots": shots, "camera": camera, "audio": audio, "speech": speech,
         "telops": T.telop_stats(frames, rows), "look": T.look_stats(frames)}
    scores, parts = N.tone_scores(m)
    tone = max(scores, key=scores.get)
    m["tone_scores"] = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
    m["tone"] = tone
    m["why"] = N.why(m, tone, parts[tone])
    m["suggest"] = N.suggest(m, tone, layout)
    m["files"] = {"contact": T.contact_sheet(frames, os.path.join(out, "contact.jpg")),
                  "telops": T.telop_sheet(frames, rows, os.path.join(out, "telops.jpg")),
                  "ref_json": os.path.join(out, "ref.json")}
    json.dump(m, open(m["files"]["ref_json"], "w"), ensure_ascii=False, indent=1)
    print(summary(m))


def summary(m: dict) -> str:
    """What the user and Claude read first (the numbers are all in ref.json)."""
    s, a, t = m["shots"], m["audio"], m["telops"]
    p = a.get("pauses")
    lines = [f"参考動画を {m['analysed_s']:.0f} 秒ぶん測りました（{m['layout']}）",
             f"  カット        1分に {s['cuts_per_min']} 回（ショット中央値 {s['median_shot_s']} 秒）"]
    if m["camera"]:
        c = m["camera"]
        lines.append(f"  画の動き      カットで顔の大きさが変わる回数 1分に {c['zoom_switch_per_min']} 回"
                     f" / ゆっくりズーム 1分に {c['push_per_min']} 回")
    jet = f" / 残っている間は {p['p97_s']} 秒まで（ジェットカット{'あり' if p['jet_cut'] else 'なし'}）" if p else ""
    lines.append(f"  話・間        話している割合 {a['speech_share']}{jet}")
    if m["speech"]:
        sp = m["speech"]
        rate = f"{sp['chars_per_sec']} 文字" if sp["lang"] == "ja" else f"{sp['words_per_sec']} 語（{sp['chars_per_sec']} 文字）"
        lines.append(f"  話す速さ      1秒に {rate}")
    if a.get("lufs") is not None:
        mu = a["music"]
        bgm = {"yes": f"声の {mu['under_voice_db']} dB 下", "maybe": "あるかも（要確認）", "no": "なし"}[mu["present"]] if mu else "不明"
        lines.append(f"  音            {a['lufs']:.1f} LUFS / BGM {bgm} / 効果音 {a['sfx']['level']}")
    if "subtitle" in t:
        sb = t["subtitle"]
        colors = "・".join(f"{c['name']}{int(c['share'] * 100)}%" for c in t.get("colors", [])[:3])
        lines.append(f"  テロップ      画面の {t['frames_with_text_pct']:.0f}% に文字 / 字幕は{sb['position']}（y={sb['y']}）"
                     f"・文字高 {sb['line_height_pct']}%" + (f" / 色: {colors}" if colors else ""))
    scores = ", ".join(f"{k} {v}" for k, v in m["tone_scores"].items())
    lines.append(f"  いちばん近いトンマナ: {m['tone']}  （{scores}）")
    lines += [f"    - {w}" for w in m["why"]]
    lines.append(f"  look の提案: {json.dumps(m['suggest']['look'], ensure_ascii=False)}")
    lines.append("  見る画像: " + " / ".join(f for f in (m["files"]["contact"], m["files"]["telops"]) if f))
    lines.append(f"  詳細: {m['files']['ref_json']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
