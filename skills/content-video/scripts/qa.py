#!/usr/bin/env python3
"""Technical QA of a finished video: length, frames, loudness, true peak, judder, and the pacing numbers of the edit.

  python qa.py final.mp4 [--cues work/<V>/cues_out.json] [--project edit.json --version V] [--json report.json]

From the file:  duration, frames, fps, size, LUFS, true peak (dBTP, measured on the AAC), audio_ok (-15..-13 LUFS and
                <= -1 dBTP), judder = frames that repeat the previous one while the picture around them moves
                (slow motion from a 30 fps clip, a frame-mapping bug); static cards and still faces are not counted.
From cues_out.json (found next to the project's work folder when --cues is not given):
                subtitle time on screen (min, how many are shorter than 1.2 s), characters per second (95th
                percentile), shot changes per minute, punch-in count (`--project/--version` rebuilds the camera plan
                when cues_out.json carries no per-shot "punch").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIN_SHOW = 1.2
DUP_ABS = 0.35               # mean |difference| (0-255, 192 px wide gray) below which two frames are the same
JUDDER_REL = 0.2             # ... and below this fraction of the local median step, with real motion around


def probe_streams(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,nb_frames,duration:format=size,duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _fps(rate: str) -> float:
    num, den = (rate + "/1").split("/")[:2]
    return float(num) / float(den)


def frame_steps(path: str, w: int, h: int, fps: float, skip: list[tuple[float, float]]) -> dict:
    """Step size between consecutive frames (192 px gray), streamed; judder = a repeated frame inside real motion."""
    gw, gh = 192, max(2, int(round(192 * h / w / 2)) * 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"scale={gw}:{gh},format=gray", "-f",
                             "rawvideo", "-"], stdout=subprocess.PIPE)
    size, steps, prev = gw * gh, [], None
    while True:
        buf = proc.stdout.read(size * 240)
        if not buf:
            break
        fr = np.frombuffer(buf[: len(buf) // size * size], np.uint8).reshape(-1, gh, gw).astype(np.float32)
        if prev is not None:
            fr = np.concatenate([prev[None], fr])
        steps.append(np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2)))
        prev = fr[-1]
    proc.wait()
    d = np.concatenate(steps) if steps else np.zeros(0)
    t = (np.arange(len(d)) + 1) / fps
    live = np.ones(len(d), bool)
    for a, b in skip:
        live &= ~((t >= a - 0.05) & (t < b + 0.05))
    med = median_filter(d, size=31, mode="nearest")
    dup = (d < DUP_ABS) & live
    judder = (d < JUDDER_REL * med) & (med > 0.8) & live
    runs = np.flatnonzero(np.diff(np.concatenate([[0], judder.astype(int), [0]])) == 1)
    lens = np.flatnonzero(np.diff(np.concatenate([[0], judder.astype(int), [0]])) == -1) - runs
    per_sec: dict[int, int] = {}
    for x in t[judder]:
        per_sec[int(x)] = per_sec.get(int(x), 0) + 1
    return {"frames_decoded": int(len(d) + 1), "near_duplicates": int(dup.sum()), "judder_frames": int(judder.sum()),
            "judder_runs": int(len(runs)), "judder_longest_run": int(lens.max()) if len(lens) else 0,
            "judder_at": [round(float(t[r]), 2) for r in runs[:12]],
            "judder_seconds": {str(k): n for k, n in sorted(per_sec.items()) if n >= 3}}


def subtitle_stats(subs: list[dict]) -> dict:
    from cv import cues as C
    if not subs:
        return {"subtitles": 0}
    dur = np.array([s["t1"] - s["t0"] for s in subs])
    cps = np.array([C.n_chars(s["text"]) / max(d, 1e-3) for s, d in zip(subs, dur)])
    return {"subtitles": len(subs), "subtitle_min_s": round(float(dur.min()), 2),
            "subtitles_shorter_than_1_2s": int((dur < MIN_SHOW - 1e-3).sum()),
            "chars_per_s_p95": round(float(np.percentile(cps, 95)), 1), "chars_per_s_max": round(float(cps.max()), 1)}


def punch_ins(cues: dict, project: str | None, version: str | None) -> int | None:
    """Slow push-ins of the camera plan: from cues_out.json when it lists them, else by rebuilding the plan."""
    if any("punch" in s for s in cues["shots"]):
        return sum(1 for s in cues["shots"] if s.get("punch"))
    if not (project and version):
        return None
    try:
        import render
        render.setup(project, version)
        return sum(1 for s in render.Comp(project, version).tl.shots if s.view.get("punch"))
    except Exception as e:  # noqa: BLE001 - QA must not stop the delivery
        print(f"[qa] punch-in count unavailable: {e}", file=sys.stderr)
        return None


def pacing(cues: dict, project: str | None = None, version: str | None = None) -> dict:
    minutes = cues["duration"] / 60
    out = {"shot_changes": len(cues["shots"]) - 1,
           "shot_changes_per_min": round((len(cues["shots"]) - 1) / minutes, 1) if minutes else 0.0}
    out.update(subtitle_stats(cues.get("subs", [])))
    out["punch_ins"] = punch_ins(cues, project, version)
    return out


def find_cues(video: str, duration: float) -> str | None:
    """cues_out.json of the version this file was made from: the work folder next to out_dir, matched by length."""
    base = os.path.dirname(os.path.abspath(video))
    for root in (base, os.path.dirname(base)):
        for p in sorted(glob.glob(os.path.join(root, "work", "*", "cues_out.json"))):
            try:
                if abs(json.load(open(p, encoding="utf-8"))["duration"] - duration) < 0.15:
                    return p
            except (OSError, KeyError, ValueError):
                continue
    return None


def report(path: str, cues: dict | None = None, loud: tuple[float, float] | None = None, project: str | None = None,
           version: str | None = None) -> dict:
    """loud: (LUFS, true peak) already measured on this file (skips a second ebur128 pass)."""
    from cv.audio import dsp
    info = probe_streams(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    fps = _fps(v["r_frame_rate"])
    out = {"file": os.path.basename(path), "duration": round(float(info["format"]["duration"]), 3),
           "size_mb": round(int(info["format"]["size"]) / 1e6, 1), "width": int(v["width"]), "height": int(v["height"]),
           "fps": round(fps, 3), "frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None}
    if any(s["codec_type"] == "audio" for s in info["streams"]):
        m = dsp.measure_file(path) if loud is None else {"I": loud[0], "TP": loud[1]}
        out.update(lufs=round(m["I"], 2), true_peak=round(m["TP"], 2),
                   audio_ok=bool(-15.0 <= m["I"] <= -13.0 and m["TP"] <= -1.0))
    skip = [(s["t0"], s["t1"]) for s in cues["shots"] if s["kind"] == "card"] if cues else []
    out.update(frame_steps(path, int(v["width"]), int(v["height"]), fps, skip))
    if out["frames"] is None:
        out["frames"] = out["frames_decoded"]
    if cues:
        out.update(pacing(cues, project, version))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="+", help="the finished .mp4 (a leading word 'video' is ignored)")
    ap.add_argument("--cues", help="work/<V>/cues_out.json (default: looked up by length next to the file)")
    ap.add_argument("--project")
    ap.add_argument("--version")
    ap.add_argument("--json", nargs="?", const="-", help="write the report to this file (print only with no value)")
    a = ap.parse_args()
    video = [x for x in a.video if x != "video"][-1]
    from cv import media
    cpath = a.cues or find_cues(video, media.duration(video))
    cues = json.load(open(cpath, encoding="utf-8")) if cpath else None
    if cues is None:
        print("[qa] no cues_out.json found: subtitle / pacing numbers skipped (pass --cues)", file=sys.stderr)
    r = report(video, cues, project=a.project, version=a.version)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.json and a.json != "-":
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
