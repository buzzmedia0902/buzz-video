#!/usr/bin/env python3
"""Technical QA of a finished video: length, frames, loudness, true peak, size, frame-step stutter.

  python qa.py final.mp4 [--until 20.3] [--json report.json]

Near-duplicate consecutive frames inside footage segments usually mean stutter: slow motion made from a
30 fps clip, or a frame-mapping bug. Static UI / end cards produce duplicates on purpose, so pass --until
(the end-card time) and judge the per-second list rather than the raw total.
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


def frame_steps(path: str, until: float | None) -> dict:
    info = media.stream_info(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (v["r_frame_rate"] + "/1").split("/")[:2]
    fps = float(num) / float(den)
    w, h = 192, int(round(192 * int(v["height"]) / int(v["width"]) / 2)) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    dup_t = [(i + 1) / fps for i in np.nonzero(d < 0.35)[0]]
    if until is not None:
        dup_t = [t for t in dup_t if t < until]
    per_sec: dict[int, int] = {}
    for t in dup_t:
        per_sec[int(t)] = per_sec.get(int(t), 0) + 1
    return {"frames": int(len(fr)), "fps": round(fps, 3), "near_duplicates": len(dup_t),
            "dup_seconds": {str(k): n for k, n in sorted(per_sec.items()) if n >= 3}}


def report(path: str, until: float | None, loud: tuple[float, float] | None = None) -> dict:
    """loud: (LUFS, true peak) already measured on this file (skips a second ebur128 pass)."""
    info = media.stream_info(path)
    fmt = info["format"]
    out = {"file": os.path.basename(path), "duration": round(float(fmt["duration"]), 3),
           "size_mb": round(int(fmt["size"]) / 1e6, 1)}
    if any(s["codec_type"] == "audio" for s in info["streams"]):
        i, tp = loud or media.loudness(path)
        out.update(lufs=i, true_peak=tp, audio_ok=(-15.0 <= i <= -13.0 and tp <= -1.0))
    out.update(frame_steps(path, until))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--until", type=float, help="ignore duplicates from this time on (end card)")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = report(a.video, a.until)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
