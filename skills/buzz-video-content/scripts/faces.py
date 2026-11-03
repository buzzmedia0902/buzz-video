#!/usr/bin/env python3
"""Face boxes of the sources (Apple Vision, 2 frames per second) for framing, face-aware panels and thumbnails.

  python faces.py edit.json [--src talk]      -> work/faces/<src>.jsonl  {"f": "000001.jpg", "faces": [[x, y, w, h, conf, yaw]]}

Coordinates are normalised to the upright frame (origin top-left); frame k (1-based) is at (k - 1) / 2 s.
Run it once per source before rendering talk videos: medium shots centre on the face and cards / name plates /
big words are placed where they do not cover anyone's face.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BIN = os.path.expanduser("~/.local/share/content-video/bin/faces")


def binary() -> str:
    src = os.path.join(HERE, "tools", "faces.swift")
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(BIN), exist_ok=True)
        subprocess.run(["swiftc", "-O", src, "-o", BIN], check=True)
    return BIN


def scan(path: str, out: str, rot: int = 0) -> None:
    rotf = {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}.get(rot, "")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", path, "-vf",
                        f"fps=2,{rotf}scale=640:-2", "-q:v", "4", os.path.join(d, "%06d.jpg")], check=True)
        res = subprocess.run([binary(), d], capture_output=True, text=True, check=True).stdout
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--src", nargs="*")
    a = ap.parse_args()
    if not shutil.which("swiftc"):
        raise SystemExit("swiftc not found: faces are optional (framing then uses 'center', panels default spots)")
    from cv import spec
    p = spec.load(a.project)
    for sid in a.src or sorted(p.sources):
        out = p.work("faces", f"{sid}.jsonl")
        scan(p.src_path(sid), out, p.sources[sid]["rot"])
        n = sum(1 for _ in open(out))
        print(f"{sid}: {n} frames -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
