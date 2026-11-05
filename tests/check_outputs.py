#!/usr/bin/env python3
"""Check the deliverables a smoke run left: every master video exists, plays, and sits at -14 +-1 LUFS.

  python tests/check_outputs.py <delivery folder> [...]
"""
import glob
import os
import re
import subprocess
import sys


def loudness(path: str) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def main() -> int:
    bad = 0
    for d in sys.argv[1:]:
        masters = [p for p in glob.glob(os.path.join(d, "*.mp4")) if "軽量版" not in os.path.basename(p)]
        if not masters:
            print(f"NG   no video in {d}")
            bad += 1
        for p in masters:
            lufs = loudness(p)
            ok = abs(lufs + 14) <= 1
            bad += not ok
            print(f"{'ok' if ok else 'NG'}   {lufs:6.1f} LUFS  {os.path.basename(p)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
