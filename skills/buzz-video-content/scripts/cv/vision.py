"""Which face / expression / cut-out / text helper runs: Apple Vision (tools/*.swift) on a Mac with swiftc, else the
MediaPipe / RapidOCR version (tools/vision.py, own venv from env.sh --vision). Both take the same arguments and print
the same JSON, so callers only prepend command(tool):

  faces | landmarks | textboxes <folder of .jpg>        lift in.png out_prefix [in2.png out_prefix2 ...]

BUZZ_VISION=python forces the Python helpers on a Mac (to compare the two).
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess

from cv import osdep

SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.expanduser("~/.local/share/content-video/bin")


def apple() -> bool:
    return osdep.MAC and os.environ.get("BUZZ_VISION") != "python" and shutil.which("swiftc") is not None


@functools.lru_cache(maxsize=None)
def python_env() -> str:
    r = subprocess.run(["bash", os.path.join(SCRIPTS, "env.sh"), "--vision"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"could not prepare the vision environment: {r.stderr.strip()[-400:]}")
    return r.stdout.strip().splitlines()[-1]


def command(tool: str) -> list[str]:
    """argv prefix of the helper for tool (compiles the Swift one on first use)."""
    if apple():
        src, out = os.path.join(SCRIPTS, "tools", f"{tool}.swift"), os.path.join(BIN, tool)
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
            os.makedirs(BIN, exist_ok=True)
            subprocess.run(["swiftc", "-O", src, "-o", out], check=True)
        return [out]
    return [python_env(), os.path.join(SCRIPTS, "tools", "vision.py"), tool]
