"""Decode one shot: sequential RGBA frames already cropped, scaled to the shot's box, graded (ffmpeg pipe).

Frame k of a shot shows source time a + k / FPS * speed: `setpts` + `fps` resample any source rate (24/25/30/60,
29.97/59.94) and speed, VideoToolbox decodes. HLG (iPhone HDR) goes through the HDR -> SDR LUT.
"""
from __future__ import annotations

import subprocess
import tempfile

import numpy as np

from cv import media
from cv.canvas import FPS, fps_arg

ROT = {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}


def _even(v: float) -> int:
    return int(round(v / 2)) * 2


def _floor_even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def box_size(view: dict) -> tuple[int, int]:
    return max(2, _even(view["box"][2])), max(2, _even(view["box"][3]))


def vf(path: str, view: dict, speed: float, grade: str) -> str:
    x, y, w, h = view["crop"]
    ow, oh = box_size(view)
    info = media.probe(path)
    smooth = bool(view.get("smooth")) and not info["image"]
    head = "" if info["image"] else f"setpts=(PTS-STARTPTS)/{speed:.6f}," + ("" if smooth else f"fps={fps_arg()},")
    # 24/25 fps or Auto-FPS clips in a 30 fps edit repeat a frame every few: "smooth" interpolates the missing ones
    # instead (motion-compensated, at output size so it stays affordable)
    mi = f"minterpolate=fps={fps_arg()}:mi_mode=mci:mc_mode=aobmc:vsbmc=1," if smooth else ""
    sw, sh = view.get("src_size", (x + w, y + h))
    cw, ch = min(_floor_even(w), _floor_even(sw)), min(_floor_even(h), _floor_even(sh))  # odd sizes: stay inside
    cx = min(max(0, int(x) // 2 * 2), max(0, (int(sw) - cw) // 2 * 2))
    cy = min(max(0, int(y) // 2 * 2), max(0, (int(sh) - ch) // 2 * 2))
    crop = f"{ROT.get(view.get('rot', 0), '')}crop={cw}:{ch}:{cx}:{cy},"
    if info["hlg"]:
        return (f"{head}{crop}scale={ow}:{oh}:flags=lanczos:in_color_matrix=bt2020,{mi}format=rgb48le,"
                f"lut3d=file='{media.hlg_lut()}':interp=tetrahedral,format=rgba")
    g = f"{grade}," if grade else ""
    # the input range comes from the stream (full-range iPhone Cinematic / yuvj clips must not be stretched)
    return (f"{head}{crop}scale={ow}:{oh}:flags=lanczos:in_color_matrix=bt709:out_range=full,"
            f"{mi}{g}format=rgba")


class Reader:
    """Sequential frames of `shot` from its k0-th frame on."""

    def __init__(self, path: str, shot, k0: int, grade: str = ""):
        self.w, self.h = box_size(shot.view)
        self.size = self.w * self.h * 4
        self.left = shot.n - k0
        info = media.probe(path)
        start = shot.src_time(k0)
        if info["image"]:
            cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf(path, shot.view, 1.0, grade), "-frames:v", "1",
                   "-f", "rawvideo", "-"]
        else:
            dur = (shot.n - k0 + 2) / FPS * shot.speed + 0.3
            cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{max(0.0, start):.4f}", "-t",
                   f"{dur:.4f}", "-i", path, "-map", "0:v:0", "-vf", vf(path, shot.view, shot.speed, grade),
                   "-frames:v", str(self.left), "-f", "rawvideo", "-"]
        self.err = tempfile.TemporaryFile()
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.err, bufsize=self.size * 2)
        self.last, self.path, self.start = None, path, start

    def next(self) -> np.ndarray:
        buf = self.p.stdout.read(self.size) if self.p.stdout else b""
        if len(buf) == self.size:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)
        elif self.last is None:  # nothing decoded at all: say why instead of rendering black
            self.p.wait()
            self.err.seek(0)
            msg = self.err.read().decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError(f"ffmpeg decoded no frame of {self.path} at {self.start:.2f} s: {' | '.join(msg)}")
        return self.last

    def close(self) -> None:
        try:
            if self.p.stdout:
                self.p.stdout.close()
            self.p.kill()
            self.p.wait(timeout=5)
            self.err.close()
        except Exception:
            pass


def frame_at(path: str, shot, k: int, grade: str = "") -> np.ndarray:
    r = Reader(path, shot, k, grade)
    try:
        return r.next()
    finally:
        r.close()
