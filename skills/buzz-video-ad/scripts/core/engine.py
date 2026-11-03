"""Frame compositor for buzz-video-ad.

Shots and sources with aspect-aware framing (cover or blurred fill), clean-up (blur patches, portrait
focus, soft bottom band), colour grading and light effects.

Coordinates of everything that refers to a SOURCE clip are normalized to the upright source frame:
x 0..1 (left -> right), y 0..1 (top -> bottom). Radii (patch / focus ellipses) are in units of the
source WIDTH, so rx == ry is a circle. Camera keys are (t relative to shot start, zoom, cx, cy) where
zoom 1.0 = the source just fills the output (cover) or the foreground box (blur fit).

Frames are float32 H x W x 3 in 0..1.

Fonts/text sprites (core.text) and easing/colour/sprite helpers (core.draw) are re-exported
here, so templates only need  from core import engine as E.
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

from core.canvas import FPS, H, W
from core.draw import (clamp01, ease_out_cubic, ease_in_cubic, ease_in_out_sine, ease_in_out_cubic, ease_out_back,
    smoothstep, window, leave, rgb, mix, premult, blit, blit_img, solid_sprite, shadow_for, with_shadow, tight)  # noqa: F401 (re-exported)
from core.text import (FONT_KEYS, font, SQUEEZE, OPEN, _advance, text_img, text_width, fit_size, vtext_img,
    pill_img, sparkle_img, TextLine, VText)  # noqa: F401 (re-exported)

LUT_HLG = os.path.expanduser(os.environ.get("SNS_HLG_LUT", "~/.local/share/sns-ad-video/hlg2sdr_65.cube"))
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


# ---------------------------------------------------------------- shots
@dataclass(frozen=True)
class Shot:
    """One use of a source clip on the output timeline. See module docstring for coordinate units."""
    name: str
    path: str
    t0: float
    t1: float
    src_in: float = 0.0
    speed: float = 1.0  # source seconds per output second (0.5 = half-speed slow motion)
    rot: int = 0  # extra rotation after the file's own: 0 / 90 / 180 / 270 (clockwise)
    cam: tuple = ((0.0, 1.0, 0.5, 0.5),)
    fit: str = "cover"  # cover | blur (whole source in a box over a blurred copy of itself)
    grade: str = "natural"
    exposure: float = 1.0
    tint: float = 0.0  # added colour temperature (negative = cooler)
    patches: tuple = ()  # (cx, cy, rx, ry, strength) or (((src_t, cx, cy), ...), rx, ry, strength) tracked
    focus: tuple | None = None  # (cx, cy, rx, ry, strength): sharp inside, blurred outside
    soft_below: float = 0.0  # >0: blur the source below this y (prints / logos along the bottom edge)
    pre: float = 0.0  # handle before t0 (output s) for transitions
    post: float = 0.0  # handle after t1
    shake: float = 0.0  # handheld wobble amplitude (output px)
    hdr: bool = False  # HLG source: converted to SDR BT.709 with the baked LUT while decoding
    flash_guard: bool = False  # hold the previous frame over single-frame flashes (laser devices)


def cam_lin(z0: float, z1: float, c0: tuple, c1: tuple, dur: float, pre: float = 0.0, post: float = 0.0) -> tuple:
    """Linear (eased) camera move over the shot including its handles."""
    return ((-pre, z0, *c0), (dur + post, z1, *c1))


def cam_at(s: Shot, t: float) -> tuple[float, float, float]:
    rel = t - s.t0
    keys = s.cam
    if rel <= keys[0][0]:
        return keys[0][1:]
    for a, b in zip(keys, keys[1:]):
        if rel <= b[0]:
            u = ease_in_out_sine((rel - a[0]) / max(b[0] - a[0], 1e-6))
            return tuple(a[k] + (b[k] - a[k]) * u for k in (1, 2, 3))
    return keys[-1][1:]


def _wobble(t: float, seed: float) -> float:
    return (math.sin(2 * math.pi * 0.37 * t + seed) * 0.55 + math.sin(2 * math.pi * 0.83 * t + seed * 2.1) * 0.3
            + math.sin(2 * math.pi * 1.9 * t + seed * 3.7) * 0.15)


@lru_cache(maxsize=None)
def probe(path: str) -> dict:
    """Upright size, fps, duration and HLG flag of a video or image."""
    if path.lower().endswith(IMAGE_EXT):
        w, h = ImageOps.exif_transpose(Image.open(path)).size
        return {"w": w, "h": h, "fps": 0.0, "duration": 1e9, "hlg": False, "image": True}
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,color_transfer:stream_side_data=rotation:format=duration",
                          "-of", "default=nw=1", path], capture_output=True, text=True, check=True).stdout
    kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    w, h = int(kv["width"]), int(kv["height"])
    if abs(int(float(kv.get("rotation", 0) or 0))) in (90, 270):
        w, h = h, w
    num, den = (kv.get("r_frame_rate", "30/1") + "/1").split("/")[:2]
    return {"w": w, "h": h, "fps": float(num) / max(float(den), 1.0), "duration": float(kv.get("duration", 0) or 0),
            "hlg": kv.get("color_transfer") == "arib-std-b67", "image": False}


@dataclass(frozen=True)
class Window:
    """Visible part of a decoded source frame and where it lands in the output."""
    left: float
    top: float
    ww: float
    wh: float
    vw: int  # view (foreground) size in output px
    vh: int
    dw: int  # decoded source size
    dh: int

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        """Normalized source point -> view px."""
        return (x * self.dw - self.left) * self.vw / self.ww, (y * self.dh - self.top) * self.vh / self.wh

    def scale(self) -> float:
        """View px per source-width unit."""
        return self.dw * self.vw / self.ww


class Source:
    """Sequential reader of one shot's source (or random access in still mode, or a still image)."""

    def __init__(self, s: Shot):
        self.s = s
        self.start, self.end = s.t0 - s.pre, s.t1 + s.post
        info = probe(s.path)
        sw, sh = (info["h"], info["w"]) if s.rot in (90, 270) else (info["w"], info["h"])
        self.is_image = info["image"]
        self.clip_end = info["duration"] - 1.5 / max(info["fps"], 1.0)  # last safely seekable time
        self.hdr = s.hdr or info["hlg"]
        self.view_w, self.view_h, scale = _view_box(sw, sh, s.fit)
        k = min(1.0, scale * 1.5)  # decode at 1.5x the fitted size (room to zoom), never above native
        self.dw, self.dh = max(2, int(round(sw * k / 2)) * 2), max(2, int(round(sh * k / 2)) * 2)
        self.proc, self.k, self.last = None, -1, None
        self.still_mode = False
        self._hist: list[tuple[float, float]] = []
        self._rejected = 0

    def src_time(self, t: float) -> float:
        return self.s.src_in + (t - self.s.t0) * self.s.speed

    def _vf(self) -> str:
        rot = {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}.get(self.s.rot, "")
        if self.hdr:
            if not os.path.exists(LUT_HLG):
                raise SystemExit(f"HLG source {self.s.path}: run prep_hdr.py once to create {LUT_HLG}")
            return (f"{rot}scale={self.dw}:{self.dh}:flags=lanczos:in_color_matrix=bt2020:in_range=tv,"
                    f"format=rgb48le,lut3d=file='{LUT_HLG}':interp=tetrahedral,format=rgb24")
        return f"{rot}scale={self.dw}:{self.dh}:flags=lanczos,format=rgb24"

    def frame(self, t: float) -> np.ndarray:
        n = self.dw * self.dh * 3
        if self.is_image:
            if self.last is None:
                img = ImageOps.exif_transpose(Image.open(self.s.path)).convert("RGB")
                if self.s.rot:
                    img = img.rotate(-self.s.rot, expand=True)
                self.last = np.asarray(img.resize((self.dw, self.dh), Image.LANCZOS))
            return self.last
        k = int(math.floor((t - self.start) * FPS + 1e-6))  # floor, never round (round duplicates/skips frames)
        if self.still_mode:
            if self.s.flash_guard:  # same flash filter as the video: stream the frames just before t
                self.close()
                self.last, self._hist, self._rejected = None, [], 0
                self._open(max(0, k - 8), k - max(0, k - 8) + 1)
                self._read_to(k)
                self.close()
                if self.last is not None:
                    return self.last
            # hold the last frame past the clip end, like the streaming reader does
            st = min(max(0.0, self.src_time(t)), max(0.0, self.clip_end))
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{st:.4f}", "-i", self.s.path, "-frames:v", "1", "-vf", self._vf(),
                   "-f", "rawvideo", "-"]
            buf = subprocess.run(cmd, capture_output=True, check=True).stdout
            if len(buf) < n:
                raise RuntimeError(f"{os.path.basename(self.s.path)}: no frame at {st:.2f}s (clip too short?)")
            return np.frombuffer(buf[:n], np.uint8).reshape(self.dh, self.dw, 3)
        if self.proc is None:  # open at the first requested frame (render chunks may start mid-shot)
            self._open(max(0, k))
        self._read_to(k)
        if self.last is None:
            raise RuntimeError(f"{os.path.basename(self.s.path)}: no frames at {t:.2f}s (check in-point/length)")
        return self.last

    def _open(self, k0: int, frames: int = 0) -> None:
        """Stream output frames k0, k0 + 1, ... (at most `frames` of them when > 0)."""
        ss = max(0.0, self.src_time(self.start + k0 / FPS))
        dur = (self.end - self.start - k0 / FPS) * self.s.speed + 0.5
        vf = f"setpts={1 / self.s.speed:.6f}*PTS,fps={FPS},{self._vf()}"
        limit = ["-frames:v", str(frames)] if frames else []
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.4f}", "-t", f"{dur:.4f}", "-i",
                                      self.s.path, "-vf", vf, *limit, "-f", "rawvideo", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.k = k0 - 1

    def _read_to(self, k: int) -> None:
        n = self.dw * self.dh * 3
        while self.k < k:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(self.dh, self.dw, 3)
            if not (self.s.flash_guard and self._is_flash(fr)):
                self.last = fr
            self.k += 1

    def _is_flash(self, fr: np.ndarray) -> bool:
        """Single-frame flashes (e.g. laser pulses): hold the previous frame instead."""
        sub = fr[::24, ::24].astype(np.float32)
        stat = (float(sub.mean()), float((sub[..., 0] - sub[..., 1]).mean()))
        if len(self._hist) >= 2 and self._rejected < 3:
            m = float(np.median([h[0] for h in self._hist]))
            r = float(np.median([h[1] for h in self._hist]))
            if stat[0] - m > 7.0 or stat[1] - r > 7.0:
                self._rejected += 1
                return True
        self._hist = (self._hist + [stat])[-6:]
        self._rejected = 0
        return False

    def close(self) -> None:
        if self.proc:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
            self.proc = None


def _view_box(sw: int, sh: int, fit: str) -> tuple[int, int, float]:
    """(view width, view height, output px per native source px at zoom 1)."""
    if fit == "blur":
        c = min(W / sw, H / sh)
        return max(2, int(round(sw * c))), max(2, int(round(sh * c))), c
    return W, H, max(W / sw, H / sh)


def crop_window(src: Source, t: float, extra_zoom: float = 1.0) -> Window:
    s = src.s
    z, cx, cy = cam_at(s, t)
    z = max(1.0, z * extra_zoom)
    vw, vh = src.view_w, src.view_h
    if s.fit == "blur":
        ww, wh = src.dw / z, src.dh / z
    else:
        c = max(W / src.dw, H / src.dh)
        ww, wh = W / (c * z), H / (c * z)
    per_px = ww / vw  # decoded px per view px
    sx = _wobble(t, 1.3) * s.shake * per_px
    sy = _wobble(t, 4.1) * s.shake * per_px
    left = min(max(cx * src.dw - ww / 2 + sx, 0.0), src.dw - ww)
    top = min(max(cy * src.dh - wh / 2 + sy, 0.0), src.dh - wh)
    return Window(left, top, ww, wh, vw, vh, src.dw, src.dh)


def view(src: Source, t: float, extra_zoom: float = 1.0) -> tuple[np.ndarray, Window, np.ndarray]:
    """Camera window of the current source frame resampled to the view size (uint8), its window, raw frame."""
    raw = src.frame(t)
    win = crop_window(src, t, extra_zoom)
    box = (win.left, win.top, win.left + win.ww, win.top + win.wh)
    return np.asarray(Image.fromarray(raw).resize((win.vw, win.vh), Image.BICUBIC, box=box)), win, raw


# ---------------------------------------------------------------- clean-up
def _ellipse_d(cx, cy, rx, ry, x0, y0, x1, y1) -> np.ndarray:
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return np.sqrt(((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2)


def downsample(x: np.ndarray, f: int) -> np.ndarray:
    """Box-average by an integer factor (edge rows/cols that do not fill a block are dropped)."""
    h, w = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:h, :w].reshape(h // f, f, w // f, f, *x.shape[2:]).mean(axis=(1, 3))


def upsample(small: np.ndarray, w: int, h: int) -> np.ndarray:
    chans = [np.asarray(Image.fromarray(small[..., c].astype(np.float32), "F").resize((w, h), Image.BILINEAR))
             for c in range(small.shape[2])]
    return np.dstack(chans)


def patch_at(p: tuple, src_t: float) -> tuple[float, float, float, float, float]:
    """Static (cx, cy, rx, ry, strength) or tracked (keys, rx, ry, strength) -> values at source time."""
    if isinstance(p[0], tuple):
        keys, rx, ry, strength = p
        ts = [k[0] for k in keys]
        return (float(np.interp(src_t, ts, [k[1] for k in keys])), float(np.interp(src_t, ts, [k[2] for k in keys])),
                rx, ry, strength)
    return p


def cleanup(img: np.ndarray, s: Shot, win: Window, t: float) -> np.ndarray:
    """Blur patches, focus (portrait) blur and soft bottom band, mapped from source coords to the view."""
    if not s.patches and s.focus is None and not s.soft_below:
        return img
    out = img.astype(np.float32)
    vh, vw = out.shape[:2]
    k = win.scale()
    zoomish = win.vw / win.ww * win.dw / 1080.0  # blur strengths are given for a 1080-wide source at zoom 1
    src_t = s.src_in + (t - s.t0) * s.speed
    for p in s.patches:
        pcx, pcy, prx, pry, strength = patch_at(p, src_t)
        cx, cy = win.to_view(pcx, pcy)
        rx, ry, sg = prx * k, pry * k, max(1.0, strength * zoomish)
        x0, x1 = int(max(0, cx - rx * 1.6)), int(min(vw, cx + rx * 1.6))
        y0, y1 = int(max(0, cy - ry * 1.6)), int(min(vh, cy + ry * 1.6))
        if x0 >= x1 or y0 >= y1:
            continue
        region = out[y0:y1, x0:x1]
        m = np.clip((1.3 - _ellipse_d(cx, cy, rx, ry, x0, y0, x1, y1)) / 0.3, 0, 1)[..., None]
        out[y0:y1, x0:x1] = region * (1 - m) + gaussian_filter(region, (sg, sg, 0)) * m
    if s.focus is not None:
        fcx, fcy, frx, fry, strength = s.focus
        cx, cy = win.to_view(fcx, fcy)
        small = downsample(out, 2)
        blurred = upsample(gaussian_filter(small, (strength * zoomish / 2, strength * zoomish / 2, 0)), vw, vh)
        m = np.clip((_ellipse_d(cx, cy, frx * k, fry * k, 0, 0, vw, vh) - 1.0) / 0.35, 0, 1)[..., None]
        out = out * (1 - m) + blurred * m
    if s.soft_below:
        _, y0 = win.to_view(0.0, s.soft_below)
        out = soften_below(out, int(y0))
    return np.clip(out, 0, 255).astype(np.uint8)


def soften_below(img: np.ndarray, y0: int, ramp: int = 180) -> np.ndarray:
    """Progressively blur everything below y0 (ramp px soft edge)."""
    h, w = img.shape[:2]
    ys = max(0, y0 - ramp)
    if ys >= h - 8:
        return img
    region = img[ys:]
    small = downsample(region, 4)
    blurred = upsample(gaussian_filter(small, (7, 7, 0)), w, region.shape[0])
    k = np.clip((np.arange(region.shape[0], dtype=np.float32) + ys - (y0 - ramp)) / ramp, 0, 1)[:, None, None]
    out = img.copy()
    out[ys:] = region * (1 - k) + blurred * k
    return out


# ---------------------------------------------------------------- shot rendering
def render_shot(src: Source, t: float, extra_zoom: float = 1.0, grade_name: str | None = None) -> np.ndarray:
    """Final graded W x H float frame of a shot at output time t."""
    s = src.s
    fg, win, raw = view(src, t, extra_zoom)
    fg = cleanup(fg, s, win, t)
    frame = _blur_fill(src, fg, raw) if s.fit == "blur" else fg
    return grade(frame, grade_name or s.grade, s.exposure, s.tint)


_SHADE: dict = {}


def _blur_fill(src: Source, fg: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Foreground box centred over a blurred, dimmed cover-fit copy of the same frame."""
    small = downsample(raw.astype(np.float32), 8)
    sh_, sw_ = small.shape[:2]
    c = max(W / sw_, H / sh_)
    cw, ch = W / c, H / c
    box = ((sw_ - cw) / 2, (sh_ - ch) / 2, (sw_ + cw) / 2, (sh_ + ch) / 2)
    bg_small = np.dstack([np.asarray(Image.fromarray(small[..., i], "F").resize((W // 8, H // 8), Image.BILINEAR, box=box))
                          for i in range(3)])
    bg = upsample(gaussian_filter(bg_small, (3.0, 3.0, 0)), W, H)
    luma = (bg @ LUMA)[..., None]
    bg = (luma + (bg - luma) * 0.7) * 0.55
    x0, y0 = (W - fg.shape[1]) // 2, (H - fg.shape[0]) // 2
    key = fg.shape[:2]
    if key not in _SHADE:  # soft drop shadow around the box, same for every frame of this size
        mask = np.zeros((H, W), np.float32)
        mask[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = 1
        _SHADE[key] = gaussian_filter(mask, 18)[..., None]
    bg = bg * (1 - 0.45 * _SHADE[key])
    bg[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    return bg.astype(np.float32)


# ---------------------------------------------------------------- grading
PRESETS = {
    "natural": dict(exp=1.02, con=1.03, sat=0.98, temp=0.015, lift=0.012, bloom=0.07, mist=0.03, vig=0.12),
    "beauty": dict(exp=1.03, con=1.05, sat=0.96, temp=0.025, lift=0.02, bloom=0.20, mist=0.09, vig=0.26),
    "soft": dict(exp=1.03, con=0.97, sat=0.90, temp=0.022, lift=0.03, bloom=0.16, mist=0.07, vig=0.16,
                 tone=((0.010, 0.002, -0.010), (0.016, 0.008, -0.010))),
    "bright": dict(exp=1.05, con=1.02, sat=0.95, temp=0.015, lift=0.02, bloom=0.10, mist=0.04, vig=0.10,
                   tone=((0.0, 0.0, 0.0), (0.010, 0.006, -0.006))),
    "pop": dict(exp=1.04, con=1.08, sat=1.06, temp=0.02, lift=0.01, bloom=0.12, mist=0.03, vig=0.14),
    "cine": dict(exp=0.97, con=1.10, sat=0.84, temp=0.03, lift=0.035, bloom=0.24, mist=0.11, vig=0.40,
                 tone=((-0.012, 0.004, 0.018), (0.02, 0.008, -0.02))),
    # muted everyday scenes / counselling room / daylight finale (from a salon ad)
    "daily": dict(exp=1.0, con=1.04, sat=0.88, temp=0.0, lift=0.018, bloom=0.07, mist=0.05, vig=0.24),
    "counsel": dict(exp=1.02, con=1.03, sat=0.93, temp=0.02, lift=0.02, bloom=0.12, mist=0.07, vig=0.24),
    "daylight": dict(exp=1.01, con=1.03, sat=0.94, temp=0.015, lift=0.02, bloom=0.12, mist=0.07, vig=0.22),
    "none": dict(exp=1.0, con=1.0, sat=1.0, temp=0.0, lift=0.0, bloom=0.0, mist=0.0, vig=0.0),
}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_VIG: dict = {}


def vignette(h: int, w: int) -> np.ndarray:
    """0 in the centre .. 1 in the corners, (h, w, 1) float32; cached per size."""
    if (h, w) not in _VIG:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        v = (((xx - w / 2) / (w / 2)) ** 2 * 0.8 + ((yy - h / 2) / (h / 2)) ** 2) ** 1.25
        _VIG[(h, w)] = (v / v.max())[..., None].astype(np.float32)
    return _VIG[(h, w)]


def grade(img: np.ndarray, name: str, exposure: float = 1.0, tint: float = 0.0,
          vig_scale: float = 1.0) -> np.ndarray:
    """uint8 or 0..255 float RGB (any size) -> graded float 0..1 (exposure, temperature, saturation, contrast,
    lift, split-tone, highlight bloom + mist, vignette x vig_scale)."""
    p = PRESETS[name]
    h, w = img.shape[:2]
    temp = p["temp"] + tint
    x = img.astype(np.float32) * (p["exp"] * exposure / 255.0)
    x[..., 0] *= 1 + temp
    x[..., 1] *= 1 + temp * 0.2
    x[..., 2] *= 1 - temp
    luma = (x @ LUMA)[..., None]
    x = luma + (x - luma) * p["sat"]
    x = 0.5 + (x - 0.5) * p["con"]
    x = p["lift"] + x * (1 - p["lift"])
    if "tone" in p:
        sh, hi = (np.array(v, np.float32) for v in p["tone"])
        lum = np.clip(luma, 0, 1)
        x += sh * (1 - lum) ** 2 + hi * lum ** 2
    np.clip(x, 0, 1, out=x)
    if p["bloom"] or p["mist"]:
        small = downsample(x, 4)
        hl = np.clip((small - 0.62) / 0.38, 0, 1) ** 1.5 * small
        glow = gaussian_filter(hl, (9, 9, 0)) * p["bloom"] + gaussian_filter(small, (5, 5, 0)) * p["mist"]
        x = 1 - (1 - x) * (1 - np.clip(upsample(glow, w, h), 0, 1))
    if p["vig"]:
        x *= 1 - vig_scale * p["vig"] * vignette(h, w)
    return x


# ---------------------------------------------------------------- effects
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)  # canvas pixel grid (light_leak, flat_bg)
_GRAIN: list | None = None


def add_grain(frame: np.ndarray, t: float, amount: float) -> None:
    """Film grain (in place), strongest in the mid-tones."""
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(11)
        _GRAIN = [gaussian_filter(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), 0.6) for _ in range(8)]
    g = _GRAIN[int(t * FPS) % 8]
    g = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)[:H, :W, None]
    frame += g * amount * (0.35 + 0.65 * (1 - np.abs(frame - 0.5) * 2))


def light_leak(t: float, strength: float, seed: float = 0.0) -> np.ndarray:
    """Warm drifting light-leak layer (screen-blend it)."""
    h, w = H // 8, W // 8
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w, 3), np.float32)
    for k, (col, sp) in enumerate((((1.0, 0.62, 0.32), 0.07), ((1.0, 0.82, 0.55), 0.05), ((0.95, 0.45, 0.35), 0.09))):
        px = w * (0.5 + 0.55 * math.sin(t * sp * 6.28 + seed + k * 2.1))
        py = h * (0.5 + 0.5 * math.cos(t * sp * 5.1 + seed * 1.7 + k))
        r = max(w, h) * (0.32 + 0.09 * k)
        g = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * r * r)))
        out += g[..., None] * np.array(col, np.float32) * (0.6 - 0.15 * k)
    return np.clip(upsample(out, W, H) * strength, 0, 1)


def screen(frame: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return 1 - (1 - frame) * (1 - layer)


def ring_sprite(radius: float, thickness: float, color, progress: float = 1.0, start_deg: float = -90) -> np.ndarray:
    r = int(radius + thickness * 2 + 2)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + yy ** 2)
    a = np.clip(1 - np.abs(d - radius) / (thickness / 2 + 0.75), 0, 1)
    if progress < 1:
        ang = (np.degrees(np.arctan2(yy, xx)) - start_deg) % 360
        a *= np.clip((progress * 360 - ang) / 2 + 0.5, 0, 1)
    return solid_sprite(a, color)


def iris(frame_a: np.ndarray, frame_b: np.ndarray, cx: float, cy: float, radius: float, feather: float = 3.0,
         ring=None) -> np.ndarray:
    """frame_b inside a circle over frame_a; optional (thickness, colour) ring on the edge."""
    d = np.sqrt((_xx - cx) ** 2 + (_yy - cy) ** 2)
    m = np.clip((radius - d) / feather + 0.5, 0, 1)[..., None]
    out = frame_a * (1 - m) + frame_b * m
    if ring:
        th, col = ring
        ra = np.clip(1 - np.abs(d - radius) / (th / 2 + 0.75), 0, 1)[..., None]
        out = out * (1 - ra) + np.array(rgb(col), np.float32) / 255 * ra
    return out


def flat_bg(center, edge, seed: int = 3, focus_y: float = 0.45) -> np.ndarray:
    """Soft radial paper/night background between two colours, with very light noise."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(((_xx - W / 2) / W) ** 2 + ((_yy - H * focus_y) / H) ** 2)
    c, e = np.array(rgb(center), np.float32) / 255, np.array(rgb(edge), np.float32) / 255
    k = np.clip(r / 0.75, 0, 1)[..., None] ** 1.5
    return (c * (1 - k) + e * k + rng.normal(0, 0.006, (H, W, 1))).astype(np.float32)


def to_u8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
