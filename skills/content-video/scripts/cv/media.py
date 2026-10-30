"""ffmpeg / ffprobe helpers: source facts (upright size, fps, duration, HLG), audio extraction, loudness."""
from __future__ import annotations

import functools
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile

SR = 48000
LUT_PATHS = [os.path.expanduser(os.environ.get("CV_HLG_LUT", "~/.local/share/content-video/hlg2sdr_65.cube")),
             os.path.expanduser("~/.local/share/sns-ad-video/hlg2sdr_65.cube")]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic")


@functools.lru_cache(maxsize=None)
def probe(path: str) -> dict:
    """Upright w/h (rotation metadata applied), fps, duration, hlg, has_audio, image."""
    path = os.path.expanduser(path)
    if path.lower().endswith(IMAGE_EXT):
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(path))
        return {"w": im.width, "h": im.height, "fps": 0.0, "duration": 1e9, "hlg": False, "audio": False, "image": True}
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,color_transfer:"
                          "stream_side_data=rotation:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    # iPhone 4K HDR lists several side-data entries and only one carries the rotation
    rot = next((int(float(sd["rotation"])) for sd in v.get("side_data_list", []) if sd.get("rotation")), 0)
    w, h = (v["height"], v["width"]) if abs(rot) in (90, 270) else (v["width"], v["height"])
    num, den = (v["r_frame_rate"] + "/1").split("/")[:2]
    fps = float(num) / float(den)
    an, ad = (v.get("avg_frame_rate", "0/1") + "/1").split("/")[:2]
    avg = float(an) / float(ad) if float(ad) else 0.0
    if avg and avg < fps * 0.8:  # iPhone Auto-FPS: a 120 / 60 fps timebase over 24-30 real frames
        fps = avg
    return {"w": int(w), "h": int(h), "fps": fps, "duration": float(j["format"]["duration"]),
            "hlg": v.get("color_transfer") == "arib-std-b67",
            "audio": any(s["codec_type"] == "audio" for s in j["streams"]), "image": False}


def hlg_lut() -> str:
    for p in LUT_PATHS:
        if os.path.exists(p):
            return p
    raise SystemExit("HLG (iPhone HDR) source: run  prep_hdr.py  once to create the HDR -> SDR LUT")


def extract_audio(path: str, out: str, sr: int = SR, mono: bool = True) -> str:
    """Source audio as float WAV (first audio stream), cached by path."""
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(os.path.expanduser(path)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.expanduser(path), "-map", "0:a:0", "-vn",
                        "-af", "aresample=async=1:first_pts=0",  # keep the audio's start offset (A/V sync)
                        "-ac", "1" if mono else "2", "-ar", str(sr), "-c:a", "pcm_f32le", out], check=True)
    return out


def read_wav(path: str) -> np.ndarray:
    """float64 (n,) mono or (2, n) stereo."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / (2 ** 31 if x.dtype == np.int32 else 2 ** 15 if x.dtype == np.int16 else 1)
    return x.T if x.ndim == 2 else x


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    a = x if x.ndim == 1 else x.T
    wavfile.write(path, sr, np.ascontiguousarray(a, dtype=np.float32))


def loudness(path: str) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) via ffmpeg ebur128 (audio only)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    s = r[r.rfind("Summary:"):]
    return float(s.split("I:")[1].split("LUFS")[0]), float(s.split("Peak:")[1].split("dBFS")[0])


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())
