"""Small ffmpeg helpers shared by tts.py, finish.py and qa.py (48 kHz float audio, loudness, probing)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
from scipy.io import wavfile

SR = 48000


def read_audio(path: str, channels: int = 2) -> np.ndarray:
    """Any audio file -> float32 array (channels, n) at 48 kHz."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(channels), "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).reshape(-1, channels).T.copy()


def write_wav(path: str, audio: np.ndarray) -> None:
    """(channels, n) or (n,) float -> 32-bit float WAV."""
    a = audio if audio.ndim == 1 else audio.T
    wavfile.write(path, SR, np.ascontiguousarray(a, dtype=np.float32))


def loudness(path: str) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) via ffmpeg ebur128."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f",
                        "null", "-"], capture_output=True, text=True).stderr  # -vn: never decode the picture
    s = r[r.rfind("Summary:"):]
    return float(s.split("I:")[1].split("LUFS")[0]), float(s.split("Peak:")[1].split("dBFS")[0])


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def stream_info(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration:format=size,duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def normalize(src: str, dst: str, target: float = -14.0, limit: float = 0.79, pcm: str = "pcm_s24le") -> None:
    """Gain to the target integrated loudness, then a peak limiter (limit = linear ceiling)."""
    i, _ = loudness(src)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af",
                    f"volume={target - i:.2f}dB,alimiter=limit={limit}:attack=1:release=60:level=false",
                    "-ar", str(SR), "-c:a", pcm, dst], check=True)
