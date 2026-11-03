"""Finding the frames for thumbnails: face crops of the whole talk, the sharpest frame near a time, role picks.

Lessons from the first YouTube thumbnails (all from a 1.5 s face-crop scan of the 11-minute talk):
  * look at faces over the WHOLE talk, not at a few guessed times -> one frame every `STEP` s
  * then take the sharpest frame (Laplacian variance of the face) within +-0.3 s of the pick
  * reject blurred hands: moving hands smear, so the frame-to-frame change below the face must be small
  * the face must be whole, frontal enough and large; the body must not be cut by the frame right under the chin
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", ".."))
STEP = 1.5
BURST = 0.3
ROT = {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}
CROP = 192        # face crops are compared at this size
ROLES = ("reaction", "before_after", "authority")


@dataclass
class Frame:
    t: float                       # source seconds
    path: str
    face: tuple | None = None      # px (x, y, w, h), Vision box brows..chin
    yaw: float = 0.0
    sharp: float = 0.0             # Laplacian variance of the face crop
    motion: float = 0.0            # mean change below the face vs the neighbouring frames (0 = still)
    size: float = 0.0              # face height / frame height
    ok: bool = False               # usable as a thumbnail face
    why: str = ""
    eye: float = 0.0               # eye height / width (about 0.32 open, < 0.19 blinking)
    eye_l: float = 0.0
    eye_r: float = 0.0
    mouth_open: float = 0.0        # inner lips height / face height
    mouth_w: float = 0.0           # outer lips width / face width
    lift: float = 0.0              # mouth corners above the lip centre (smile > 0.015)
    brow: float = 0.0              # eyebrows above the eyes
    score: float = 0.0

    @property
    def blink(self) -> bool:
        return min(self.eye_l, self.eye_r) < 0.12 or self.eye < 0.19


# ---------------------------------------------------------------- decoding
def _hdr(path: str) -> str:
    """ffmpeg filters that turn iPhone HLG into SDR RGB frames (same LUT as the renderer), '' for SDR."""
    from cv import media
    if not media.probe(path)["hlg"]:
        return ""
    return (",scale=iw:ih:flags=lanczos:in_color_matrix=bt2020:in_range=tv,format=rgb48le,"
            f"lut3d=file='{media.hlg_lut()}':interp=tetrahedral,format=rgb24")


def _run_frames(path: str, rot: int, t0: float, dur: float, select: str, out_dir: str, ext: str = "jpg",
                q: str = "2") -> list[tuple[float, str]]:
    """Decode frames of [t0, t0 + dur] chosen by `select` (an ffmpeg select expression); [(source seconds, file)]."""
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.startswith("f") and old.endswith("." + ext):
            os.remove(os.path.join(out_dir, old))
    vf = f"{ROT.get(rot, '')}select='{select}',showinfo{_hdr(path)}"
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-hwaccel", "videotoolbox", "-ss", f"{max(0.0, t0):.3f}", "-t",
           f"{dur:.3f}", "-i", os.path.expanduser(path), "-map", "0:v:0", "-vf", vf, "-fps_mode", "vfr"]
    cmd += ["-q:v", q] if ext == "jpg" else []
    r = subprocess.run(cmd + [os.path.join(out_dir, f"f%04d.{ext}")], capture_output=True, text=True)
    times = [float(m) + max(0.0, t0) for m in re.findall(r"pts_time:\s*([0-9.]+)", r.stderr)]
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("f") and f.endswith("." + ext))
    return [(t, os.path.join(out_dir, f)) for t, f in zip(times, files)]


def scan_frames(path: str, rot: int, t0: float, t1: float, out_dir: str, step: float = STEP):
    return _run_frames(path, rot, t0, t1 - t0, f"isnan(prev_selected_t)+gte(t-prev_selected_t,{step})", out_dir)


def burst_frames(path: str, rot: int, t: float, out_dir: str, half: float = BURST):
    return _run_frames(path, rot, t - half, 2 * half, "1", out_dir)


def grab(path: str, rot: int, t: float, out: str) -> str:
    """One lossless PNG (for the matte) at source time t."""
    vf = f"{ROT.get(rot, '')}null{_hdr(path)}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", os.path.expanduser(path),
                    "-map", "0:v:0", "-vf", vf, "-frames:v", "1", out], check=True)
    return out


# ---------------------------------------------------------------- faces
def detect_faces(paths: list[str]) -> list[list[list[float]]]:
    """Apple Vision faces per image: [[x, y, w, h, conf, yaw], ...] normalised, origin top-left."""
    sys.path.insert(0, SCRIPTS)
    import faces
    with tempfile.TemporaryDirectory() as d:
        for i, p in enumerate(paths):
            os.symlink(os.path.abspath(p), os.path.join(d, f"{i + 1:06d}.jpg"))   # Vision sniffs the content
        rows = {}
        for line in subprocess.run([faces.binary(), d], capture_output=True, text=True, check=True).stdout.splitlines():
            r = json.loads(line)
            rows[r["f"]] = r["faces"]
    return [rows.get(f"{i + 1:06d}.jpg", []) for i in range(len(paths))]


def landmarks_binary() -> str:
    src = os.path.join(SCRIPTS, "tools", "landmarks.swift")
    out = os.path.expanduser("~/.local/share/content-video/bin/landmarks")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        if not shutil.which("swiftc"):
            raise SystemExit("swiftc not found: expressions need Apple Vision (Xcode command line tools)")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(["swiftc", "-O", src, "-o", out], check=True)
    return out


def detect_expressions(paths: list[str]) -> list[dict]:
    """Expression numbers of the largest face per image (tools/landmarks.swift); {} where there is none."""
    with tempfile.TemporaryDirectory() as d:
        for i, p in enumerate(paths):
            os.symlink(os.path.abspath(p), os.path.join(d, f"{i + 1:06d}.jpg"))
        rows = {}
        for line in subprocess.run([landmarks_binary(), d], capture_output=True, text=True,
                                   check=True).stdout.splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                rows[r["f"]] = max(r["faces"], key=lambda f: f["box"][3], default={})
    return [rows.get(f"{i + 1:06d}.jpg", {}) for i in range(len(paths))]


def _fill_expression(fr: "Frame", e: dict) -> None:
    for k in ("eye", "eye_l", "eye_r", "mouth_open", "mouth_w", "lift", "brow"):
        setattr(fr, k, float(e.get(k, 0.0)))


def _largest(fs: list[list[float]], size: tuple[int, int]):
    fs = [f for f in fs if f[4] > 0.5 and f[3] > 0.04]
    if not fs:
        return None, 0.0
    f = max(fs, key=lambda f: f[3])
    w, h = size
    return (f[0] * w, f[1] * h, f[2] * w, f[3] * h), float(f[5])


# ---------------------------------------------------------------- metrics
def lapvar(gray: np.ndarray) -> float:
    return float(ndimage.laplace(gray.astype(np.float32)).var())


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), np.float32)


def face_crop(img: Image.Image, face: tuple, grow: float = 1.9, size: int = CROP) -> Image.Image:
    x, y, w, h = face
    cx, cy, s = x + w / 2, y + h / 2 + h * 0.05, max(w, h) * grow
    box = (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))
    return img.crop(box).resize((size, size), Image.LANCZOS)


def measure(fr: Frame, size: tuple[int, int]) -> Image.Image | None:
    """Fill sharp / size from the frame's file; returns the face crop."""
    if fr.face is None:
        return None
    crop = face_crop(Image.open(fr.path).convert("RGB"), fr.face)
    fr.sharp = lapvar(_gray(crop))
    fr.size = fr.face[3] / size[1]
    return crop


def body_motion(paths: list[str], faces: list[tuple | None]) -> list[float]:
    """Per frame of a burst: the largest mean change to a neighbour in the region below the face (hands, arms)."""
    grays = []
    for p in paths:
        im = Image.open(p).convert("L")
        grays.append(np.asarray(im.resize((im.width // 4, im.height // 4), Image.BILINEAR), np.float32))
    ref = next((f for f in faces if f), None)
    out = []
    for i, g in enumerate(grays):
        f = faces[i] or ref
        h, w = g.shape
        x, y, fw, fh = (v / 4 for v in f) if f else (w * 0.3, h * 0.2, w * 0.4, h * 0.2)
        y0, y1 = int(min(h - 2, y + fh)), int(min(h, y + fh * 3.2))
        x0, x1 = int(max(0, x - fw * 1.2)), int(min(w, x + fw * 2.2))
        d = [float(np.abs(g[y0:y1, x0:x1] - grays[j][y0:y1, x0:x1]).mean()) for j in (i - 1, i + 1)
             if 0 <= j < len(grays)]
        out.append(max(d) if d else 0.0)
    return out


# ---------------------------------------------------------------- the scan
def usable(fr: Frame, frame_size: tuple[int, int], med_sharp: float) -> None:
    """Gate a frame as thumbnail face (sets ok / why)."""
    w, h = frame_size
    if fr.face is None:
        fr.ok, fr.why = False, "no face"
        return
    x, y, fw, fh = fr.face
    if fr.size < 0.07:
        fr.ok, fr.why = False, "face small"
    elif abs(fr.yaw) > 0.55:
        fr.ok, fr.why = False, "face turned"
    elif x < fw * 0.1 or x + fw > w - fw * 0.1 or y < fh * 0.6:
        fr.ok, fr.why = False, "face at the edge"
    elif fr.sharp < 0.55 * med_sharp:
        fr.ok, fr.why = False, "blurred"
    elif fr.blink:
        fr.ok, fr.why = False, "eyes shut"
    else:
        fr.ok, fr.why = True, ""


def scan(path: str, rot: int, t0: float, t1: float, work: str, step: float = STEP, refresh: bool = False) -> list[Frame]:
    """Every `step` s of [t0, t1]: face, sharpness, size, expression numbers. Cached in work/scan.json."""
    cache = os.path.join(work, "scan.json")
    key = [os.path.getmtime(os.path.expanduser(path)), t0, t1, step]
    if not refresh and os.path.exists(cache):
        data = json.load(open(cache))
        if data["key"] == key and all(os.path.exists(r["path"]) for r in data["frames"]):
            return [Frame(**{**r, "face": tuple(r["face"]) if r["face"] else None}) for r in data["frames"]]
    got = scan_frames(path, rot, t0, t1, os.path.join(work, "scan"), step)
    if not got:
        raise SystemExit(f"no frames decoded from {path} between {t0:.1f} and {t1:.1f} s")
    size = Image.open(got[0][1]).size
    frames = [Frame(t, p) for t, p in got]
    paths = [p for _, p in got]
    for fr, fs in zip(frames, detect_faces(paths)):
        fr.face, fr.yaw = _largest(fs, size)
        measure(fr, size)
    for fr, e in zip(frames, detect_expressions(paths)):
        _fill_expression(fr, e)
    med = float(np.median([fr.sharp for fr in frames if fr.face])) if any(fr.face for fr in frames) else 1.0
    for fr in frames:
        usable(fr, size, med)
    json.dump({"key": key, "size": size, "frames": [asdict(f) for f in frames]}, open(cache, "w"), indent=1)
    return frames


# ---------------------------------------------------------------- expression scores
def _z(values: np.ndarray) -> np.ndarray:
    """Robust z-score (median / MAD) so one talk's own range of expressions is the yardstick."""
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826 or float(values.std()) or 1.0
    return (values - med) / mad


def expression_scores(frames: list[Frame]) -> dict[str, np.ndarray]:
    """Per frame (same order): smile = wide mouth + lifted corners, reaction = open mouth / wide / raised brows."""
    f = lambda k: np.array([getattr(fr, k) for fr in frames])
    smile = _z(f("mouth_w")) + _z(f("lift")) + 0.5 * _z(f("mouth_open"))
    react = _z(f("mouth_open")) + 0.6 * _z(f("mouth_w")) + 0.6 * _z(f("brow"))
    return {"smile": smile, "reaction": react}


def sharpest(path: str, rot: int, t: float, work: str, role: str = "") -> Frame:
    """The best frame within +-BURST s of t: sharp face, eyes open, still hands (and the role's expression at its
    peak). Returns a Frame with a lossless PNG path."""
    burst = burst_frames(path, rot, t, os.path.join(work, "burst"))
    if not burst:
        raise SystemExit(f"no frames decoded near {t:.2f} s")
    paths = [p for _, p in burst]
    size = Image.open(paths[0]).size
    frames = [Frame(tt, p) for tt, p in burst]
    for fr, fs in zip(frames, detect_faces(paths)):
        fr.face, fr.yaw = _largest(fs, size)
        measure(fr, size)
    for fr, e in zip(frames, detect_expressions(paths)):
        _fill_expression(fr, e)
    motion = body_motion(paths, [fr.face for fr in frames])
    top = max((fr.sharp for fr in frames), default=1.0) or 1.0
    expr = expression_scores(frames)
    bonus = expr["smile"] if role == "authority" else expr["reaction"] if role == "reaction" else np.zeros(len(frames))
    for fr, m, b in zip(frames, motion, np.clip(bonus, -3, 3)):
        fr.motion = m
        fr.score = (fr.sharp / top if fr.face else -9.0) - 0.04 * m - (0.8 if fr.blink else 0.0) + 0.06 * float(b)
    best = max(frames, key=lambda f: f.score)
    png = os.path.join(work, "frames", f"t{best.t:08.3f}.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    grab(path, rot, best.t, png)
    best.path = png
    best.face, best.yaw = _largest(detect_faces([png])[0], size)
    if best.face is None:
        raise SystemExit(f"no face found at {best.t:.2f} s (frame {png})")
    return best


# ---------------------------------------------------------------- roles
def quality(fr: Frame) -> float:
    return fr.sharp * (0.6 + fr.size) * (1.0 - min(abs(fr.yaw), 0.55) * 0.8) if fr.ok else 0.0


def pick_roles(frames: list[Frame], want: tuple = ROLES, gap: float = 3.0) -> dict[str, Frame]:
    """reaction = the most expressive face, authority = the broadest smile, before_after = the cleanest frame."""
    good = [f for f in frames if f.ok]
    if not good:
        raise SystemExit("no usable face in the scan (all frames blurred / turned / small / blinking)")
    qual = np.array([quality(f) for f in good])
    qual = qual / (qual.max() or 1.0)
    expr = expression_scores(good)
    scores = {"reaction": expr["reaction"] + 2.0 * qual,
              "authority": expr["smile"] + 2.0 * qual,
              "before_after": 3.0 * qual + 0.5 * np.clip(expr["smile"], -1, 2)}
    out: dict[str, Frame] = {}
    for role in want:
        order = np.argsort(-scores[role])
        for i in order:
            fr = good[int(i)]
            if all(abs(fr.t - o.t) >= gap for o in out.values()):
                out[role] = fr
                break
        else:
            out[role] = good[int(order[0])]
    return out


# ---------------------------------------------------------------- the sheet Claude looks at
def face_sheet(frames: list[Frame], out: str, picks: dict[str, Frame] | None = None, cols: int = 10,
               cell: int = 150) -> str:
    """Face crops of the whole scan with time and expression numbers; picks are boxed. Choose faces by looking,
    then re-run with --times."""
    mark = {id(f): role for role, f in (picks or {}).items()}
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    for i, fr in enumerate(frames):
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        if fr.face is not None and os.path.exists(fr.path):
            sheet.paste(face_crop(Image.open(fr.path).convert("RGB"), fr.face, size=cell), (x, y + 30))
        d.text((x + 4, y + 3), f"{fr.t:.1f}s  {'ok' if fr.ok else fr.why}",
               fill=(255, 255, 255) if fr.ok else (255, 140, 120))
        d.text((x + 4, y + 16), f"mouth {fr.mouth_open:.2f}  smile {fr.lift * 100:+.1f}", fill=(170, 170, 170))
        if id(fr) in mark:
            d.rectangle((x + 1, y + 30, x + cell - 2, y + cell + 28), outline=(254, 219, 46), width=4)
            d.text((x + 6, y + cell + 12), mark[id(fr)], fill=(254, 219, 46))
    sheet.save(out, quality=88)
    return out
