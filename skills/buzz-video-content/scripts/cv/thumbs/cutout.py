"""Speaker cut-out of a still frame: Apple Vision matte -> clean alpha -> sticker.

Generalised so no frame needs hand fixes:
  * the person matte (.accurate) has the best hair, the foreground-instance mask drops chair / table bits:
    trust only what both call foreground
  * small pockets the matte kept (wall between thumb and jaw, between arm and body) are cleared when the two
    models disagree or the pixels look like the wall
  * edge pixels carry the wall colour: F = (I - (1 - a) B) / a
  * sticker = white outline + thin black edge + soft drop shadow (YouTube thumbnails read like stickers)
"""
from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import skia
from PIL import Image
from scipy import ndimage

from cv import vision

HERE = os.path.dirname(os.path.abspath(__file__))


def matte_frames(frames: Sequence[str]) -> None:
    """Write <frame stem>_person.png and _fg<N>.png next to every frame that has none yet (one process)."""
    todo = [f for f in frames if not os.path.exists(os.path.splitext(f)[0] + "_person.png")]
    if not todo:
        return
    args = vision.command("lift")
    for f in todo:
        stem = os.path.splitext(f)[0]
        for old in glob.glob(stem + "_fg*.png"):
            os.remove(old)
        args += [f, stem]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cut-out failed: {(r.stderr or r.stdout).strip()[:300]}")


@dataclass(frozen=True)
class Cutout:
    rgb: np.ndarray        # (h, w, 3) float32 0..255, edge-decontaminated and lightly graded
    alpha: np.ndarray      # (h, w) float32 0..1
    face: tuple            # Vision box brows..chin in px: x, y, w, h
    top: float             # hair top (px)
    head_w: float          # widest row of the head (hair, ears) in px

    @property
    def chin(self) -> float:
        return self.face[1] + self.face[3]

    @property
    def head_h(self) -> float:
        return self.chin - self.top

    @property
    def face_cx(self) -> float:
        return self.face[0] + self.face[2] / 2


# ---------------------------------------------------------------- matte
def _mask(path: str, size: tuple[int, int], resample) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, resample), np.float32) / 255


def _alpha(frame: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """(alpha, the foreground-instance mask that is the person or None)."""
    h, w = shape
    stem = os.path.splitext(frame)[0]
    person = _mask(stem + "_person.png", (w, h), Image.BICUBIC)
    fgs = [_mask(p, (w, h), Image.BILINEAR) for p in sorted(glob.glob(stem + "_fg*.png"))]
    if not fgs:
        return person, None
    fg = max(fgs, key=lambda m: float(np.minimum(m, person).sum()))
    k = max(5, int(round(w / 125)) | 1)
    a = np.minimum(person, ndimage.maximum_filter(fg, size=k))        # what both call foreground
    lab, n = ndimage.label(a > 0.3)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
        a *= ndimage.binary_dilation(lab == 1 + int(np.argmax(sizes)), iterations=max(2, w // 480))
    return a, fg


def _clear_pockets(a: np.ndarray, fg: np.ndarray | None, face: tuple) -> np.ndarray:
    """Enclosed patches the person matte kept but the foreground mask calls background (wall between thumb and jaw,
    between arm and body). The face box is never touched (glasses lenses, teeth)."""
    if fg is None:
        return a
    solid = a > 0.5
    area = float(solid.sum())
    cand = ndimage.binary_opening(solid & (fg < 0.3), iterations=2)
    fx, fy, fw, fh = face
    cand[int(max(fy - fh * 0.9, 0)):int(fy + fh), int(max(fx - fw * 0.15, 0)):int(fx + fw * 1.15)] = False
    lab, n = ndimage.label(cand)
    out = a.copy()
    for i in range(1, n + 1):
        comp = lab == i
        size = float(comp.sum())
        if size < 30 or size > 0.03 * area:
            continue
        rim = ndimage.binary_dilation(comp, iterations=6) & ~comp
        if solid[rim].mean() > 0.92:                         # surrounded by the person: a hole, not an edge
            out *= 1 - ndimage.gaussian_filter(ndimage.binary_dilation(comp, iterations=2).astype(np.float32), 1.0)
    return out


def _bust(a: np.ndarray, cx: float, chin: float, head_w: float, head_h: float) -> np.ndarray:
    """Close-up selfies are cut by the frame right under the chin (shoulders wider than the picture). Cut the body
    ourselves instead, on a clean vertical line a little inside the frame, so no frame edge or smear shows up."""
    h, w = a.shape
    reach = 1.18 * head_w
    x = np.arange(w, dtype=np.float32)
    over = np.where(x < cx, (cx - x) - min(reach, 0.93 * cx), (x - cx) - min(reach, 0.93 * (w - cx)))
    below = np.clip((np.arange(h, dtype=np.float32) - (chin - 0.30 * head_h)) / (0.03 * head_h), 0, 1)
    return a * (1.0 - np.clip(over / 3.0, 0, 1)[None, :] * below[:, None])


def _decontaminate(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Remove the background colour from semi-transparent edge pixels (hair): F = (I - (1 - a) B) / a."""
    sigma = max(6.0, rgb.shape[1] / 160)
    inv = 1.0 - a
    bg = ndimage.gaussian_filter(rgb * inv[..., None], (sigma, sigma, 0))
    bg /= ndimage.gaussian_filter(inv, sigma)[..., None] + 1e-4
    fg = (rgb - inv[..., None] * bg) / np.maximum(a, 0.08)[..., None]
    edge = (a > 0.01) & (a < 0.99)
    out = rgb.copy()
    out[edge] = np.clip(fg[edge], 0, 255)
    return out


def _grade(rgb: np.ndarray, a: np.ndarray, face: tuple) -> np.ndarray:
    """Thumbnails want a little more punch: lift a dark face to a readable level, then contrast + saturation."""
    fx, fy, fw, fh = (int(v) for v in face)
    patch = rgb[max(fy, 0):fy + fh, max(fx, 0):fx + fw]
    luma = float((patch @ np.array([0.299, 0.587, 0.114], np.float32)).mean()) if patch.size else 128.0
    gamma = float(np.clip(np.log(150 / 255) / np.log(max(luma, 40.0) / 255), 0.78, 1.0))
    x = 255 * (np.clip(rgb, 0, 255) / 255) ** gamma
    x = (x - 128) * 1.05 + 128
    grey = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    return np.clip(grey + (x - grey) * 1.10, 0, 255)


def load(frame: str, face: tuple) -> Cutout:
    """Cut-out of a PNG frame (mattes made by matte_frames); face = Vision box in px (x, y, w, h)."""
    rgb = np.asarray(Image.open(frame).convert("RGB"), np.float32)
    h, w = rgb.shape[:2]
    a, fg = _alpha(frame, (h, w))
    a = _clear_pockets(a, fg, face)
    rows = np.where((a > 0.5).sum(1) > 0.012 * w)[0]
    top = float(rows[0]) if len(rows) else float(face[1])
    chin = face[1] + face[3]
    band = (a[int(top):int(top + 0.7 * (chin - top))] > 0.5).sum(1)      # hair and ears, above the shoulders
    head_w = float(np.percentile(band, 85)) if len(band) else float(face[2])
    a = _bust(a, face[0] + face[2] / 2, chin, head_w, chin - top)
    clean = _grade(_decontaminate(rgb, a), a, face)
    return Cutout(clean, a, tuple(float(v) for v in face), top, head_w)


# ---------------------------------------------------------------- sticker
def _over(top_c: np.ndarray, top_a: np.ndarray, bot_c: np.ndarray, bot_a: np.ndarray):
    """Premultiplied 'over'."""
    return top_c + bot_c * (1 - top_a)[..., None], top_a + bot_a * (1 - top_a)


def _crop_pad(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """arr[y0:y1, x0:x1] where outside the array the edge pixels continue (no outline along the frame edge)."""
    h, w = arr.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w))) + ((0, 0),) * (arr.ndim - 2)
    sub = arr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return np.pad(sub, pad, mode="edge") if any(p for pp in pad for p in pp) else sub


@dataclass(frozen=True)
class Sticker:
    img: skia.Image
    pad: int
    scale: float
    origin: tuple          # source px that sits at the image's (pad, pad) pixel

    def draw(self, c: skia.Canvas, anchor_src: tuple, anchor_dst: tuple) -> None:
        """Draw so that the source point anchor_src lands on the canvas point anchor_dst."""
        x = anchor_dst[0] + (self.origin[0] - anchor_src[0]) * self.scale - self.pad
        y = anchor_dst[1] + (self.origin[1] - anchor_src[1]) * self.scale - self.pad
        c.drawImage(self.img, x, y, skia.SamplingOptions(skia.CubicResampler.Mitchell()))


def sticker(cut: Cutout, scale: float, view: tuple, stroke: float = 10.0, edge: float = 3.0, ring=(255, 255, 255),
            edge_rgb=(13, 13, 13), shadow: float = 0.55) -> Sticker:
    """The speaker at `scale` with an outline. view = (x0, y0, x1, y1) of the source that can show up on the canvas
    (everything else is cropped away, which keeps the distance transform small)."""
    pad = int((stroke + edge) * 3 + 30)
    m = int(np.ceil(pad / scale))
    x0, y0, x1, y1 = (int(np.floor(view[0])) - m, int(np.floor(view[1])) - m,
                      int(np.ceil(view[2])) + m, int(np.ceil(view[3])) + m)
    rgb = _crop_pad(cut.rgb, x0, y0, x1, y1)
    a = _crop_pad(cut.alpha, x0, y0, x1, y1)
    size = (max(2, int(round(rgb.shape[1] * scale))), max(2, int(round(rgb.shape[0] * scale))))
    im = Image.fromarray(np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8), "RGBA").resize(size, Image.LANCZOS)
    arr = np.asarray(im, np.float32) / 255
    al = np.pad(arr[..., 3], pad)
    person = np.pad(arr[..., :3], ((pad, pad), (pad, pad), (0, 0))) * al[..., None]
    dist = ndimage.distance_transform_edt(ndimage.gaussian_filter(al, 1.0) < 0.5)
    ring_a = np.maximum(np.clip(stroke + 0.5 - dist, 0, 1), al)
    edge_a = np.maximum(np.clip(stroke + edge + 0.5 - dist, 0, 1), al) if edge > 0 else np.zeros_like(al)
    sh = ndimage.shift(ndimage.gaussian_filter(np.maximum(ring_a, edge_a), 9), (10, 7), order=1) * shadow
    c, a2 = _over(np.array(edge_rgb, np.float32) / 255 * edge_a[..., None], edge_a, np.zeros_like(person), sh)
    c, a2 = _over(np.array(ring, np.float32) / 255 * ring_a[..., None], ring_a, c, a2)
    c, a2 = _over(person, al, c, a2)
    straight = c / np.maximum(a2, 1e-4)[..., None]
    out = np.dstack([straight, a2]).clip(0, 1) * 255
    img = skia.Image.fromarray(out.astype(np.uint8), colorType=skia.kRGBA_8888_ColorType)
    return Sticker(img, pad, scale, (float(x0), float(y0)))
