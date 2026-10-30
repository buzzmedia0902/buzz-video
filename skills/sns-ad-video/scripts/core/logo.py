"""Animated logo built from logo_prep.py output (<stem>_on_light/_on_dark.png + <stem>_parts.json).

Styles
  rise  letters of the wordmark fade in one by one (rise + blur-to-sharp); other rows (mark, tagline,
        ornament) open from the centre afterwards. Falls back to the whole logo if no wordmark row.
  pop   letters pop in with overshoot (motion taste).
  fade  whole logo rises out of a blur (safest for emblems / complex marks).
  wipe  whole logo revealed left -> right with a soft edge.
glint: a soft diagonal highlight sweeping across the logo (kept subtle so coloured ink never turns white).
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageFilter

from core.engine import blit, blit_img, ease_in_out_sine, ease_out_back, ease_out_cubic, premult

GLINT = np.array([255, 241, 196], np.float32)


class Logo:
    def __init__(self, image_path: str, parts_path: str, width: int):
        img = Image.open(image_path).convert("RGBA")
        info = json.load(open(parts_path))
        sc = width / img.width
        self.img = img.resize((int(width), max(1, int(img.height * sc))), Image.LANCZOS)
        self.w, self.h = self.img.size
        self.pieces = self._pieces(info, sc)

    def _pieces(self, info: dict, sc: float) -> list[dict]:
        """Glyph boxes of the wordmark row + whole-width boxes for the other rows (scaled logo px)."""
        lb = info.get("letters_band")
        if lb is None:
            return [{"box": (0, 0, self.w, self.h), "kind": "whole", "i": 0}]
        pieces, bands = [], info["bands"]
        pad = 5
        for bi, band in enumerate(bands):
            y0 = max(0, int(band["y"][0] * sc) - pad)
            y1 = min(self.h, int(band["y"][1] * sc) + pad + 1)
            if bi == lb:
                for gi, (x0, x1) in enumerate(band["glyphs"]):
                    pieces.append({"box": (max(0, int(x0 * sc) - 4), y0, min(self.w, int(x1 * sc) + 5), y1),
                                   "kind": "glyph", "i": gi})
            else:
                pieces.append({"box": (0, y0, self.w, y1), "kind": "band", "i": bi})
        return pieces

    @property
    def n_glyphs(self) -> int:
        return sum(p["kind"] == "glyph" for p in self.pieces)

    def _glinted(self, g: float) -> Image.Image:
        a = np.asarray(self.img, np.float32).copy()
        h, w = a.shape[:2]
        xs = np.arange(w)[None, :] + np.arange(h)[:, None] * 0.5
        band = np.exp(-((xs - (-150 + g * (w + 300))) / (w * 0.07 + 12)) ** 2)
        k = (band * 0.42 * (a[..., 3] / 255))[..., None]
        a[..., :3] += k * (GLINT - a[..., :3])
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")

    def draw(self, frame: np.ndarray, t: float, cx: float, top: float, t0: float, style: str = "rise",
             gap: float = 0.07, sub_at: float | None = None, glint_at: float | None = None,
             opacity: float = 1.0, scale: float = 1.0) -> None:
        """Draw the logo (top edge at `top`, centred on cx) as it looks at time t; animation starts at t0."""
        if t < t0 or opacity <= 0.003:
            return
        g = (t - glint_at) / 0.75 if glint_at is not None else -1.0
        base = self._glinted(g) if 0 < g < 1 else self.img
        if abs(scale - 1) > 1e-3:
            base = base.resize((max(1, int(self.w * scale)), max(1, int(self.h * scale))), Image.LANCZOS)
        n = max(1, self.n_glyphs)
        gap = min(gap, 0.9 / n)
        sub_at = t0 + n * gap + 0.2 if sub_at is None else sub_at
        if style in ("fade", "wipe") or self.pieces[0]["kind"] == "whole":
            self._draw_whole(frame, base, t, cx, top, t0, style, opacity)
            return
        for p in self.pieces:
            x0, y0, x1, y1 = (int(v * scale) for v in p["box"])
            crop = base.crop((x0, y0, x1, y1))
            px, py = cx - base.width / 2 + (x0 + x1) / 2, top + (y0 + y1) / 2
            if p["kind"] == "glyph":
                self._draw_glyph(frame, crop, px, py, (t - t0 - p["i"] * gap), style, opacity)
            else:
                self._draw_band(frame, crop, cx - base.width / 2 + x0, top + y0, (t - sub_at) / 0.7, opacity)

    @staticmethod
    def _draw_glyph(frame, crop, px, py, dt, style, opacity) -> None:
        p = dt / (0.45 if style == "pop" else 0.6)
        if p <= 0:
            return
        if style == "pop":
            blit_img(frame, crop, px, py, scale=max(0.05, ease_out_back(p, 2.2)), opacity=ease_out_cubic(p * 2) * opacity)
            return
        e = ease_out_cubic(p)
        blur = (1 - e) * 6
        im = crop.filter(ImageFilter.GaussianBlur(blur)) if blur > 0.3 else crop
        blit_img(frame, im, px, py + (1 - e) * 30, opacity=e * opacity)

    @staticmethod
    def _draw_band(frame, crop, x, y, p, opacity) -> None:
        e = ease_in_out_sine(p)
        if e <= 0:
            return
        arr = premult(crop)
        xs = np.arange(arr.shape[1])
        half = e * (arr.shape[1] / 2 + 30)
        arr *= np.clip((half - np.abs(xs - arr.shape[1] / 2)) / 24, 0, 1).astype(np.float32)[None, :, None]
        blit(frame, arr, x, y, opacity)

    @staticmethod
    def _draw_whole(frame, base, t, cx, top, t0, style, opacity) -> None:
        p = (t - t0) / 0.9
        if style == "wipe":
            e = ease_in_out_sine(p)
            arr = premult(base)
            xs = np.arange(arr.shape[1])
            arr *= np.clip((e * (arr.shape[1] + 60) - xs) / 60, 0, 1).astype(np.float32)[None, :, None]
            blit(frame, arr, cx - base.width / 2, top, opacity)
            return
        e = ease_out_cubic(p)
        blur = (1 - e) * 8
        im = base.filter(ImageFilter.GaussianBlur(blur)) if blur > 0.3 else base
        blit_img(frame, im, cx, top + base.height / 2 + (1 - e) * 24, opacity=e * opacity)
