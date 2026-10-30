"""Prepare a client logo for animation.

  python logo_prep.py brand/logo.png [--keep-bg] [--ink auto|dark|light|keep]

Writes next to the input:
  <stem>_on_light.png   transparent logo for light backgrounds (original colours)
  <stem>_on_dark.png    transparent logo for dark backgrounds (neutral dark ink turned off-white, colours kept)
  <stem>_parts.json     row bands + glyph spans (lets the animator reveal a wordmark letter by letter)
  <stem>_preview.png    both variants on paper and night backgrounds -- LOOK AT THIS before using the logo

A logo on a plain background (white, cream, any flat colour) is made transparent with colour-to-alpha
(keeps anti-aliased edges). Use --keep-bg if the background is part of the design (badge, photo).
Never redesign or recolour the mark beyond the dark/light ink swap.

Which side the logo was made for (--ink auto): the flat background it came on decides (light background ->
dark ink); a transparent file counts as light ink ("white logo") only when most of it is near-white.
Override with --ink dark | light, or --ink keep (both variants = the original colours, nothing swapped;
the default with --keep-bg).
Warnings: a swap that recoloured much of the logo, and a logo that is hard to read (contrast < 3:1 for
most of its ink) on light or dark backgrounds -> pick the card theme / logo placement accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

PAPER, NIGHT, OFF_WHITE, INK = (247, 243, 234), (30, 25, 21), (248, 245, 239), (34, 30, 27)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(path: str) -> Image.Image:
    if path.lower().endswith((".heic", ".heif")):
        tmp = os.path.splitext(path)[0] + "_conv.png"
        subprocess.run(["sips", "-s", "format", "png", path, "--out", tmp], check=True, capture_output=True)
        path = tmp
    return Image.open(path).convert("RGBA")


def color_to_alpha(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove a flat background (median border colour) keeping soft edges (GIMP colour-to-alpha).
    Returns (RGBA float, background RGB)."""
    rgb = rgba[..., :3].astype(np.float32)
    h, w = rgb.shape[:2]
    b = max(2, int(min(h, w) * 0.02))
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3),
                             rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    up = np.where(rgb > bg, (rgb - bg) / np.maximum(255 - bg, 1), 0)
    down = np.where(rgb < bg, (bg - rgb) / np.maximum(bg, 1), 0)
    alpha = np.clip(np.maximum(up, down).max(axis=2), 0, 1)
    alpha = np.where(alpha < 0.04, 0, alpha)
    ink = alpha[alpha > 0]
    top = float(np.percentile(ink, 99)) if ink.size else 1.0
    if top < 0.9:  # pale ink close to the background: make its solid parts opaque (else a faint dark ghost)
        alpha = np.clip(alpha / top, 0, 1)
    col = bg + (rgb - bg) / np.maximum(alpha, 1e-3)[..., None]
    out = np.dstack([np.clip(col, 0, 255), alpha * 255])
    print(f"background {bg.round().astype(int).tolist()} removed", file=sys.stderr)
    return out.astype(np.float32), bg


def trim(arr: np.ndarray, margin: float = 0.04) -> np.ndarray:
    ys, xs = np.nonzero(arr[..., 3] > 5)
    if len(xs) == 0:
        raise SystemExit("logo: no visible pixels after background removal (try --keep-bg)")
    m = int(max(arr.shape[:2]) * margin)
    y0, y1 = max(0, ys.min() - m), min(arr.shape[0], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(arr.shape[1], xs.max() + m + 1)
    return arr[y0:y1, x0:x1]


def neutral_weight(rgb: np.ndarray, dark: bool) -> np.ndarray:
    """1 for neutral (unsaturated) dark ink (dark=True) or light ink (dark=False), 0 for coloured parts."""
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)) / 255  # absolute chroma: stable for near-black ink
    lum = (rgb @ LUMA) / 255
    w_sat = np.clip((0.14 - chroma) / 0.07, 0, 1)
    w_lum = np.clip((0.62 - lum) / 0.2, 0, 1) if dark else np.clip((lum - 0.7) / 0.15, 0, 1)
    return w_sat * w_lum


def light_ink(arr: np.ndarray, bg: np.ndarray | None) -> bool:
    """True for light ink made for dark backgrounds (a "white logo")."""
    if bg is not None:  # delivered on a flat background: that background shows which side it was made for
        return float(bg @ LUMA) / 255 < 0.45
    rgb, opaque = arr[..., :3], arr[..., 3] > 128
    if not opaque.any():
        return False
    lum = (rgb[opaque] @ LUMA) / 255
    chroma = (rgb[opaque].max(axis=1) - rgb[opaque].min(axis=1)) / 255
    return float(((lum > 0.86) & (chroma < 0.10)).mean()) >= 0.5  # mostly near-white (pale logos are not)


def variants(arr: np.ndarray, ink: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(on_light, on_dark, share of the opaque ink recoloured) for ink dark | light | keep. Neutral ink is
    swapped only when there is a real amount of it (>= 3 %) and it is hard to read on the other side: a black
    wordmark is, the grey shading of an illustrated mark that already reads there is not (it would streak)."""
    if ink == "keep":
        return arr, arr, 0.0
    rgb, a = arr[..., :3], arr[..., 3:4]
    w = neutral_weight(rgb, dark=ink == "dark")[..., None]
    opaque = a[..., 0] > 128
    cand = opaque & (w[..., 0] > 0.5)
    if cand.sum() < 0.03 * max(1, opaque.sum()) or \
            low_share(rgb[cand], NIGHT if ink == "dark" else PAPER) < 0.5:
        return arr, arr, 0.0
    swapped = float(w[..., 0][opaque].mean())
    if ink == "light":
        return np.dstack([rgb * (1 - w) + np.array(INK, np.float32) * w, a]), arr, swapped
    return arr, np.dstack([rgb * (1 - w) + np.array(OFF_WHITE, np.float32) * w, a]), swapped


def low_share(pixels: np.ndarray, bg) -> float:
    """Share of RGB pixels (n, 3) with WCAG contrast < 3:1 against bg (hard to read there)."""
    if not len(pixels):
        return 0.0

    def rel(c):
        c = np.asarray(c, np.float32) / 255
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ LUMA

    li, lb = rel(pixels), float(rel(bg))
    ratio = (np.maximum(li, lb) + 0.05) / (np.minimum(li, lb) + 0.05)
    return float((ratio < 3.0).mean())


def low_contrast(arr: np.ndarray, bg) -> float:
    """Share of the logo's opaque ink that is hard to read on bg."""
    return low_share(arr[..., :3][arr[..., 3] > 128], bg)


def parts(alpha: np.ndarray) -> dict:
    """Row bands and per-band glyph spans (column runs separated by >3 px gaps)."""
    h, w = alpha.shape
    rows = np.nonzero(alpha.sum(axis=1) > 0.5)[0]
    gap = max(4, int(h * 0.02))
    bands = [(int(b[0]), int(b[-1])) for b in np.split(rows, np.nonzero(np.diff(rows) > gap)[0] + 1) if len(b)]
    out = []
    for y0, y1 in bands:
        cols = np.nonzero(alpha[y0:y1 + 1].sum(axis=0) > 0.3)[0]
        spans = [(int(s[0]), int(s[-1])) for s in np.split(cols, np.nonzero(np.diff(cols) > 3)[0] + 1) if len(s)]
        out.append({"y": [y0, y1], "glyphs": spans})
    letters = max(range(len(out)), key=lambda i: len(out[i]["glyphs"])) if out else None
    if letters is not None and len(out[letters]["glyphs"]) < 3:
        letters = None
    return {"size": [w, h], "bands": out, "letters_band": letters}


def accent_of(arr: np.ndarray) -> tuple[str | None, bool]:
    """(brand colour, muted): the median saturated ink; else the median mid-tone tinted ink (greige,
    taupe ...) marked muted; else None (black / white logo: take the colour from the shop or website)."""
    rgb, a = arr[..., :3], arr[..., 3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    lum = (rgb @ LUMA) / 255
    for sel, muted in (((a > 160) & (sat > 0.3) & (mx > 60), False),
                       ((a > 160) & ((mx - mn) / 255 > 0.025) & (lum > 0.15) & (lum < 0.85), True)):
        if sel.sum() >= 50:
            c = np.median(rgb[sel], axis=0).round().astype(int)
            return "#{:02X}{:02X}{:02X}".format(*c), muted
    return None, False


def preview(on_light: Image.Image, on_dark: Image.Image, path: str) -> None:
    w, h = on_light.size
    pad = 40
    img = Image.new("RGB", (w + 2 * pad, 2 * (h + 2 * pad)), PAPER)
    img.paste(Image.new("RGB", (w + 2 * pad, h + 2 * pad), NIGHT), (0, h + 2 * pad))
    img.paste(on_light, (pad, pad), on_light)
    img.paste(on_dark, (pad, h + 3 * pad), on_dark)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logo")
    ap.add_argument("--keep-bg", action="store_true", help="keep the background (badge/photo logos)")
    ap.add_argument("--ink", choices=("auto", "dark", "light", "keep"), default="auto",
                    help="dark: made for light backgrounds, light: white logo for dark ones, keep: never recolour")
    a = ap.parse_args()
    src = os.path.abspath(os.path.expanduser(a.logo))
    arr = np.asarray(load(src), np.float32)
    has_alpha = (arr[..., 3] < 250).mean() > 0.02
    bg = None
    if not has_alpha and not a.keep_bg:
        arr, bg = color_to_alpha(arr)
    arr = trim(arr)
    if a.ink != "auto":
        ink = a.ink
    elif a.keep_bg:  # badge / photo: its background is part of the design, recolouring would repaint it
        ink = "keep"
    else:
        ink = "light" if light_ink(arr, bg) else "dark"
    on_light, on_dark, swapped = variants(arr, ink)
    light_logo = ink == "light"
    stem = os.path.splitext(src)[0]
    imgs = {k: Image.fromarray(np.clip(v, 0, 255).astype(np.uint8), "RGBA") for k, v in
            (("light", on_light), ("dark", on_dark))}
    imgs["light"].save(f"{stem}_on_light.png")
    imgs["dark"].save(f"{stem}_on_dark.png")
    info = parts(arr[..., 3] / 255.0)
    accent, muted = accent_of(arr)
    hard = {"on_paper": round(low_contrast(on_light, PAPER), 2), "on_night": round(low_contrast(on_dark, NIGHT), 2)}
    info.update(accent=accent, accent_muted=muted, light_logo=light_logo, ink=ink, low_contrast=hard)
    with open(f"{stem}_parts.json", "w") as f:
        json.dump(info, f, ensure_ascii=False)
    preview(imgs["light"], imgs["dark"], f"{stem}_preview.png")
    print(json.dumps({"size": info["size"], "bands": len(info["bands"]), "letters_band": info["letters_band"],
                      "glyphs": [len(b["glyphs"]) for b in info["bands"]], "accent": accent, "accent_muted": muted,
                      "ink": ink, "recoloured": round(swapped, 2), "low_contrast": hard,
                      "preview": f"{stem}_preview.png"}, ensure_ascii=False))
    if swapped:
        side, swap = ("on_light", "light neutral ink -> dark") if ink == "light" else ("on_dark", "dark neutral ink -> off-white")
        print(f"ink {ink}: {side} recolours {swapped:.0%} of the logo ({swap}). Check the preview; if that is wrong "
              f"rerun with --ink dark | light | keep.", file=sys.stderr)
    if hard["on_paper"] >= 0.8:
        print("NOTE: pale logo, hard to read on light backgrounds: use dark end cards and place it over dark parts "
              "of the footage (references/tastes.md, pale logos).", file=sys.stderr)
    if hard["on_night"] >= 0.8:
        print("NOTE: the dark-background variant is hard to read on dark backgrounds: keep end cards light.",
              file=sys.stderr)
    if accent is None:
        print("NOTE: no brand colour in the logo: take brand.colors.accent from the shop / website "
              "(or keep the default).", file=sys.stderr)


if __name__ == "__main__":
    main()
