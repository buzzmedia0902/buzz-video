#!/usr/bin/env python3
"""Normalized-coordinate grid on source frames, for reading patch / focus / center values.

  python grid.py "~/Downloads/clips/IMG_0996.MOV" 5.0 [--rot 180] [--out grid.jpg]
  python grid.py photo.jpg

Lines every 0.05 (labels every 0.1) in the coordinate system the spec uses: x 0..1 left->right,
y 0..1 top->bottom of the upright (and --rot rotated) source. Radii in the spec are in units of the
source WIDTH, so read rx/ry against the x labels.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps


def frame(path: str, t: float, rot: int) -> Image.Image:
    if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    else:
        buf = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-f", "image2pipe",
                              "-vcodec", "png", "-"], capture_output=True, check=True).stdout
        im = Image.open(io.BytesIO(buf)).convert("RGB")
    return im.rotate(-rot, expand=True) if rot else im


def draw_grid(im: Image.Image) -> Image.Image:
    im = im.resize((900, int(900 * im.height / im.width)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    for k in range(1, 20):
        v, major = k / 20, k % 2 == 0
        col, wd = ((255, 40, 40), 2) if major else ((255, 150, 150), 1)
        x, y = v * im.width, v * im.height
        d.line([(x, 0), (x, im.height)], fill=col, width=wd)
        d.line([(0, y), (im.width, y)], fill=col, width=wd)
        if major:
            d.text((x + 3, 3), f"{v:.1f}", fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
            d.text((3, y + 3), f"{v:.1f}", fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("t", nargs="?", type=float, default=0.0)
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    src = os.path.expanduser(a.source)
    out = a.out or f"grid_{os.path.splitext(os.path.basename(src))[0]}_{a.t:.1f}.jpg"
    draw_grid(frame(src, a.t, a.rot)).save(out, quality=88)
    print(out)


if __name__ == "__main__":
    main()
