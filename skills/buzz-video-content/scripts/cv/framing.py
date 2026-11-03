"""Framing: which part of the upright source each shot shows, and where it lands on the canvas.

  cover  the crop fills the canvas (zoom >= 1, centred on the face or `center`)
  band   the source fits the canvas width (16:9 footage in a 9:16 clip) or height, over a background the tone
         draws (blurred copy / colour); `band_y` = centre of the band as a fraction of the canvas height

Talk parts get a camera plan from the tone (Look.CAMERA): wide <-> medium only at a jump cut and only after the shot
has been held `hold` seconds, plus at most one slow push-in per `punch_every` seconds. Viewers said constant
reframing and snap zooms were tiring: tempo comes from cuts, telops and sound, not from the camera.
"""
from __future__ import annotations

import functools
import json
import math
import os

import numpy as np

from cv import media
from cv.canvas import ASPECT, FPS, H, W


@functools.lru_cache(maxsize=None)
def face_track(work_dir: str, sid: str) -> dict | None:
    """Largest face per 0.5 s from work/faces/<sid>.jsonl (faces.py): arrays t, cx, cy, h (normalised)."""
    path = os.path.join(work_dir, "faces", f"{sid}.jsonl")
    if not os.path.exists(path):
        return None
    rows = sorted((json.loads(line) for line in open(path)), key=lambda r: r["f"])
    t, cx, cy, hh, every = [], [], [], [], []
    for k, r in enumerate(rows):
        fs = [f for f in r["faces"] if f[4] > 0.5 and f[3] > 0.04]
        t.append(k * 0.5)
        every.append([(f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], f[3]) for f in fs])
        if fs:
            f = max(fs, key=lambda f: f[3])
            cx.append(f[0] + f[2] / 2)
            cy.append(f[1] + f[3] / 2)
            hh.append(f[3])
        else:
            cx.append(np.nan)
            cy.append(np.nan)
            hh.append(np.nan)
    return {"t": np.array(t), "cx": np.array(cx), "cy": np.array(cy), "h": np.array(hh), "all": every}


def face_center(work_dir: str, sid: str, a: float, b: float) -> tuple[float, float, float] | None:
    """Median face (cx, cy, h) over [a, b] (widened to 2 s)."""
    tr = face_track(work_dir, sid)
    if tr is None:
        return None
    if b - a < 2.0:
        m = (a + b) / 2
        a, b = m - 1.0, m + 1.0
    sel = (tr["t"] >= a) & (tr["t"] <= b) & ~np.isnan(tr["cx"])
    if not sel.any():
        sel = ~np.isnan(tr["cx"])
        if not sel.any():
            return None
    return float(np.median(tr["cx"][sel])), float(np.median(tr["cy"][sel])), float(np.median(tr["h"][sel]))


def crop(sw: int, sh: int, ar: float, cx: float, cy: float, zoom: float) -> tuple[float, float, float, float]:
    """Crop (x, y, w, h) in source px with aspect `ar` (w/h), 1/zoom of the largest such crop, centred at (cx, cy)
    (normalised), kept inside the source."""
    w, h = (sw, sw / ar) if sw / sh < ar else (sh * ar, sh)
    w, h = w / max(zoom, 1.0), h / max(zoom, 1.0)
    x = min(max(cx * sw - w / 2, 0.0), sw - w)
    y = min(max(cy * sh - h / 2, 0.0), sh - h)
    return x, y, w, h


def _default_fit(fmt: str, kind: str, src_ar: float) -> str:
    canvas_ar = W / H
    mismatch = abs(math.log(src_ar / canvas_ar)) > 0.5  # 16:9 in 9:16 or the reverse
    if not mismatch or fmt == "vlog":  # a vlog fills the screen (pick the part with "center")
        return "cover"
    if fmt == "clip":
        return "band"
    return "cover" if kind == "talk" else "band"


def plan(tl) -> None:
    """Set shot.view for every shot of the timeline."""
    project, version, look = tl.project, tl.version, tl.look
    fmt = version["format"]
    cam = {**look.CAMERA, **version.get("camera", {})}
    by_part: dict[int, list] = {}
    for s in tl.shots:
        by_part.setdefault(s.part, []).append(s)
    for i, shots in by_part.items():
        p = version["parts"][i]
        if p["kind"] == "card":
            for s in shots:
                s.view = {"fit": "card"}
            continue
        info = media.probe(project.src_path(p["src"]))
        rot = project.sources[p["src"]]["rot"]
        sw, sh = (info["h"], info["w"]) if rot in (90, 270) else (info["w"], info["h"])
        fit = p.get("fit") or _default_fit(fmt, p["kind"], sw / sh)
        aspect_over = p.get("aspect", {}).get(ASPECT, {})
        fit = aspect_over.get("fit", fit)
        zoom = float(aspect_over.get("zoom", p.get("zoom", 1.0)))
        center = aspect_over.get("center", p.get("center"))
        band_y = float(aspect_over.get("band_y", p.get("band_y", look.band_y(fmt))))
        moving = p["kind"] == "talk" and fit == "cover" and cam.get("med", 1.0) > 1.0 and not p.get("static")
        held, med, since_punch = 0.0, False, 0.0
        for j, s in enumerate(shots):
            fc = face_center(project.work_dir, p["src"], s.a, s.b)
            cx, cy = (center if center else ((fc[0], fc[1]) if fc else (0.5, 0.42)))
            dur = s.n / FPS
            if moving and j > 0 and held >= cam["hold"]:
                med, held = not med, 0.0
            z = zoom * (cam["med"] if med else 1.0)
            if med and fc:  # medium: face a little above the middle
                cy = fc[1] + 0.04
            punch = None
            if moving and j > 0 and since_punch >= cam["punch_every"] and dur >= 3.0:
                punch, since_punch = (0.3, cam["punch_dur"], cam["punch_zoom"]), 0.0
            held += dur
            since_punch += dur
            if fit == "band":
                ar = sw / sh
                if ar >= W / H:
                    ow, oh = W, W / ar
                    oy = min(max(band_y * H - oh / 2, 0.0), H - oh)
                    box = (0.0, oy, ow, oh)
                else:
                    ow, oh = H * ar, H
                    box = ((W - ow) / 2, 0.0, ow, oh)
                cr = crop(sw, sh, box[2] / box[3], cx, cy, z)
            else:
                box = (0.0, 0.0, float(W), float(H))
                cr = crop(sw, sh, W / H, cx, cy, z)
            s.view = {"fit": fit, "crop": cr, "box": box, "punch": punch, "src_size": (sw, sh), "rot": rot,
                      "med": med, "smooth": bool(p.get("smooth"))}
