"""Face-aware placement: cards, name plates and big words go where they cover nobody's face.

Viewers watch faces and reactions; a panel on a head hides the point and looks careless. For each such overlay the
faces on screen during its whole time (work/faces/<src>.jsonl, grown 1.7 x 2.0 for hair, mapped through the shot's
crop) are scored against the tone's candidate spots; the first clear spot wins, then a smaller size, and only then
the least-covering spot drawn see-through.
"""
from __future__ import annotations

import numpy as np

from cv import framing
from cv.canvas import FPS, TALL

PLACED = ("card", "name", "emphasis")


def face_at(comp, t: float) -> tuple[float, float, float] | None:
    """(x, y, h) in canvas px of the largest face on screen at output time t."""
    shot = comp.tl.at(min(int(t * FPS), comp.tl.total - 1))
    if shot.kind == "card":
        return None
    tr = framing.face_track(comp.p.work_dir, shot.src)
    if tr is None:
        return None
    st = (shot.a + (t - shot.t0) * shot.speed) / 0.5
    k0 = int(st)
    if not 0 <= k0 < len(tr["cx"]) - 1 or np.isnan(tr["cx"][k0]) or np.isnan(tr["cx"][k0 + 1]):
        return None
    w = st - k0
    cx, cy, fh = ((1 - w) * tr[k][k0] + w * tr[k][k0 + 1] for k in ("cx", "cy", "h"))
    sw, sh = shot.view["src_size"]
    x, y, cw, ch = shot.view["crop"]
    ox, oy, ow, oh = shot.view["box"]
    return ox + (cx * sw - x) / cw * ow, oy + (cy * sh - y) / ch * oh, fh * sh / ch * oh


def face_rects(comp, t0: float, t1: float, step: float = 0.5) -> list[list[tuple[float, float, float, float]]]:
    """Per sample (every `step` s over [t0, t1]): canvas boxes of the faces on screen."""
    out = []
    t = t0
    while t <= t1 + 1e-6:
        shot = comp.tl.at(min(int(t * FPS), comp.tl.total - 1))
        tr = framing.face_track(comp.p.work_dir, shot.src) if shot.kind != "card" else None
        sample = []
        if tr is not None:
            k = int(round((shot.a + (t - shot.t0) * shot.speed) / 0.5))
            if 0 <= k < len(tr["all"]):
                sw, sh = shot.view["src_size"]
                x, y, w, h = shot.view["crop"]
                ox, oy, ow, oh = shot.view["box"]
                for cx, cy, fw, fh in tr["all"][k]:  # Vision box = brows..chin: add hair above, ears aside
                    x0 = ox + ((cx - fw * 0.65) * sw - x) / w * ow
                    x1 = ox + ((cx + fw * 0.65) * sw - x) / w * ow
                    y0 = oy + ((cy - fh * 0.85) * sh - y) / h * oh
                    y1 = oy + ((cy + fh * 0.60) * sh - y) / h * oh
                    sample.append((x0, y0, x1, y1))
        out.append(sample)
        t += step
    return out


def _face_cover(rect, samples) -> float:
    """Coverage the viewer sees most of the time: the 80th percentile over the samples (a brief lean of the
    speaker into the panel is fine, a panel sitting on the face is not)."""
    vals = sorted(_cover(rect, s) for s in samples) or [0.0]
    return vals[min(len(vals) - 1, int(0.8 * len(vals)))]


def _cover(rect, faces) -> float:
    x, y, w, h = rect
    if w <= 0 or h <= 0 or not faces:
        return 0.0
    tot = 0.0
    for fx0, fy0, fx1, fy1 in faces:
        ix = max(0.0, min(x + w, fx1) - max(x, fx0))
        iy = max(0.0, min(y + h, fy1) - max(y, fy0))
        tot = max(tot, ix * iy / (w * h))
    return tot


OK = 0.04  # a panel may touch this share of its area (chin / hair edge) and still count as clear


def apply(comp) -> None:
    look = comp.look
    for ov in comp.ovs:
        if ov.type not in PLACED or "_place" in ov.d:
            continue
        if TALL and ov.type == "card":  # 9:16 close-ups leave no room: card mode (blurred picture, face bubble)
            ov.d["_place"] = {"center": look.CARD_MODE_CENTER, "scale": 1.0, "alpha": 1.0}
            ov.d["_mode"] = "card"
            continue
        faces = face_rects(comp, ov.t0, ov.t1)
        if not any(faces):
            continue
        blockers = [look.box(o) for o in comp.ovs if o.type == "title_band" and o.t0 < ov.t1 and ov.t0 < o.t1]
        if ov.type != "emphasis":
            blockers.append(look.sub_zone())
        blockers = [(x, y, x + w, y + h) for x, y, w, h in blockers]
        best = None
        for scale in (1.0, 0.85, 0.72):
            for center in look.spots(ov):
                rect = look.box(ov, center, scale)
                cov = max(_face_cover(rect, faces), _cover(rect, blockers))
                if best is None or cov < best[0] - 1e-6:
                    best = (cov, center, scale)
                if cov <= OK:
                    break
            if best[0] <= OK:
                break
        cov, center, scale = best
        ov.d["_place"] = {"center": center, "scale": scale, "alpha": 1.0 if cov <= OK else 0.82}
