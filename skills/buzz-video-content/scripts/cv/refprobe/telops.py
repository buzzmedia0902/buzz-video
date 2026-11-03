"""On-screen text of a reference video (Apple Vision boxes + colour analysis), the sheets Claude looks at, the look."""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def color_name(rgb) -> str:
    r, g, b = (v / 255 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 0.22:
        return "黒"
    if mx - mn < 0.14:
        return "白" if mx > 0.78 else "グレー"
    import colorsys
    h = colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return ("赤" if h < 14 or h >= 346 else "オレンジ" if h < 40 else "黄" if h < 68 else "緑" if h < 170 else
            "水色" if h < 200 else "青" if h < 258 else "紫" if h < 300 else "ピンク")


def kmeans(px: np.ndarray, k: int = 4, iters: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """(labels, centres) of a small k-means over RGB pixels."""
    rng = np.random.default_rng(3)
    cen = px[rng.choice(len(px), min(k, len(px)), replace=False)].astype(np.float64)
    for _ in range(iters):
        lab = np.argmin(((px[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        for j in range(len(cen)):
            if (lab == j).any():
                cen[j] = px[lab == j].mean(0)
    return lab, cen


def palette(px: np.ndarray, k: int = 5) -> list[dict]:
    """Telop fill colours: clusters of the fill pixels, each shown by its purest pixels (anti-aliased edge pixels would
    dull it), clusters with the same colour name merged."""
    lab, cen = kmeans(px, k)
    found: dict[str, dict] = {}
    for j in range(len(cen)):
        sel = px[lab == j]
        if not len(sel):
            continue
        mx, mn = sel.max(-1), sel.min(-1)
        pure = sel[np.argsort(-((mx - mn) / np.maximum(mx, 1) * 0.5 + mx / 255))[: max(3, len(sel) * 3 // 20)]]
        rgb = np.median(pure, 0)
        name = color_name(rgb)
        share = len(sel) / len(px)
        if name in found:
            found[name]["share"] += share
        else:
            found[name] = {"rgb": [int(v) for v in rgb], "hex": "#%02X%02X%02X" % tuple(int(v) for v in rgb),
                           "name": name, "share": share}
    out = sorted(found.values(), key=lambda c: -c["share"])
    return [{**c, "share": round(c["share"], 2)} for c in out if c["share"] >= 0.05]


def line_pixels(img: np.ndarray, box: tuple) -> dict | None:
    """Text pixels of one recognised line, by what the text sits on (the colour just outside its box):
    light bg -> dark text on a plate; else bright / saturated pixels hugging dark ones (outlined telop) or, without an
    outline, pixels far from the surrounding colour. Also: ink coverage, outline (the pixels next to the text are
    darker than the picture a little further out) and plate (a calm patch behind the text)."""
    from scipy import ndimage
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x0, y0, x1, y1 = int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)
    lh = max(8, y1 - y0)
    m = int(0.75 * lh)
    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
    crop = img[cy0:min(h, y1 + m), cx0:min(w, x1 + m)].astype(np.float32)
    ix0, iy0, ix1, iy1 = x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0
    if iy1 - iy0 < 8 or ix1 - ix0 < 8:
        return None
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    d_out = np.maximum(np.maximum(ix0 - xx, xx - ix1), np.maximum(iy0 - yy, yy - iy1))      # px outside the box
    box_m = d_out <= int(0.04 * lh)                                                         # the glyph box
    near = (d_out > 0) & (d_out <= max(3, int(0.18 * lh)))
    wide = (d_out > 0.4 * lh) & (d_out <= 0.75 * lh)
    if near.sum() < 20 or wide.sum() < 20:
        return None
    luma_w = np.array([0.299, 0.587, 0.114], np.float32)
    luma = crop @ luma_w
    sat = (crop.max(-1) - crop.min(-1)) / np.maximum(crop.max(-1), 1)
    bg = np.median(crop[near], 0)
    outline = None                                                        # None: too dark a picture to tell
    if float(bg @ luma_w) >= 165:                                        # dark text on a light plate
        outline = False
        fill = box_m & (luma <= float(bg @ luma_w) - 70)
    else:
        dark = luma < 60
        hug = ndimage.binary_dilation(dark, iterations=3) & ~dark
        bright = (luma > 170) | ((sat > 0.45) & (luma > 90))
        fill = ndimage.binary_opening(box_m & hug & bright, iterations=1)
        fill = ndimage.binary_dilation(fill, iterations=max(3, int(0.14 * lh))) & bright & box_m     # the stroke interior
        if fill.sum() < 12:                                              # no outline: text is what is far from the bg
            fill = ndimage.binary_opening(box_m & (np.sqrt(((crop - bg) ** 2).sum(-1)) >= 90) & (luma > 140), iterations=1)
            outline = False if float(bg @ luma_w) < 120 else None
        elif fill.sum() > 30:
            r_adj, r_far = max(3, int(0.07 * lh)), max(8, int(0.40 * lh))        # heavy outlines are ~0.18 em thick
            adj = ndimage.binary_dilation(fill, iterations=r_adj) & ~ndimage.binary_dilation(fill, iterations=1)
            beyond = (ndimage.binary_dilation(fill, iterations=r_far)
                      & ~ndimage.binary_dilation(fill, iterations=int(0.55 * r_far)))
            if adj.sum() > 20 and beyond.sum() > 20 and np.median(luma[beyond]) >= 75:     # a picture bright enough to tell
                adj_dark, far_dark = float((luma[adj] < 70).mean()), float((luma[beyond] < 70).mean())
                outline = True if adj_dark >= 0.6 and far_dark <= 0.4 else False if adj_dark <= 0.3 else None
    if fill.sum() < 12:
        return None
    plate = bool(float(bg @ luma_w) >= 185 and crop[near].std(0).mean() < 14
                 and np.linalg.norm(np.median(crop[near], 0) - np.median(crop[wide], 0)) >= 9)    # a light, calm patch
    px = crop[fill]
    sel = np.random.default_rng(1).choice(len(px), min(len(px), 150), replace=False)
    ink = float(fill.sum() / max(1, ((d_out <= 0) & box_m).sum()))
    return {"px": px[sel], "ink": ink, "outline": outline, "plate": plate}


def telop_stats(frames: list[tuple[float, str]], rows: list[dict]) -> dict:
    """Where / how big / what colour the on-screen text is. Text that stays (a channel label, a title band) is reported
    as title_band; the rest is the subtitle / telop layer."""
    lines, per_frame = [], []
    for (t, path), row in zip(frames, rows):
        keep = [ln for ln in row.get("lines", []) if ln["box"][3] >= 0.02 and ln["box"][2] >= 0.04
                and len(re.sub(r"\W", "", ln["text"])) >= 2 and ln["box"][1] > 0.03 and ln["conf"] >= 0.3]
        per_frame.append(keep)
        lines += [{**ln, "t": t, "path": path, "key": re.sub(r"\W", "", ln["text"])} for ln in keep]
    n_frames = len(frames)
    out: dict = {"frames_with_text_pct": round(100 * sum(1 for f in per_frame if f) / max(n_frames, 1), 0),
                 "lines": len(lines)}
    if not lines:
        return out
    seen_in: dict[str, set] = {}
    for ln in lines:
        seen_in.setdefault(ln["key"][:6], set()).add(ln["t"])
    static = [ln for ln in lines if len(seen_in[ln["key"][:6]]) >= max(6, 0.4 * n_frames)]
    moving = [ln for ln in lines if ln not in static]
    if static:
        out["title_band"] = {"y": round(float(np.median([ln["box"][1] + ln["box"][3] / 2 for ln in static])), 3),
                             "line_height_pct": round(100 * float(np.median([ln["box"][3] for ln in static])), 1),
                             "frames_pct": round(100 * len(set(ln["t"] for ln in static)) / max(n_frames, 1), 0),
                             "text": static[0]["text"][:30]}
    if moving:
        cy = np.array([ln["box"][1] + ln["box"][3] / 2 for ln in moving])
        wgt = np.array([ln["box"][2] * ln["box"][3] for ln in moving])
        hist, edges = np.histogram(cy, bins=np.arange(0, 1.0001, 0.05), weights=wgt)
        k = int(np.argmax(hist))
        lo, hi = edges[max(0, k - 1)], edges[min(len(edges) - 1, k + 2)]
        band = [ln for ln, c in zip(moving, cy) if lo <= c < hi]
        ids = {id(ln["box"]) for ln in band}
        per_band = [c for c in (sum(1 for ln in f if id(ln["box"]) in ids) for f in per_frame) if c] or [1]
        bcx = np.median([ln["box"][0] + ln["box"][2] / 2 for ln in band])
        y = float(np.median([ln["box"][1] + ln["box"][3] / 2 for ln in band]))
        out["subtitle"] = {"y": round(y, 3), "position": "下" if y >= 0.62 else "中央" if y >= 0.38 else "上",
                           "line_height_pct": round(100 * float(np.median([ln["box"][3] for ln in band])), 1),
                           "lines_per_frame": round(float(np.median(per_band)), 1),
                           "one_line_pct": round(100 * float((np.array(per_band) <= 1).mean()), 0),
                           "chars_per_line": round(float(np.median([len(ln["text"]) for ln in band])), 1),
                           "align": "中央" if abs(bcx - 0.5) < 0.08 else "左" if bcx < 0.5 else "右",
                           "frames_pct": round(100 * len(set(ln["t"] for ln in band)) / max(n_frames, 1), 0)}
    pool, inks, outlines, plates = [], [], [], []
    cache: dict[str, np.ndarray] = {}
    sample = moving or lines
    for ln in sample[:: max(1, len(sample) // 160)]:
        img = cache.setdefault(ln["path"], np.asarray(Image.open(ln["path"]).convert("RGB")))
        info = line_pixels(img, ln["box"])
        if info:
            pool.append(info["px"])
            inks.append(info["ink"])
            outlines.append(info["outline"])
            plates.append(info["plate"])
    if pool:
        out["colors"] = palette(np.concatenate(pool))
        out["ink_coverage"] = round(float(np.median(inks)), 2)
        known = [o for o in outlines if o is not None]
        if len(known) >= 3:
            out["outline_pct"] = round(100 * float(np.mean(known)), 0)
        out["plate_pct"] = round(100 * float(np.mean(plates)), 0)
    return out


# ---------------------------------------------------------------- the sheets Claude looks at


def label_font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(frames: list[tuple[float, str]], out: str, step: float = 5.0, cols: int | None = None) -> str:
    pick, nxt = [], 0.0
    for t, p in frames:
        if t >= nxt - 1e-6:
            pick.append((t, p))
            nxt += step
    im0 = Image.open(pick[0][1])
    tall = im0.height > im0.width
    tw = 220 if tall else 360
    th = round(tw * im0.height / im0.width)
    cols = cols or (8 if tall else 5)
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    for i, (t, p) in enumerate(pick):
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 4, y + 4), f"{int(t // 60)}:{t % 60:04.1f}", fill=(255, 255, 255), font=label_font(14))
    sheet.save(out, quality=85)
    return out


def telop_sheet(frames: list[tuple[float, str]], rows: list[dict], out: str, count: int = 28) -> str | None:
    """Crops of the text on screen (the lines around the most common text height), spread over the video."""
    cands = []
    for (t, p), row in zip(frames, rows):
        ls = [ln for ln in row.get("lines", []) if ln["box"][3] >= 0.02 and ln["conf"] >= 0.3]
        if ls:
            cands.append((t, p, ls))
    if not cands:
        return None
    pick = [cands[int(i)] for i in np.linspace(0, len(cands) - 1, min(count, len(cands)))]
    tiles = []
    for t, p, ls in pick:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        x0 = min(ln["box"][0] for ln in ls)
        y0 = min(ln["box"][1] for ln in ls)
        x1 = max(ln["box"][0] + ln["box"][2] for ln in ls)
        y1 = max(ln["box"][1] + ln["box"][3] for ln in ls)
        if y1 - y0 > 0.5:                       # text all over the frame: show the biggest line only
            big = max(ls, key=lambda ln: ln["box"][2] * ln["box"][3])
            x0, y0, x1, y1 = big["box"][0], big["box"][1], big["box"][0] + big["box"][2], big["box"][1] + big["box"][3]
        pad_y = 0.6 * max(ln["box"][3] for ln in ls)          # heavy outlines reach past the recognised box
        crop = img.crop((int(max(0, x0 - 0.03) * w), int(max(0, y0 - pad_y) * h), int(min(1, x1 + 0.03) * w),
                         int(min(1, y1 + pad_y) * h)))
        k = min(1.0, 520 / crop.width, 150 / crop.height) if crop.width > 520 or crop.height > 150 else 1.0
        tiles.append((t, crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS)))
    cols, cw, ch = 2, 540, 176
    rws = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rws * ch), (24, 24, 27))
    d = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(tiles):
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(im, (x + 8, y + 22))
        d.text((x + 8, y + 3), f"{int(t // 60)}:{t % 60:04.1f}", fill=(255, 255, 255), font=label_font(14))
    sheet.save(out, quality=88)
    return out


# ---------------------------------------------------------------- the look of the pictures


def look_stats(frames: list[tuple[float, str]]) -> dict:
    sats, stds, means = [], [], []
    for _, p in frames[:: max(1, len(frames) // 40)]:
        a = np.asarray(Image.open(p).convert("RGB").resize((160, 160)), np.float32) / 255
        mx, mn = a.max(-1), a.min(-1)
        sats.append(float(((mx - mn) / np.maximum(mx, 0.05))[mx > 0.15].mean()))
        luma = a @ np.array([0.299, 0.587, 0.114], np.float32)
        stds.append(float(luma.std()))
        means.append(float(luma.mean()))
    return {"saturation": round(float(np.median(sats)), 2), "contrast": round(float(np.median(stds)), 2),
            "brightness": round(float(np.median(means)), 2)}


# ---------------------------------------------------------------- the nearest tone
