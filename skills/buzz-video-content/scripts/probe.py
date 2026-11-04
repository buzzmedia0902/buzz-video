#!/usr/bin/env python3
"""Footage inventory + contact sheets: the first thing to run on a new project.

  python probe.py ~/Downloads/clips [--out work/footage] [--per 12] [--flash]

Point it at a folder that holds only this project's footage (every video / image in it is read).
For every video / image: upright size, orientation, fps (60 fps clips allow 0.5x slow motion), duration,
HDR (HLG -> needs prep_hdr.py once), audio presence and how much of it is speech (talk vs b-roll). Writes footage.json and one contact sheet per clip
(<out>/<clip>.jpg, thumbnails with time stamps) plus overview.jpg (one frame per clip) to look at.
HEIC stills are converted to JPEG next to the sheets (sips).

--flash also decodes every frame to find single-frame flashes (laser / IPL pulses, camera flashes, strobe
lights) that contact sheets miss; clips with flashes need "flash_guard": true. It decodes every frame (seconds
on an idle Mac, minutes on a busy one): use it when the footage shows treatment devices, flash photography or
blinking lights.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cv import osdep  # noqa: E402

VIDEO = (".mov", ".mp4", ".m4v", ".avi", ".mts")
IMAGE = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
FONT = osdep.ui_font()
# same LUT as core/engine.py: HLG clips are tested for flashes in the SDR the renderer sees
LUT_HLG = next((p for p in (os.path.expanduser(os.environ.get("CV_HLG_LUT", "~/.local/share/content-video/hlg2sdr_65.cube")),
                             os.path.expanduser("~/.local/share/sns-ad-video/hlg2sdr_65.cube")) if os.path.exists(p)),
               os.path.expanduser("~/.local/share/content-video/hlg2sdr_65.cube"))


def probe_video(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,color_transfer:"
                          "stream_side_data=rotation:"
                          "format=duration", "-of", "json", path], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    # iPhone 4K HDR lists several side-data entries and only one carries the rotation
    rot = next((int(float(sd["rotation"])) for sd in v.get("side_data_list", []) if sd.get("rotation")), 0)
    w, h = (v["height"], v["width"]) if abs(rot) in (90, 270) else (v["width"], v["height"])
    num, den = (v["r_frame_rate"] + "/1").split("/")[:2]
    fps = round(float(num) / float(den), 2)
    an, ad = (v.get("avg_frame_rate", "0/1") + "/1").split("/")[:2]
    avg = float(an) / float(ad) if float(ad) else 0.0
    if avg and avg < fps * 0.8:  # iPhone Auto-FPS: a 120 / 60 fps timebase over 24-30 real frames, not slow motion
        fps = round(avg, 2)
    return {"kind": "video", "w": w, "h": h, "orientation": "vertical" if h > w else "horizontal" if w > h else "square",
            "fps": fps, "slowmo_ok": fps >= 50, "duration": round(float(j["format"]["duration"]), 2),
            "hdr": v.get("color_transfer") in ("arib-std-b67", "smpte2084"),
            "hlg": v.get("color_transfer") == "arib-std-b67",
            "audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def flashes(path: str, fps: float, hlg: bool = False) -> list[float]:
    """Times of single-frame flashes: frames clearly brighter or redder than the frames around them (about the
    test engine.Source applies with flash_guard, a little more sensitive: a false alarm only costs a check)."""
    side = 32
    vf = f"scale={side}:{side},format=rgb24"
    if hlg and os.path.exists(LUT_HLG):
        vf = (f"scale={side}:{side}:in_color_matrix=bt2020:in_range=tv,format=rgb48le,"
              f"lut3d=file='{osdep.ffpath(LUT_HLG)}':interp=tetrahedral,format=rgb24")
    raw = subprocess.run(["ffmpeg", "-v", "error", *osdep.hwaccel(), "-i", path, "-vf", vf,
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, side, side, 3).astype(np.float32)
    stat = np.stack([fr.mean(axis=(1, 2, 3)), (fr[..., 0] - fr[..., 1]).mean(axis=(1, 2))], axis=1)
    out = []
    for i in range(len(stat)):
        around = np.concatenate([stat[max(0, i - 3):i], stat[i + 1:i + 4]])
        if len(around) >= 2 and ((stat[i] - np.median(around, axis=0)) > 6.0).any():
            out.append(round(i / fps, 2))
    return out


def speech_share(path: str) -> float:
    """Share of the clip with speech (150-4000 Hz energy over an adaptive floor): > 0.4 is a talk."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "8000",
                          "-af", "highpass=f=150,lowpass=f=4000", "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    if len(x) < 800:
        return 0.0
    n = len(x) // 80
    db = 20 * np.log10(np.sqrt((x[: n * 80].reshape(n, 80) ** 2).mean(1)) + 1e-9)
    thr = np.clip(np.percentile(db, 15) + 12, -55, -34)
    return round(float((db > thr).mean()), 2)


def thumbs(path: str, dur: float, n: int, width: int = 240) -> list[tuple[float, Image.Image]]:
    out = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        buf = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf",
                              f"scale={width}:-2,format=rgb24", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True).stdout
        if buf:
            import io
            out.append((t, Image.open(io.BytesIO(buf)).convert("RGB")))
    return out


def sheet(items: list[tuple[str, Image.Image]], path: str, cols: int = 6) -> None:
    w, h = items[0][1].size
    rows = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * w, rows * (h + 28)), "white")
    d, f = ImageDraw.Draw(img), ImageFont.truetype(FONT, 18)
    for i, (label, im) in enumerate(items):
        x, y = (i % cols) * w, (i // cols) * (h + 28)
        img.paste(im.resize((w, h)), (x, y + 28))
        d.text((x + 4, y + 4), label, fill="black", font=f)
    img.save(path, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="work/footage")
    ap.add_argument("--per", type=int, default=12, help="thumbnails per clip")
    ap.add_argument("--flash", action="store_true", help="scan every frame for single-frame flashes (slow)")
    a = ap.parse_args()
    src = os.path.expanduser(a.folder)
    os.makedirs(a.out, exist_ok=True)
    inv, overview = {}, []
    for name in sorted(os.listdir(src)):
        path, ext = os.path.join(src, name), os.path.splitext(name)[1].lower()
        if ext in VIDEO:
            info = probe_video(path)
            if info["audio"]:
                info["speech"] = speech_share(path)
            if a.flash:
                info["flashes"] = flashes(path, info["fps"], info["hlg"])
            th = thumbs(path, info["duration"], a.per)
            if th:
                sheet([(f"{t:.1f}s", im) for t, im in th], os.path.join(a.out, f"{os.path.splitext(name)[0]}.jpg"))
                overview.append((f"{name[:14]} {info['duration']:.0f}s {info['fps']:.0f}fps", th[len(th) // 2][1]))
        elif ext in IMAGE:
            if ext in (".heic", ".heif"):
                jpg = os.path.join(a.out, os.path.splitext(name)[0] + ".jpg")
                osdep.convert_still(path, jpg)
                path = jpg
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            info = {"kind": "image", "w": im.width, "h": im.height, "path": path,
                    "orientation": "vertical" if im.height > im.width else "horizontal"}
            overview.append((name[:20], im.resize((240, int(240 * im.height / im.width)))))
        else:
            continue
        inv[name] = info
        print(f"{name:28s} {info['kind']:5s} {info['w']}x{info['h']} {info.get('fps', '')}fps "
              f"{info.get('duration', '')}s{'  HDR' if info.get('hdr') else ''}"
              f"{'  speech %d%%' % (100 * info['speech']) if 'speech' in info else ''}", file=sys.stderr)
    if not inv:
        raise SystemExit(f"no video or image files in {src}")
    with open(os.path.join(a.out, "footage.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)
    sheet(overview, os.path.join(a.out, "overview.jpg"))
    hdr = [k for k, v in inv.items() if v.get("hdr")]
    flash = [k for k, v in inv.items() if len(v.get("flashes", [])) >= 2]
    summary = {"clips": len(inv), "vertical": sum(v["orientation"] == "vertical" for v in inv.values()),
               "slowmo_ok": [k for k, v in inv.items() if v.get("slowmo_ok")], "hdr": hdr,
               "overview": os.path.join(a.out, "overview.jpg")}
    if a.flash:
        summary["flash"] = flash
    print(json.dumps(summary, ensure_ascii=False))
    if hdr:
        print("HDR (HLG) clips found: run prep_hdr.py once per machine; shots of them are converted automatically.",
              file=sys.stderr)
    if flash:
        print(f"single-frame flashes in {', '.join(flash)}: set \"flash_guard\": true on those clips "
              "(times in footage.json).", file=sys.stderr)


if __name__ == "__main__":
    main()
