#!/usr/bin/env python3
"""Render one version of a project at one aspect ratio.

  python render.py project.json --version vlog --aspect 4:5 --info
  python render.py project.json --version vlog --aspect 4:5 --stills 0.5 3 12.2 [--guides]
  python render.py project.json --version vlog --aspect 4:5 --sheet 0.5 [--guides]
  python render.py project.json --version vlog --aspect 4:5 --video [--jobs 4]

Outputs (under the project's work_dir):
  review/<version>_<9x16>/t03.00.png      stills          review/<version>_<9x16>_sheet.jpg  contact sheet
  render/<version>_<9x16>.mp4             silent master   <version>/cues.json  timings for bgm/sfx/finish
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--version", required=True)
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "4:5", "1:1", "16:9"])
    ap.add_argument("--platform", help="safe-zone preset for 9:16: meta | tiktok (default: project.json)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--sheet", type=float, help="contact sheet every N seconds")
    ap.add_argument("--guides", action="store_true", help="draw safe zones on stills / sheet")
    ap.add_argument("--video", nargs="?", const="", help="render the silent master (optional path)")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--range", nargs=2, type=int, help=argparse.SUPPRESS)  # frame range for a parallel job
    ap.add_argument("--info", action="store_true", help="print shots and cues, render nothing")
    return ap.parse_args()


# BT.709 tags on the frames (ffmpeg 8 ignores -color_primaries/-color_trc here); re-encodes inherit them
BT709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"


def encoder(path: str, fps: int, w: int, h: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,{BT709}", "-c:v", "libx264",
           "-preset", "slow", "-crf", "16", "-profile:v", "high", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def encode_range(tl, E, path: str, n0: int, n1: int, fps: int, w: int, h: int, label: str) -> None:
    enc = encoder(path, fps, w, h)
    t_start = time.time()
    for n in range(n0, n1):
        t = n / fps
        enc.stdin.write(E.to_u8(tl.frame(t)).tobytes())
        tl.release(t)
        if (n - n0) % 60 == 0:
            print(f"{label} frame {n}/{n1} {time.time() - t_start:.0f}s", file=sys.stderr, flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {path}")
    tl.close()


def parallel(a: argparse.Namespace, out: str, n_frames: int) -> None:
    """Split the timeline into a.jobs chunks rendered by sub-processes, then concatenate losslessly."""
    bounds = [round(n_frames * k / a.jobs) for k in range(a.jobs + 1)]
    parts, procs = [], []
    for k in range(a.jobs):
        part = f"{out[:-4]}_part{k}.mp4"
        parts.append(part)
        cmd = [sys.executable, __file__, a.project, "--version", a.version, "--aspect", a.aspect, "--fps", str(a.fps),
               "--range", str(bounds[k]), str(bounds[k + 1]), "--video", part]
        if a.platform:
            cmd += ["--platform", a.platform]
        procs.append(subprocess.Popen(cmd))
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a render job failed (see messages above)")
    lst = out[:-4] + "_parts.txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    for p in parts + [lst]:
        os.remove(p)


def sheet(images: list, path: str, cols: int = 8) -> None:
    from PIL import Image, ImageDraw, ImageFont
    tw = 270
    th = int(tw * images[0][1].height / images[0][1].width)
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * (th + 24)), "white")
    d = ImageDraw.Draw(out)
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    for i, (t, im) in enumerate(images):
        x, y = (i % cols) * tw, (i // cols) * (th + 24)
        out.paste(im.resize((tw, th)), (x, y + 24))
        d.text((x + 4, y + 3), f"{t:.2f}s", fill="black", font=f)
    out.save(path, quality=85)


def main() -> None:
    a = parse()
    raw = json.load(open(os.path.expanduser(a.project), encoding="utf-8"))
    os.environ["SNS_ASPECT"] = a.aspect
    os.environ["SNS_PLATFORM"] = a.platform or raw.get("platform", "meta")
    os.environ["SNS_FPS"] = str(a.fps)
    sys.path.insert(0, HERE)
    from core import canvas, layout, spec
    from core import engine as E
    project = spec.load(a.project)
    v = project.version(a.version)
    still = a.stills is not None or a.sheet is not None
    ctx = spec.Ctx(project, a.version, v, layout.Layout(), still=still)
    tl = importlib.import_module(f"tastes.{v['taste']}").build(ctx)
    tag = canvas.tag()
    if a.range is None:
        with open(ctx.path("cues.json"), "w") as f:
            json.dump(tl.cues(), f, ensure_ascii=False, indent=1)
    if a.info:
        for s in tl.shots:
            print(f"{s.name:10s} {s.t0:6.2f}-{s.t1:6.2f}  {os.path.basename(s.path)} in {s.src_in:.2f} x{s.speed} "
                  f"fit={s.fit} grade={s.grade}")
        print(json.dumps(tl.cues(), ensure_ascii=False))
        return
    review = os.path.join(project.work_dir, "review")
    os.makedirs(review, exist_ok=True)
    if still:
        times = a.stills or [round(k * a.sheet, 3) for k in range(int(tl.duration / a.sheet) + 1)
                             if k * a.sheet < tl.duration]
        from PIL import Image
        thumbs, d = [], os.path.join(review, f"{a.version}_{tag}")
        os.makedirs(d, exist_ok=True)
        for t in times:
            u8 = E.to_u8(tl.frame(t))
            if a.guides:
                u8 = layout.draw_guides(u8)
            img = Image.fromarray(u8)
            if a.stills:
                img.save(os.path.join(d, f"t{t:05.2f}.png"))
            thumbs.append((t, img))
            print("still", a.version, tag, t, flush=True)
        if a.sheet:
            path = os.path.join(review, f"{a.version}_{tag}_sheet.jpg")
            sheet(thumbs, path)
            print(path)
        tl.close()
    if a.video is not None:
        n_frames = int(round(tl.duration * canvas.FPS))
        if a.range:
            encode_range(tl, E, a.video, a.range[0], a.range[1], canvas.FPS, canvas.W, canvas.H, f"{a.version}/{tag}")
            return
        out = a.video or os.path.join(project.work_dir, "render", f"{a.version}_{tag}.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        if a.jobs > 1 and n_frames > 120:
            tl.close()
            parallel(a, out, n_frames)
        else:
            encode_range(tl, E, out, 0, n_frames, canvas.FPS, canvas.W, canvas.H, f"{a.version}/{tag}")
        print(f"video {out}  {n_frames} frames  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # spec errors should read as one clear line
        if type(e).__name__ == "SpecError":
            raise SystemExit(f"project.json: {e}")
        raise
