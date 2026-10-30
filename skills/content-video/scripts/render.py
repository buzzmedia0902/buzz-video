#!/usr/bin/env python3
"""Render one version of edit.json: review stills, a contact sheet, or the silent master video.

  python render.py edit.json --version short --info                 # timeline summary (+ work/<v>/cues_out.json)
  python render.py edit.json --version short --stills 1.0 12.5      # work/review/<v>_<9x16>/tXXX.XX.jpg
  python render.py edit.json --version short --sheet 2.0            # one frame every 2 s -> work/review/<v>_<tag>_sheet.jpg
  python render.py edit.json --version short --video [--jobs 6] [--half] [--only 12.5,40]
                                                                    # work/render/<v>_<tag>.mp4 (no audio: finish.py)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# BT.709 tags on the frames (ffmpeg 8 ignores -color_primaries here); re-encodes inherit them
BT709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"


def setup(project_path: str, version: str) -> None:
    """Set CV_ASPECT / CV_FPS for the version before anything that draws is imported."""
    from cv import media, spec
    p = spec.load(project_path)
    v = p.version(version)
    os.environ["CV_ASPECT"] = v.get("aspect", spec.DEFAULT_ASPECT[v["format"]])
    fps = 30.0
    for part in v["parts"]:
        if part["kind"] != "card":
            f = media.probe(p.src_path(part["src"]))["fps"]
            fps = 29.97 if f and abs(f - round(f)) > 0.01 and round(f) in (30, 60) else 30.0
            break
    os.environ.setdefault("CV_FPS", f"{fps}")


class Comp:
    """Builds one frame: footage (cropped / banded / punched in) or a card, then the tone's overlays."""

    def __init__(self, project_path: str, version: str):
        import skia
        from cv import cues as C
        from cv import framing, overlays, spec, tones
        from cv import gfx as G
        from cv.timeline import Timeline
        self.skia, self.G = skia, G
        self.p = spec.load(project_path)
        self.name = version
        self.v = self.p.version(version)
        tone = self.v.get("tone", self.p.tone)
        from cv.tones.base import _merge
        look_over = _merge(self.p.look, self.v.get("look", {}))  # a version's look.cut keeps the project's other keys
        self.look = tones.get(tone)(self.p.brand, look_over, self.p.speakers, self.v["format"])
        self.tl = Timeline(self.p, self.v, self.look)
        framing.plan(self.tl)
        cue_list = C.load(self.p.cues_path) if self.p.cues_path and os.path.exists(self.p.cues_path) else []
        if self.v.get("subs", True) is False:  # テロップなし: title / names / cards only
            cue_list = []
        self.subs = C.for_version(cue_list, self.tl, self.look.SUBS["min_show"], self.look.SUBS.get("tidy", True))
        self.ovs = overlays.build(self.v, self.tl)
        from cv import layout
        layout.apply(self)  # cards / name plates / big words off the faces
        self.surf = skia.Surface(G.W, G.H)
        self.c = self.surf.getCanvas()
        self._bg_cache: dict = {}

    # -------------------------------------------------------------- footage
    def reader(self, shot, k0: int):
        from cv.reader import Reader
        return Reader(self.p.src_path(shot.src), shot, k0, self.look.GRADE)

    def card_bg(self, shot):
        """Blurred still behind a card: the part's "bg": {"src", "at"} or the nearest talk / clip shot."""
        if shot.part in self._bg_cache:
            return self._bg_cache[shot.part]
        from cv.reader import frame_at
        from cv.timeline import Shot
        ref = None
        bg = shot.data.get("bg")
        if isinstance(bg, dict):
            near = next((s for s in self.tl.shots if s.src == bg.get("src")), None)
            if near is not None:
                ref = Shot(near.kind, near.part, near.src, float(bg.get("at", near.a)), float(bg.get("at", near.a)) + 1,
                           1.0, 0, 1, data=near.data, view=near.view)
        if ref is None and bg is not False:
            others = [s for s in self.tl.shots if s.kind != "card"]
            if others:
                near = min(others, key=lambda s: abs(s.t0 - shot.t0))
                mid = near.a + (near.b - near.a) / 2
                ref = type(near)(near.kind, near.part, near.src, mid, mid + 1, 1.0, 0, 1, data=near.data,
                                 view=near.view)
        img = None
        if ref is not None and ref.view.get("fit") in ("cover", "band"):
            if ref.view["fit"] == "band":  # a card background always fills the canvas
                from cv import framing
                sw, sh = ref.view["src_size"]
                ref.view = {**ref.view, "fit": "cover", "box": (0.0, 0.0, float(self.G.W), float(self.G.H)),
                            "crop": framing.crop(sw, sh, self.G.W / self.G.H, 0.5, 0.42, 1.0)}
            img = self.G.image_from_array(frame_at(self.p.src_path(ref.src), ref, 0, self.look.GRADE))
        self._bg_cache[shot.part] = img
        return img

    def compose(self, n: int, frame):
        skia, G, c = self.skia, self.G, self.c
        t = n / G.FPS
        shot = self.tl.at(n)
        c.clear(skia.ColorBLACK)
        if shot.kind == "card":
            self.look.draw_card_part(c, shot.data, t - shot.t0, shot.n / G.FPS, self.card_bg(shot))
            return self.surf.makeImageSnapshot()
        img = G.image_from_array(frame)
        ox, oy, ow, oh = shot.view["box"]
        if shot.view["fit"] == "band":
            self.look.draw_band_bg(c, img, shot.view["box"])
        z = 1.0
        if shot.view.get("punch"):
            p0, pdur, pz = shot.view["punch"]
            z = 1 + (pz - 1) * G.ease_io((t - shot.t0 - p0) / pdur)
        c.save()
        if z > 1.0001:
            cx, cy = ox + ow / 2, oy + oh * 0.42
            c.clipRect(skia.Rect.MakeXYWH(ox, oy, ow, oh))
            c.translate(cx, cy)
            c.scale(z, z)
            c.translate(-cx, -cy)
        c.drawImageRect(img, skia.Rect.MakeXYWH(ox, oy, ow, oh), G.SAMPLING)
        c.restore()
        k = self.card_mode(t)
        if k > 0:
            from cv import layout
            self.look.draw_card_mode(c, img, shot.view["box"], k, layout.face_at(self, t))
        self.draw_overlays(t)
        return self.surf.makeImageSnapshot()

    def card_mode(self, t: float) -> float:
        """0..1: how far a 9:16 card-mode card is in (eased 0.35 s in and out)."""
        k = 0.0
        for o in self.ovs:
            if o.d.get("_mode") == "card" and o.t0 - 0.05 <= t < o.t1 + 0.05:
                k = max(k, self.G.ease_io((t - o.t0) / 0.35) * self.G.ease_io((o.t1 - t) / 0.35))
        return k

    def draw_overlays(self, t: float) -> None:
        from cv import cues as C
        from cv.overlays import active
        act = active(self.ovs, t)
        order = {"card": 0, "name": 1, "caption": 1, "note": 1, "chapter": 2, "credit": 4, "title_band": 4,
                 "emphasis": 5}
        for ov in sorted((o for o in act if order[o.type] < 3), key=lambda o: order[o.type]):
            self.look.draw_overlay(self.c, ov, t)
        sub = C.at(self.subs, t)
        if sub is not None:
            self.look.draw_sub(self.c, sub, t)
        for ov in sorted((o for o in act if order[o.type] >= 3), key=lambda o: order[o.type]):
            self.look.draw_overlay(self.c, ov, t)

    # -------------------------------------------------------------- cues for audio / finish
    def cues_out(self) -> dict:
        from cv import cues as C
        from cv.canvas import ASPECT, FPS
        tl, look = self.tl, self.look
        sfx = []
        if self.v.get("sfx", True):
            for ov in self.ovs:
                sfx += look.sfx(ov)
            for s in tl.shots:
                if s.kind == "card" and s.f0 == min(x.f0 for x in tl.shots if x.part == s.part):
                    sfx += look.card_sfx(s.data, s.t0, s.n / FPS)
        vb = dict(self.v.get("bgm", {}))
        audio = {**look.AUDIO, **{k: v for k, v in vb.items() if k != "style"}}
        if "style" in vb:
            audio["bgm"] = vb["style"]
        if self.v.get("snap") == "beat" or "bpm" in vb:
            audio["bpm"] = tl.bpm()
        return {
            "version": self.name, "format": self.v["format"], "tone": look.key, "aspect": ASPECT, "fps": FPS,
            "duration": tl.duration, "frames": tl.total, "audio": audio,
            "shots": [{"kind": s.kind, "part": s.part, "src": s.src,
                       "path": self.p.src_path(s.src) if s.src else None, "a": s.a, "b": s.b, "speed": s.speed,
                       "t0": s.t0, "t1": s.t1, "audio": C.audio_mode(s.data, self.v["format"]),
                       "punch": bool(s.view.get("punch")), "med": bool(s.view.get("med"))}
                      for s in tl.shots],
            "sections": [{"part": i, "kind": p["kind"], "card": p.get("card"), "t0": a, "t1": b}
                         for i, ((a, b), p) in enumerate(zip(tl.part_spans, self.v["parts"]))],
            "subs": [{"t0": round(s.t0, 3), "t1": round(s.t1, 3), "text": s.text, "speaker": s.speaker}
                     for s in self.subs],
            "overlays": [{"type": o.type, "t0": round(o.t0, 3), "t1": round(o.t1, 3), "text": o.text} for o in self.ovs],
            "chapters": [[round(o.t0, 2), o.text] for o in self.ovs if o.type == "chapter"],
            "sfx": sorted([[round(t, 3), n, g] for t, n, g in sfx]),
        }


def work_path(comp: Comp, *parts: str) -> str:
    return comp.p.work(*parts)


def write_cues(comp: Comp) -> str:
    path = work_path(comp, comp.name, "cues_out.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comp.cues_out(), f, ensure_ascii=False, indent=1)
    return path


def render_range(args: tuple) -> str:
    project_path, version, a, b, path, half = args
    setup(project_path, version)
    comp = Comp(project_path, version)
    from cv.canvas import H, W, fps_arg
    scale = ",scale=trunc(iw/4)*2:trunc(ih/4)*2" if half else ""
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", fps_arg(),
         "-i", "-", "-vf", f"scale=in_range=full:out_color_matrix=bt709:out_range=tv{scale},format=yuv420p,{BT709}",
         "-c:v", "libx264", "-preset", "veryfast" if half else "medium", "-crf", "26" if half else "19",
         "-x264-params", "keyint=60:min-keyint=30", "-threads", "3", path],
        stdin=subprocess.PIPE, bufsize=W * H * 4 * 2)
    reader, rs = None, None
    for n in range(a, b):
        shot = comp.tl.at(n)
        frame = None
        if shot.kind != "card":
            if rs is not shot:
                if reader:
                    reader.close()
                reader, rs = comp.reader(shot, n - shot.f0), shot
            frame = reader.next()
        img = comp.compose(n, frame)
        enc.stdin.write(img.toarray(colorType=comp.skia.kRGBA_8888_ColorType).tobytes())
    if reader:
        reader.close()
    enc.stdin.close()
    enc.wait()
    return path


def chunks(comp: Comp, workers: int) -> list[tuple[int, int]]:
    """~3 chunks per worker, cut at shot starts."""
    total = comp.tl.total
    target = max(30, total // (workers * 3))
    out, cur = [], 0
    for s in comp.tl.shots[1:]:
        if s.f0 - cur >= target:
            out.append((cur, s.f0))
            cur = s.f0
    out.append((cur, total))
    return out


def video(project_path: str, version: str, comp: Comp, workers: int, half: bool, only: list[float]) -> str:
    from cv.canvas import FPS, tag
    parts = chunks(comp, workers)
    kind = "half" if half else "full"
    pdir = work_path(comp, "render", f"{version}_{tag()}_{kind}", "x")[:-2]
    jobs = [(project_path, version, a, b, os.path.join(pdir, f"part_{k:03d}.mp4"), half)
            for k, (a, b) in enumerate(parts)]
    todo = jobs
    plan_path = os.path.join(pdir, "plan.json")
    plan = {"chunks": parts, "shots": [[s.src, round(s.a, 4), round(s.b, 4), s.speed, s.n] for s in comp.tl.shots]}
    if only:
        old = json.load(open(plan_path)) if os.path.exists(plan_path) else None
        if old != json.loads(json.dumps(plan)):  # the timeline changed since those chunks: they are stale
            print("--only ignored: the timeline changed since the last render, re-rendering every chunk",
                  file=sys.stderr)
        else:
            todo = [j for j in jobs if any(j[2] <= int(t * FPS) < j[3] for t in only) or not os.path.exists(j[4])]
            print(f"re-rendering {len(todo)} of {len(jobs)} chunks", file=sys.stderr)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(render_range, todo))
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    lst = os.path.join(pdir, "parts.txt")
    with open(lst, "w") as f:
        for j in jobs:
            f.write(f"file '{j[4]}'\n")
    out = work_path(comp, "render", f"{version}_{tag()}{'_half' if half else ''}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    print(f"{out}: {len(parts)} chunks, {comp.tl.duration:.1f} s rendered in {time.time() - t0:.0f} s", file=sys.stderr)
    return out


def still(comp: Comp, t: float):
    from cv.canvas import FPS
    from cv.reader import frame_at
    n = min(int(round(t * FPS)), comp.tl.total - 1)
    shot = comp.tl.at(n)
    frame = None
    if shot.kind != "card":
        frame = frame_at(comp.p.src_path(shot.src), shot, n - shot.f0, comp.look.GRADE)
    return comp.compose(n, frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--version", required=True)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--sheet", type=float, help="seconds between frames of the contact sheet")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--half", action="store_true", help="half-size quick preview")
    ap.add_argument("--only", default="", help="re-render only chunks containing these output seconds (12.5,40)")
    a = ap.parse_args()
    from cv.spec import SpecError
    try:
        setup(a.project, a.version)
        comp = Comp(a.project, a.version)
    except SpecError as e:
        raise SystemExit(f"edit.json: {e}")
    from cv.canvas import tag
    cues_path = write_cues(comp)
    if a.info:
        tl = comp.tl
        print(json.dumps({"duration": round(tl.duration, 2), "shots": len(tl.shots),
                          "parts": [[round(x, 2), round(y, 2)] for x, y in tl.part_spans], "subs": len(comp.subs),
                          "overlays": len(comp.ovs), "cues_out": cues_path}, ensure_ascii=False))
    if a.stills:
        d = work_path(comp, "review", f"{a.version}_{tag()}", "x")[:-2]
        for t in a.stills:
            p = os.path.join(d, f"t{t:06.2f}.jpg")
            still(comp, t).save(p, comp.skia.kJPEG, 90)
            print(p)
    if a.sheet:
        sheet(comp, a.sheet, work_path(comp, "review", f"{a.version}_{tag()}_sheet.jpg"))
    if a.video:
        video(a.project, a.version, comp, a.jobs, a.half, [float(x) for x in a.only.split(",") if x])


def sheet(comp: Comp, step: float, path: str) -> None:
    from PIL import Image, ImageDraw
    from cv.canvas import H, W
    times = [round(i * step, 2) for i in range(int(comp.tl.duration / step) + 1)]
    tw = 270 if H > W else 384
    th = round(tw * H / W)
    cols = 8 if H > W else 6
    rows = (len(times) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * (th + 26)), "white")
    d = ImageDraw.Draw(out)
    for i, t in enumerate(times):
        img = still(comp, t)
        arr = img.toarray(colorType=comp.skia.kRGBA_8888_ColorType)
        im = Image.fromarray(arr[..., :3]).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * (th + 26)
        out.paste(im, (x, y + 26))
        d.text((x + 4, y + 6), f"{t:.1f}s", fill="black")
    out.save(path, quality=85)
    print(path)


if __name__ == "__main__":
    main()
