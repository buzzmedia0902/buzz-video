#!/usr/bin/env python3
"""Thumbnails (YouTube 1280x720) and covers (Shorts / clips 1080x1920) of one version of edit.json, in the tone's look.

  python thumb.py edit.json --version yt --copy "AIで変わる｜{動画編集}｜{10選}" \\
         --badge "経営者｜{AI導入}｜の現場から" --line "これ、｜{手作業やめよ！}" --sub "編集の時間、ほぼゼロに" \\
         --before "手作業｜8時間" --after "AI｜10分"
  python thumb.py edit.json --version yt --scan            # only the face-crop sheet (work/thumbs/<v>/faces.jpg)
  python thumb.py edit.json --version yt --copy "…" --times 67.6 33.0 82.8   # faces by time (in direction order)

YouTube: three directions for "Test & compare" (upload order = file number; the first one is the fallback):
  1 reaction      his own line (--line, else --copy) in a speech bubble + an expressive face (--thing: what he
                  reacts to)
  2 before_after  the SAME thing wrong -> right (--before / --after: text `label｜value`, an image file or `src@seconds`)
  3 authority     credibility badge (--badge) + huge copy + a confident smile
Short / clip / vlog: two 1080x1920 covers (poster on the tone's ground, and the photo with the title).
Copy marks: `｜` breaks a line, {..} the words that get the loud colour (else the last / the later lines do).
Everything lands in <out_dir>/サムネ/ with a phone-feed preview <version>_feed.jpg. Faces: the whole talk is scanned
every 1.5 s (Apple Vision: sharpness, eyes open, smile / open mouth), then the sharpest frame within 0.3 s of the pick.
Needs ffmpeg and swiftc (Apple Vision: faces, expressions, the cut-out); the tools compile on first use.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROLE_ORDER = ("reaction", "before_after", "authority")
RE_FRAME = re.compile(r"^(\w+)@([0-9.]+)$")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic")


def setup(project_path: str, version: str):
    """Load the project and set CV_ASPECT before anything that imports cv.canvas."""
    from cv import spec
    p = spec.load(project_path)
    v = p.version(version)
    os.environ["CV_ASPECT"] = v.get("aspect", spec.DEFAULT_ASPECT[v["format"]])
    return p, v


def theme_for(p, v, tone: str | None = None):
    """(Look, Theme) of the version's tone; a tone that is not written yet falls back to the base look."""
    from cv import tones
    from cv.thumbs import theme as T
    tone = tone or v.get("tone", p.tone)
    from cv.tones.base import _merge
    look_over = _merge(p.look, v.get("look", {}))
    try:
        cls = tones.get(tone)
    except SystemExit as e:
        print(f"[thumb] {e}: using the base look with the {tone} poster colours", file=sys.stderr)
        from cv.tones.base import Look as cls
    look = cls(p.brand, look_over, p.speakers, v["format"])
    return look, T.for_look(look, tone, look_over.get("fonts", {}).get("thumb"))


def source_range(p, v, want: str | None) -> tuple[str, float, float, list[tuple[float, float]]]:
    """(source id, from, to, used ranges) whose faces to scan: the version's first talk / clip source. Only frames
    inside the ranges the version really plays count (a vlog clip with "dur" uses a few seconds of a long file)."""
    from cv import media
    parts = [x for x in v["parts"] if x["kind"] in ("talk", "clip")]
    sid = want or (parts[0]["src"] if parts else next(iter(p.sources)))
    info = media.probe(p.src_path(sid))
    if info["image"]:
        raise SystemExit(f"source {sid} is a picture: thumbnails are cut from a video (use --src)")
    used = []
    for x in (x for x in parts if x["src"] == sid):
        a = float(x.get("from", 0.0))
        b = float(x["to"]) if "to" in x else a + float(x["dur"]) * float(x.get("speed", 1.0)) if "dur" in x \
            else info["duration"]
        used.append((a, min(b, info["duration"])))
    used = used or [(0.0, info["duration"])]
    return sid, min(a for a, _ in used), max(b for _, b in used), used


def best_frame(path: str, rot: int, t: float, role: str, work: str):
    """cv.thumbs.scan.sharpest, remembered in work/picks.json (the burst decode and landmarks take a few seconds)."""
    from dataclasses import asdict
    from cv.thumbs import scan as S
    cache_path = os.path.join(work, "picks.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    key = f"{role}@{t:.3f}@{os.path.getmtime(os.path.expanduser(path)):.0f}"
    hit = cache.get(key)
    if hit and os.path.exists(hit["path"]):
        return S.Frame(**{**hit, "face": tuple(hit["face"])})
    fr = S.sharpest(path, rot, t, work, role)
    cache[key] = asdict(fr)
    json.dump(cache, open(cache_path, "w"), ensure_ascii=False, indent=1)
    return fr


def parse_times(tokens: list[str] | None) -> tuple[dict[str, float], list[float]]:
    """`role=12.5` or bare seconds (assigned to the directions in order)."""
    named, bare = {}, []
    for tok in tokens or []:
        if "=" in tok:
            role, _, val = tok.partition("=")
            if role not in ROLE_ORDER:
                raise SystemExit(f"--times {tok}: role must be one of {', '.join(ROLE_ORDER)}")
            named[role] = float(val)
        else:
            bare.append(float(tok))
    return named, bare


def content(spec: str | None, p, work: str):
    """Text as is; a picture file or `src@seconds` (a frame of a source) becomes a skia image."""
    if not spec:
        return None
    from cv import gfx as G
    from cv.thumbs import scan as S
    m = RE_FRAME.match(spec)
    if m and m.group(1) in p.sources:
        s = p.sources[m.group(1)]
        png = os.path.join(work, "content", f"{m.group(1)}_{m.group(2)}.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        return G.image(S.grab(s["path"], s["rot"], float(m.group(2)), png))
    path = os.path.expanduser(spec)
    path = path if os.path.isabs(path) else os.path.join(p.root, path)
    if path.lower().endswith(IMAGE_EXT) and os.path.exists(path):
        return G.image(path)
    return spec


def dominant(img) -> tuple | None:
    """A vivid colour of a picture (for the glow behind a reaction): mean of its most saturated pixels."""
    import numpy as np
    import skia
    from PIL import Image
    small = Image.fromarray(np.ascontiguousarray(img.toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]))
    arr = np.asarray(small.resize((48, 48)), np.float32)
    mx, mn = arr.max(-1), arr.min(-1)
    sel = ((mx - mn) / np.maximum(mx, 1) > 0.35) & (mx > 80)
    return tuple(int(v) for v in arr[sel].mean(0)) if sel.sum() >= 20 else None


def save(arr, base: str) -> tuple[str, str]:
    """PNG master + JPEG (the upload: under YouTube's 2 MB limit)."""
    from PIL import Image
    im = Image.fromarray(arr)
    png, jpg = base + ".png", base + ".jpg"
    im.save(png)
    for q in (93, 90, 86, 80):
        im.save(jpg, quality=q, subsampling=0)
        if os.path.getsize(jpg) < 1_900_000:
            break
    return png, jpg


def duration_text(p, v, name: str) -> str:
    """m:ss of the version for the preview's duration chip (cues_out.json after render.py, else the parts)."""
    path = p.work(name, "cues_out.json")
    sec = None
    if os.path.exists(path):
        sec = float(json.load(open(path)).get("duration", 0)) or None
    if sec is None:
        sec = sum(float(x["dur"]) if x["kind"] == "card" else float(x.get("to", 0)) - float(x.get("from", 0))
                  for x in v["parts"])
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def plain(text: str) -> str:
    return re.sub(r"[{}]", "", text).replace("｜", " ")


def repeats(copy: str, title: str) -> bool:
    """True when the thumbnail words just say the title again (a weaker thumbnail: add something the title lacks)."""
    from difflib import SequenceMatcher
    norm = lambda t: re.sub(r"[\s{}｜【】、。！？!?「」]", "", t)
    a, b = norm(copy), norm(title)
    return bool(a and b) and (a in b or b in a or SequenceMatcher(None, a, b).ratio() > 0.8)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--version", required=True)
    ap.add_argument("--copy", help="main copy (｜ = new line, {..} = loud words)")
    ap.add_argument("--line", help="reaction: his own line for the speech bubble (= --copy-for reaction=…)")
    ap.add_argument("--sub", default="", help="support line (a label under / beside the copy)")
    ap.add_argument("--badge", default="", help="credibility plate: `tag｜{HERO}｜rest`")
    ap.add_argument("--thing", help="reaction: what he reacts to (text, picture file or src@seconds)")
    ap.add_argument("--before", help="before / after: the 'before' (label｜value, picture file or src@seconds)")
    ap.add_argument("--after", help="before / after: the 'after'")
    ap.add_argument("--times", nargs="*", help="face times in source seconds: bare numbers in output order (covers: "
                                               "poster, photo) or reaction=12.5 before_after=40 authority=60")
    ap.add_argument("--dirs", nargs="*", choices=ROLE_ORDER, help="which YouTube directions (default: all possible)")
    ap.add_argument("--kind", choices=("wide", "cover"), help="1280x720 directions or 1080x1920 covers (default: by format)")
    ap.add_argument("--copy-for", action="append", default=[], metavar="DIRECTION=TEXT",
                    help="copy of one direction (reaction / before_after / authority), else --copy; repeatable")
    ap.add_argument("--tone", choices=("trust", "friendly", "variety", "wa", "minimal"),
                    help="look of another tone (default: the version's) - to compare tones")
    ap.add_argument("--src", help="source whose faces to use (default: the first talk part)")
    ap.add_argument("--scan", action="store_true", help="only scan the faces and write faces.jpg")
    ap.add_argument("--rescan", action="store_true", help="redo the face scan")
    ap.add_argument("--title", help="video title for the feed preview (default: the version's title or the copy)")
    ap.add_argument("--out", help="output folder (default: <out_dir>/サムネ)")
    return ap


def plan_roles(a, wide: bool) -> list[str]:
    """Which faces to make: the YouTube directions that have what they need, or the two covers."""
    if not wide:
        return ["authority", "reaction"]
    roles = list(a.dirs or ROLE_ORDER)
    if "before_after" in roles and not (a.before and a.after):
        if a.dirs:
            raise SystemExit("--dirs before_after needs --before and --after (text `label｜value`, a picture file "
                             "or src@seconds)")
        roles.remove("before_after")
        print("[thumb] no --before / --after: skipping the before_after direction", file=sys.stderr)
    return roles


def pick_times(a, roles: list[str], auto: dict) -> dict[str, float]:
    """Source time of every face: --times (named or in order), else the scan's pick for the role."""
    named, bare = parse_times(a.times)
    times, bare_iter = {}, iter(bare)
    for r in roles:
        if r in named:
            times[r] = named[r]
        elif (t := next(bare_iter, None)) is not None:
            times[r] = t
        elif r in auto:
            times[r] = auto[r].t
        else:
            raise SystemExit("no usable face in the scan: pass --times (see faces.jpg)")
    return times


def copy_per_direction(a) -> dict[str, str]:
    own = dict(x.split("=", 1) for x in a.copy_for if "=" in x)
    if a.line:
        own.setdefault("reaction", a.line)
    bad = [k for k in own if k not in ROLE_ORDER]
    if bad:
        raise SystemExit(f"--copy-for {bad[0]}: direction must be one of {', '.join(ROLE_ORDER)}")
    return own


def render_all(a, theme, roles: list[str], chosen: dict, cp, own: dict, out: str, wide: bool) -> list[dict]:
    from cv import gfx as G
    from cv.thumbs import cutout, layouts
    made = []
    for n, r in enumerate(roles, 1):
        fr = chosen[r]
        cut = cutout.load(fr.path, fr.face)
        if wide:
            arr = layouts.DIRECTIONS[r](theme, layouts.Copy(**{**cp.__dict__, "copy": own.get(r) or a.copy}), cut)
            name = f"{a.version}_{n}_{r}"
        elif r == "authority":
            arr, name = layouts.cover_poster(theme, cp, cut), f"{a.version}_cover_a_poster"
        else:
            arr, name = layouts.cover_photo(theme, cp, G.image(fr.path), fr.face), f"{a.version}_cover_b_photo"
        png, jpg = save(arr, os.path.join(out, name))
        made.append({"direction": r, "label": layouts.LABELS.get(r, r), "t": round(fr.t, 2), "jpg": jpg, "png": png,
                     "kb": os.path.getsize(jpg) // 1024})
    return made


def main() -> None:
    a = build_parser().parse_args()
    from cv.spec import SpecError
    try:
        p, v = setup(a.project, a.version)
    except SpecError as e:
        raise SystemExit(f"edit.json: {e}")
    wide = (a.kind or ("wide" if v["format"] == "youtube" else "cover")) == "wide"
    if not a.scan and not (a.copy or a.copy_for or a.line):
        raise SystemExit("--copy is required (the main words of the thumbnail), or use --scan to look at faces first")

    from cv.thumbs import cutout, feed, layouts, scan as S
    sid, t0, t1, used = source_range(p, v, a.src)
    path, rot, work = p.src_path(sid), p.sources[sid]["rot"], p.work("thumbs", a.version, "x")[:-2]
    frames = S.scan(path, rot, t0, t1, work, refresh=a.rescan)
    for f in frames:  # a face from a moment the video does not show must not become its thumbnail
        if not any(x - 0.3 <= f.t <= y + 0.3 for x, y in used):
            f.ok, f.why = False, "not in the video"
    good = [f for f in frames if f.ok]
    sheet = S.face_sheet(frames, os.path.join(work, "faces.jpg"))
    print(f"faces: {len(good)} usable of {len(frames)} frames ({sid} {t0:.0f}-{t1:.0f} s) -> {sheet}", file=sys.stderr)
    if a.scan:
        picks = S.pick_roles(frames) if good else {}
        sheet = S.face_sheet(frames, sheet, picks)
        print(json.dumps({"faces_sheet": sheet, "usable": len(good), "frames": len(frames),
                          "auto_picks": {r: round(f.t, 2) for r, f in picks.items()}}, ensure_ascii=False))
        return

    _, theme = theme_for(p, v, a.tone)
    roles = plan_roles(a, wide)
    times = pick_times(a, roles, S.pick_roles(frames, tuple(roles)) if good else {})
    chosen = {r: best_frame(path, rot, t, r, work) for r, t in times.items()}
    cutout.matte_frames([f.path for f in chosen.values()])

    a.copy = a.copy or a.line or a.copy_for[0].split("=", 1)[-1]
    own = copy_per_direction(a)
    cp = layouts.Copy(copy=a.copy, sub=a.sub, badge=a.badge, thing=content(a.thing, p, work),
                      before=content(a.before, p, work), after=content(a.after, p, work), title=a.title or "")
    if cp.thing is not None and not isinstance(cp.thing, str):
        cp.extra["glow"] = dominant(cp.thing)
    out = os.path.abspath(os.path.expanduser(a.out)) if a.out else os.path.join(p.out_dir, "サムネ")
    os.makedirs(out, exist_ok=True)
    made = render_all(a, theme, roles, chosen, cp, own, out, wide)

    video_title = a.title or v.get("title", "")
    if video_title and repeats(a.copy, video_title):
        print("[thumb] the copy repeats the video title: a thumbnail works better with words the title lacks "
              "(his own line, a number, a promise)", file=sys.stderr)
    labels = [f"サムネ{i + 1}  {m['label']}" if wide else f"カバー{'AB'[i]}  {['ポスター型', '写真型'][i]}"
              for i, m in enumerate(made)]
    feed_path = feed.render([(lab, m["jpg"]) for lab, m in zip(labels, made)], plain(video_title) or plain(a.copy),
                            p.brand.get("channel") or p.brand.get("name") or "チャンネル名",
                            os.path.join(out, f"{a.version}_feed.jpg"), duration_text(p, v, a.version),
                            p.brand.get("logo"), theme.accent, vertical=not wide)
    json.dump({"tone": theme.key, "wide": wide, "files": made, "feed": feed_path, "faces_sheet": sheet, "copy": a.copy,
               "times": {r: round(f.t, 2) for r, f in chosen.items()}},
              open(os.path.join(work, "thumbs.json"), "w"), ensure_ascii=False, indent=1)
    print(f"サムネ（{theme.key}）{len(made)}枚 -> {out}")
    for i, m in enumerate(made, 1):
        print(f"  {i} {m['label']:<8} 顔 {m['t']:>6.1f}s  {os.path.basename(m['jpg'])} ({m['kb']} KB)")
    print(f"  スマホのフィードでの見え方: {feed_path}")
    print(f"  顔の一覧（選び直すとき --times）: {sheet}")


if __name__ == "__main__":
    main()
