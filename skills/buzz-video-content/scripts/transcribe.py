#!/usr/bin/env python3
"""Transcribe the talk sources of edit.json and write draft subtitles (cues.json) to correct by hand.

  python transcribe.py edit.json [--src talk] [--range 600 720] [--all] [--lang ja] [--max-chars 18] [--jobs 3]
  python transcribe.py edit.json --align        # after correcting cues.json: re-time the cues to the speech

Only the source ranges the versions use are transcribed (the parts' from / to), unless --range or --all (the whole
source: needed before find.py looks for clips in a long episode; 49 min take ~5-8 min). Cues of other sources and
other ranges already in cues.json are kept.

Every speech burst (cv.vad) is transcribed on its own with whisper-cli (large-v3-turbo): whole-file Whisper drifts
up to ~1 s and hallucinates on silence. Output:
  work/asr/<src>.json   [[t0, t1, text], ...] per burst (source seconds)
  cues.json             [{"src", "t0", "t1", "text", "speaker": null}] split at punctuation / word boundaries,
                        <= max-chars per cue -> fix names and mis-hearings, mark {key words}, then --align.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = os.path.expanduser(os.environ.get("CV_WHISPER_MODEL", "~/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin"))
HALLU = ("ご視聴ありがとうございました", "チャンネル登録", "字幕をご覧いただき")  # Whisper's stock lines on noise
MAX_CPS = 14.0  # characters per second nobody speaks: text that long for the burst is invented
DROP = re.compile(r"[\s、。，．！？!?「」『』（）()…・ー〜\-{}／,.｜]")


def asr_burst(wav16: str, b: tuple[float, float], lang: str, tmp: str) -> list:
    t0, t1 = b
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        w = os.path.join(d, "b.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, t0 - 0.10):.3f}", "-t", f"{t1 - t0 + 0.20:.3f}",
                        "-i", wav16, w], check=True)
        r = subprocess.run(["whisper-cli", "-m", MODEL, "-l", lang, "-nt", "-np", "-mc", "0", "-f", w],
                           capture_output=True)
    if r.returncode != 0:  # never let a failed run wipe cues: stop instead
        raise SystemExit(f"whisper-cli failed on {t0:.2f}-{t1:.2f} s: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    txt = r.stdout.decode("utf-8", "replace").strip().replace("\n", "")
    n = len(DROP.sub("", txt))
    if txt.startswith("[") or (any(h in txt for h in HALLU) and t1 - t0 < 2.0) or n / max(t1 - t0, 0.1) > MAX_CPS:
        txt = ""
    return [round(t0, 3), round(t1, 3), txt]


def ranges_for(project, sid: str) -> list[tuple[float, float]] | None:
    """Union of the parts' [from, to] of a source over all versions; None = whole source (a part without bounds)."""
    spans = []
    for v in project.versions.values():
        for p in v["parts"]:
            if p["kind"] == "card" or p.get("src") != sid:
                continue
            if "from" not in p or ("to" not in p and "dur" not in p):
                return None
            a = float(p["from"])
            b = float(p["to"]) if "to" in p else a + float(p["dur"]) * float(p.get("speed", 1.0))
            spans.append((max(0.0, a - 1.0), b + 1.0))
    spans.sort()
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out or None


def inside(t0: float, t1: float, ranges: list[tuple[float, float]] | None) -> bool:
    """The one test for "this burst / cue belongs to the transcribed ranges" (overlap): old ones that pass are
    replaced by the new transcription, so nothing is doubled at a range seam."""
    return ranges is None or any(t0 < b and a < t1 for a, b in ranges)


def bursts_for(project, sid: str, ranges: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    from cv import vad
    sp = vad.for_source(project.src_path(sid), project.work_dir, sid)
    out = []
    for s, e in sp.bursts(min_len=0.12):
        if not inside(s, e, ranges):
            continue
        if out and s - out[-1][1] < 0.25:  # tiny gaps: one burst
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def split_text(text: str, max_chars: int) -> list[str]:
    """Split a burst's text into cue-sized pieces at sentence ends, commas, then word boundaries."""
    from cv import linebreak
    text = text.strip()
    pieces = [p for p in re.split(r"(?<=[。！？!?])", text) if p.strip()]
    out = []
    for p in pieces:
        while len(p) > max_chars:
            costs = linebreak.break_costs(p)
            cands = [(c + abs(i - len(p) / 2) * 0.8, i) for i, c in costs.items() if 4 <= i <= len(p) - 4]
            cands = [x for x in cands if x[1] <= max_chars + 4] or cands
            if not cands:
                break
            i = min(cands)[1]
            out.append(p[:i])
            p = p[i:]
        out.append(p)
    return [o.strip("、 ") or o for o in out if o.strip()]


def draft_cues(asr: list, sid: str, max_chars: int) -> list[dict]:
    cues = []
    for t0, t1, txt in asr:
        if not txt:
            continue
        parts = split_text(txt, max_chars)
        total = sum(len(DROP.sub("", p)) or 1 for p in parts)
        t = t0
        for p in parts:
            d = (t1 - t0) * (len(DROP.sub("", p)) or 1) / total
            cues.append({"src": sid, "t0": round(t, 2), "t1": round(t + d, 2), "text": p, "speaker": None})
            t += d
    return cues


def align(project, cues: list[dict]) -> list[dict]:
    """Re-time corrected cues: match their characters to the per-burst ASR text (difflib) and snap the ends to
    speech onsets / offsets. Text, speakers and other keys are kept."""
    from cv import vad
    out = []
    for sid in sorted({c["src"] for c in cues}):
        asr = json.load(open(project.work("asr", f"{sid}.json"), encoding="utf-8"))
        chars, times = [], []
        for t0, t1, txt in asr:
            s = DROP.sub("", txt)
            for i, ch in enumerate(s):
                chars.append(ch)
                times.append(t0 + (i + 0.5) / len(s) * (t1 - t0))
        bs = "".join(chars)
        mine = [c for c in cues if c["src"] == sid]
        texts = [DROP.sub("", c["text"]) for c in mine]
        cs = "".join(texts)
        sm = difflib.SequenceMatcher(None, cs, bs, autojunk=False)
        t_of = [None] * len(cs)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                t_of[a + k] = times[b + k]
        known = [i for i, t in enumerate(t_of) if t is not None]
        for i in range(len(cs)):
            if t_of[i] is None and known:
                lo = max((k for k in known if k < i), default=None)
                hi = min((k for k in known if k > i), default=None)
                t_of[i] = (t_of[hi] if lo is None else t_of[lo] if hi is None
                           else t_of[lo] + (t_of[hi] - t_of[lo]) * (i - lo) / (hi - lo))
        sp = vad.for_source(project.src_path(sid), project.work_dir, sid)
        matched = set(known)
        pos, fixed = 0, []
        for c, s in zip(mine, texts):
            a, b = pos, pos + len(s) - 1
            pos += len(s)
            if not s or not any(k in matched for k in range(a, b + 1)):  # nothing heard for it: keep its times
                fixed.append(dict(c))
                continue
            t0 = sp.onset_near(t_of[a] - 0.12, 0.30)
            t1 = sp.offset_near(t_of[b] + 0.12, 0.30)
            fixed.append({**c, "t0": round(t0, 3), "t1": round(max(t1, t0 + 0.4), 3)})
        for i, c in enumerate(fixed):  # monotonic, no overlaps, bridge tiny gaps
            if i and c["t0"] < fixed[i - 1]["t1"]:
                c["t0"] = fixed[i - 1]["t1"]
            if i + 1 < len(fixed) and fixed[i + 1]["t0"] - c["t1"] < 0.35:
                c["t1"] = max(c["t0"] + 0.3, fixed[i + 1]["t0"])
            c["t1"] = round(max(c["t1"], c["t0"] + 0.3), 3)
        out += fixed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--src", nargs="*", help="source ids (default: every source used by a talk part)")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--max-chars", type=int, default=18, help="characters per draft cue (9:16 ~ 14-16, 16:9 ~ 20)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--range", nargs=2, type=float, metavar=("FROM", "TO"), help="only this source range")
    ap.add_argument("--all", action="store_true", help="the whole source (for find.py on a long episode)")
    ap.add_argument("--align", action="store_true", help="re-time the corrected cues.json")
    a = ap.parse_args()
    from cv import media, spec
    p = spec.load(a.project)
    cues_path = p.cues_path or os.path.join(p.root, "cues.json")
    if a.align:
        cues = json.load(open(cues_path, encoding="utf-8"))
        missing = sorted({c["src"] for c in cues} - {s for s in p.sources if os.path.exists(p.work("asr", f"{s}.json"))})
        if missing:
            raise SystemExit(f"no transcription for {', '.join(missing)}: run transcribe.py first (cues.json unchanged)")
        out = align(p, cues)  # all work first, then write: a failure never leaves cues.json missing
        shutil.copy2(cues_path, cues_path + ".bak")
        with open(cues_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(cues_path + ".tmp", cues_path)
        print(f"re-timed {len(cues)} cues -> {cues_path} (previous: {cues_path}.bak)", file=sys.stderr)
        return
    from cv.cues import shows_subs
    srcs = a.src or sorted({part["src"] for v in p.versions.values() for part in v["parts"]
                            if part["kind"] != "card" and shows_subs(part, v["format"])})
    if not srcs:
        raise SystemExit("nothing to transcribe: no talk part and no clip with \"audio\": \"keep\" (a vlog without "
                         "kept speech needs no subtitles). Pass --src <id> [--all] to transcribe anyway, e.g. to "
                         "search a long clip for things that must not be shown.")
    old = json.load(open(cues_path, encoding="utf-8")) if os.path.exists(cues_path) else []
    keep, new = list(old), []
    for sid in srcs:
        ranges = None if a.all else [tuple(a.range)] if a.range else ranges_for(p, sid)
        wav16 = media.extract_audio(p.src_path(sid), p.work("audio", f"{sid}_16k.wav"), sr=16000)
        bs = bursts_for(p, sid, ranges)
        where = "whole source" if ranges is None else ", ".join(f"{x:.0f}-{y:.0f} s" for x, y in ranges)
        print(f"{sid}: {len(bs)} speech bursts ({where})", file=sys.stderr)
        tmp = p.work("tmp", "x")[:-2]
        with ThreadPoolExecutor(a.jobs) as ex:
            asr = list(ex.map(lambda b: asr_burst(wav16, b, a.lang, tmp), bs))
        asr_path = p.work("asr", f"{sid}.json")
        prev = json.load(open(asr_path, encoding="utf-8")) if os.path.exists(asr_path) else []
        asr_all = sorted([x for x in prev if not inside(x[0], x[1], ranges)] + asr)
        json.dump(asr_all, open(asr_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        keep = [c for c in keep if c["src"] != sid or not inside(float(c["t0"]), float(c["t1"]), ranges)]
        new += draft_cues(asr, sid, a.max_chars)
    if os.path.exists(cues_path):
        shutil.copy2(cues_path, cues_path + ".bak")
    cues = sorted(keep + new, key=lambda c: (c["src"], float(c["t0"])))
    with open(cues_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cues, f, ensure_ascii=False, indent=1)
    os.replace(cues_path + ".tmp", cues_path)
    print(f"{len(new)} draft cues ({len(keep)} kept) -> {cues_path}: correct names / mis-hearings, mark "
          f"{{key words}}, then run with --align", file=sys.stderr)


if __name__ == "__main__":
    main()
