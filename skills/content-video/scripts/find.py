#!/usr/bin/env python3
"""Find 切り抜き segments in a long, transcribed video (cues.json + speech pauses), ranked, with a hook line each.

  python find.py edit.json --src ep [--who 山田] [--grep 出資 100万 ...] [--min 30] [--max 75] [--top 10]
                 [--range A B] [--show]
  python find.py --download URL [--out src] [--dry-run]      # long video from YouTube (yt-dlp, see below)

Needs cues.json for --src (transcribe.py; run speakers.py first when you pass --who: it ranks by the speaker tags).
A candidate starts at a pause and ends at a pause that is also a sentence end (長い間 or 「。？！」), so the cut never
lands inside a word; length is --min..--max seconds. Score (0-100) = the wanted speaker's share of the talk (--who) +
keyword hits (--grep, regex or plain words) + density of strong words / questions / laughter / numbers per 10 s +
how much of the time is speech, with small bonuses for ending on the wanted speaker's line. Segments dominated by
other voices, or without any --grep hit, are halved. The hook is the most quotable single line of the segment
(strong wording, a question, the wanted speaker, a keyword, 8-28 characters): use it as the cold open.

Writes work/find/<src>.json (all details, the lines of each segment and ready-to-paste parts) and prints a table.
Ranking is a shortlist, not a verdict: watch the stills / listen to the top ones before cutting.

--download: plain yt-dlp (its own client choice); keeps the first file that has audio and is at least 720p:
  1. -f "bv*[height<=1080]+ba/b[height<=1080]"
  2. -f 270+ba                                          (the 1080p HLS stream, when 1. gives no usable file)
If neither works, update yt-dlp or ask the user for the file. Only for videos the user may use (their own
appearance, a licensed clip, the show's 切り抜き rules): this mode only runs when asked.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cv.speakers_lib import clip_text, is_laugh, pad  # noqa: E402

STRONG = ("マジ", "めっちゃ", "めちゃ", "ほんま", "絶対", "全然", "本当に", "ぶっちゃけ", "正直", "結局", "要は", "逆に",
          "そもそも", "やばい", "ヤバ", "最高", "最悪", "最強", "無理", "ありえ", "天才", "衝撃", "腹立", "ダサ", "かっこ",
          "大事", "重要", "本質", "失敗", "後悔", "覚悟", "本気", "責任", "やめ", "嘘", "勝", "負け", "稼", "儲", "怖")
QUESTION = re.compile(r"[？?]|(?:ですか|ますか|でしょうか|のか|なの|なん|やろ|でしょ|やん)[。、]?$")
EXCLAIM = re.compile(r"[！!]")
NUMBER = re.compile(r"\d[\d,.]*\s*(?:万|億|円|%|％|倍|人|社|年|割)|[一二三四五六七八九十百千]+(?:万|億|円|人|社|年|割)")
SENTENCE_END = re.compile(r"[。！？!?]$")
FILLER = re.compile(r"^(?:えっと|えー|あの|その|まあ|ま、|なんか|で、|うん|はい|そう)+[、。]?")
SOFT_PAUSE, HARD_PAUSE = 0.12, 0.30   # a pause this long ends a sentence if the text agrees / on its own
LEAD, TAIL = 0.10, 0.12                # breathing room kept before / after the cut
MAX_OVERLAP = 0.5                      # seconds two suggestions may share (so no two get the same hook line)
UNTAGGED_CREDIT = 0.2                  # an untagged line counts a little for --who (speakers.py leaves doubtful ones)
FORMATS = (("default", ["-f", "bv*[height<=1080]+ba/b[height<=1080]"]),
           ("hls1080", ["-f", "270+ba"]))


def norm(text: str) -> str:
    """Comparable text: NFKC, no spaces / markers (cues carry {emphasis} and ｜ marks)."""
    return re.sub(r"[\s{}｜]", "", unicodedata.normalize("NFKC", text))


def show(text: str) -> str:
    return re.sub(r"[{}｜]", "", text)


@dataclass
class Line:
    t0: float
    t1: float
    speaker: str | None
    text: str
    plain: str


def load_lines(cues: list[dict], src: str, lo: float, hi: float) -> list[Line]:
    out = [Line(float(c["t0"]), float(c["t1"]), c.get("speaker"), c["text"], norm(c["text"]))
           for c in cues if c["src"] == src and not c.get("cut") and c.get("text")
           and float(c["t1"]) > lo and float(c["t0"]) < hi]
    return sorted(out, key=lambda x: x.t0)


def is_who(speaker: str | None, who: str | None) -> bool:
    return bool(who and speaker and (who in speaker or speaker in who))


class Features:
    """Per-line numbers as prefix sums, so any run of lines is scored in O(1)."""

    def __init__(self, lines: list[Line], who: str | None, grep: list[re.Pattern]):
        n = len(lines)
        self.mids = [(x.t0 + x.t1) / 2 for x in lines]
        col = {k: np.zeros(n) for k in ("speech", "who", "other", "strong", "quest", "excl", "laugh", "num")}
        kw = np.zeros((n, len(grep)))
        for i, x in enumerate(lines):
            d = x.t1 - x.t0
            col["speech"][i] = d
            if who and x.speaker:
                col["who" if is_who(x.speaker, who) else "other"][i] = d
            col["strong"][i] = min(2, sum(x.plain.count(w) for w in STRONG))
            col["quest"][i] = bool(QUESTION.search(x.plain))
            col["excl"][i] = bool(EXCLAIM.search(x.text))
            col["laugh"][i] = is_laugh(x.text)
            col["num"][i] = min(2, len(NUMBER.findall(x.plain)))
            for k, pat in enumerate(grep):
                kw[i, k] = len(pat.findall(x.plain))
        self.P = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in col.items()}
        self.K = np.vstack([np.zeros(len(grep)), np.cumsum(kw, axis=0)]) if grep else np.zeros((n + 1, 0))

    def span(self, i0: int, i1: int) -> dict:
        s = {k: float(v[i1] - v[i0]) for k, v in self.P.items()}
        hits = self.K[i1] - self.K[i0]
        s["kw_distinct"], s["kw_hits"] = int((hits > 0).sum()), hits
        return s


def pause_list(sp, lo: float, hi: float) -> list[tuple[float, float]]:
    """Silences between speech bursts (>= SOFT_PAUSE) of the source, as (start, end)."""
    bs = sp.bursts(lo, hi)
    return [(a1, b0) for (_, a1), (b0, _) in zip(bs, bs[1:]) if b0 - a1 >= SOFT_PAUSE]


def clean_cuts(pauses: list[tuple[float, float]], lines: list[Line]) -> list[tuple[float, float]]:
    """Pauses where a sentence ends: long enough on their own, or short ones next to 「。？！」."""
    ends = [x.t1 for x in lines]
    out = []
    for g0, g1 in pauses:
        k = bisect.bisect_right(ends, g0 + 0.3) - 1
        punct = k >= 0 and abs(ends[k] - g0) < 0.4 and bool(SENTENCE_END.search(lines[k].text.strip()))
        if g1 - g0 >= HARD_PAUSE or punct:
            out.append((g0, g1))
    return out


def score(f: dict, dur: float, a, first: Line, last: Line) -> tuple[float, dict]:
    """0-100 and the parts. Terms in [0, 1]; weights of the terms the user asked for are active."""
    speech = max(f["speech"], 1e-6)
    terms, weights = {}, {}
    if a.who:
        terms["who"] = (f["who"] + UNTAGGED_CREDIT * (speech - f["who"] - f["other"])) / speech
        weights["who"] = 3.0
    if a.grep:
        terms["kw"] = 0.7 * f["kw_distinct"] / len(a.grep) + 0.3 * min(1.0, float(f["kw_hits"].sum()) / 3)
        weights["kw"] = 3.0
    events = f["strong"] + f["quest"] + 0.7 * f["excl"] + f["laugh"] + 0.6 * f["num"]
    terms["int"] = 1 - math.exp(-events / (dur / 10) / 2.5)
    terms["dens"] = min(1.0, f["speech"] / dur)
    target = a.min + 0.55 * (a.max - a.min)
    terms["len"] = max(0.0, 1 - abs(dur - target) / max(1.0, a.max - a.min))
    weights |= {"int": 2.0, "dens": 1.0, "len": 0.4}
    total = sum(weights[k] * terms[k] for k in weights)
    if a.who:
        total += 0.3 * is_who(last.speaker, a.who) + 0.15 * is_who(first.speaker, a.who)
    s = 100 * total / sum(weights.values())
    if (a.who and terms["who"] < 0.35) or (a.grep and terms["kw"] == 0):
        s *= 0.5
    return s, terms


def utterances(lines: list[Line]) -> list[Line]:
    """Sentence-sized pieces of a segment: split at 「。？！」, speaker changes, pauses and 45 characters."""
    out: list[Line] = []
    for x in lines:
        prev = out[-1] if out else None
        if prev and prev.speaker == x.speaker and x.t0 - prev.t1 < 0.25 and not SENTENCE_END.search(prev.text.strip()) \
                and len(prev.plain) + len(x.plain) <= 45:
            out[-1] = Line(prev.t0, x.t1, prev.speaker, prev.text + x.text, prev.plain + x.plain)
        else:
            out.append(Line(x.t0, x.t1, x.speaker, x.text, x.plain))
    return out


def quote_value(u: Line, a, grep: list[re.Pattern]) -> float:
    """How quotable one sentence is: the cold-open candidate."""
    body = FILLER.sub("", u.plain)
    n = len(body)
    if n < 5 or n > 45:
        return 0.0
    v = 1.0 if 8 <= n <= 28 else 0.5
    if a.who:
        if u.speaker and not is_who(u.speaker, a.who):
            return 0.0
        v += 1.5 if u.speaker else 0.4
    v += 1.2 * any(p.search(u.plain) for p in grep)
    v += 0.8 * min(2, sum(u.plain.count(w) for w in STRONG))
    v += 0.6 * bool(QUESTION.search(u.plain)) + 0.4 * bool(EXCLAIM.search(u.text)) + 0.4 * bool(NUMBER.search(u.plain))
    return v


def build_hooks(lines: list[Line], a, grep: list[re.Pattern]) -> list[dict]:
    ranked = sorted(utterances(lines), key=lambda u: quote_value(u, a, grep), reverse=True)
    return [{"t0": round(u.t0, 2), "t1": round(u.t1, 2), "speaker": u.speaker, "text": show(u.text)}
            for u in ranked[:3] if quote_value(u, a, grep) > 0]


def search(lines: list[Line], cuts: list[tuple[float, float]], feats: Features, a) -> list[dict]:
    """Every (start pause, end pause) pair of the right length, scored; then best-first without overlaps."""
    cands = []
    for p, (_, s0) in enumerate(cuts):
        start = s0 - LEAD
        i0 = bisect.bisect_left(feats.mids, start)
        for q in range(p + 1, len(cuts)):
            end = cuts[q][0] + TAIL
            if end - start > a.max:
                break
            if end - start < a.min:
                continue
            i1 = bisect.bisect_right(feats.mids, end)
            if i1 - i0 < 3:
                continue
            f = feats.span(i0, i1)
            sc, terms = score(f, end - start, a, lines[i0], lines[i1 - 1])
            cands.append({"t0": start, "t1": end, "i0": i0, "i1": i1, "score": sc, "terms": terms, "f": f})
    picked: list[dict] = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        if all(min(c["t1"], q["t1"]) - max(c["t0"], q["t0"]) < MAX_OVERLAP for q in picked):
            picked.append(c)
        if len(picked) >= a.top:
            break
    return picked


def describe(rank: int, c: dict, lines: list[Line], a, grep: list[re.Pattern], src: str) -> dict:
    seg = lines[c["i0"]:c["i1"]]
    f, terms = c["f"], c["terms"]
    hooks = build_hooks(seg, a, grep)
    out = {"rank": rank, "t0": round(c["t0"], 2), "t1": round(c["t1"], 2), "dur": round(c["t1"] - c["t0"], 1),
           "score": round(c["score"], 1), "who_share": round(terms["who"], 2) if "who" in terms else None,
           "kw_hits": {p.pattern: int(h) for p, h in zip(grep, f["kw_hits"]) if h},
           "strong": int(f["strong"]), "questions": int(f["quest"]), "laugh": int(f["laugh"]),
           "speech_share": round(terms["dens"], 2), "hook": hooks[0] if hooks else None, "alts": hooks[1:],
           "part": {"kind": "talk", "src": src, "from": round(c["t0"], 2), "to": round(c["t1"], 2)},
           "lines": [{"t0": round(x.t0, 2), "t1": round(x.t1, 2), "speaker": x.speaker, "text": show(x.text)} for x in seg]}
    if hooks:
        out["cold_open"] = {"kind": "clip", "src": src, "from": round(hooks[0]["t0"] - 0.05, 2),
                            "to": round(hooks[0]["t1"] + 0.1, 2)}
    return out


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def print_table(rows: list[dict], a) -> None:
    print(f"{'順':>2} {'点':>3}  {'範囲':<15}{'秒':>4}  {'本人':>4} {'語':>3} {'強':>2} {'問':>2} {'笑':>2}  フック")
    for r in rows:
        who = "  - " if r["who_share"] is None else f"{r['who_share'] * 100:3.0f}%"
        kw = sum(r["kw_hits"].values())
        hook = f"「{clip_text(r['hook']['text'], 44)}」 {mmss(r['hook']['t0'])}" if r["hook"] else "-"
        print(f"{r['rank']:>2} {r['score']:>3.0f}  {pad(mmss(r['t0']) + '-' + mmss(r['t1']), 15)}{r['dur']:>4.0f}  "
              f"{who:>4} {kw:>3} {r['strong']:>2} {r['questions']:>2} {r['laugh']:>2}  {hook}")
        if a.show:
            for x in r["lines"]:
                print(f"       {mmss(x['t0'])} {pad(x['speaker'] or '-', 6)} {clip_text(x['text'], 70)}")


def ytdlp(url: str, out: str, dry: bool) -> str | None:
    """Download with the first format that gives a usable file (audio + >= 720p); None if none did."""
    from cv import media
    if not shutil.which("yt-dlp"):
        raise SystemExit("yt-dlp not found (brew install yt-dlp)")
    os.makedirs(out, exist_ok=True)
    for name, extra in FORMATS:
        cmd = ["yt-dlp", "--no-playlist", "--progress", *extra, "--merge-output-format", "mp4",
               "-o", os.path.join(out, "%(id)s.%(ext)s"), "--print", "after_move:filepath", url]
        print(f"[{name}] " + shlex.join(cmd), file=sys.stderr)
        if dry:
            continue
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        path = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        if path and os.path.exists(path):
            info = media.probe(path)
            if info["audio"] and info["h"] >= 720:
                print(f"{path}  {info['w']}x{info['h']}  {info['duration'] / 60:.1f} min  (via {name})")
                return path
            print(f"[{name}] unusable ({info['w']}x{info['h']}, audio={info['audio']}): trying the next one", file=sys.stderr)
        else:
            print(f"[{name}] failed (exit {r.returncode}): trying the next one", file=sys.stderr)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?")
    ap.add_argument("--src", help="source id (default: the one the cues use most)")
    ap.add_argument("--who", help="wanted speaker (a name used in cues' speaker tags)")
    ap.add_argument("--grep", nargs="+", default=[], help="keywords / regexes that should appear")
    ap.add_argument("--min", type=float, default=30.0, help="shortest segment, seconds (default 30)")
    ap.add_argument("--max", type=float, default=75.0, help="longest segment, seconds (default 75)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--range", nargs=2, type=float, metavar=("A", "B"), help="search only A-B source seconds")
    ap.add_argument("--show", action="store_true", help="print the lines of every segment")
    ap.add_argument("--download", metavar="URL", help="fetch a long video with yt-dlp (see --help)")
    ap.add_argument("--out", default="src", help="download folder (default ./src)")
    ap.add_argument("--dry-run", action="store_true", help="--download: print the commands only")
    a = ap.parse_args()
    if a.min >= a.max:
        ap.error("--min must be smaller than --max")
    if a.download:
        if not ytdlp(a.download, a.out, a.dry_run) and not a.dry_run:
            raise SystemExit("no format worked: update yt-dlp (brew upgrade yt-dlp), or ask the user for the file")
        return
    if not a.project:
        ap.error("edit.json is required (or use --download URL)")
    from cv import spec, vad
    p = spec.load(a.project)
    cues = json.load(open(p.cues_path or os.path.join(p.root, "cues.json"), encoding="utf-8"))
    src = a.src or max({c["src"] for c in cues}, key=lambda s: sum(c["src"] == s for c in cues))
    lo, hi = a.range or (0.0, 1e12)
    lines = load_lines(cues, src, lo, hi)
    if len(lines) < 3:
        raise SystemExit(f"no cues for source {src!r} in that range: run transcribe.py first")
    tagged = sum(1 for x in lines if x.speaker)
    if a.who and not any(is_who(x.speaker, a.who) for x in lines):
        print(f"! no cue is tagged {a.who!r}: run speakers.py first (the ranking would ignore the speaker)", file=sys.stderr)
        a.who = None
    grep = []
    for g in a.grep:
        try:
            grep.append(re.compile(norm(g), re.I))
        except re.error:
            grep.append(re.compile(re.escape(norm(g)), re.I))
    sp = vad.for_source(p.src_path(src), p.work_dir, src)
    cuts = clean_cuts(pause_list(sp, max(0.0, lines[0].t0 - 1), lines[-1].t1 + 1), lines)
    feats = Features(lines, a.who, grep)
    best = search(lines, cuts, feats, a)
    if not best:
        raise SystemExit(f"no {a.min:.0f}-{a.max:.0f} s segment starts and ends at a pause here ({len(cuts)} usable pauses): "
                         "widen --min / --max or --range")
    rows = [describe(i, c, lines, a, grep, src) for i, c in enumerate(best, 1)]
    out = p.work("find", f"{src}.json")
    json.dump({"src": src, "who": a.who, "grep": a.grep, "min": a.min, "max": a.max, "cues": len(lines),
               "tagged_share": round(tagged / len(lines), 2), "segments": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{src}: {len(lines)} 行 / {len(cuts)} 区切り / 話者タグ {tagged * 100 // len(lines)}%"
          + (f" / 本人={a.who}" if a.who else "") + (f" / 語={' '.join(a.grep)}" if a.grep else "") + f"  → {len(rows)} 件\n")
    print_table(rows, a)
    print(f"\n-> {out}（行・フック・part / cold_open の貼り付け用が入っている）", file=sys.stderr)


if __name__ == "__main__":
    main()
