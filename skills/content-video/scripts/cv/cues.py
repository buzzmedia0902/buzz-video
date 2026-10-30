"""Subtitles: cues.json (source time, corrected by hand) -> output-time subtitles of one version.

cues.json: [{"src": "talk", "t0": 12.3, "t1": 14.8, "text": "AIで{動画編集}が変わる", "speaker": "山田"}, ...]
  {..}   emphasis (the tone's accent colour / marker)
  ｜     forced line break (one spoken phrase per line)
  "cut": true   keeps the cue out of the picture (e.g. a retake you also listed in a part's "remove")

Rules from viewer feedback: every subtitle stays >= MIN_SHOW s (grown into the silence around it, else joined to a
neighbour, max 40 characters / 2 lines), and pure fillers are trimmed from the text only (the audio keeps them).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

JOIN = "｜"
MERGE_MAX = 40
EARLY = 0.25
TIDY = (  # display-only clean-up of fillers; the voice keeps every word
    (r"^(で|でね|えー|ええと|えっと|あの|あのー|その)、", ""),
    (r"^(まあ|なんか|まぁ)、?", ""),
    (r"^実際ね?、", ""),
    (r"^本当にね?[、…]", ""),
    (r"ちょっと(こう|ね)、?", ""),
    (r"(?<=[をがもはてでかにとら])ですね(、|$)", r"\1"),
    (r"(?<=[のが日])ね、", ""),
)


@dataclass
class Sub:
    t0: float
    t1: float
    text: str
    speaker: str | None = None


def load(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: cues must be a list of {{src, t0, t1, text}}")
    return data


def tidy(text: str) -> str:
    out = text
    for pat, rep in TIDY:
        out = re.sub(pat, rep, out)
    return out if re.sub(r"[{}、…\s" + JOIN + "]", "", out) else text


def plain(text: str) -> str:
    return re.sub(r"[{}" + JOIN + "]", "", text)


def n_chars(text: str) -> int:
    return len(plain(text))


def min_show(subs: list[Sub], spans: list[tuple[float, float]], min_s: float) -> list[Sub]:
    """Every subtitle >= min_s: grow it into the silence around it (never past its part), else join a neighbour."""
    out = [Sub(s.t0, s.t1, s.text, s.speaker) for s in subs]
    for i, cu in enumerate(out):
        if cu.t1 - cu.t0 >= min_s:
            continue
        lo, hi = next(((a, b) for a, b in spans if a - 0.01 <= cu.t0 < b), (cu.t0, cu.t1))
        nxt = min(out[i + 1].t0 if i + 1 < len(out) else hi, hi)
        cu.t1 = max(cu.t1, min(nxt, cu.t0 + min_s))
        prv = max(out[i - 1].t1 if i > 0 else lo, lo)
        cu.t0 = min(cu.t0, max(prv, cu.t1 - min_s, cu.t0 - EARLY))
    def span(s: Sub) -> int:
        return next((k for k, (a, b) in enumerate(spans) if a - 0.01 <= s.t0 < b), -1)

    def joinable(a: Sub, b: Sub) -> bool:  # same speaker, same part, short gap, two single phrases that fit
        return b.t0 - a.t1 < 0.6 and a.speaker == b.speaker and span(a) == span(b) and \
            JOIN not in a.text and JOIN not in b.text and n_chars(a.text + b.text) <= MERGE_MAX

    joined = True
    while joined:
        joined = False
        for i, s in enumerate(out):
            if s.t1 - s.t0 >= min_s:
                continue
            opts = []
            if i > 0 and joinable(out[i - 1], s):
                opts.append((n_chars(out[i - 1].text), i - 1))
            if i + 1 < len(out) and joinable(s, out[i + 1]):
                opts.append((n_chars(out[i + 1].text), i))
            if opts:
                j = min(opts)[1]
                a, b = out[j], out[j + 1]
                out[j:j + 2] = [Sub(a.t0, b.t1, a.text + JOIN + b.text, a.speaker)]
                joined = True
                break
    return out


def audio_mode(part: dict, fmt: str) -> str:
    """keep | duck | mute: a part's own sound. Talk and cold-open clips keep it; vlog b-roll is ducked under the
    music by default."""
    if part["kind"] == "card":
        return "mute"
    return part.get("audio") or ("duck" if fmt == "vlog" and part["kind"] == "clip" else "keep")


def shows_subs(part: dict, fmt: str = "youtube") -> bool:
    """Subtitles only where the voice is heard (audio keep); "subs": true / false overrides."""
    if "subs" in part:
        return bool(part["subs"])
    return audio_mode(part, fmt) == "keep"


def for_version(cues: list[dict], tl, min_s: float, do_tidy: bool = True) -> list[Sub]:
    """Cues of the sources this version plays, mapped to output seconds, tidied, >= min_s where possible."""
    subs = []
    parts, fmt = tl.version["parts"], tl.version["format"]
    for c in cues:
        if c.get("cut") or not c.get("text"):
            continue
        t0, t1 = float(c["t0"]), float(c["t1"])
        for part in tl.parts_playing(c["src"], t0, t1):  # once per part that plays these words
            if not shows_subs(parts[part], fmt):
                continue
            o0 = tl.src_to_out(c["src"], t0, "next", part)
            o1 = tl.src_to_out(c["src"], t1, "prev", part)
            # a cue only grazed by a part (its last syllable at the edge of a cold-open clip) is not shown there
            if o0 is None or o1 is None or o1 - o0 < min(0.5, 0.4 * (t1 - t0)):
                continue
            subs.append(Sub(o0, o1, tidy(c["text"]) if do_tidy else c["text"], c.get("speaker")))
    subs.sort(key=lambda s: s.t0)
    for a, b in zip(subs, subs[1:]):  # no overlaps
        if b.t0 < a.t1:
            a.t1 = max(a.t0 + 0.05, b.t0)
    return min_show(subs, tl.talk_spans(), min_s)


def at(subs: list[Sub], t: float) -> Sub | None:
    lo, hi = 0, len(subs)
    while lo < hi:
        mid = (lo + hi) // 2
        if subs[mid].t0 <= t:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    return subs[i] if i >= 0 and subs[i].t0 <= t < subs[i].t1 else None


def to_srt(subs: list[Sub], path: str) -> None:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            f.write(f"{i}\n{ts(s.t0)} --> {ts(s.t1)}\n{plain(s.text.replace(JOIN, chr(10)))}\n\n")
