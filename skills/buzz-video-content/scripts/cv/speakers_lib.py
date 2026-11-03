"""Voice-ID decisions (pure numpy): reference centroids, per-cue verdicts, "never guess" thresholds.

The torch / speechbrain part (ECAPA embeddings) lives in speakers.py, so this module imports in any env.
Why so strict: a wrong name tag puts words in someone's mouth, the on-screen name plate stays
on one person while others shout (plate != speaker), and ECAPA margins collapse in laughter and overlap - so a cue
is tagged only when its voice is clearly one known speaker; everything else stays untagged.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

MIN_COS = 0.50        # the best centroid must be at least this close ...
MIN_MARGIN = 0.15     # ... and beat the runner-up by this much
MIN_DUR = 1.0         # shorter cues give unstable embeddings
WIN, HOP = 2.0, 1.0   # analysis windows of a long cue (a speaker change inside it shows up as a disagreement)
CONFLICT_SHARE = 0.25  # this share of windows owned by someone else = mixed voices
REF_CHUNK = 3.0       # reference ranges are embedded in chunks of about this length

WHY_JA = {"short": "短", "laugh": "笑", "low": "低", "close": "差", "mixed": "混", "manual": "手"}
WHY_HELP = {"短": "短すぎる", "笑": "笑い・拍手・音楽", "低": "似た声なし", "差": "2人が僅差", "混": "声が混在", "手": "手動タグを維持"}
LAUGH = re.compile(r"[（(\[【]\s*(?:笑[いう]?|拍手|音楽|歓声|BGM)\s*[）)\]】]|[はハふフへヘ]{3,}|[wｗ]{2,}")


@dataclass
class Verdict:
    label: str | None
    why: str = ""                       # short | laugh | low | close | mixed ("" when tagged)
    cos: float = 0.0
    margin: float = 0.0
    sims: dict = field(default_factory=dict)
    wins: str = ""                      # one char per analysis window: speaker letter, or . when not confident


@dataclass
class Ref:
    speaker: str
    src: str
    t0: float
    t1: float
    X: np.ndarray                       # (chunks, dim) unit rows


@dataclass
class RefReport:
    ref: Ref
    own: float | None                   # similarity to the speaker's other refs (None: only one ref)
    other: float                        # best similarity to another speaker's centroid
    other_name: str
    ok: bool


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def spans(a: float, b: float) -> list[tuple[float, float]]:
    """Analysis windows of a cue: the cue itself when it is short, else WIN-long windows every HOP (last one flush)."""
    if b - a < 2 * WIN:
        return [(a, b)]
    starts = list(np.arange(a, b - WIN, HOP)) + [b - WIN]
    return [(round(s, 3), round(s + WIN, 3)) for s in starts]


def chunks(t0: float, t1: float) -> list[tuple[float, float]]:
    """A reference range cut into ~REF_CHUNK pieces (no piece shorter than 1.5 s)."""
    n = max(1, int((t1 - t0) // REF_CHUNK))
    step = (t1 - t0) / n
    return [(round(t0 + i * step, 3), round(t0 + (i + 1) * step, 3)) for i in range(n)]


def centroids(refs: list[Ref]) -> dict[str, np.ndarray]:
    out = {}
    for name in dict.fromkeys(r.speaker for r in refs):
        out[name] = unit(np.concatenate([r.X for r in refs if r.speaker == name]).mean(0))
    return out


def check_refs(refs: list[Ref], min_cos: float, min_margin: float) -> list[RefReport]:
    """Does each reference look like its own speaker and unlike the others? (leave-one-out: a ref never votes for
    itself). A ref that fails would poison the centroid and cause wrong tags."""
    allc = centroids(refs)
    out = []
    for r in refs:
        v = unit(r.X.mean(0))
        rest = [x for x in refs if x.speaker == r.speaker and x is not r]
        own = float(v @ centroids(rest)[r.speaker]) if rest else None
        others = {n: float(v @ c) for n, c in allc.items() if n != r.speaker}
        name, best = max(others.items(), key=lambda kv: kv[1]) if others else ("", -1.0)
        ok = (own >= min_cos and own - best >= min_margin) if own is not None else best < min_cos
        out.append(RefReport(r, own, best, name, ok))
    return out


def usable_refs(reports: list[RefReport]) -> list[Ref]:
    """Refs that passed; a speaker whose refs all failed keeps them all (and the caller warns)."""
    out = []
    for name in dict.fromkeys(rp.ref.speaker for rp in reports):
        mine = [rp for rp in reports if rp.ref.speaker == name]
        out += [rp.ref for rp in mine if rp.ok] or [rp.ref for rp in mine]
    return out


def sims_to(e: np.ndarray, cents: dict[str, np.ndarray]) -> dict[str, float]:
    e = unit(e)
    return {n: float(e @ c) for n, c in cents.items()}


def decide(sims: dict[str, float], min_cos: float, min_margin: float) -> tuple[str | None, str, float, float]:
    """(speaker | None, why, cos, margin). Tagged only when close enough AND clearly ahead of the runner-up."""
    if not sims:
        return None, "low", 0.0, 0.0
    order = sorted(sims, key=sims.get, reverse=True)
    best = sims[order[0]]
    margin = best - (sims[order[1]] if len(order) > 1 else 0.0)
    if best < min_cos:
        return None, "low", best, margin
    if margin < min_margin:
        return None, "close", best, margin
    return order[0], "", best, margin


def judge(embs: np.ndarray, cents: dict[str, np.ndarray], min_cos: float, min_margin: float) -> Verdict:
    """Verdict of one cue from the embeddings of its analysis windows (one row for a short cue)."""
    sims = sims_to(embs.mean(0), cents)
    label, why, cos, margin = decide(sims, min_cos, min_margin)
    if len(embs) == 1:
        return Verdict(label, why, cos, margin, sims)
    votes = [decide(sims_to(e, cents), min_cos, min_margin)[0] for e in embs]
    letters = legend(cents)
    wins = "".join(letters[v] if v else "." for v in votes)
    seen = Counter(v for v in votes if v)
    if len(seen) > 1:
        # two voices clearly heard: untagged if the cue has no clear owner, or a quarter of it belongs to the other
        others = sum(n for v, n in seen.items() if v != label)
        if label is None or others >= max(1, math.ceil(CONFLICT_SHARE * len(votes))):
            return Verdict(None, "mixed", cos, margin, sims, wins)
    return Verdict(label, why, cos, margin, sims, wins)


def legend(names) -> dict[str, str]:
    """Speaker -> letter used in the window strings (A, B, ... in edit.json order)."""
    return {n: chr(65 + i) for i, n in enumerate(names)}


def is_laugh(text: str) -> bool:
    """Laughter / applause / music markers: the voice is overlapped, so the cue is left untagged."""
    return bool(LAUGH.search(text))


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, right: bool = False) -> str:
    """Pad to w terminal columns (Japanese characters count double)."""
    gap = " " * max(0, w - width(s))
    return gap + s if right else s + gap


def clip_text(s: str, w: int) -> str:
    out, used = "", 0
    for c in s:
        d = 2 if unicodedata.east_asian_width(c) in "WF" else 1
        if used + d > w - 1:
            return out + "…"
        out, used = out + c, used + d
    return out
