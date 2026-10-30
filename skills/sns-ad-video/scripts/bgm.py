#!/usr/bin/env python3
"""Original background music for short SNS ads - synthesized from scratch, seeded and reproducible.

  python scripts/bgm.py --style piano|pop|cinematic|lofi --dur 24 --card 20.3 \
         [--bpm 100] [--seed 7] [--cues cues.json] --out bgm.wav [--info bgm.wav.json]

  --card / card_at  time (s) the end card (logo + CTA) appears. Bar lines are fitted so a bar starts exactly
                    there: build-up/swell before it, then a tonic chord + FM bell chime whose onset lands on
                    the card (sample-accurate), ringing out to the end with a 0.3-0.8 s fade.
  --bpm             tempo the fit starts from (default per style: piano 100, pop 128, cinematic 70, lofi 86);
                    the fitted tempo stays within about +-8 % of it (whole bars), else whole beats with a
                    partial first bar (pick-up) for very short lead-ins. If this tempo already puts the card on
                    a whole beat it is kept exactly (cuts on a beat grid stay on the music's beats).
  --beats           whole beats before the card (the cut grid's count, cues bgm.beats): tempo = 60 * beats / card.
  --seed            picks key, chord loops, voicings, melody and patterns (same seed + style = same piece).
  --cues            JSON written by the video side, e.g. {"duration": 24.0, "card_at": 20.3, "cuts": [2.1, 4.0],
                    "sfx": [...], "voice": [...], "bgm": {"style": "piano", "bpm": 100, "seed": 7}};
                    duration / card_at / cuts / bgm.* are used unless given on the command line.
                    pop adds subtle swish accents on the cuts; the other styles ignore them.

Output: 48 kHz stereo 24-bit PCM WAV of exactly round(dur * 48000) samples, -14 LUFS integrated, true peak
<= -1 dBTP, plus the info JSON next to it (<out>.json, e.g. bgm.wav.json; or --info PATH) with {style, bpm, beat,
bar, bars, card_at, lufs, true_peak, key, seed, pickup_beats, beats_before_card, sections, ...}.
Importable: render_bgm(style, dur, card_at, bpm=None, seed=7, cuts=(), beats=None) -> (np.ndarray[2, n], info) and
key_for(style, seed) -> "D" (the key of a piece without rendering it, e.g. to tune sfx.render_sfx(key=...)).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from core import master as M  # noqa: E402
from core.composer.context import Ctx  # noqa: E402
from core.composer.plan import KEYS, STYLE_ID, Grid, fit_tempo, make_plan  # noqa: E402
from core.composer.theory import Key, make_chord  # noqa: E402

STYLES = tuple(STYLE_ID)
TARGET_LUFS = -14.0
TP_CEILING = -1.3  # internal ceiling; the spec is <= -1.0 dBTP


def _clean_cuts(cuts: Sequence[float], card_at: float) -> tuple[float, ...]:
    """Sorted cut times that are usable as accents: after 0.2 s, 0.4 s before the card, >= 0.25 s apart."""
    out: list[float] = []
    for c in sorted(float(x) for x in cuts or ()):
        if 0.2 < c < card_at - 0.4 and (not out or c - out[-1] >= 0.25):
            out.append(c)
    return tuple(out)


def _fade_out(tail: float) -> float:
    return float(min(max(0.25 * tail, 0.3), 0.8, 0.8 * tail))


def build_context(style: str, dur: float, card_at: float, bpm: float | None = None, seed: int = 7,
                  cuts: Sequence[float] = (), beats: int | None = None) -> tuple[ModuleType, Ctx]:
    """Style module + render context (tempo grid, key, form, chord plan) for one piece."""
    style = str(style).strip().lower()
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {', '.join(STYLES)}")
    if not (dur > 0 and 0 < card_at < dur):
        raise ValueError(f"need 0 < card_at < dur (got dur={dur}, card_at={card_at})")
    mod = importlib.import_module(f"core.composer.{style}")
    if beats:  # the video's cut grid dictates the beat count: no fitting
        n_beats = max(2, int(round(beats)))
        fitted = 60.0 * n_beats / card_at
    else:  # an explicit bpm that already puts the card on a whole beat is kept exactly
        n_beats, fitted = fit_tempo(card_at, float(bpm or mod.CFG.bpm), exact=bool(bpm))
    grid = Grid(fitted, 60.0 / fitted, n_beats, n_beats % 4, float(card_at), float(dur))
    sid, sd = STYLE_ID[style], abs(int(seed))
    comp = np.random.default_rng([sd, sid, 1])
    plan = make_plan(mod.CFG, grid, comp)
    ctx = Ctx(plan, np.random.default_rng([sd, sid, 2, int(round(dur * 1000)), int(round(card_at * 1000))]), comp,
              _clean_cuts(cuts, card_at))
    return mod, ctx


def key_for(style: str, seed: int = 7) -> str:
    """Key name ('D', 'Bb', ...) of a (style, seed) piece without rendering it (to tune sfx.py's pitched sounds)."""
    rng = np.random.default_rng([abs(int(seed)), STYLE_ID[str(style).strip().lower()], 1])
    return Key(int(rng.choice(KEYS))).name


def render_bgm(style: str, dur: float, card_at: float, bpm: float | None = None, seed: int = 7,
               cuts: Sequence[float] = (), beats: int | None = None) -> tuple[np.ndarray, dict]:
    """Render one piece. Returns (stereo float array of shape (2, round(dur*48000)), info dict).
    `beats` (optional) = whole beats before the card taken from the video's cut grid; it overrides the tempo fit."""
    mod, ctx = build_context(style, dur, card_at, bpm, seed, cuts, beats)
    grid, plan = ctx.g, ctx.plan
    mix = mod.render(ctx)
    st, meas = M.master(mix, dur, fade_in=0.008, fade_out=_fade_out(dur - card_at), target=TARGET_LUFS,
                        tp_ceiling=TP_CEILING, **getattr(mod, "MASTER", {}))
    info = {"style": style, "bpm": round(grid.bpm, 3), "beat": round(grid.beat, 6), "bar": round(grid.bar, 6),
            "bars": round(grid.bars, 3), "card_at": card_at, "dur": dur, "lufs": meas["lufs"],
            "true_peak": meas["true_peak"], "seed": int(seed), "key": plan.key.name + " major",
            "pickup_beats": grid.pickup, "beats_before_card": grid.n_beats, "bpm_base": float(bpm or mod.CFG.bpm),
            "progression": [make_chord(plan.key, d, mod.CFG.ext[d]).label for d in plan.loop_a],
            "sections": [[s.kind, round(grid.t(s.beat0), 3), round(grid.t(s.end), 3)] for s in plan.sections],
            "glue_gr_db": meas["glue_gr_db"], "limiter_gr_db": meas["limiter_gr_db"], "bands": meas["bands"]}
    return st, info


def _write(out: Path, st: np.ndarray, info_path: Path, info: dict) -> dict:
    """Write the WAV, re-measure with ffmpeg (if installed) and trim the gain if it disagrees."""
    M.write_wav24(out, st)
    meas = M.ffmpeg_loudness(out)
    if meas and (abs(meas["I"] - TARGET_LUFS) > 0.1 or meas["TP"] > -1.1):
        gain = TARGET_LUFS - meas["I"]
        if meas["TP"] + gain > -1.2:  # never trade the true-peak limit for loudness
            gain = -1.2 - meas["TP"]
        M.write_wav24(out, st * 10 ** (gain / 20))
        meas = M.ffmpeg_loudness(out)
    if meas:
        info = {**info, "lufs_internal": info["lufs"], "lufs": meas["I"], "true_peak": meas["TP"]}
    info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    return info


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description="Original BGM generator (piano / pop / cinematic / lofi).")
    ap.add_argument("--style", type=str.lower, choices=STYLES)
    ap.add_argument("--dur", type=float, help="total duration in seconds (8-60 is tuned)")
    ap.add_argument("--card", type=float, help="end-card time in seconds")
    ap.add_argument("--bpm", type=float, help="tempo the fit starts from (an exact fit is kept as is)")
    ap.add_argument("--beats", type=int, help="whole beats before the card (cut grid); overrides the tempo fit")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cues", help="cues.json from the video render")
    ap.add_argument("--out", required=True, help="output .wav")
    ap.add_argument("--info", help="info JSON path (default: <out>.json)")
    a = ap.parse_args(argv)
    cues = json.loads(Path(a.cues).read_text()) if a.cues else {}
    bcfg = cues.get("bgm", {}) if isinstance(cues.get("bgm"), dict) else {}
    style = a.style or bcfg.get("style")
    dur = a.dur or cues.get("duration")
    card = a.card or cues.get("card_at")
    if not (style and dur and card):
        ap.error("need --style, --dur and --card (or a --cues file providing style/duration/card_at)")
    seed = a.seed if a.seed is not None else int(bcfg.get("seed", 7))
    bpm = a.bpm or bcfg.get("bpm")
    beats = a.beats or bcfg.get("beats")
    if not 8 <= dur <= 60:
        print(f"note: duration {dur}s is outside the tuned 8-60 s range", file=sys.stderr)
    try:
        st, info = render_bgm(style, float(dur), float(card), bpm, seed, cues.get("cuts", []), beats)
    except ValueError as e:
        ap.error(str(e))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = _write(out, st, Path(a.info) if a.info else Path(str(out) + ".json"), info)
    print(f"{style} {dur:g}s card {card:g}: {info['bpm']:.2f} bpm ({info['bars']:g} bars), key {info['key']}, "
          f"{info['lufs']:.2f} LUFS, {info['true_peak']:.2f} dBTP -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
