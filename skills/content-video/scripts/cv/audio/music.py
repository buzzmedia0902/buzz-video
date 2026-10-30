"""Original background music for any length from 15 s to 20 min, synthesized in numpy (no samples, no existing tunes).

  compose(style, duration, sections, bpm=None, seed=7, mood=None) -> (float32 (2, round(duration * 48000)), info)
  grid(style, duration, sections, bpm=None, seed=7, mood=None)    -> info without rendering (tempo, beat times)
  default_bpm(style) -> the style's own tempo

styles   lofi (trust)  acoustic (friendly)  pop (variety)  wa (和楽器: koto, shakuhachi, taiko)  ambient (minimal)  piano
sections the `sections` of cues_out.json: title / end / eyecatch cards swell the arrangement, the end card is where the
         music lands (tonic chord, bells, ring-out); without an end card it lands about 2 s before the end.
bpm      None: the style's tempo, fitted a few percent so the landing is on a bar line; a number (beat-snapped vlogs):
         exactly that tempo with a downbeat at t = 0 (cues_out.json audio.bpm).
mood     wa only: "calm" (都節, default) or "bright" (陽音階).
info     style, bpm, beat, bars, key, seed, t_land, beats[], downbeats[], sections[[kind, t0, t1]], lufs, true_peak.

A piece is planned in blocks (form.py), composed up front (melody, patterns, voicings: `prepare`) and rendered chunk by
chunk (`render`) with the ring-out overlap-added, so a 20-minute track needs no more memory than its output and never
repeats a stretch: every block has its own eight-chord progression, melodies are re-composed per block, patterns rotate.
"""
from __future__ import annotations

import importlib

import numpy as np

from cv.audio import dsp, form, instruments
from cv.audio.dsp import SR

STYLES = ("lofi", "acoustic", "pop", "wa", "ambient", "piano")
TARGET_LUFS = -18.0            # level of the returned track (the mix sets the real level relative to the voice)
PEAK_CEILING = -3.0            # dBTP of the returned track
MAX_CHUNK_BARS = 16


def _module(style: str):
    if style not in STYLES:
        raise ValueError(f"unknown BGM style {style!r}; choose from {', '.join(STYLES)}")
    return importlib.import_module(f"cv.audio.style_{style}")


def default_bpm(style: str) -> float:
    """The tempo a style starts from when no bpm is given (a beat-snapping vlog planner may want to start from it)."""
    return _module(style).CFG.bpm


def plan_piece(style: str, duration: float, sections=(), bpm: float | None = None, seed: int = 7, **opts
               ) -> form.Plan:
    """The composed plan of a piece (form + melody + patterns), not yet rendered. opts: style options (wa: mood)."""
    mod = _module(style)
    plan = form.build_plan(style, mod.CFG, float(duration), list(sections), bpm, seed)
    plan.extra.update({k: v for k, v in opts.items() if v is not None})
    mod.prepare(plan, np.random.default_rng([abs(int(seed)), form.STYLE_ID[style], 2]))
    return plan


def chunks(plan: form.Plan) -> list[list[form.Bar]]:
    """Bars grouped by block (at most MAX_CHUNK_BARS); the landing bar stays with the bars before it."""
    out: list[list[form.Bar]] = []
    cur: list[form.Bar] = []
    for bar in plan.bars:
        if cur and bar.kind != "land" and (bar.block != cur[-1].block or len(cur) >= MAX_CHUNK_BARS):
            out.append(cur)
            cur = []
        cur.append(bar)
    out.append(cur)
    return out


def _info(plan: form.Plan, mod, seed: int) -> dict:
    beats = mod.beat_times(plan) if hasattr(mod, "beat_times") else plan.beat_times()
    key = mod.key_name(plan) if hasattr(mod, "key_name") else f"{plan.key.name} major"
    return {"style": plan.style, "bpm": round(plan.bpm, 3), "beat": round(plan.beat, 5), "bars": len(plan.bars),
            "key": key, "seed": int(seed), "duration": plan.duration,
            "t_land": round(plan.t_land, 3), "beats": beats,
            "downbeats": [round(b.t0, 4) for b in plan.bars if b.nbeats == 4 and b.kind != "land"],
            "sections": [[b.kind, round(plan.bars[b.bar0].t0, 3),
                          round(plan.bars[min(b.bar0 + b.bars, len(plan.bars) - 1)].t0, 3)] for b in plan.blocks]}


def grid(style: str, duration: float, sections=(), bpm: float | None = None, seed: int = 7, **opts) -> dict:
    """Tempo, key and beat times of the piece `compose` would render (no audio)."""
    plan = plan_piece(style, duration, sections, bpm, seed, **opts)
    return _info(plan, _module(style), seed)


def compose(style: str, duration: float, sections=(), bpm: float | None = None, seed: int = 7, **opts
            ) -> tuple[np.ndarray, dict]:
    mod = _module(style)
    plan = plan_piece(style, duration, sections, bpm, seed, **opts)
    n = dsp.n_of(duration)
    out = np.zeros((2, n + dsp.n_of(8.0)), np.float32)
    ctx = mod.make_ctx(plan, seed)
    for k, bars in enumerate(chunks(plan)):
        rng = np.random.default_rng([abs(int(seed)), form.STYLE_ID[style], 3, k])
        seg = mod.render(plan, bars, ctx, rng)
        s = dsp.n_of(bars[0].t0)
        e = min(out.shape[1], s + seg.shape[1])
        out[:, s:e] += seg[:, : e - s].astype(np.float32)
    instruments.clear_caches()
    out = out[:, :n]
    _fade_edges(out, plan)
    out *= np.float32(dsp.from_db(TARGET_LUFS - dsp.lufs(out)))
    out, _ = dsp.limit_true_peak(out, PEAK_CEILING)
    info = _info(plan, mod, seed)
    info.update(lufs=round(dsp.lufs(out), 2), true_peak=round(dsp.true_peak_db(out), 2))
    return out.astype(np.float32), info


def _fade_edges(out: np.ndarray, plan: form.Plan) -> None:
    """In place: 8 ms fade-in; after the landing the ring-out fades to exactly zero at the last sample."""
    head = dsp.n_of(0.008)
    out[:, :head] *= np.linspace(0.0, 1.0, head, dtype=np.float32)[None, :]
    t0 = plan.t_land + 0.35 * max(0.0, plan.duration - plan.t_land)
    s = dsp.n_of(t0)
    t = np.arange(s, out.shape[1]) / SR
    out[:, s:] *= (np.cos(0.5 * np.pi * np.clip((t - t0) / max(plan.duration - t0, 1e-3), 0.0, 1.0)) ** 2).astype(np.float32)[None, :]


if __name__ == "__main__":
    import argparse
    import json
    import sys
    ap = argparse.ArgumentParser(description="Render a BGM to a WAV (debug / listening).")
    ap.add_argument("--style", choices=STYLES, required=True)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sections", help="cues_out.json to take the sections from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    secs = json.load(open(a.sections, encoding="utf-8")).get("sections", []) if a.sections else []
    audio, meta = compose(a.style, a.dur, secs, a.bpm, a.seed)
    dsp.write_wav(a.out, audio)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("beats", "downbeats")}, ensure_ascii=False),
          file=sys.stderr)
