#!/usr/bin/env python3
"""Sound-effect track for short SNS ads - synthesized from scratch, rendered to one stereo float WAV.

  python scripts/sfx.py --cues cues.json --out sfx.wav [--dur 24] [--key D | --bgm-info bgm.wav.json] [--seed 0]

  cues.json   {"duration": 24.0, "sfx": [[3.1, "pop", 1.0], [5.0, "swish"], [9.2, "pop", 0.8, "E6"], ...],
               "bgm": {"key": "D"}}                      -> events are [time s, name, level = 1.0, pitch = None]
  names       send receive sticker link pop swish whoosh tick chime impact type
              - `time` is when the sound is HEARD: swish / whoosh peak on `time` (they start 0.2 / 0.5 s earlier,
                so put them on the cut), everything else starts on `time`.
              - `level` scales the event (1.0 = the calibrated level below). Levels are absolute: nothing is
                normalised here, the mixer decides the balance (-3..-8 dBFS peaks sit well on top of a
                -14 LUFS music bed at unity gain).
              - pitched sounds (pop, link, tick, chime) follow the music key when `--key` / `bgm.key` /
                `--bgm-info` (the info JSON written by bgm.py) is given; an explicit 4th value (note name such
                as "E6", or Hz) sets the base pitch of that event instead.
Output: 48 kHz stereo 32-bit float WAV of exactly round(duration * 48000) samples (silence where nothing plays).
Importable: render_sfx(events, dur, key=None, seed=0) -> np.ndarray[2, n].
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from core import synth as S  # noqa: E402
from core.master import write_wav_f32  # noqa: E402

Rng = np.random.Generator
D6 = S.hz("D6")  # reference pitches (the library is written in D; `ratio` transposes them)


def _send(rng: Rng, r: float) -> np.ndarray:
    return S.glide_tone(1250, 1650, 0.09, 0.03)


def _receive(rng: Rng, r: float) -> np.ndarray:
    a, b = S.glide_tone(700, 1050, 0.07, 0.025), 0.82 * S.glide_tone(1050, 1400, 0.08, 0.03)
    out = np.zeros(int(0.2 * S.SR))
    out[: len(a)] += a
    s = int(0.065 * S.SR)
    out[s:s + len(b)] += b
    return out


def _sticker(rng: Rng, r: float) -> np.ndarray:
    return S.sticker_wobble()


def _link(rng: Rng, r: float) -> np.ndarray:
    return S.chime([(S.hz("E6") * r, 0.09, 3.5, 2.2), (S.hz("B6") * r, 0.07, 3.5, 2.2)], spread=0.12, length=1.6,
                   a_tau=1.4)


def _pop(rng: Rng, r: float) -> np.ndarray:
    return S.bubble_pop(rng, S.hz("A6") * r)


def _swish(rng: Rng, r: float) -> np.ndarray:
    return S.swish(rng, pre=0.2, post=0.14, pan=(-0.75, 0.75))


def _whoosh(rng: Rng, r: float) -> np.ndarray:
    return S.swish(rng, pre=0.5, post=0.45, f_lo=220.0, f_hi=4200.0, pan=(-0.6, 0.6), bw_oct=1.8, curve=2.0)


def _tick(rng: Rng, r: float) -> np.ndarray:
    return S.tick(r)


def _chime(rng: Rng, r: float) -> np.ndarray:
    notes = [(D6 * r, 0.10, 3.5, 2.0), (S.hz("A6") * r, 0.08, 3.5, 1.5), (S.hz("F#7") * r, 0.05, 3.5, 1.1)]
    return S.chime(notes, spread=0.04, length=2.8, a_tau=1.3)


def _impact(rng: Rng, r: float) -> np.ndarray:
    return S.impact(rng, 1.0, thump_level=0.3)


def _type(rng: Rng, r: float) -> np.ndarray:
    return S.key_tap(rng, pitch=float(rng.uniform(0.92, 1.1)))


# name -> (builder, seconds the sound starts BEFORE the event time, per-channel peak at level 1.0, base Hz or None)
SFX: dict[str, tuple[Callable[[Rng, float], np.ndarray], float, float, float | None]] = {
    "send": (_send, 0.0, 0.50, None),
    "receive": (_receive, 0.0, 0.50, None),
    "sticker": (_sticker, 0.0, 0.45, None),
    "link": (_link, 0.0, 0.40, S.hz("E6")),
    "pop": (_pop, 0.0, 0.50, S.hz("A6")),
    "swish": (_swish, 0.2, 0.40, None),
    "whoosh": (_whoosh, 0.5, 0.45, None),
    "tick": (_tick, 0.0, 0.50, 1175.0),
    "chime": (_chime, 0.0, 0.50, D6),
    "impact": (_impact, 0.0, 0.70, None),
    "type": (_type, 0.0, 0.45, None),
}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(key: str | int | None) -> int | None:
    """'D', 'Bb', 'F# major', 7 -> pitch class (None stays None)."""
    if key is None or key == "":
        return None
    if isinstance(key, (int, float)):
        return int(key) % 12
    m = re.match(r"\s*([A-Ga-g])([#b]?)", str(key))
    if not m:
        raise ValueError(f"cannot read key {key!r} (use e.g. 'D', 'Bb', 'F#')")
    return (_PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12


def _ratio(pitch: str | float | None, base: float | None, key_pc: int | None) -> float:
    """Transposition ratio: explicit pitch wins, else the music key (reference tonal centre D), else 1."""
    if pitch not in (None, "") and base:
        f = S.hz(pitch) if isinstance(pitch, str) else float(pitch)
        return f / base
    if key_pc is None:
        return 1.0
    shift = (key_pc - 2 + 6) % 12 - 6  # semitones from D, folded into -6..5
    return 2.0 ** (shift / 12.0)


def render_sfx(events: Sequence[Sequence], dur: float, key: str | int | None = None, seed: int = 0) -> np.ndarray:
    """Mix `events` ([t, name, level=1.0, pitch=None] each) into a (2, round(dur*48000)) float array."""
    n = int(round(dur * S.SR))
    bus = S.Bus(dur, tail=3.0)
    key_pc = parse_key(key)
    for i, ev in enumerate(sorted(events, key=lambda e: float(e[0]))):
        t, name = float(ev[0]), str(ev[1]).lower()
        if name not in SFX:
            raise ValueError(f"unknown sfx {ev[1]!r}; choose from {', '.join(SFX)}")
        level = float(ev[2]) if len(ev) > 2 and ev[2] is not None else 1.0
        pitch = ev[3] if len(ev) > 3 else None
        build, lead, peak, base = SFX[name]
        sig = build(np.random.default_rng([seed, i]), _ratio(pitch, base, key_pc))
        sig = sig / (np.abs(sig).max() + 1e-12) * peak * level
        bus.add(sig if sig.ndim == 2 else np.stack([sig, sig]), t - lead)  # dual-mono: no -3 dB pan-law loss
    out = bus.st[:, :n].copy()
    return out * S.fade_end(n, int(0.005 * S.SR))  # no click if a tail runs past the end


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description="Render the SFX events of a cues.json to a stereo float WAV.")
    ap.add_argument("--cues", help="cues.json (uses duration, sfx, bgm.key)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dur", type=float, help="length in seconds (default: cues.duration)")
    ap.add_argument("--key", help="music key for pitched sounds, e.g. D or Bb")
    ap.add_argument("--bgm-info", help="info JSON written by bgm.py (its 'key' is used)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cues = json.loads(Path(a.cues).read_text()) if a.cues else {}
    dur = a.dur or cues.get("duration")
    if not dur:
        ap.error("need --dur or a --cues file with 'duration'")
    key = a.key or (json.loads(Path(a.bgm_info).read_text()).get("key") if a.bgm_info else None) \
        or (cues.get("bgm") or {}).get("key") or cues.get("key")
    events = cues.get("sfx", [])
    try:
        st = render_sfx(events, float(dur), key, a.seed)
    except ValueError as e:
        ap.error(str(e))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav_f32(out, st)
    peak = 20 * np.log10(np.abs(st).max() + 1e-12)
    print(f"sfx: {len(events)} events, {dur:g}s, key {key or '-'}, peak {peak:.1f} dBFS -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
