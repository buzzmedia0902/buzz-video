"""Final mix: voice + BGM ducked under speech + SFX -> -14 LUFS master with a true-peak limit.

Levels are set by measurement, not by guess:
  voice      integrated -16 LUFS as it plays from both channels (the reference everything else is relative to)
  music      loudness of the music *while someone speaks* = voice - |music_db| (14 dB for the trust tone: 21 dB was
             "no BGM at all" for viewers); in pauses it comes up by `lift` dB, on title / end / eyecatch cards it
             swells by `CARD_BOOST_DB`
  ambience   b-roll sound ("duck" shots): voice.py brings every source to PRE_LUFS, here it ends 12 dB under the voice
  sfx        rendered at absolute levels by sfx.py (event gain + the tone's sfx_db)
  master     -14 LUFS integrated, then `limit_true_peak` (oversampled envelope), re-normalised after limiting
Speech detection for the ducking: 20 ms RMS windows, opened 60 ms early, held 400 ms (a plain sample threshold
under-ducks and pumps).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cv.audio import dsp
from cv.audio.dsp import SR
from cv.audio.voice import PRE_LUFS

VOICE_LUFS = -16.0           # of the (dual-mono) voice in the stereo mix
VOICE_PEAK_DB = -2.0         # the voice gain never pushes a sample peak above this (shouts stay out of the limiter)
MASTER_LUFS = -14.0
STEREO_DB = 3.0103           # BS.1770 sums channel powers: a mono signal played from both channels measures 3 dB louder
AMBIENCE_DB = -12.0
LIFT_DB = 6.0                # music level in pauses over its level under speech
CARD_BOOST_DB = {"title": 3.0, "end": 3.0, "eyecatch": 3.0, "text": 1.5}
NO_SPEECH_MUSIC_LUFS = -22.0 # pre-master loudness of the music when nobody speaks at all
FRAME = 960                  # 20 ms


@dataclass
class Mixed:
    pre: np.ndarray          # (2, n) float32 sum of the tracks, not yet normalised (re-master with another ceiling)
    master: np.ndarray       # (2, n) float32 after normalisation and the limiter
    info: dict = field(default_factory=dict)
    stems: dict = field(default_factory=dict)   # voice / music / sfx as mixed (before the master gain), if asked for


def speech_env(voice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(smooth 0..1 ducking envelope per sample, per-frame speech flag at 20 ms)."""
    nf = len(voice) // FRAME
    fr = voice[: nf * FRAME].astype(np.float64).reshape(nf, FRAME)
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    live = db[db > -90.0]
    floor = float(np.percentile(live, 10)) if len(live) else -60.0
    act = (db > np.clip(floor + 12.0, -54.0, -34.0)).astype(float)
    act = np.maximum(act, np.concatenate([act[3:], np.zeros(3)]))               # open 60 ms early
    held = act.copy()
    for k in range(1, 20):                                                       # hold 400 ms
        held[k:] = np.maximum(held[k:], act[:-k])
    env = np.empty(nf)
    prev = 0.0
    for i, a in enumerate(held.tolist()):
        prev = a + (prev - a) * (0.5 if a > prev else 0.95)                      # ~40 ms attack, ~400 ms release
        env[i] = prev
    full = np.interp(np.arange(len(voice)), (np.arange(nf) + 0.5) * FRAME, env) if nf else np.zeros(len(voice))
    return full.astype(np.float32), held > 0.5


def card_boost(cues: dict, n: int) -> np.ndarray:
    """dB the music swells on title / end / eyecatch / text cards: full over the card, 0.3 s ramps outside it."""
    db = np.zeros(n, np.float32)
    ramp = 0.3
    for s in cues.get("sections", []):
        boost = CARD_BOOST_DB.get(s.get("card") or "", 0.0) if s["kind"] == "card" else 0.0
        if not boost:
            continue
        lo, hi = max(0, dsp.n_of(s["t0"] - ramp)), min(n, dsp.n_of(s["t1"] + ramp))
        t = np.arange(lo, hi) / SR
        shape = np.clip(np.minimum((t - (s["t0"] - ramp)) / ramp, ((s["t1"] + ramp) - t) / ramp), 0.0, 1.0)
        db[lo:hi] = np.maximum(db[lo:hi], boost * shape)
    return db


def _speech_blocks(env: np.ndarray, n: int, cues: dict) -> np.ndarray:
    """Bool per 400 ms loudness block: the speaker is talking the whole block, and it is not a card."""
    nsub = n // dsp.HOP
    e = env[: nsub * dsp.HOP].reshape(nsub, dsp.HOP).mean(axis=1) > 0.95
    for s in cues.get("sections", []):
        if s["kind"] == "card":
            e[int(s["t0"] * 10): int(np.ceil(s["t1"] * 10))] = False
    return dsp.block_mask(e, 0.99)


def _pocket(music: np.ndarray) -> np.ndarray:
    """Keep the voice's presence band free: -2 dB around 2.8 kHz, rumble out."""
    return dsp.eq(dsp.hp(music, 45.0, 2), dsp.peaking(2800.0, -2.0, 0.7))


def _place_music(music: np.ndarray, voice: np.ndarray, env: np.ndarray, cues: dict, has_speech: bool,
                 music_db: float, lift: float) -> tuple[np.ndarray, dict]:
    """Music with the pocket EQ, pause lift and card swells, scaled so it sits `music_db` under the voice while
    someone speaks (both measured as they play in the stereo mix, on the blocks where the speaker talks throughout)."""
    n = len(voice)
    m = _pocket(music[:, :n].astype(np.float32))
    m = m * dsp.from_db(lift * (1.0 - env) + card_boost(cues, n)).astype(np.float32)[None, :]
    sel = _speech_blocks(env, n, cues)
    em = dsp.k_energy(m)
    if has_speech and sel.any():
        l_v = dsp.gated_lufs(2.0 * dsp.k_energy(voice), sel)                # dual mono
        gain = (l_v + music_db) - dsp.gated_lufs(em, sel)
    else:
        l_v, gain = None, NO_SPEECH_MUSIC_LUFS - dsp.gated_lufs(em)
    m = m * np.float32(dsp.from_db(gain))
    info = {"music_gain_db": round(float(gain), 2), "speech_blocks": int(sel.sum())}
    if l_v is not None:
        l_m = dsp.gated_lufs(dsp.k_energy(m), sel)
        info.update(voice_under_speech_lufs=round(l_v, 2), music_under_speech_lufs=round(l_m, 2),
                    music_vs_voice_db=round(l_m - l_v, 2))
    return m, info


def build(cues: dict, dialogue: np.ndarray, ambience: np.ndarray, music: np.ndarray | None,
          sfx: np.ndarray | None, ceiling_db: float = -1.5, keep_stems: bool = False) -> Mixed:
    """Voice + music + sfx tracks (music / sfx: (2, n) or None) -> limited master and the measured levels."""
    n = dsp.n_of(cues["duration"])
    audio = cues.get("audio", {})
    music_db = -abs(float(audio.get("music_db", -14.0)))              # "under the voice", whichever sign the tone wrote
    loud_d = dsp.lufs(dialogue) + STEREO_DB
    has_speech = loud_d > -60.0
    g_ref = VOICE_LUFS - loud_d if has_speech else 0.0
    # a quiet clip with one shout (a vlog with a few kept words) must not be lifted into the limiter: keep the
    # loudest voice peak at VOICE_PEAK_DB before the master stage
    peak = float(np.max(np.abs(dialogue))) if dialogue.size else 0.0
    cut = 0.0  # dB the peak cap took off the loudness gain (a few dB is normal speech; > 6 dB means a lone shout)
    if has_speech and peak > 0 and 20 * np.log10(peak) + g_ref > VOICE_PEAK_DB:
        cut = 20 * np.log10(peak) + g_ref - VOICE_PEAK_DB
        g_ref -= cut
    voice = dialogue * np.float32(dsp.from_db(g_ref))
    info: dict = {"voice_gain_db": round(g_ref, 2), "voice_peak_cap_db": round(float(cut), 2),
                  "has_speech": bool(has_speech), "music_db_target": music_db}
    env = speech_env(voice)[0] if has_speech else np.zeros(n, np.float32)
    mix = np.zeros((2, n), np.float32)
    mix += (voice + ambience * np.float32(dsp.from_db(VOICE_LUFS - STEREO_DB - PRE_LUFS + AMBIENCE_DB)))[None, :]
    stems = {"voice": voice.copy()} if keep_stems else {}
    if sfx is not None and sfx.size:
        mix += sfx[:, :n]
        if keep_stems:
            stems["sfx"] = sfx[:, :n]
    if music is not None and music.size:
        m, minfo = _place_music(music, voice, env, cues, has_speech, music_db, float(audio.get("gap_lift_db", LIFT_DB)))
        mix += m
        info.update(minfo)
        if keep_stems:
            stems["music"] = m
    fade = dsp.n_of(0.05)
    mix[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[None, :]
    out, minfo = master(mix, ceiling_db)
    return Mixed(mix, out, {**info, **minfo}, stems)


def master(mix: np.ndarray, ceiling_db: float = -1.5, target: float = MASTER_LUFS) -> tuple[np.ndarray, dict]:
    """Normalise to `target` LUFS, limit on the oversampled envelope, re-normalise what the limiter took."""
    gain = target - dsp.lufs(mix)
    out, gr = mix * np.float32(dsp.from_db(gain)), 0.0
    for _ in range(4):
        out, gr = dsp.limit_true_peak(mix * np.float32(dsp.from_db(gain)), ceiling_db)
        err = target - dsp.lufs(out)
        if abs(err) <= 0.08:
            break
        gain += err
    return out, {"master_gain_db": round(float(gain), 2), "limiter_gr_db": round(float(gr), 2),
                 "lufs": round(dsp.lufs(out), 2), "true_peak": round(dsp.true_peak_db(out), 2)}
