"""wa_arrange - form plan + ensemble parts: turns (landing time, seed, tempo, mood) into timed events.

Form: intro (opening gesture, optional shakuhachi / soft taiko roll) -> phrase pairs A, A2, B, A3, ... with 1-2 bar
interludes ("ma") -> final pair whose last bar leads (with a ritardando) onto the tonic stroke at `t_final`.
A long piece is told in movements of at most PAIRS_PER_MOVE pairs; every movement has its own theme (rhythm cells,
cadence plans, accompaniment figures, motif memory), so ten minutes do not repeat two tunes.
Parts: melody koto, 2nd koto (light broken patterns), 17-string bass koto, sho-like drone, sparse shamisen accents,
shakuhachi (intro / between phrases / ending), taiko (title swell, lifts, the landing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv.audio.wa_score import Mood, Note, Theme, TimeMap, cell_notes, make_theme, search_line, write_pair

PAIRS_PER_MOVE = 10
# (slot, voice 0=low 1=mid 2=high, velocity) per bar
ACC = {
    "sparse": ((0, 0, 0.72), (4, 1, 0.55)),
    "lilt": ((0, 0, 0.72), (3, 2, 0.42), (4, 1, 0.55), (6, 2, 0.45)),
    "pulse": ((0, 0, 0.70), (2, 2, 0.42), (4, 1, 0.55), (6, 2, 0.42)),
    "flow": ((0, 0, 0.72), (2, 1, 0.48), (3, 2, 0.42), (4, 1, 0.55), (6, 2, 0.48)),
    "dotted": ((0, 0, 0.72), (3, 1, 0.42), (4, 2, 0.50), (7, 1, 0.38)),
}
JITTER = {"koto": 0.006, "koto2": 0.008, "bass": 0.005, "shami": 0.005, "shaku": 0.015}
SHAMI_FIGURES = (((0.0, 74, 0.55), (0.5, 69, 0.42)),                       # chin-ton
                 ((0.0, 62, 0.48), (0.5, 69, 0.42), (1.0, 74, 0.5)),       # ton-ton-chin
                 ((0.0, 69, 0.32), (0.25, 74, 0.55)))                      # sukui grace


@dataclass(frozen=True)
class Section:
    kind: str       # intro | A | A2 | A3 | B | B2 | Af | inter | short
    bar: int
    bars: int
    move: int = 0   # movement (theme) the section belongs to


@dataclass(frozen=True)
class Event:
    inst: str
    t: float
    dur: float
    pitch: int
    vel: float
    orn: str = ""
    open_pitch: int | None = None
    hold: float = 0.0
    extra: tuple = ()


@dataclass(frozen=True)
class Score:
    events: tuple
    t_final: float
    tmap: TimeMap
    mood: Mood
    themes: tuple
    form: tuple


# =================================================================== form
def _pair_kinds(k: int) -> tuple[str, ...]:
    table = {1: ("Af",), 2: ("A", "Af"), 3: ("A", "B", "Af"), 4: ("A", "A2", "B", "Af"),
             5: ("A", "A2", "B", "B2", "Af"), 6: ("A", "A2", "B", "A3", "B2", "Af")}
    if k in table:
        return table[k]
    extra = tuple(("A3", "B2")[j % 2] for j in range(k - 6))
    return ("A", "A2", "B", "A3", "B2") + extra + ("Af",)


def plan_form(n_bars: int) -> tuple[Section, ...]:
    intro = 2 if n_bars >= 10 else (1 if n_bars >= 5 else 0)
    body = n_bars - intro
    k = int(round(body / 4.5))
    while k > 0 and 4 * k > body:
        k -= 1
    if k == 0:
        head = (Section("intro", 0, intro),) if intro else ()
        return head + (Section("short", intro, body),)
    rest = body - 4 * k
    gaps = [0] * (k - 1)
    for j in range(rest):
        if gaps:
            gaps[j % len(gaps)] += 1
        else:
            intro += 1
    out = [Section("intro", 0, intro)] if intro else []
    bar = intro
    for j, kind in enumerate(_pair_kinds(k)):
        move = min(j // PAIRS_PER_MOVE, (k - 1) // PAIRS_PER_MOVE)      # the final pair belongs to the last movement
        out.append(Section(kind, bar, 4, move))
        bar += 4
        if j < len(gaps) and gaps[j]:
            out.append(Section("inter", bar, gaps[j], move))
            bar += gaps[j]
    return tuple(out)


def _plan_for(sec: Section, theme: Theme) -> tuple:
    if sec.kind == "Af":
        return (theme.plan_a[0], theme.plan_a[1], "S", "D")
    return theme.plan_a if sec.kind.startswith("A") else theme.plan_b


def bar_frames(form: tuple[Section, ...], themes: tuple) -> tuple[str, ...]:
    frames: list[str] = []
    for sec in form:
        theme = themes[sec.move]
        if sec.kind in ("A", "A2", "A3", "B", "B2", "Af"):
            frames += list(_plan_for(sec, theme))
        elif sec.kind == "inter":
            frames += list(("T", "T", "S", "D")[-sec.bars:]) if sec.bars <= 4 else ["T"] * sec.bars
        elif sec.kind == "intro":
            frames += ["T"] * max(0, sec.bars - 1) + (["D"] if sec.bars >= 2 else ["T"] * sec.bars)
        else:
            frames += ["T"] * max(0, sec.bars - 1) + ["D"] * min(1, sec.bars)
    return tuple(frames)


# =================================================================== melody koto
def melody_part(rng: np.random.Generator, mood: Mood, themes: tuple, form: tuple[Section, ...]) -> tuple[list[Note], int]:
    """All phrase pairs (+ short form) as koto notes; returns (notes, final tonic pitch)."""
    raw: list[tuple[float, float, int, float]] = []
    memos: dict[int, dict] = {}
    prev, final_tonic = None, 62
    for sec in form:
        theme, memo = themes[sec.move], memos.setdefault(sec.move, {})
        if sec.kind == "intro" and not theme.shaku_intro and (sec.bars >= 2 or theme.opening == "drone"):
            rhy = cell_notes(theme.leadin)
            line = search_line(rng, mood, rhy, "D", None, None, 3, 7)
            bar = sec.bar + sec.bars - 1
            raw += [(bar * 4 + s / 2, ln / 2, mood.mel[i], mood.vel - 0.06) for (s, ln), i in zip(rhy, line)]
            prev = line[-1]
        if sec.kind in ("intro", "inter"):
            continue
        if sec.kind == "short":
            line, final_tonic = _short_line(rng, mood, theme, sec.bars)
            raw += [(sec.bar * 4 + b, ln, p, v) for b, ln, p, v in line]
            continue
        res = write_pair(rng, mood, theme, sec.kind, _plan_for(sec, theme), prev, memo, final=sec.kind == "Af")
        key = "A" if sec.kind.startswith("A") else "B"
        if key not in memo:
            memo[key] = (res.motif, res.call2)
        prev = res.last
        for slot, length, idx, vel in res.notes:
            raw.append((sec.bar * 4 + slot / 2, length / 2, mood.mel[idx], vel))
        if sec.kind == "Af":
            final_tonic = mood.mel[res.last - 1]
        else:
            raw.append(_awase(rng, raw[-1]))
    return ornament(rng, mood, [r for r in raw if r is not None]), final_tonic


def _awase(rng: np.random.Generator, last: tuple) -> tuple | None:
    """Optional awasezume (octave dyad) on a full-cadence note."""
    beat, beats, pitch, vel = last
    if rng.random() > 0.35:
        return None
    return (beat, beats, pitch - 12 if pitch - 12 >= 55 else pitch + 12, vel * 0.8)


def _short_line(rng: np.random.Generator, mood: Mood, theme: Theme, bars: int) -> tuple[list, int]:
    """Very short pieces: motif bars then a final approach bar onto the upper neighbour."""
    out, prev = [], None
    for b in range(bars):
        last = b == bars - 1
        rhy = cell_notes(theme.final_cell if last else theme.cell_a)
        target = 4 if last else None
        line = search_line(rng, mood, rhy, "D" if last else "T", prev, target, 3, 9)
        out += [(b * 4 + s / 2, ln / 2, mood.mel[i], mood.vel) for (s, ln), i in zip(rhy, line)]
        prev = line[-1]
    return out, mood.mel[3]


def ornament(rng: np.random.Generator, mood: Mood, raw: list) -> list[Note]:
    """押し手 / 揺り / 突き色 / grace notes on the melody (raw = (beat, beats, pitch, vel))."""
    raw = sorted(raw, key=lambda r: (r[0], r[2]))
    open_set = set(mood.koto)
    out: list[Note] = []
    i = 0
    while i < len(raw):
        beat, beats, pitch, vel = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        if (nxt and pitch - nxt[2] == mood.press and nxt[2] in open_set
                and 0 < nxt[0] - beat <= 1.01 and rng.random() < 0.45):
            out.append(Note("koto", beat, nxt[0] - beat + nxt[1], nxt[2], vel, "hanashi", nxt[2], nxt[0] - beat))
            i += 2
            continue
        orn, open_p = "", pitch
        r = rng.random()
        if beats >= 1.5 and (pitch - mood.press) in open_set and r < 0.2:
            orn, open_p = "ato", pitch - mood.press
        elif pitch not in open_set:
            orn, open_p = "press", pitch - mood.press
        elif beats >= 2 and r < 0.38:
            orn = "yuri"
        elif beats <= 0.5 and r < 0.06:
            orn = "tsuki"
        gap = beat - (raw[i - 1][0] + raw[i - 1][1]) if i else 9.0
        if beats >= 1 and gap >= 0.9 and beat % 2 == 0 and rng.random() < 0.15:
            up = next((p for p in mood.mel if p > pitch), pitch)
            out.append(Note("koto", beat - 0.12, 0.12, up, vel * 0.55))
        out.append(Note("koto", beat, beats, pitch, vel, orn, open_p))
        i += 1
    return out


# =================================================================== accompaniment
def accomp_part(rng: np.random.Generator, mood: Mood, themes: tuple, form: tuple[Section, ...],
                frames: tuple[str, ...]) -> list[Note]:
    styles: list[str | None] = []
    for sec in form:
        theme = themes[sec.move]
        if sec.kind == "intro":
            first = None if theme.opening != "drone" else "sparse"
            styles += ([first] + ["sparse"] * (sec.bars - 2) + [theme.acc_a])[: sec.bars] if sec.bars >= 2 else [first]
        elif sec.kind == "inter":
            styles += ["flow"] * sec.bars
        elif sec.kind == "Af":
            styles += [theme.acc_a] * 3 + ["sparse"]
        elif sec.kind == "short":
            styles += ["sparse"] * sec.bars
        else:
            styles += [theme.acc_a if sec.kind.startswith("A") else theme.acc_b] * sec.bars
    notes = []
    for bar, (frame, style) in enumerate(zip(frames, styles)):
        if style is None:
            continue
        voicing = mood.voicing[frame]
        for slot, voice, v in ACC[style]:
            pitch = voicing[voice]
            if voice == 2 and rng.random() < 0.12:
                pitch = next((p for p in mood.mel if p > pitch), pitch)
            vel = float(np.clip(v * (0.92 + rng.normal(0.0, 0.05)), 0.15, 0.9))
            notes.append(Note("koto2", bar * 4 + slot / 2, 0.5, pitch, vel))
    return notes


def bass_part(rng: np.random.Generator, mood: Mood, frames: tuple[str, ...]) -> list[Note]:
    notes, prev = [], None
    for bar, frame in enumerate(frames):
        changed = frame != prev
        if changed or rng.random() < 0.3:
            notes.append(Note("bass", bar * 4, 4, mood.bass[frame], 0.6 if changed else 0.45))
        prev = frame
    return notes


def pad_part(mood: Mood, frames: tuple[str, ...]) -> list[Note]:
    notes, start = [], 0
    for bar in range(1, len(frames) + 1):
        if bar == len(frames) or frames[bar] != frames[start]:
            f = frames[start]
            notes.append(Note("pad", start * 4, (bar - start) * 4, 0, 0.5, extra=mood.pad[f]))
            start = bar
    return notes


# =================================================================== colour parts
def shaku_part(rng: np.random.Generator, mood: Mood, themes: tuple, form: tuple[Section, ...]) -> list[Note]:
    notes: list[Note] = []
    intro = next((s for s in form if s.kind == "intro"), None)
    t0 = themes[0]
    if t0.shaku_intro and intro and intro.bars >= 2:
        start = {"drone": 0.5, "sararin": 1.5, "kororin": 3.0}[t0.opening]
        notes += _shaku_phrase(rng, mood, start, intro.bars * 4 - start - 0.5)
    fills = 0
    for sec in form:
        if sec.kind == "inter" and rng.random() < 0.55:
            notes += _shaku_phrase(rng, mood, sec.bar * 4 + 0.5, sec.bars * 4 - 0.5, short=True)
        elif sec.kind in ("A2", "B", "A3", "B2") and rng.random() < 0.2:
            notes += _shaku_phrase(rng, mood, sec.bar * 4 + 14.0, 2.0, short=True)
        else:
            continue
        fills += 1
        if fills >= 2 + len(form) // 12:                # a few long notes per movement, not a constant flute
            break
    return notes


def _shaku_phrase(rng: np.random.Generator, mood: Mood, beat: float, span: float, short: bool = False) -> list[Note]:
    """2-3 long notes on / around nuclear tones (A4, D5, G4) within `span` beats."""
    mel = mood.mel
    a4, d5 = mel.index(69), mel.index(74)
    first = int(rng.choice([a4, d5]))
    if short or span < 5:
        pitches = [mel[first]] if rng.random() < 0.5 else [mel[first + 1], mel[first]]
    else:
        nb = int(rng.choice([-1, 1]))
        pitches = [mel[first], mel[first + nb], mel[int(rng.choice([a4, d5]))]]
    if len(pitches) == 1:
        lens = [span]
    else:
        w = rng.uniform(0.25, 0.4, len(pitches) - 1)
        lens = [float(x) * span for x in w] + [span * (1 - float(w.sum()))]
    out, b = [], beat
    for p, ln in zip(pitches, lens):
        out.append(Note("shaku", b, max(ln, 0.75), p, float(rng.uniform(0.42, 0.55))))
        b += ln
    return out


def shami_part(rng: np.random.Generator, mood: Mood, form: tuple[Section, ...]) -> list[Note]:
    notes: list[Note] = []
    for sec in form:
        p = {"B": 0.7, "B2": 0.5, "inter": mood.shami_p, "intro": 0.25}.get(sec.kind, 0.0)
        if sec.bars and rng.random() < p:
            fig = SHAMI_FIGURES[rng.integers(len(SHAMI_FIGURES))]
            at = sec.bar * 4 + (1.0 if sec.kind in ("inter", "intro") else 0.0)
            if sec.kind == "intro":
                at = sec.bar * 4 + max(0.0, sec.bars * 4 - 2.0)
            notes += [Note("shami", at + d, 0.5, pitch, v) for d, pitch, v in fig]
    return notes


def opening_part(rng: np.random.Generator, mood: Mood, theme: Theme, spb: float) -> list[Note]:
    if theme.opening == "sararin":
        strings = mood.koto[1:10]
        out = [Note("koto", k * 0.045 / spb, 2.0, p, 0.26 + 0.035 * k, extra=("nojitter",)) for k, p in enumerate(strings)]
        last = len(strings) * 0.045 / spb
        return out + [Note("koto", last, 3.0, 62, 0.55, extra=("nojitter",)), Note("koto", last, 3.0, 74, 0.62, extra=("nojitter",))]
    if theme.opening == "kororin":
        third = 63 if mood.press == 1 else 64
        return [Note("koto", 0.0, 1.0, 62, 0.45), Note("koto", 0.0, 1.0, 74, 0.5), Note("koto", 1.0, 0.5, 69, 0.44),
                Note("koto", 1.5, 0.5, 67, 0.4), Note("koto", 2.0, 0.5, third, 0.38), Note("koto", 2.5, 1.5, 62, 0.46)]
    return []


def ending_part(theme: Theme, mood: Mood, fb: float, tonic: int, spb: float) -> list[Note]:
    octave = tonic + 12 if tonic <= 62 else tonic - 12
    if theme.final_stroke == "roll":
        chord = sorted({tonic, octave, min(tonic, octave) + 7})
        out = [Note("koto", fb + k * 0.035 / spb, 8.0, p, 0.5 + 0.04 * k, extra=("nojitter",)) for k, p in enumerate(chord)]
    else:
        out = [Note("koto", fb, 8.0, tonic, 0.58), Note("koto", fb, 8.0, octave, 0.52)]
    out += [Note("koto2", fb, 8.0, 50, 0.42), Note("koto2", fb, 8.0, 57, 0.36), Note("bass", fb, 8.0, mood.bass["T"], 0.6),
            Note("pad", fb, 8.0, 0, 0.5, extra=mood.pad["T"])]
    if theme.shaku_end:
        out.append(Note("shaku", fb + 0.25, 8.0, theme.shaku_end_pitch, 0.48))
    return out


def taiko_part(rng: np.random.Generator, form: tuple[Section, ...], fb: float, spb: float, start: bool, end: bool,
               lifts: bool) -> list[Note]:
    """Taiko: a soft roll at the very start (title card), a don at the head of B pairs when `lifts`, and the roll +
    don that lands the piece (end card). Beats are in the piece's own beats."""
    out = []
    if start:
        out.append(Note("roll", 0.0, 1.7 / spb, 0, 0.45, extra=(0.45,)))
    if lifts:
        for sec in form:
            if sec.kind in ("B", "B2"):
                out += [Note("taiko", sec.bar * 4, 4.0, 0, 0.36), Note("taiko", sec.bar * 4 + 8.0, 4.0, 0, 0.3)]
            elif sec.kind == "inter" and rng.random() < 0.4:
                out.append(Note("taiko", sec.bar * 4, 4.0, 0, 0.3))
    if end:
        out.append(Note("roll", fb - 3.0, 3.0, 0, 0.42, extra=(0.97,)))
        out.append(Note("taiko", fb, 4.0, 0, 0.5))
    return out


# =================================================================== assembly
def to_events(rng: np.random.Generator, notes: list[Note], tmap: TimeMap) -> tuple[Event, ...]:
    evs = []
    for nt in sorted(notes, key=lambda x: (x.beat, x.inst, x.pitch)):
        t = tmap.sec(max(nt.beat, 0.0))
        jit = JITTER.get(nt.inst, 0.0) if "nojitter" not in nt.extra else 0.0
        if nt.beat > 0 and jit:
            t += float(np.clip(rng.normal(0.0, jit), -2.5 * jit, 2.5 * jit))
        t = max(t, 0.004)
        dur = tmap.sec(nt.beat + nt.beats) - tmap.sec(nt.beat)
        hold = tmap.sec(nt.beat + nt.hold_beats) - tmap.sec(nt.beat)
        vel = float(np.clip(nt.vel * (1.0 + rng.normal(0.0, 0.05)), 0.05, 1.0))
        evs.append(Event(nt.inst, t, dur, nt.pitch, vel, nt.orn, nt.open_pitch, hold, nt.extra))
    return tuple(evs)


def compose(t_final: float, seed: int, bpm: float, mood: Mood, exact: bool = False, title: bool = False,
            end_card: bool = False) -> Score:
    """Whole piece as timed events. Independent RNG streams keep each movement's theme stable.
    `exact`: `bpm` is kept (no ritardando) and the final stroke lands on the last whole bar at or before t_final."""
    if exact:
        n = max(2, int(t_final * bpm / 240.0))
        tmap = TimeMap(n, 60.0 / bpm, 1.0, 0.0)
    else:
        tmap = TimeMap.fit(t_final, bpm)
    form = plan_form(tmap.n_bars)
    n_moves = max(s.move for s in form) + 1
    themes = tuple(make_theme(np.random.default_rng([seed, 11, m]), mood) for m in range(n_moves))
    frames = bar_frames(form, themes)
    fb = tmap.final_beat
    mel, tonic = melody_part(np.random.default_rng([seed, 12]), mood, themes, form)
    r_parts = np.random.default_rng([seed, 13])
    th0, th_last = themes[0], themes[-1]
    notes = (opening_part(r_parts, mood, th0, tmap.spb) + mel + accomp_part(r_parts, mood, themes, form, frames)
             + bass_part(r_parts, mood, frames) + pad_part(mood, frames) + shaku_part(r_parts, mood, themes, form)
             + shami_part(r_parts, mood, form) + ending_part(th_last, mood, fb, tonic, tmap.spb)
             + taiko_part(r_parts, form, fb, tmap.spb, title or th0.taiko_start, end_card or th_last.taiko_end,
                          title or end_card or tmap.n_bars >= 24))
    events = to_events(np.random.default_rng([seed, 14]), notes, tmap)
    return Score(events, tmap.spb * tmap.final_beat if exact else t_final, tmap, mood, themes,
                 tuple((s.kind, s.bar, s.bars) for s in form))
