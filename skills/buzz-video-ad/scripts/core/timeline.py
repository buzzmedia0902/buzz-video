"""Base class for taste templates: a list of shots with lazily opened sources, plus cues for the audio.

A taste module exposes build(ctx) -> Timeline. render.py calls tl.frame(t) for every output frame
(float32 H x W x 3, 0..1) and writes tl.cues() next to the render for bgm.py / sfx.py / finish.py.
"""
from __future__ import annotations

import numpy as np

from core import engine as E


class Timeline:
    duration: float = 0.0
    card_at: float = 0.0  # end card start (music resolves here)
    cover_at: float = 0.0  # frame used for the cover image
    bgm_style: str = "piano"

    def __init__(self, ctx, shots: list[E.Shot]):
        self.ctx = ctx
        self.shots = shots
        self.sources = [E.Source(s) for s in shots]
        for s in self.sources:
            s.still_mode = ctx.still
        self.sfx: list[list] = []  # [t, name, level]
        self.voice: list[list] = []  # [t, wav path]

    # -------------------------------------------------------------- shots
    def index_at(self, t: float) -> int:
        for i, s in enumerate(self.shots):
            if s.t0 - 1e-6 <= t < s.t1 - 1e-6:
                return i
        return len(self.shots) - 1 if t >= self.shots[-1].t1 - 1e-6 else 0

    def shot_frame(self, i: int, t: float, extra_zoom: float = 1.0, grade: str | None = None) -> np.ndarray:
        return E.render_shot(self.sources[i], t, extra_zoom, grade)

    def release(self, t: float) -> None:
        """Close decoders of shots that are over (render.py calls this every frame)."""
        for s in self.sources:
            if t > s.end + 0.1:
                s.close()

    def close(self) -> None:
        for s in self.sources:
            s.close()

    # -------------------------------------------------------------- output
    def frame(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def cuts(self) -> list[float]:
        return [round(s.t0, 3) for s in self.shots[1:]]

    def cues(self) -> dict:
        v = self.ctx.v
        bgm = {"style": self.bgm_style, **v.get("bgm", {})}
        return {"version": self.ctx.name, "taste": v["taste"], "duration": round(self.duration, 3),
                "card_at": round(self.card_at, 3), "cover_at": round(self.cover_at, 3), "cuts": self.cuts(),
                "sfx": self.sfx, "voice": self.voice, "bgm": bgm}
