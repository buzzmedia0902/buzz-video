"""Render context handed to every style module."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.composer.plan import Grid, Plan


@dataclass
class Ctx:
    plan: Plan
    rng: np.random.Generator    # performance: timing jitter, noise textures, reverb IRs
    comp: np.random.Generator   # composition: patterns and melodies (independent of the duration)
    cuts: tuple[float, ...] = ()
    log: dict = field(default_factory=dict)   # style modules park debug info here (melody, ...)

    @property
    def g(self) -> Grid:
        return self.plan.grid

    def t(self, beat: float) -> float:
        return self.plan.grid.t(beat)
