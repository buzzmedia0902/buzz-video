"""End card: logo + label + tagline + CTA on a paper (light) or night (dark) background.

The stack is laid out inside the safe zone and rebuilt smaller when the format is short (1:1, 16:9).
Timings are relative to the card start t0 and can be overridden per template.
Keep the CTA on screen for at least 2 s (ads that cut away sooner lose the click).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from core import engine as E
from core.canvas import H, W
from core.logo import Logo

TIMES = {"logo": 0.12, "label": 0.42, "sub": 0.62, "tagline": 1.05, "glint": 1.25, "cta": 1.2, "note": 0.1}
GAPS = {"note": 30, "label": 26, "logo": 64, "tag": 70, "cta": 0}


def cta_outline(text: str, key: str, size: int, ink, line) -> Image.Image:
    """CTA as an outlined pill with a trailing chevron."""
    f = E.font(key, size)
    tw = E.text_width(text, key, size, 0.06)
    pw, ph = int(tw + size * 3.6), int(size * 2.6)
    img = Image.new("RGBA", (pw + 8, ph + 8), (*E.rgb(ink), 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, pw + 3, ph + 3), radius=ph // 2, outline=(*E.rgb(line), 255), width=max(2, size // 20))
    x = (pw + 8 - tw - size * 0.85) / 2
    for ch in text:
        d.text((x, (ph + 8) / 2 + size * 0.35), ch, font=f, fill=(*E.rgb(ink), 255), anchor="ls")
        x += E._advance(f, ch)[1] + 0.06 * size
    d.text((x + size * 0.35, (ph + 8) / 2 + size * 0.3), "›", font=E.font("optima", size), fill=(*E.rgb(line), 255),
           anchor="ls")
    return img


class EndCard:
    def __init__(self, ctx, theme: str = "light", cta_style: str = "outline", logo_style: str = "rise",
                 note: str | None = None, particles: bool = True, times: dict | None = None,
                 show_label: bool = True, show_tagline: bool = True, logo_width: float | None = None,
                 tagline: str | None = None, logo_gap: float = 0.075):
        """tagline overrides the brand tagline for this card; logo_gap = seconds between letter reveals."""
        b, L = ctx.brand, ctx.layout
        self.b, self.L, self.cta_text = b, L, ctx.cta
        self.theme, self.cta_style, self.logo_style, self.particles = theme, cta_style, logo_style, particles
        self.note_text, self.logo_gap = note, logo_gap
        self.tagline_text = b.tagline if tagline is None else tagline
        self.show_label, self.show_tagline = show_label and bool(b.label), show_tagline and bool(self.tagline_text)
        self.t = {**TIMES, **(times or {})}
        dark = theme == "dark"
        self.ink = (244, 238, 226) if dark else b.color("ink")
        self.acc = b.color("accent_light") if dark else b.color("accent_deep")
        self.accent = b.color("accent")
        self.bg = (E.flat_bg(E.mix(b.color("night"), (255, 255, 255), 0.08), b.color("night"), seed=5) if dark
                   else E.flat_bg(E.mix(b.color("paper"), (255, 255, 255), 0.5), E.mix(b.color("paper"), (0, 0, 0), 0.05)))
        self.logo_w = logo_width or min(0.72 * L.sw, 780)
        self._build(1.0)
        total = self._stack_height()
        k = L.fit(total, L.sh * 0.92)
        if k < 0.98:
            self._build(k)
        self._place()
        rng = np.random.default_rng(3)
        n = 44
        self.parts = dict(x=rng.uniform(40, W - 40, n), y=rng.uniform(40, H - 40, n), v=rng.uniform(14, 40, n),
                          s=rng.uniform(2.0, 6.5, n), a=rng.uniform(0.10, 0.38, n), ph=rng.uniform(0, 6.28, n))
        self._dots: dict = {}

    def _build(self, k: float) -> None:
        b, dark = self.b, self.theme == "dark"
        self.k = k
        self.logo = Logo(b.logo_on_dark if dark else b.logo_on_light, b.parts, int(self.logo_w * k))
        self.label = E.TextLine(b.label, b.font("latin"), int(31 * k), self.acc, tracking=0.42) if self.show_label else None
        if self.show_tagline:
            size = E.fit_size([self.tagline_text], b.font("serif"), int(46 * k), self.L.sw * 0.96, 0.12)
            self.tag = E.TextLine(self.tagline_text, b.font("serif"), size, self.ink, tracking=0.12)
        else:
            self.tag = None
        self.note = (E.text_img(self.note_text, b.font("hand"), int(50 * k), E.mix(self.ink, self.acc, 0.4),
                                tracking=0.04, pad=8) if self.note_text else None)
        if self.cta_style == "solid":
            self.cta = E.pill_img(f"{self.cta_text}  ›", b.font("body"), int(46 * k), (255, 255, 255), b.color("ink"),
                                  pad_x=int(54 * k), pad_y=int(24 * k))
        elif self.cta_style == "text":
            self.cta = E.text_img(self.cta_text, b.font("vertical"), int(50 * k), self.ink, tracking=0.14, pad=6)
        else:
            self.cta = cta_outline(self.cta_text, b.font("serif"), int(40 * k), self.ink, self.acc)

    def _items(self) -> list[tuple[str, float]]:
        items = []
        if self.note is not None:
            items.append(("note", self.note.height))
        if self.label is not None:
            items.append(("label", self.label.box_h))
        items.append(("logo", self.logo.h))
        if self.tag is not None:
            items.append(("tag", self.tag.box_h))
        items.append(("cta", self.cta.height))
        return items

    def _stack_height(self) -> float:
        return sum(h + GAPS[n] * self.k for n, h in self._items())

    def _place(self) -> None:
        """Centre y of every element, stack centred slightly below the middle of the safe zone."""
        y = self.L.cy - self._stack_height() / 2 + self.L.sh * 0.02
        self.pos = {}
        for name, h in self._items():
            self.pos[name] = y + h / 2
            y += h + GAPS[name] * self.k

    def _dot(self, size: float) -> np.ndarray:
        key = round(size * 2) / 2
        if key not in self._dots:
            r = int(key * 3) + 2
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            a = np.exp(-(xx ** 2 + yy ** 2) / (2 * (key * 0.6) ** 2)).astype(np.float32)
            self._dots[key] = E.solid_sprite(a, self.accent)
        return self._dots[key]

    def background(self, t: float, t0: float) -> np.ndarray:
        frame = self.bg.copy()
        if self.theme == "dark":
            return E.screen(frame, E.light_leak(t, 0.10 * E.clamp01((t - t0) / 1.5), seed=2.0))
        if self.particles:
            p, tt = self.parts, t - t0 + 3.0
            for i in range(len(p["x"])):
                y = (p["y"][i] - p["v"][i] * tt) % (H + 100) - 50
                x = p["x"][i] + 14 * math.sin(tt * 0.6 + p["ph"][i])
                tw = 0.6 + 0.4 * math.sin(tt * 2.1 + p["ph"][i] * 3)
                spr = self._dot(p["s"][i])
                E.blit(frame, spr, x - spr.shape[1] / 2, y - spr.shape[0] / 2, p["a"][i] * tw)
        return frame

    def draw_on(self, frame: np.ndarray, t: float, t0: float, opacity: float = 1.0) -> None:
        """Card elements (no background) at time t; the card started at t0."""
        T, cx = self.t, self.L.cx
        if self.note is not None:
            E.blit_img(frame, self.note, cx, self.pos["note"], angle=-2.0,
                       opacity=E.ease_out_cubic((t - t0 - T["note"]) / 0.5) * opacity)
        if self.label is not None:
            self.label.draw_centered(frame, cx, self.pos["label"], t, t0 + T["label"], 0.015, 0.6, opacity, rise=0)
        self.logo.draw(frame, t, cx, self.pos["logo"] - self.logo.h / 2, t0 + T["logo"], style=self.logo_style,
                       gap=self.logo_gap, sub_at=t0 + T["sub"], glint_at=t0 + T["glint"], opacity=opacity)
        if self.tag is not None:
            self.tag.draw_centered(frame, cx, self.pos["tag"], t, t0 + T["tagline"], 0.04, 0.5, opacity, rise=0)
        p = (t - t0 - T["cta"]) / (0.35 if self.cta_style == "solid" else 0.5)
        if p > 0:
            sc = max(0.05, E.ease_out_back(p, 1.8)) if self.cta_style == "solid" else 1.0
            E.blit_img(frame, self.cta, cx, self.pos["cta"], scale=sc, opacity=E.clamp01(p * 2) * opacity)
            if self.cta_style == "text":
                self._rules(frame, t - t0 - T["cta"], opacity)

    def _rules(self, frame: np.ndarray, dt: float, opacity: float) -> None:
        """Short accent rules either side of a text CTA."""
        e = E.ease_out_cubic(dt / 0.7) * opacity
        y = int(self.pos["cta"])
        for sgn in (-1, 1):
            x0 = self.L.cx + sgn * (self.cta.width / 2 + 24)
            x1 = x0 + sgn * 70 * e
            region = frame[y - 1:y + 1, int(min(x0, x1)):int(max(x0, x1))]
            region *= 1 - 0.9
            region += np.array(self.accent, np.float32) / 255 * 0.9

    def frame(self, t: float, t0: float) -> np.ndarray:
        out = self.background(t, t0)
        self.draw_on(out, t, t0)
        return out
