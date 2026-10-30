"""Vlog taste: a narrated diary ("the day I went to ...") driven by the narration timings.

Each narration line gets its own shot, cut `lead` s before the line starts. Rounded subtitles, optional
time-of-day chips, a sticker title card on the first shot, a quick punch-in on every cut, and a crossfade
into a light end card whose CTA pops while the last line ("ご予約は、リンクから。") is spoken.

Version spec (see references/tastes.md):
  {"taste": "vlog", "title": "美容鍼に行ってきた日",
   "lines": [{"sub": ["row 1", "row 2"], "speech": "読み上げ用の文", "chip": "AM 9:00", "sub_pos": 0.25,
              "shot": {shot dict without "dur"}}, ...,
             {"sub": ["ご予約はリンクから。"], "speech": "ご予約は、リンクから。", "card": true}],
   "start": 0.3, "gaps": [0.3, 0.38, ...] or "gap": 0.32, "lead": 0.18, "sub_y": 0.88, "duration": 24,
   "voice": {...for tts.py...}, "bgm": {"style": "lofi"}}
Narration timings come from work/<version>/voice/timings.json (tts.py); before that exists, durations are
estimated from the text so stills can be checked early.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

from core import engine as E
from core.cards import EndCard
from core.spec import SpecError, make_shot
from core.timeline import Timeline


def narration(ctx) -> tuple[list[float], list[str | None]]:
    """Per-line durations and wav paths from timings.json, or estimates (0.13 s per character)."""
    lines = ctx.v["lines"]
    path = ctx.path("voice", "timings.json")
    if os.path.exists(path):
        t = json.load(open(path, encoding="utf-8"))
        if len(t) != len(lines):
            raise SpecError(f"{path}: {len(t)} lines but the spec has {len(lines)} (re-run tts.py)")
        return [float(x["duration"]) for x in t], [os.path.join(os.path.dirname(path), x["file"]) for x in t]
    print("[vlog] no narration yet: using estimated line lengths", flush=True)
    return [0.3 + 0.13 * len(ln.get("speech", "".join(ln["sub"]))) for ln in lines], [None] * len(lines)


class Vlog(Timeline):
    bgm_style = "lofi"

    def __init__(self, ctx):
        v, L, b = ctx.v, ctx.layout, ctx.brand
        self.lines = v["lines"]
        card_idx = [i for i, ln in enumerate(self.lines) if ln.get("card")]
        if len(card_idx) != 1 or card_idx[0] != len(self.lines) - 1:
            raise SpecError("vlog: exactly one line, the last, needs \"card\": true (the CTA line)")
        self.durs, self.wavs = narration(ctx)
        gaps = v.get("gaps") or [v.get("gap", 0.32)] * len(self.lines)
        self.lead = float(v.get("lead", 0.18))
        self.starts, t = [], float(v.get("start", 0.3))
        for i, d in enumerate(self.durs):
            self.starts.append(t)
            t += d + (gaps[i] if i < len(gaps) else gaps[-1])
        self.card_at = self.starts[-1] - self.lead
        self.duration = float(v.get("duration", self.card_at + max(2.6, self.durs[-1] + 1.4)))
        shots = []
        for i, ln in enumerate(self.lines[:-1]):
            t0 = 0.0 if i == 0 else self.starts[i] - self.lead
            t1 = self.starts[i + 1] - self.lead
            shots.append(make_shot(ctx.project, ln["shot"], f"l{i + 1}", t0, t1, grade="natural", shake=3.0))
        super().__init__(ctx, shots)
        self.voice = [[round(s, 3), w] for s, w in zip(self.starts, self.wavs) if w]
        self.cover_at = min(1.2, self.starts[1] - self.lead - 0.3)
        self.L, self.b = L, b
        self._assets(v.get("title", ""))

    # -------------------------------------------------------------- assets
    def _assets(self, title: str) -> None:
        b, L = self.b, self.L
        ink = b.color("ink")
        self.subs = []
        for ln in self.lines:
            size = E.fit_size(ln["sub"], b.font("body"), 58, L.text_w, 0.03)
            imgs = [E.text_img(r, b.font("body"), size, (255, 255, 255), tracking=0.03, stroke=7, stroke_color=ink)
                    for r in ln["sub"]]
            self.subs.append((imgs, [E.shadow_for(im, 10, 1.2, pad=30) for im in imgs]))
        self.chips = {i: self._chip(ln["chip"]) for i, ln in enumerate(self.lines) if ln.get("chip")}
        self.k_sticker = L.pick(tall=1.0, portrait=0.9, default=0.78)
        self.sticker = self._sticker(title) if title else None
        if self.sticker is not None:
            self.sticker_sh = E.shadow_for(self.sticker, 16, 0.9, pad=40)
        self.spark = E.sparkle_img(30, b.color("accent_light"))
        self.card = EndCard(self.ctx, theme="light", cta_style="solid", note=title or None, show_tagline=False,
                            particles=False, times={"note": 0.2, "label": 0.7, "logo": 0.35, "sub": 0.8,
                                                    "glint": 1.2, "cta": 0.5})

    def _sticker(self, title: str) -> Image.Image:
        b = self.b
        size = E.fit_size([title], b.font("hand"), 70, self.L.text_w - 110, 0.05)
        t = E.tight(E.text_img(title, b.font("hand"), size, b.color("ink"), tracking=0.05, pad=0))
        logo = Image.open(b.logo_on_light)
        logo = logo.resize((250, int(logo.height * 250 / logo.width)), Image.LANCZOS)
        w, h = max(t.width, logo.width) + 110, t.height + logo.height + 120
        img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=36, fill=(255, 255, 255, 242))
        img.alpha_composite(t, ((w - t.width) // 2, 46))
        img.alpha_composite(logo, ((w - logo.width) // 2, 46 + t.height + 30))
        return img

    def _chip(self, text: str) -> Image.Image:
        b = self.b
        t = E.tight(E.text_img(text, b.font("latin"), 40, b.color("ink"), tracking=0.06, pad=0))
        w, h = t.width + 108, t.height + 38
        img = Image.new("RGBA", (w + 4, h + 4), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((2, 2, w + 1, h + 1), radius=h // 2, fill=(255, 255, 255, 236))
        cx, cy, r = 2 + 34, 2 + h // 2, 15  # clock icon
        acc = b.color("accent_deep")
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*acc, 255), width=4)
        d.line((cx, cy, cx, cy - 9), fill=(*acc, 255), width=4)
        d.line((cx, cy, cx + 7, cy + 2), fill=(*acc, 255), width=4)
        img.alpha_composite(t, (2 + 64, 2 + 19))
        return img

    # -------------------------------------------------------------- drawing
    def _shot(self, i: int, t: float) -> np.ndarray:
        s = self.shots[i]
        dt = t - s.t0
        extra = 1 + 0.06 * (1 - E.ease_out_cubic(dt / 0.25)) if i > 0 and dt < 0.25 else 1.0  # punch-in
        return self.shot_frame(i, t, extra)

    def _sub(self, frame: np.ndarray, t: float, i: int) -> None:
        t_in, t_out = self.starts[i] - 0.05, self.starts[i] + self.durs[i] + 0.25
        if not t_in <= t <= t_out:
            return
        imgs, shs = self.subs[i]
        p = (t - t_in) / 0.18
        out = E.ease_out_cubic((t_out - t) / 0.12)
        gap = 84
        base = self.L.y(float(self.lines[i].get("sub_pos", self.ctx.v.get("sub_y", 0.88))))
        y0 = min(base - (len(imgs) - 1) * gap / 2, self.L.y1 - (len(imgs) - 0.5) * gap)
        for k, (im, sh) in enumerate(zip(imgs, shs)):
            y = y0 + k * gap
            sc = 0.92 + 0.08 * E.ease_out_back(p, 2.0)
            E.blit_img(frame, sh, self.L.cx, y + 4, scale=sc, opacity=0.55 * E.clamp01(p) * out)
            E.blit_img(frame, im, self.L.cx, y, scale=sc, opacity=E.clamp01(p * 1.5) * out)

    def _chip_draw(self, frame: np.ndarray, t: float, i: int) -> None:
        if i not in self.chips:
            return
        t_in, t_out = self.shots[i].t0 + 0.12, self.shots[i].t1 - 0.05
        if not t_in <= t <= t_out:
            return
        img = self.chips[i]
        p = (t - t_in) / 0.35
        E.blit_img(frame, img, self.L.x0 + 25 + img.width / 2, self.L.y(0.062), scale=max(0.05, E.ease_out_back(p, 2.2)),
                   opacity=E.clamp01(p * 2) * E.ease_out_cubic((t_out - t) / 0.12))

    def _title(self, frame: np.ndarray, t: float) -> None:
        end = self.starts[1] - self.lead
        if self.sticker is None or t >= end:
            return
        k, cx, cy = self.k_sticker, self.L.cx, self.L.y(0.267)
        e = E.ease_out_cubic((end - 0.05 - t) / 0.2)
        p = (t - 0.25) / 0.45
        sc = max(0.05, E.ease_out_back(p, 1.8)) * k
        E.blit_img(frame, self.sticker_sh, cx, cy + 15 * k, scale=sc, opacity=0.6 * E.clamp01(p) * e, angle=-2.5)
        E.blit_img(frame, self.sticker, cx, cy, scale=sc, opacity=E.clamp01(p * 2) * e, angle=-2.5)
        for j, (dx, dy) in enumerate(((-330, -90), (335, 60), (300, -110))):
            env = E.clamp01((t - 0.7 - j * 0.15) / 0.3) * e
            E.blit_img(frame, self.spark, cx + dx * k, cy + dy * k, scale=0.5 + 0.2 * np.sin(t * 6 + j), opacity=env)

    def frame(self, t: float) -> np.ndarray:
        if t >= self.card_at + 0.25:
            frame = self.card.frame(t, self.card_at)
        else:
            i = self.index_at(t)
            frame = self._shot(i, t)
            self._chip_draw(frame, t, i)
            self._title(frame, t)
            if t >= self.card_at:
                k = E.smoothstep((t - self.card_at) / 0.25)
                frame = frame * (1 - k) + self.card.frame(t, self.card_at) * k
        for i in range(len(self.lines) - 1):  # the CTA line is shown by the card itself
            self._sub(frame, t, i)
        return frame


def build(ctx) -> Timeline:
    if "lines" not in ctx.v:
        raise SpecError("vlog: 'lines' is required")
    return Vlog(ctx)
