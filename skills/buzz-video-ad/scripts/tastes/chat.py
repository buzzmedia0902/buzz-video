"""Chat taste: a LINE-style chat screen (no LINE logo or name anywhere) that tells the story as messages.

Messages pop in on a fast cadence and scroll the conversation; video messages play real footage inside the
bubble, original stickers, a booking link card, a typing indicator, an iOS status bar + header with the brand
banner pinned under it, the Japanese 12-key keyboard; at the end the screen pushes into the link card and
dissolves to the brand end card. 9:16 / 4:5 = full-screen phone UI (4:5 shows fewer bubbles); 1:1 / 16:9 =
phone mockup on a brand-colour background (16:9 adds logo + tagline beside the phone).

Version spec (see references/tastes.md); only "messages" is required:
  {"taste": "chat", "title": "あや", "banner": "美容鍼、気になってた？", "clock": "20:41",
   "messages": [
     {"side": "L", "kind": "video", "t": 0.25, "dur": 2.4,
      "shot": {"clip": "IMG_2742.MOV", "in": 2.85, "speed": 0.5, "zoom": [1.45, 1.38], "center": [0.46, 0.38]}},
     {"side": "L", "text": "美容鍼いってきた〜！"},
     {"side": "R", "kind": "sticker", "text": "えっ！？", "face": "surprised"},
     {"side": "L", "kind": "typing", "dur": 0.6},
     {"side": "L", "kind": "link", "title": "SALON SAMPLE｜美容鍼のご予約", "sub": "ご予約はこちらから"},
     {"side": "R", "text": "ありがと！さっそく予約する！"}],
   "end": {"tagline": "気になってたなら、いまが行きどき。", "theme": "light", "cta_style": "solid"},
   "cta": "ご予約はリンクから", "bgm": {"style": "lofi"}}

Message keys
  side   "L" = the friend (white bubble, avatar)  |  "R" = me (green bubble, 既読)           [default "L"]
  kind   text | video | sticker | link | typing                       [text; "video" when "shot" is given]
  text   message text ("\\n" = line break, long lines wrap) | sticker caption | link card title
  shot   video: shot dict as in the other tastes (clip, in, speed, zoom, center, grade, rot, patches, focus,
         soft_below ...) without "dur"; clip settings from project.json apply. Grade defaults to "natural".
  dur    video: seconds the clip plays in the bubble, then it holds its last frame        [2.2]
         typing: seconds the dots show before the next message                           [0.6]
  face   sticker face: surprised | happy | love                  [surprised when the caption has ！？, else happy]
  sub    link card caption under the title                                                 [the CTA text]
  t      absolute time (s) the message appears
  gap    seconds after the previous message started, when "t" is absent            [by previous message: text
         0.55-1.15 s growing with its length, sticker 0.9, link 1.0, video 1.2, typing = its "dur"]
  clock  explicit time stamp "20:42"; stamps never go backwards and the status-bar clock follows the latest one

Version keys
  title     name in the chat header                                                       ["友だち"]
  avatar    letter in the friend's round avatar                                           [first letter of title]
  banner    pinned banner text next to the brand logo; false = no banner                  [brand tagline]
  clock     first time stamp                                                              ["20:41"]
  minute_every  stamps advance one minute every N seconds (0 = never)                     [5]
  start     time of the first message (s)                                                 [0.25]
  zoom      push into the last link card (or last message) before the end card            [true]
  zoom_gap  seconds from the last message to the push                                     [1.1]
  hold      seconds the end card stays                                                    [3.15]
  duration  optional total length; with no explicit "t" the gaps stretch / shrink (0.55-1.7x) to fit it,
            otherwise the end card absorbs the difference
  cover_at  time of the cover frame                                                       [2nd video + 0.9 s]
  note      small "※画面はイメージです" (string = other wording, false = off)              [true]
  palette   {"wall": "#98B1D4", "mine": "#86DE62"}  chat wallpaper and my bubble colour
  end       {"tagline": ..., "theme": "light"|"dark", "cta_style": "solid"|"outline"|"text", "note": ...}
            end-card copy (the brand tagline is the default; "cta" at version level sets the button text)
  beside    16:9 only, {"tagline": ...}: copy beside the phone (default: brand tagline)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from core import engine as E
from core.canvas import FPS
from core.cards import EndCard
from core.logo import Logo
from core.spec import SpecError, make_shot
from core.timeline import Timeline
from tastes.chat_ui import (WALL, MINE, STAMP_INK, FONT, FONT_BOLD, TEXT_PX, GAP_SAME, GAP_SWITCH, X_AVATAR,
    X_LEFT, EDGE, POP, GROW, ZOOM_LEN, CARD_FADE, WORK_W, KINDS, SIDES, FACES, clamp, Stage, make_stage, rounded,
    wrap, bubble, sticker, link_card, avatar, typing_bubble, chrome, Phone)


# ================================================================= spec -> messages
@dataclass(frozen=True)
class Msg:
    t: float
    side: str  # "L" friend | "R" me
    kind: str  # text | video | sticker | link | typing
    text: str = ""
    sub: str = ""
    face: str = ""
    dur: float = 0.0
    stamp: str = ""
    avatar: bool = False  # friend's avatar shown beside this message
    shot: int = -1  # index into Timeline.shots (video)


def read_messages(ctx) -> list[dict]:
    """Validated message dicts with kind / side / dur / text normalised."""
    raw = ctx.v.get("messages")
    if not isinstance(raw, list) or not raw:
        raise SpecError("chat: 'messages' must be a non-empty list")
    out = []
    for i, d in enumerate(raw):
        where = f"chat: messages[{i}]"
        if not isinstance(d, dict):
            raise SpecError(f"{where} must be an object")
        kind = d.get("kind") or ("video" if "shot" in d else "text")
        side = SIDES.get(str(d.get("side", "L")).upper())
        if kind not in KINDS:
            raise SpecError(f"{where}: kind {kind!r} must be one of {', '.join(KINDS)}")
        if side is None:
            raise SpecError(f"{where}: side must be \"L\" (friend) or \"R\" (me)")
        if kind in ("text", "sticker") and not str(d.get("text", "")).strip():
            raise SpecError(f"{where}: a {kind} message needs 'text'")
        if kind == "video" and not isinstance(d.get("shot"), dict):
            raise SpecError(f"{where}: a video message needs 'shot' ({{\"clip\": ..., \"in\": ...}})")
        if kind == "typing" and side != "L":
            raise SpecError(f"{where}: the typing indicator belongs to the friend (side \"L\")")
        if kind == "sticker" and d.get("face", "happy") not in FACES:
            raise SpecError(f"{where}: face must be one of {', '.join(FACES)}")
        dur = float(d.get("dur", (d.get("shot") or {}).get("dur", {"video": 2.2, "typing": 0.6}.get(kind, 0.0))))
        if kind in ("video", "typing") and dur <= 0:
            raise SpecError(f"{where}: dur must be > 0")
        out.append({**d, "kind": kind, "side": side, "dur": dur, "text": str(d.get("text", ""))})
    return out


def _gap_after(prev: dict) -> float:
    kind = prev["kind"]
    if kind == "typing":
        return prev["dur"]
    if kind == "text":
        return clamp(0.42 + 0.04 * len(prev["text"].replace("\n", "")), 0.55, 1.15)
    return {"sticker": 0.9, "link": 1.0, "video": 1.2}[kind]


def schedule(msgs: list[dict], start: float) -> list[float]:
    """Message times: explicit "t", else previous time + "gap" (default by previous kind)."""
    times: list[float] = []
    for i, d in enumerate(msgs):
        if "t" in d:
            t = float(d["t"])
        elif i == 0:
            t = start
        else:
            t = times[-1] + float(d.get("gap", _gap_after(msgs[i - 1])))
        if times and t < times[-1] - 1e-6:
            raise SpecError(f"chat: messages[{i}] appears at {t:.2f}s, before the previous message ({times[-1]:.2f}s)")
        times.append(t)
    return times


def fit_to_duration(times: list[float], total: float, tail: float) -> list[float]:
    """Stretch / shrink the gaps so the last message lands `tail` seconds before `total` (end of the chat part)."""
    span = times[-1] - times[0]
    if span <= 1e-6:
        return times
    k = clamp((total - tail - times[0]) / span, 0.55, 1.7)
    return [times[0] + (t - times[0]) * k for t in times]


def _minutes(s: str) -> int:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(s))
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        raise SpecError(f"chat: clock {s!r} must look like \"20:41\"")
    return int(m[1]) * 60 + int(m[2])


def stamps(msgs: list[dict], times: list[float], first: str, every: float) -> list[str]:
    """Time stamp per message: +1 minute every `every` s, explicit "clock" respected, never decreasing."""
    base = _minutes(first)
    cur, out = base, []
    for d, t in zip(msgs, times):
        auto = base + int(t / every) if every > 0 else base
        cur = max(cur, auto, _minutes(d["clock"]) if d.get("clock") else 0)
        out.append(f"{cur // 60 % 24}:{cur % 60:02d}")
    return out


# ---------------------------------------------------------------- video bubbles
def bubble_view(src: E.Source, shot: E.Shot, t: float, size: tuple[int, int]) -> np.ndarray:
    """The shot's camera window cut to the bubble's aspect (cover), cleaned up and graded at WORK_W px wide
    (engine.cleanup scales its blur radii to the view), then scaled to the bubble; float (h, w, 3)."""
    bw, bh = size
    raw = src.frame(t)
    dh, dw = raw.shape[:2]
    z, cx, cy = E.cam_at(shot, t)
    z, asp = max(1.0, z), bw / bh
    ww0, wh0 = (dh * asp, float(dh)) if dw / dh > asp else (float(dw), dw / asp)
    ww, wh = ww0 / z, wh0 / z
    left = clamp(cx * dw - ww / 2, 0.0, dw - ww)
    top = clamp(cy * dh - wh / 2, 0.0, dh - wh)
    vw, vh = WORK_W, round(WORK_W * bh / bw)
    view = np.asarray(Image.fromarray(raw).resize((vw, vh), Image.BICUBIC, box=(left, top, left + ww, top + wh)))
    win = E.Window(left, top, ww, wh, vw, vh, dw, dh)
    view = E.cleanup(view, replace(shot, soft_below=0.0), win, t)
    if shot.soft_below:  # engine.cleanup's 180 px ramp is sized for the full frame; scale it to this view
        view = E.soften_below(view, int(win.to_view(0.0, shot.soft_below)[1]), ramp=round(180 * vw / 1080))
    # a bubble shows only the middle of what the full frame was graded on: ease the vignette
    graded = E.grade(view, shot.grade, shot.exposure, shot.tint, vig_scale=0.7)
    return np.asarray(Image.fromarray(E.to_u8(graded)).resize((bw, bh), Image.LANCZOS), np.float32) / 255


# ================================================================= the phone screen
@dataclass
class Item:
    msg: Msg
    w: int = 0
    h: int = 0
    top: float = 0.0  # content-space y
    img: Image.Image | None = None  # straight RGBA (None for video / typing)
    spr: np.ndarray | None = None  # premultiplied float sprite of img
    held: tuple | None = None  # video: (frame key, sprite)


class Screen:
    """The phone screen at design size: static chrome + the scrolling conversation; frame(t) -> float (dh, dw, 3)."""

    def __init__(self, ctx, st: Stage, msgs: list[Msg], tl: Timeline):
        v, b = ctx.v, ctx.brand
        self.st, self.tl, self.b = st, tl, b
        pal = v.get("palette", {})
        self.wall = E.rgb(pal.get("wall", WALL))
        self.mine = E.rgb(pal.get("mine", MINE))
        title = str(v.get("title", "友だち"))
        note = v.get("note", True)
        self.note = ("※画面はイメージです" if note is True else str(note)) if note else None
        banner = v.get("banner", b.tagline or b.label)
        text = None if banner is False else str(banner or "")
        bar = E.mix(self.wall, (0, 0, 0), 0.06)
        self.base = chrome(b, st, title, text, None if st.mockup else self.note, self.wall, bar)
        self.avatar = E.premult(avatar(b, str(v.get("avatar", title[:1]))))
        self.shell = bubble(["　　"], "L", self.mine)
        self.vmask = np.asarray(rounded(st.media_w, st.media_h, 30, (0, 0, 0)).split()[3], np.float32)[..., None] / 255
        self.read = E.premult(E.text_img("既読", FONT, 26, STAMP_INK, pad=0))
        self._clk: dict[str, np.ndarray] = {}
        self.items = self._layout([Item(m) for m in msgs])

    # -------------------------------------------------------------- layout
    def _make(self, it: Item) -> None:
        m, st = it.msg, self.st
        if m.kind == "text":
            it.img = bubble(wrap(m.text, TEXT_PX, st.bubble_max - 68), m.side, self.mine)
        elif m.kind == "sticker":
            it.img = sticker(m.text, m.face)
        elif m.kind == "link":  # on short screens shrink the card so it still fits with a reply under it
            k = clamp((0.95 * st.view_h - 114) / 400, 0.7, 1.0)
            it.img = link_card(self.b, round(min(640, st.dw - X_LEFT - 100) * k), m.text, m.sub, k)
        elif m.kind == "typing":
            it.img = self.shell
        if it.img is not None:
            it.w, it.h = it.img.size
            it.spr = E.premult(it.img)
        else:
            it.w, it.h = st.media_w, st.media_h

    def _layout(self, items: list[Item]) -> list[Item]:
        y, prev = 0.0, None
        for it in items:
            self._make(it)
            if it.msg.kind == "typing":
                it.top = y + GAP_SWITCH
                continue
            y += 0 if prev is None else (GAP_SAME if prev == it.msg.side else GAP_SWITCH)
            it.top = y
            y += it.h
            prev = it.msg.side
        return items

    # -------------------------------------------------------------- time -> geometry
    def _typing_room(self, t: float) -> float:
        """Space reserved for the typing bubble; it eases away while the next message grows into its place."""
        room = 0.0
        for it in self.items:
            m = it.msg
            if m.kind == "typing" and m.t <= t < m.t + m.dur + GROW:
                room += ((GAP_SWITCH + it.h) * E.ease_out_cubic((t - m.t) / 0.2)
                         * (1 - E.ease_out_cubic((t - m.t - m.dur) / GROW)))
        return room

    def content_bottom(self, t: float) -> float:
        total, prev = 0.0, None
        for it in self.items:
            m = it.msg
            if m.kind == "typing":
                continue
            e = E.ease_out_cubic((t - m.t) / GROW)
            if e <= 0:
                break
            total += ((0 if prev is None else (GAP_SAME if prev == m.side else GAP_SWITCH)) + it.h) * e
            prev = m.side
        return total + self._typing_room(t)

    def offset(self, t: float) -> float:
        return max(0.0, self.content_bottom(t) - self.st.view_h)

    def visible(self, it: Item, t: float) -> tuple[float, float]:
        """(top, bottom) of the part of an item that is inside the scrolling area at time t."""
        top = self.st.view_top + it.top - self.offset(t)
        return max(top, self.st.chat_top), min(top + it.h, self.st.view_bottom)

    def center_of(self, it: Item, t: float) -> tuple[float, float]:
        """Centre of the visible part of an item at time t (for the push-in)."""
        x = X_LEFT + it.w / 2 if it.msg.side == "L" else self.st.dw - EDGE - it.w / 2
        top, bottom = self.visible(it, t)
        return x, (top + bottom) / 2

    def clock_at(self, t: float) -> str:
        shown = [it.msg.stamp for it in self.items if it.msg.t <= t]
        return shown[-1] if shown else self.items[0].msg.stamp

    # -------------------------------------------------------------- sprites
    def _stamp(self, s: str) -> np.ndarray:
        if s not in self._clk:
            self._clk[s] = E.premult(E.text_img(s, FONT, 26, STAMP_INK, pad=0))
        return self._clk[s]

    def _status_clock(self, s: str) -> np.ndarray:
        key = "status " + s
        if key not in self._clk:
            self._clk[key] = E.premult(E.tight(E.text_img(s, FONT_BOLD, 40, (30, 30, 30), pad=0)))
        return self._clk[key]

    def _video(self, it: Item, t: float) -> np.ndarray:
        """Premultiplied sprite of a video bubble; holds the last frame once the clip has ended."""
        m = it.msg
        shot = self.tl.shots[m.shot]
        tt = min(max(t, shot.t0), shot.t1 - 1e-3)
        key = int(math.floor((tt - shot.t0) * FPS + 1e-6))
        if it.held is None or it.held[0] != key:
            rgb = bubble_view(self.tl.sources[m.shot], shot, tt, (self.st.media_w, self.st.media_h))
            it.held = (key, np.dstack([rgb * self.vmask, self.vmask[..., 0]]).astype(np.float32))
        return it.held[1]

    def _scaled(self, spr: np.ndarray, img: Image.Image | None, sc: float) -> np.ndarray:
        if abs(sc - 1) <= 1e-3:
            return spr
        if img is None:  # video: premultiplied sprite -> straight RGBA for a clean resize
            a = np.maximum(spr[..., 3:4], 1e-6)
            straight = np.dstack([spr[..., :3] / a, spr[..., 3:4]])
            img = Image.fromarray((np.clip(straight, 0, 1) * 255).astype(np.uint8), "RGBA")
        im = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BICUBIC)
        return E.premult(im)

    # -------------------------------------------------------------- drawing
    def _draw_item(self, frame: np.ndarray, it: Item, t: float, y: float) -> None:
        m, st = it.msg, self.st
        p = (t - m.t) / POP
        sc, op = 0.86 + 0.14 * E.ease_out_back(p, 2.0), E.clamp01(p * 3)
        spr = self._scaled(self._video(it, t), None, sc) if m.kind == "video" else self._scaled(it.spr, it.img, sc)
        nh, nw = spr.shape[:2]
        ck = self._stamp(m.stamp)
        if m.side == "L":
            x = X_LEFT - (16 if m.kind == "text" else 0)
            E.blit(frame, spr, x, y + (it.h - nh), op)
            if m.avatar:
                E.blit(frame, self.avatar, X_AVATAR, y - 4, op)
            E.blit(frame, ck, x + nw + 12, y + nh - ck.shape[0] - 4, op)
            return
        x = st.dw - EDGE - nw
        E.blit(frame, spr, x, y + (it.h - nh), op)
        E.blit(frame, ck, x - ck.shape[1] - 12, y + nh - ck.shape[0] - 4, op)
        if t > m.t + 0.45:
            E.blit(frame, self.read, x - self.read.shape[1] - 12, y + nh - ck.shape[0] - self.read.shape[0] - 10,
                   op * E.clamp01((t - m.t - 0.45) / 0.15))

    def _draw_typing(self, frame: np.ndarray, it: Item, t: float, y: float) -> None:
        m = it.msg
        end = m.t + m.dur
        if not m.t <= t < end + 0.08:
            return
        p = E.ease_out_cubic((t - m.t) / 0.15) * (1 - E.clamp01((t - end) / 0.08))
        E.blit(frame, E.premult(typing_bubble(t, self.shell)), X_LEFT - 16, y, p)
        E.blit(frame, self.avatar, X_AVATAR, y - 4, p)

    def frame(self, t: float) -> np.ndarray:
        st = self.st
        frame = self.base.copy()
        off = self.offset(t)
        for it in self.items:
            if t < it.msg.t:
                break
            y = st.view_top + it.top - off
            if y + it.h < st.chat_top - 40 or y > st.view_bottom + 20:
                continue
            if it.msg.kind == "typing":
                self._draw_typing(frame, it, t, y)
            else:
                self._draw_item(frame, it, t, y)
        frame[:st.chat_top] = self.base[:st.chat_top]  # header and banner stay on top of scrolled content
        frame[st.input_top:] = self.base[st.input_top:]
        clock = self._status_clock(self.clock_at(t))
        E.blit(frame, clock, 120 - clock.shape[1] / 2, 54 - clock.shape[0] / 2)
        return frame


# ================================================================= phone mockup backdrop (1:1, 16:9)
class Backdrop:
    """Mockup canvas: brand-colour background plus copy beside the phone (16:9: logo block, 1:1: vertical lines)."""

    def __init__(self, ctx, st: Stage, theme: str, tagline: str, note: str | None):
        b, L = ctx.brand, ctx.layout
        self.st, self.L, self.dark = st, L, theme == "dark"
        acc = b.color("accent")
        if self.dark:
            self.bg = E.flat_bg(E.mix(b.color("night"), acc, 0.14), b.color("night"), seed=5, focus_y=0.5)
        else:
            pale, mid = E.mix(acc, (255, 255, 255), 0.86), E.mix(acc, (255, 255, 255), 0.58)
            self.bg = E.flat_bg(pale, mid, seed=3, focus_y=0.5)
        self.ink = (244, 238, 226) if self.dark else b.color("ink")
        self.acc = b.color("accent_light") if self.dark else b.color("accent_deep")
        self.phone = Phone(st)
        self.stage = self.phone.stage(self.bg)  # background + device body, baked once
        self.wide = L.kind == "wide"
        self.note = E.text_img(note, FONT, 24, E.mix(self.ink, self.bg[0, 0] * 255, 0.35), pad=4) if note else None
        (self._build_block if self.wide else self._build_columns)(b, tagline)

    def _build_block(self, b, tagline: str) -> None:
        """16:9: label, logo and tagline stacked beside the phone."""
        L = self.L
        logo_img = b.logo_on_dark if self.dark else b.logo_on_light
        self.logo = Logo(logo_img, b.parts, int(min(0.34 * L.W - 20, 620)))
        self.label = E.TextLine(b.label, b.font("latin"), 31, self.acc, tracking=0.42) if b.label else None
        size = E.fit_size([tagline], b.font("serif"), 46, 0.36 * L.W, 0.12) if tagline else 0
        self.tag = E.TextLine(tagline, b.font("serif"), size, self.ink, tracking=0.12) if tagline else None
        parts = [("label", self.label.box_h if self.label else 0, 26), ("logo", self.logo.h, 60),
                 ("tag", self.tag.box_h if self.tag else 0, 0)]
        y, self.pos = L.cy - sum(h + g for _, h, g in parts if h) / 2 + L.sh * 0.02, {}
        for name, h, g in parts:
            self.pos[name] = y + h / 2
            y += h + g if h else 0

    def _build_columns(self, b, tagline: str) -> None:
        """1:1: the tagline runs down the left of the phone, the brand label (rotated) down the right."""
        st, L = self.st, self.L
        self.vline = E.VText(tagline, b.font("vertical"), 46, self.ink, tracking=0.3) if tagline else None
        self.vlabel = (E.text_img(b.label, b.font("latin"), 28, self.acc, tracking=0.42, pad=4).rotate(-90, expand=True)
                       if b.label else None)
        left, right = st.x - st.bezel, st.x + st.px[0] + st.bezel
        self.cols, self.mid = (left / 2, (right + L.W) / 2), st.y + st.px[1] / 2

    def _draw_block(self, frame: np.ndarray, t: float) -> None:
        cx = self.st.side_cx
        if self.label:
            self.label.draw_centered(frame, cx, self.pos["label"], t, 0.25, 0.015, 0.6, rise=0)
        self.logo.draw(frame, t, cx, self.pos["logo"] - self.logo.h / 2, 0.4, style="rise", gap=0.075,
                       sub_at=0.9, glint_at=1.5)
        if self.tag:
            self.tag.draw_centered(frame, cx, self.pos["tag"], t, 1.0, 0.04, 0.5, rise=0)

    def _draw_columns(self, frame: np.ndarray, t: float) -> None:
        if self.vline:
            self.vline.draw(frame, self.cols[0], self.mid - self.vline.height / 2, t, 0.5, 1e9)
        if self.vlabel:
            E.blit_img(frame, self.vlabel, self.cols[1], self.mid, opacity=E.ease_out_cubic((t - 0.9) / 0.6))

    def canvas(self, t: float) -> np.ndarray:
        frame = self.stage.copy()
        (self._draw_block if self.wide else self._draw_columns)(frame, t)
        if self.note is not None:
            if self.wide:
                cx, y = self.st.side_cx, self.L.y1 - 10
            else:
                cx, y = self.L.W / 2, (self.st.y + self.st.px[1] + self.st.bezel + self.L.H) / 2
            E.blit_img(frame, self.note, cx, y)
        return frame


# ================================================================= timeline
class ChatCard(EndCard):
    """EndCard whose tagline can be set per ad (the brand tagline stays the default)."""

    def __init__(self, ctx, tagline: str | None, **kw):
        self._line = tagline
        super().__init__(ctx, **kw)

    def _build(self, k: float) -> None:
        super()._build(k)
        if self._line:
            size = E.fit_size([self._line], self.b.font("serif"), int(46 * k), self.L.sw * 0.96, 0.12)
            self.tag = E.TextLine(self._line, self.b.font("serif"), size, self.ink, tracking=0.12)


class Chat(Timeline):
    bgm_style = "lofi"

    def __init__(self, ctx):
        v, b = ctx.v, ctx.brand
        raw = read_messages(ctx)
        self.zoom_on = bool(v.get("zoom", True))
        zoom_gap, hold = float(v.get("zoom_gap", 1.1)), float(v.get("hold", 3.15))
        tail = zoom_gap + (ZOOM_LEN if self.zoom_on else 0.0) + hold
        times = schedule(raw, float(v.get("start", 0.25)))
        if "duration" in v and not any("t" in d for d in raw):
            times = fit_to_duration(times, float(v["duration"]), tail)
        marks = stamps(raw, times, str(v.get("clock", "20:41")), float(v.get("minute_every", 5)))
        shots = [replace(make_shot(ctx.project, d["shot"], f"m{i}", times[i], times[i] + d["dur"], grade="natural"),
                         fit="cover") for i, d in enumerate(raw) if d["kind"] == "video"]
        for s in shots:
            if s.grade not in E.PRESETS:
                raise SpecError(f"chat: shot {s.name}: grade {s.grade!r} must be one of {', '.join(E.PRESETS)}")
        super().__init__(ctx, shots)
        self.msgs = self._msgs(ctx, raw, times, marks)
        self.zoom_at = times[-1] + zoom_gap
        self.card_at = self.zoom_at + (ZOOM_LEN if self.zoom_on else 0.0)
        self.duration = float(v["duration"]) if "duration" in v else self.card_at + hold
        if self.duration - self.card_at < 2.0:
            raise SpecError(f"chat: duration {self.duration:.1f}s is too short for {len(raw)} messages: even at the "
                            f"fastest pace the chat runs to {self.card_at:.1f}s and the end card needs >= 2 s "
                            f"(duration >= {self.card_at + 2.0:.1f}s, or fewer messages)")
        self.cover_at = self._cover(v, times, raw)
        self._build_view(ctx)
        self._cues()

    # -------------------------------------------------------------- build
    @staticmethod
    def _msgs(ctx, raw: list[dict], times: list[float], marks: list[str]) -> list[Msg]:
        out, vid = [], 0
        for i, d in enumerate(raw):
            kind, text = d["kind"], d["text"]
            if kind == "link":
                text = str(d.get("title") or text or ctx.brand.name)
            prev = raw[i - 1] if i else None
            # the friend's avatar shows with the first message of a run (and again after the typing indicator)
            new_run = prev is None or prev["side"] != "L" or prev["kind"] == "typing"
            avatar_on = d["side"] == "L" and kind != "typing" and new_run
            face = d.get("face") or ("surprised" if "！？" in text else "happy")
            sub = str(d.get("sub", ctx.cta if kind == "link" else ""))
            out.append(Msg(times[i], d["side"], kind, text, sub, face, d["dur"], marks[i], avatar_on,
                           vid if kind == "video" else -1))
            vid += kind == "video"
        return out

    def _cover(self, v: dict, times: list[float], raw: list[dict]) -> float:
        if "cover_at" in v:
            return float(v["cover_at"])
        vids = [t for t, d in zip(times, raw) if d["kind"] == "video"]
        t = vids[1] + 0.9 if len(vids) > 1 else times[0] + 0.55 * (times[-1] - times[0])
        return min(max(t, times[0] + 0.7), self.zoom_at - 0.1)

    def _build_view(self, ctx) -> None:
        v, L = ctx.v, ctx.layout
        end = v.get("end", {})
        theme, cta_style = end.get("theme", "light"), end.get("cta_style", "solid")
        if theme not in ("light", "dark") or cta_style not in ("solid", "outline", "text"):
            raise SpecError("chat: end.theme must be light | dark and end.cta_style solid | outline | text")
        self.st = make_stage(L, bool(v.get("note", True)))
        self.screen = Screen(ctx, self.st, self.msgs, self)
        times = {"logo": 0.1, "label": 0.35, "sub": 0.4, "tagline": 0.6, "glint": 0.75, "cta": 0.75}
        self.card = ChatCard(ctx, end.get("tagline"), theme=theme, cta_style=cta_style, logo_style="pop",
                             note=end.get("note"), particles=False, times=times)
        self.back = None
        if self.st.mockup:
            beside = (v.get("beside") or {}).get("tagline", ctx.brand.tagline)
            self.back = Backdrop(ctx, self.st, theme, beside, self.screen.note)
        self.target = self._push_target()
        self.z_end = clamp(1.54 * self.st.dw / self.target.w, 1.5, 2.8)

    def _push_target(self) -> Item:
        """The last link card if most of it is on screen when the push starts, else the last message."""
        items = [it for it in self.screen.items if it.msg.kind != "typing"]
        links = [it for it in items if it.msg.kind == "link"]
        if links:
            top, bottom = self.screen.visible(links[-1], self.zoom_at)
            if bottom - top >= 0.6 * links[-1].h:
                return links[-1]
        return items[-1]

    def _cues(self) -> None:
        """SFX cues: friend = receive, me = send (two-step pop / single blip), sticker wobble, link bell,
        key taps while the friend types, end-card chime then CTA pop."""
        out = self.sfx
        for m in self.msgs:
            if m.kind == "typing":
                n = min(5, max(2, int(m.dur / 0.16)))
                out += [[round(m.t + 0.08 + k * (m.dur - 0.12) / n, 3), "type", 0.5] for k in range(n)]
                continue
            out.append([round(m.t, 3), "send" if m.side == "R" else "receive", 1.0])
            if m.kind == "sticker":
                out.append([round(m.t + 0.02, 3), "sticker", 0.8])
            elif m.kind == "link":
                out.append([round(m.t + 0.05, 3), "link", 0.9])
        out += [[round(self.card_at + 0.12, 3), "chime", 1.0], [round(self.card_at + 0.75, 3), "pop", 0.7]]
        out.sort(key=lambda c: c[0])

    # -------------------------------------------------------------- frames
    def _push(self, screen: np.ndarray, t: float) -> np.ndarray:
        """Zoom the screen into the link card, easing until the dissolve to the end card is complete."""
        st = self.st
        u = E.ease_in_out_sine((t - self.zoom_at) / (self.card_at + CARD_FADE - self.zoom_at))
        z = 1 + (self.z_end - 1) * u
        cx, cy = self.screen.center_of(self.target, self.zoom_at)
        wc, hc = st.dw / z, st.dh / z
        left, top = clamp(cx - wc / 2, 0, st.dw - wc), clamp(cy - hc / 2, 0, st.dh - hc)
        box = (left, top, left + wc, top + hc)
        im = Image.fromarray(E.to_u8(screen)).resize((st.dw, st.dh), Image.BICUBIC, box=box)
        return np.asarray(im, np.float32) / 255

    def _chat_frame(self, t: float) -> np.ndarray:
        screen = self.screen.frame(min(t, self.zoom_at))
        if self.zoom_on and t > self.zoom_at:
            screen = self._push(screen, t)
        if self.back is None:
            return screen
        out = self.back.canvas(t)
        self.back.phone.place(out, screen)
        return out

    def _card_frame(self, t: float) -> np.ndarray:
        if self.back is None:
            return self.card.frame(t, self.card_at)
        out = self.back.bg.copy()
        self.card.draw_on(out, t, self.card_at)
        return out

    def frame(self, t: float) -> np.ndarray:
        if t >= self.card_at + CARD_FADE:
            return self._card_frame(t)
        out = self._chat_frame(t)
        if t >= self.card_at:
            k = E.smoothstep((t - self.card_at) / CARD_FADE)
            out = out * (1 - k) + self._card_frame(t) * k
        return out


def build(ctx) -> Timeline:
    if "messages" not in ctx.v:
        raise SpecError("chat: 'messages' is required")
    return Chat(ctx)
