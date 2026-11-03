"""Chat taste, screen art: LINE-style UI parts drawn with PIL in design px (1080-wide phone), the stage
geometry that places the phone screen on each canvas, and the device frame for the 1:1 / 16:9 mockup.
No LINE logo, name or icon anywhere (see chat.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core import engine as E

# ---------------------------------------------------------------- constants (design px = px of a 1080-wide phone)
WALL, MINE = (152, 177, 212), (134, 222, 98)
INK, STAMP_INK, NOTE_INK = (24, 24, 24), (70, 80, 100), (40, 50, 70)
FONT, FONT_BOLD = "goth4", "goth6"
TEXT_PX, TEXT_LINE = 46, 1.42
GAP_SAME, GAP_SWITCH = 14, 30
X_AVATAR, X_LEFT, EDGE = 36, 150, 40
HDR, BANNER_Y, BANNER_H, NOTE_ROW = 228, 252, 92, 30
POP, GROW, ZOOM_LEN, CARD_FADE = 0.25, 0.22, 0.6, 0.35
WORK_W = 720  # video bubbles are cleaned up and graded at this width, then scaled to the bubble
MOCK_DW, MOCK_ASPECT, MOCK_HEIGHT = 840, 0.54, 0.92  # mockup: design width, screen w/h, phone height / canvas height
KINDS = ("text", "video", "sticker", "link", "typing")
SIDES = {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}
FACES = ("surprised", "happy", "love")
NO_START = set("、。，．,.）)」』】！？!?…・ー〜～" "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")  # never start a line
NO_END = set("（(「『【")
SOFT_BREAK = set("、。！？!?…")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ================================================================= stage geometry
@dataclass(frozen=True)
class Stage:
    """Phone screen in design px (dw x dh), its vertical stack, and where it lands on the canvas."""
    dw: int
    dh: int
    scale: float  # canvas px per design px (1.0 = full-screen)
    x: float  # top-left of the screen on the canvas
    y: float
    mockup: bool
    chat_top: int  # rows above are fixed chrome (status bar, header, banner)
    view_top: int  # content origin
    view_bottom: int
    input_top: int
    input_h: int
    kb_top: int
    key_h: int
    media_h: int
    bezel: int = 0
    side_cx: float = 0.0  # 16:9: centre of the copy beside the phone
    banner_y: int = BANNER_Y  # top of the pinned banner (its logo must start inside the safe zone)
    note_right: int = 0  # right edge of the in-screen note (inside the safe zone)

    @property
    def view_h(self) -> int:
        return self.view_bottom - self.view_top

    @property
    def media_w(self) -> int:
        return round(self.media_h * 0.75)

    @property
    def bubble_max(self) -> int:
        return int(min(self.dw - 250, 800))

    @property
    def px(self) -> tuple[int, int]:
        """Screen size on the canvas."""
        return round(self.dw * self.scale), round(self.dh * self.scale)


def make_stage(L, note_row: bool) -> Stage:
    mockup = L.kind in ("square", "wide")
    bezel = round(L.H * 0.012) if mockup else 0
    if mockup:
        sh = L.H * MOCK_HEIGHT - 2 * bezel
        scale = sh * MOCK_ASPECT / MOCK_DW
        dw, dh = MOCK_DW, round(sh / scale)
        x, y, side_cx = _phone_origin(L, sh * MOCK_ASPECT, sh, bezel)
        x, y = round(x), round(y)
    else:
        dw, dh, scale, x, y, side_cx = L.W, L.H, 1.0, 0.0, 0.0, 0.0
    banner_y = BANNER_Y if mockup else max(BANNER_Y, round(L.y0) - 17)
    chat_top = banner_y + BANNER_H + 18 + (NOTE_ROW if note_row and not mockup else 0)
    view_top, compact = chat_top + 22, dh < 1700
    input_h, pad = (112, 24) if compact else (130, 40)
    home = 26 if mockup else 0  # strip under the keys for the home indicator
    room = dh - view_top - 0.36 * dh - pad - input_h - 56 - home
    key_h = int(clamp(room // 4, 62, 86))
    kb_top = dh - (56 + 4 * key_h + home)
    input_top = kb_top - input_h
    view_bottom = input_top - pad
    media_h = int(clamp(0.8 * (view_bottom - view_top), 330, 586))
    return Stage(dw, dh, scale, x, y, mockup, chat_top, view_top, view_bottom, input_top, input_h, kb_top, key_h,
                 media_h, bezel, side_cx, banner_y, min(dw - EDGE - 4, round(L.x1)))


def _phone_origin(L, sw: float, sh: float, bezel: int) -> tuple[float, float, float]:
    """(x, y) of the screen and centre x of the copy beside the phone, for the mockup formats."""
    pw = sw + 2 * bezel
    y = (L.H - sh) / 2
    if L.kind == "square":  # a little above centre: the "※画面はイメージです" note sits under the phone
        return (L.W - sw) / 2, y - 0.013 * L.H, 0.0
    block, gap = 0.34 * L.W, 0.07 * L.W
    left = (L.W - (block + gap + pw)) / 2
    return left + block + gap + bezel, y, left + block / 2


# ================================================================= UI art (PIL, design px)
def rounded(w: int, h: int, r: int, fill) -> Image.Image:
    img = Image.new("RGBA", (w, h), (*fill[:3], 0))
    alpha = fill[3] if len(fill) > 3 else 255
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=(*fill[:3], alpha))
    return img


_TOKEN = re.compile(r"[A-Za-z0-9.,'’%/:@_+&#-]+|.")


def _greedy(toks: list[str], f, max_w: float) -> list[str]:
    """Fill lines up to max_w; a closing mark never starts a line (the previous character moves down with it)
    and a line prefers to end after ！？、。 when that leaves it at least 40 % full."""
    lines: list[str] = []
    cur: list[str] = []
    soft = 0  # tokens up to the last good break point
    for tok in toks:
        if cur and f.getlength("".join(cur) + tok) > max_w:
            if tok[0] in NO_START and len(cur) > 1:
                cut = len(cur) - 1
            else:
                cut = soft if soft >= 0.4 * len(cur) else len(cur)
            if cut > 1 and cur[cut - 1][-1] in NO_END:
                cut -= 1
            lines.append("".join(cur[:cut]))
            cur, soft = cur[cut:], 0
        cur.append(tok)
        if tok in SOFT_BREAK:
            soft = len(cur)
    return lines + ["".join(cur)]


def wrap(text: str, size: int, max_w: float) -> list[str]:
    """Bubble text lines: explicit \\n kept, Latin words stay whole, kinsoku, and a paragraph that needs several
    lines is balanced (the narrowest width that still gives the same number of lines)."""
    f = E.font(FONT, size)
    out: list[str] = []
    for para in text.split("\n"):
        toks = [p for t in _TOKEN.findall(para) for p in (list(t) if f.getlength(t) > max_w else [t])]
        lines = _greedy(toks, f, max_w)
        if len(lines) > 1:
            lo, hi = 0.5 * max_w, max_w
            for _ in range(8):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if len(_greedy(toks, f, mid)) == len(lines) else (mid, hi)
            lines = _greedy(toks, f, hi)
        out += lines
    return out


def text_lines(lines: list[str], size: int, color, key: str = FONT) -> Image.Image:
    f = E.font(key, size)
    lh = int(size * TEXT_LINE)
    img = Image.new("RGBA", (int(max(f.getlength(ln) for ln in lines)) + 4, lh * len(lines)), (*color, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((0, i * lh + int(size * 1.05)), ln, font=f, fill=(*color, 255), anchor="ls")
    return img


def bubble(lines: list[str], side: str, mine) -> Image.Image:
    """Speech bubble with a small tail (top-left for the friend, top-right for me); 16 px of the image is tail room."""
    t = text_lines(lines, TEXT_PX, INK)
    px, py = 34, 22
    w, h = t.width + 2 * px, t.height + 2 * py - 8
    col = mine if side == "R" else (255, 255, 255)
    img = Image.new("RGBA", (w + 16, h), (*col, 0))
    d = ImageDraw.Draw(img)
    x0 = 16 if side == "L" else 0
    d.rounded_rectangle((x0, 0, x0 + w - 1, h - 1), radius=38, fill=(*col, 255))
    if side == "L":
        d.polygon([(0, 14), (x0 + 20, 8), (x0 + 22, 34)], fill=(*col, 255))
    else:
        d.polygon([(w + 15, 14), (w - 4, 8), (w - 6, 34)], fill=(*col, 255))
    img.alpha_composite(t, (x0 + px, py))
    return img


def _heart(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill) -> None:
    d.ellipse((cx - r, cy - r * 0.9, cx, cy + r * 0.1), fill=fill)
    d.ellipse((cx, cy - r * 0.9, cx + r, cy + r * 0.1), fill=fill)
    d.polygon([(cx - r * 0.97, cy - r * 0.2), (cx + r * 0.97, cy - r * 0.2), (cx, cy + r * 1.05)], fill=fill)


def sticker(text: str, face: str) -> Image.Image:
    """Original round-face character sticker with a caption (no third-party artwork)."""
    size = E.fit_size([text], "maru", 62, 360, min_size=34)
    cap = E.tight(E.text_img(text, "maru", size, (255, 255, 255), stroke=9, stroke_color=(92, 64, 51), pad=6))
    s = 300
    sw = max(s, cap.width + 16)
    img = Image.new("RGBA", (sw, s), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    cx, cy, r, dark, skin = sw // 2, 150, 112, (60, 40, 32, 255), (255, 221, 190, 255)
    d.ellipse((cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6), fill=(255, 255, 255, 255))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=skin, outline=(92, 64, 51, 255), width=6)
    for bx in (-70, 70):  # blush
        d.ellipse((cx + bx - 22, cy + 6, cx + bx + 22, cy + 30), fill=(255, 160, 160, 200))
    for ex in (-42, 42):  # eyes
        if face == "surprised":
            d.ellipse((cx + ex - 14, cy - 40, cx + ex + 14, cy - 2), fill=dark)
        elif face == "love":
            _heart(d, cx + ex, cy - 22, 21, (236, 82, 110, 255))
        else:
            d.arc((cx + ex - 20, cy - 34, cx + ex + 20, cy + 2), 200, 340, fill=dark, width=7)
    mouth = (140, 60, 60, 255)
    if face == "surprised":
        d.ellipse((cx - 16, cy + 18, cx + 16, cy + 58), fill=mouth)
        d.polygon([(cx + 100, 40), (cx + 112, 70), (cx + 88, 70)], fill=(120, 190, 255, 255))  # sweat drop
        d.ellipse((cx + 88, 58, cx + 112, 82), fill=(120, 190, 255, 255))
    else:
        d.chord((cx - 34, cy + 4, cx + 34, cy + 60), 0, 180, fill=mouth)
        for sx, sy in ((cx - 102, 52), (cx + 108, 92), (cx + 88, 30)):
            img.alpha_composite(E.sparkle_img(18, (255, 210, 90)), (sx - 18, sy - 18))
    img.alpha_composite(cap, ((sw - cap.width) // 2, s - cap.height - 2))
    return img


def one_line(text: str, key: str, size: int, max_w: float, min_size: int = 20) -> tuple[str, int]:
    """(text, size) that fits one line: shrunk down to min_size, then cut with an ellipsis."""
    size = E.fit_size([text], key, size, max_w, min_size=min_size)
    if E.text_width(text, key, size) > max_w:
        while len(text) > 1 and E.text_width(text + "…", key, size) > max_w:
            text = text[:-1]
        text += "…"
    return text, size


def link_card(brand, w: int, title: str, sub: str, k: float = 1.0) -> Image.Image:
    """Booking link preview: brand logo on paper, title and caption underneath (k scales the text block)."""
    h_img, h_txt = round(w * 0.39), round(150 * k)
    img = rounded(w, h_img + h_txt, 30, (255, 255, 255))
    top = Image.new("RGBA", (w, h_img), (*brand.color("paper"), 255))
    logo = Image.open(brand.logo_on_light).convert("RGBA")
    s = min(0.656 * w / logo.width, 0.62 * h_img / logo.height)
    logo = logo.resize((max(1, round(logo.width * s)), max(1, round(logo.height * s))), Image.LANCZOS)
    top.alpha_composite(logo, ((w - logo.width) // 2, (h_img - logo.height) // 2))
    mask = rounded(w, h_img + 40, 30, (0, 0, 0)).split()[3].crop((0, 0, w, h_img))
    img.paste(top, (0, 0), mask)
    text, size = one_line(title, FONT_BOLD, round(36 * k), w - 60, 22)
    img.alpha_composite(E.tight(E.text_img(text, FONT_BOLD, size, INK, pad=0)), (30, h_img + round(30 * k)))
    if sub:
        text, size = one_line(sub, FONT, round(30 * k), w - 60)
        img.alpha_composite(E.tight(E.text_img(text, FONT, size, (120, 120, 120), pad=0)), (30, h_img + round(92 * k)))
    return img


def avatar(brand, letter: str) -> Image.Image:
    s = 92
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    k = np.clip((xx + yy) / (2 * s), 0, 1)[..., None]
    col = (np.array(E.mix(brand.color("accent"), (255, 255, 255), 0.45), np.float32) * (1 - k)
           + np.array(E.mix(brand.color("accent"), (0, 0, 0), 0.08), np.float32) * k)
    a = np.clip(s / 2 - np.sqrt((xx - s / 2 + 0.5) ** 2 + (yy - s / 2 + 0.5) ** 2) + 0.5, 0, 1)[..., None] * 255
    img = Image.fromarray(np.dstack([col, a]).astype(np.uint8), "RGBA")
    t = E.tight(E.text_img(letter, FONT_BOLD, 44, (255, 255, 255), pad=0))
    img.alpha_composite(t, ((s - t.width) // 2, (s - t.height) // 2 - 2))
    return img


def typing_bubble(t: float, shell: Image.Image) -> Image.Image:
    """Three bouncing dots in a compact friend bubble (`shell` = empty bubble)."""
    img = shell.copy()
    d = ImageDraw.Draw(img)
    for k in range(3):
        y = img.height / 2 + 4 - 9 * max(0.0, math.sin((t * 6 - k * 0.8) % (2 * math.pi)))
        x = 16 + (img.width - 16) / 2 + (k - 1) * 30
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(170, 170, 170, 255))
    return img


# ---------------------------------------------------------------- static chrome
def _label(img: Image.Image, text: str, box: tuple, size: int, color=(40, 40, 40)) -> None:
    lab = E.tight(E.text_img(text, FONT, size, color, pad=0))
    x0, y0, x1, y1 = box
    img.alpha_composite(lab, (int(x0 + (x1 - x0 - lab.width) / 2), int(y0 + (y1 - y0 - lab.height) / 2)))


def status_bar(d: ImageDraw.ImageDraw, dw: int) -> None:
    """Signal bars, wifi and battery, right-aligned (the clock is drawn per frame)."""
    ink, xr = (30, 30, 30, 255), dw - 51
    for k in range(4):
        h = 12 + k * 7
        d.rounded_rectangle((xr - 183 + k * 13, 66 - h, xr - 183 + k * 13 + 8, 66), radius=2, fill=ink)
    cx, cy = xr - 99, 68
    for r in (30, 20, 10):
        d.arc((cx - r, cy - r, cx + r, cy + r), 225, 315, fill=ink, width=6)
    d.ellipse((cx - 4, cy - 6, cx + 4, cy + 2), fill=ink)
    d.rounded_rectangle((xr - 61, 40, xr - 7, 68), radius=8, outline=ink, width=3)
    d.rounded_rectangle((xr - 56, 45, xr - 19, 63), radius=4, fill=ink)
    d.rounded_rectangle((xr - 5, 49, xr, 59), radius=2, fill=ink)


def header(img: Image.Image, d: ImageDraw.ImageDraw, dw: int, title: str) -> None:
    ink = (40, 40, 40, 255)
    d.line((70, 136, 46, 160, 70, 184), fill=ink, width=6, joint="curve")  # back
    name = E.tight(E.text_img(title, FONT_BOLD, 46, (30, 30, 30), pad=0))
    img.alpha_composite(name, ((dw - name.width) // 2, 160 - name.height // 2))
    for k in range(3):  # menu
        d.line((dw - 110, 144 + k * 16, dw - 66, 144 + k * 16), fill=ink, width=5)


def banner_img(brand, w: int, text: str) -> Image.Image:
    """Pinned banner: pin icon, brand logo, one line of copy. The logo sits inside the safe zone."""
    img = rounded(w, BANNER_H, 22, (255, 255, 255, 236))
    d = ImageDraw.Draw(img)
    acc = (*brand.color("accent_deep"), 255)
    d.ellipse((26, 30, 58, 62), outline=acc, width=4)
    d.line((42, 62, 42, 76), fill=acc, width=4)
    logo = Image.open(brand.logo_on_light).convert("RGBA")
    k = min(210 / logo.width, 62 / logo.height)
    logo = logo.resize((max(1, round(logo.width * k)), max(1, round(logo.height * k))), Image.LANCZOS)
    img.alpha_composite(logo, (82, (BANNER_H - logo.height) // 2 + 4))
    x = 82 + logo.width + 34
    if text:
        text, size = one_line(text, FONT, 32, w - x - 26)
        lab = E.tight(E.text_img(text, FONT, size, (90, 90, 90), pad=0))
        img.alpha_composite(lab, (x, (BANNER_H - lab.height) // 2))
    return img


def input_bar(img: Image.Image, d: ImageDraw.ImageDraw, st: Stage) -> None:
    top, cy = st.input_top, st.input_top + st.input_h // 2
    d.rectangle((0, top, st.dw, top + st.input_h), fill=(250, 250, 250, 255))
    grey = (120, 120, 120, 255)
    d.line((60, cy, 100, cy), fill=grey, width=5)
    d.line((80, cy - 20, 80, cy + 20), fill=grey, width=5)
    d.rounded_rectangle((150, cy - 37, st.dw - 180, cy + 37), radius=37, fill=(238, 238, 238, 255))
    ph = E.tight(E.text_img("Aa", FONT, 34, (160, 160, 160), pad=0))
    img.alpha_composite(ph, (185, cy - ph.height // 2 + 4))
    d.ellipse((st.dw - 140, cy - 27, st.dw - 80, cy + 33), outline=grey, width=5)


KEY_ROWS = (("→", "あ", "か", "さ", "BS"), ("UNDO", "た", "な", "は", "空白"),
            ("ABC", "ま", "や", "ら", "改行"), ("GLOBE", "^_^", "わ", "、。?!", None))


def _key_icon(d: ImageDraw.ImageDraw, name: str, cx: float, cy: float, u: float) -> None:
    ink = (40, 40, 40, 255)
    w = max(2, round(4 * u))
    if name == "BS":
        pts = [(-34, 0), (-16, -20), (30, -20), (30, 20), (-16, 20)]
        d.polygon([(cx + x * u, cy + y * u) for x, y in pts], outline=ink, width=w)
        d.line((cx - 2 * u, cy - 9 * u, cx + 16 * u, cy + 9 * u), fill=ink, width=w)
        d.line((cx - 2 * u, cy + 9 * u, cx + 16 * u, cy - 9 * u), fill=ink, width=w)
    elif name == "UNDO":
        d.arc((cx - 22 * u, cy - 18 * u, cx + 22 * u, cy + 22 * u), 200, 120, fill=ink, width=max(3, round(5 * u)))
        d.polygon([(cx - 30 * u, cy - 4 * u), (cx - 14 * u, cy - 2 * u), (cx - 24 * u, cy + 12 * u)], fill=ink)
    else:  # globe
        d.ellipse((cx - 22 * u, cy - 22 * u, cx + 22 * u, cy + 22 * u), outline=ink, width=w)
        d.ellipse((cx - 10 * u, cy - 22 * u, cx + 10 * u, cy + 22 * u), outline=ink, width=max(2, round(3 * u)))
        d.line((cx - 22 * u, cy, cx + 22 * u, cy), fill=ink, width=max(2, round(3 * u)))


def keyboard(img: Image.Image, d: ImageDraw.ImageDraw, st: Stage) -> None:
    """iPhone Japanese 12-key pad: 5 columns, grey function keys at both sides, white kana keys between,
    a tall 改行 key spanning the last two rows."""
    d.rectangle((0, st.kb_top, st.dw, st.dh), fill=(209, 212, 219, 255))
    gap, margin, kh = 12, 12, st.key_h
    kw, u = (st.dw - 2 * margin - 4 * gap) / 5, kh / 86
    white, grey = (255, 255, 255, 255), (172, 177, 188, 255)
    for r, row in enumerate(KEY_ROWS):
        for c, label in enumerate(row):
            if label is None:
                continue
            x0, y_top = margin + c * (kw + gap), st.kb_top + 14 + r * (kh + gap)
            y_bot = y_top + kh + (kh + gap if label == "改行" else 0)
            box = (x0, y_top, x0 + kw, y_bot)
            d.rounded_rectangle(box, radius=round(14 * u), fill=white if 1 <= c <= 3 else grey)
            d.line((x0 + 6, y_bot + 2, x0 + kw - 6, y_bot + 2), fill=(150, 154, 163, 255), width=2)
            if label in ("BS", "UNDO", "GLOBE"):
                _key_icon(d, label, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, u)
            else:
                _label(img, label, box, round((46 if 1 <= c <= 3 and len(label) == 1 else 32) * u))


def chrome(brand, st: Stage, title: str, banner: str | None, note: str | None, wall, bar) -> np.ndarray:
    """Everything that never changes: wallpaper, status bar, header, banner, note, input bar, keyboard."""
    img = Image.new("RGBA", (st.dw, st.dh), (*wall, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, st.dw, HDR), fill=(*bar, 255))
    status_bar(d, st.dw)
    header(img, d, st.dw, title)
    if banner is not None:
        img.alpha_composite(banner_img(brand, st.dw - 2 * EDGE, banner), (EDGE, st.banner_y))
    if note:
        n = E.tight(E.text_img(note, FONT, 24, NOTE_INK, pad=0))
        img.alpha_composite(n, (st.note_right - n.width, st.banner_y + BANNER_H + 8))
    input_bar(img, d, st)
    keyboard(img, d, st)
    return np.asarray(img.convert("RGB"), np.float32) / 255


# ================================================================= phone mockup (1:1, 16:9)
def _rr_mask(w: int, h: int, r: int, ss: int = 3) -> Image.Image:
    big = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return big.resize((w, h), Image.LANCZOS)


class Phone:
    """Device frame (body, shadow, buttons), screen mask, dynamic island and home bar as static art.
    stage(bg) bakes the body into the background once; place() then only pastes the moving screen."""

    PAD = 80

    def __init__(self, st: Stage):
        sw, sh = st.px
        bez, rad = st.bezel, round(0.115 * sw)
        self.st, self.size, self.r = st, (sw, sh), rad + 2
        w, h, pad = sw + 2 * bez, sh + 2 * bez, self.PAD
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        shadow = Image.new("L", img.size, 0)
        shadow.paste(_rr_mask(w, h, rad + bez), (pad, pad + 26))
        shadow = shadow.filter(ImageFilter.GaussianBlur(30)).point(lambda v: int(v * 0.42))
        img.alpha_composite(Image.merge("RGBA", (*(Image.new("L", img.size, 20),) * 3, shadow)))
        d = ImageDraw.Draw(img)
        for y0, y1 in ((0.17, 0.22), (0.26, 0.33), (0.35, 0.42)):  # side buttons
            d.rounded_rectangle((pad - 4, pad + h * y0, pad + 3, pad + h * y1), radius=2, fill=(58, 58, 64, 255))
        d.rounded_rectangle((pad + w - 3, pad + h * 0.30, pad + w + 4, pad + h * 0.40), radius=2,
                            fill=(58, 58, 64, 255))
        body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        body.paste(Image.new("RGBA", (w, h), (74, 74, 82, 255)), (0, 0), _rr_mask(w, h, rad + bez))
        body.paste(Image.new("RGBA", (w - 4, h - 4), (30, 30, 35, 255)), (2, 2), _rr_mask(w - 4, h - 4, rad + bez - 2))
        glass = Image.new("RGBA", (sw + 2, sh + 2), (4, 4, 6, 255))
        body.paste(glass, (bez - 1, bez - 1), _rr_mask(sw + 2, sh + 2, rad + 1))
        img.alpha_composite(body, (pad, pad))
        self.body = E.premult(img)
        self.origin = (st.x - bez - pad, st.y - bez - pad)
        self.mask = np.asarray(_rr_mask(sw, sh, rad), np.float32)[..., None] / 255
        self.front = self._front(sw, sh)

    @staticmethod
    def _front(sw: int, sh: int) -> list[tuple[np.ndarray, int, int]]:
        """Dynamic island and home bar as small premultiplied sprites with their offsets on the screen."""
        def pill(w: int, h: int, rgb, alpha: int) -> np.ndarray:
            img = Image.new("RGBA", (w, h), (*rgb, 0))
            img.putalpha(_rr_mask(w, h, h // 2).point(lambda v: v * alpha // 255))
            return E.premult(img)
        iw, ih, bw = round(0.30 * sw), round(0.085 * sw), round(0.34 * sw)
        return [(pill(iw, ih, (6, 6, 8), 255), (sw - iw) // 2, round(0.034 * sw)),
                (pill(bw, 5, (24, 24, 28), 210), (sw - bw) // 2, sh - 15)]

    def _corners(self) -> list[tuple[slice, slice]]:
        (sw, sh), r = self.size, self.r
        return [(slice(a, a + r), slice(b, b + r)) for a in (0, sh - r) for b in (0, sw - r)]

    def stage(self, bg: np.ndarray) -> np.ndarray:
        """bg with the device body baked in; remembers what lies under the screen's four rounded corners."""
        out = bg.copy()
        E.blit(out, self.body, *self.origin)
        x, y = int(self.st.x), int(self.st.y)
        region = out[y:y + self.size[1], x:x + self.size[0]]
        self.under = [region[ys, xs].copy() for ys, xs in self._corners()]
        return out

    def place(self, canvas: np.ndarray, screen: np.ndarray) -> None:
        """Paste the screen (scaled, rounded corners) and the island / home bar onto a staged canvas (in place)."""
        sw, sh = self.size
        x, y = int(self.st.x), int(self.st.y)
        small = np.asarray(Image.fromarray(E.to_u8(screen)).resize((sw, sh), Image.LANCZOS), np.float32) / 255
        region = canvas[y:y + sh, x:x + sw]
        region[:] = small
        for (ys, xs), under in zip(self._corners(), self.under):
            m = self.mask[ys, xs]
            region[ys, xs] = small[ys, xs] * m + under * (1 - m)
        for spr, fx, fy in self.front:
            E.blit(canvas, spr, x + fx, y + fy)
