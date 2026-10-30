"""Colours, fonts and decoration of each tone for thumbnails and covers.

Fonts, the accent colour, ink and paper come from the tone's Look (so edit.json "look": {"fonts": ...} and the
brand accent apply); the rest of the poster look (backgrounds, outlines, badge style) is the table below, described
in references/tones.md. A Look may carry its own `THUMB` dict to override any field here.
`look.fonts.thumb` in edit.json sets the family of the big copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cv import gfx as G

WHITE, BLACK = (255, 255, 255), (13, 13, 13)


@dataclass(frozen=True)
class Theme:
    key: str
    dark: bool                      # dark ground: white copy, light sticker outline reads best
    bg: tuple                       # ground
    bg2: tuple                      # second ground colour (panels, split side, gradient end)
    accent: tuple                   # the brand colour (headline fill, marker, badge)
    accent_top: tuple               # lighter top of the accent gradient
    ink: tuple
    paper: tuple
    hot: tuple                      # red: speech-bubble punch line, the ✕ mark
    ok: tuple                       # green: the ○ mark
    metal: tuple                    # badge frame (gold)
    glow: tuple                     # light behind the speaker
    font: tuple                     # (family, weight) of the big copy
    font2: tuple                    # (family, weight) of badges and small lines
    deco: str = "hex"               # hex | dots | seigaiha | lines | none
    shape: str = "hexagon"          # shape behind the speaker: hexagon | circle | burst | sun | none
    skew: float = -0.10             # lean of the big copy
    hero_fill: tuple = ()           # (top, bottom) of the emphasised copy
    hero_layers: tuple = ()         # outlines (rgb, width in em), outermost first
    plain_fill: tuple = WHITE
    plain_layers: tuple = ((BLACK, 0.20),)
    marker: bool = False            # highlighter swipe behind the emphasised copy
    underline: bool = False         # accent bar under the emphasised copy (minimal)
    split: str = "accent"           # colour of the second half of a before / after thumbnail: accent | bg2 | hot
    sticker: dict = field(default_factory=dict)    # stroke / edge / ring / edge_rgb / shadow of the speaker
    badge: str = "plate"            # plate (gold frame) | pill | seal
    bubble_line: tuple = BLACK      # speech-bubble outline


def _lift(rgb, k: float = 0.55) -> tuple:
    return G.mix_rgb(rgb, WHITE, k)


# what a tone looks like as a poster (colours that are not in its Look)
POSTER = {
    "trust": dict(dark=True, bg=(13, 13, 13), bg2=(26, 24, 52), hot=(232, 28, 58), ok=(22, 190, 110),
                  metal=(222, 184, 92), glow=(139, 92, 246), deco="hex", shape="hexagon", skew=-0.10,
                  hero_layers=((WHITE, 0.30), (BLACK, 0.17)), plain_layers=((BLACK, 0.20),),
                  sticker=dict(stroke=10, edge=4)),
    "friendly": dict(dark=False, bg=(255, 249, 240), bg2=(255, 226, 200), hot=(240, 132, 118), ok=(110, 196, 164),
                     metal=(255, 196, 176), glow=(255, 196, 176), deco="dots", shape="circle", skew=-0.03, marker=True,
                     split="bg2",
                     hero_fill=((84, 62, 52), (84, 62, 52)), hero_layers=((WHITE, 0.30),),
                     plain_layers=((WHITE, 0.30),), plain_fill=(84, 62, 52),
                     sticker=dict(stroke=12, edge=0, shadow=0.35), badge="pill", bubble_line=(84, 62, 52)),
    "variety": dict(dark=False, bg=(255, 214, 0), bg2=(255, 60, 170), hot=(255, 40, 60), ok=(40, 190, 60),
                    metal=(255, 255, 255), glow=(255, 255, 255), deco="dots", shape="burst", skew=-0.13, split="bg2",
                    hero_layers=(((40, 10, 60), 0.34), (WHITE, 0.19)), plain_layers=((BLACK, 0.26),),
                    sticker=dict(stroke=12, edge=5), badge="pill"),
    "wa": dict(accent=(222, 184, 92), dark=True, bg=(18, 16, 14), bg2=(42, 14, 14), hot=(200, 30, 40), ok=(60, 150, 90),
               metal=(205, 165, 80), glow=(200, 30, 40), deco="seigaiha", shape="sun", skew=-0.04, split="hot",
               hero_fill=((255, 238, 176), (214, 160, 60)), hero_layers=((BLACK, 0.22),),
               plain_layers=((BLACK, 0.22),), sticker=dict(stroke=7, edge=3, edge_rgb=(205, 165, 80)),
               badge="seal"),
    "minimal": dict(accent=(214, 92, 55), dark=False, bg=(246, 245, 242), bg2=(232, 231, 227), hot=(220, 60, 60),
                    ok=(40, 160, 100), metal=(24, 24, 27), glow=(255, 255, 255), deco="lines", shape="circle", skew=0.0, split="bg2",
                    underline=True, hero_fill=((24, 24, 27), (24, 24, 27)), hero_layers=(), plain_layers=(),
                    plain_fill=(24, 24, 27),
                    sticker=dict(stroke=6, edge=0, shadow=0.45), badge="pill"),
}


def _family(look, role: str, fallback: tuple) -> tuple:
    entry = look.S.get(role, {})
    return (entry.get("font", fallback[0]), int(entry.get("weight", fallback[1])))


def for_look(look, tone: str, thumb_font: str | None = None) -> Theme:
    """Theme of a tone from its Look instance (accent / ink / paper / fonts) and the poster table."""
    p = {**POSTER.get(tone, POSTER["trust"]), **getattr(look, "THUMB", {})}
    S = look.S
    accent = G.hexrgb(p["accent"] if look.key == "base" and "accent" in p else S["accent"])
    ink, paper = G.hexrgb(S["ink"]), G.hexrgb(S["paper"])
    title = _family(look, "title", ("gothic", 900))
    minimal = tone == "minimal"
    font = (thumb_font or title[0], title[1] if minimal else max(title[1], 800))
    font2 = (title[0], max(title[1], 700))
    bg = p["bg"]
    return Theme(
        key=tone, dark=p["dark"], bg=bg, bg2=p["bg2"], accent=accent, accent_top=_lift(accent, 0.55), ink=ink,
        paper=paper, hot=p["hot"], ok=p["ok"], metal=p["metal"], glow=p["glow"], font=font, font2=font2,
        deco=p["deco"], shape=p["shape"], skew=p["skew"],
        hero_fill=p.get("hero_fill") or (_lift(accent, 0.55), accent), hero_layers=p["hero_layers"],
        plain_fill=p.get("plain_fill", WHITE), plain_layers=p["plain_layers"], marker=p.get("marker", False),
        underline=p.get("underline", False), split=p.get("split", "accent"),
        sticker=p["sticker"], badge=p.get("badge", "plate"), bubble_line=p.get("bubble_line", BLACK))
