"""信頼・解説 (trust): a business-talk YouTube look. White heavy gothic subtitles with a black outline and {yellow}
key words, black / yellow chapter banners, white explanation cards, calm camera (shot held >= 10 s, one slow push-in
per ~40 s), jet cut at 0.32 s pauses, soft lo-fi BGM about 14 dB under the voice, SFX only on cards and banners.

Use for: business explanations, know-how, AI / management talks, interviews where credibility matters.
"""
from __future__ import annotations

from cv.tones.base import Look


class Trust(Look):
    key = "trust"
    CUT = dict(pause=0.32, tail=0.11, lead=0.08, speed=1.0)
    CAMERA = dict(hold=10.0, med=1.14, punch_every=40.0, punch_zoom=1.08, punch_dur=0.5)
    AUDIO = dict(bgm="lofi", music_db=-14.0, sfx="soft", sfx_db=-2.0)
    GRADE = "eq=contrast=1.04:saturation=1.05"


LOOK = Trust
