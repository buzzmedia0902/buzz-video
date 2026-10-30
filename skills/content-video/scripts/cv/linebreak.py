"""Where a Japanese line may break, and how good each break is (lower = better).

Breaks are allowed only between words (janome morphological analysis), never inside one
(年の瀬, なんとなく, ひと月 stay whole), and never before a particle, auxiliary, suffix or closing mark.
Falls back to a character heuristic when janome is unavailable.
"""
from __future__ import annotations

from functools import lru_cache

NO_START = set("、。，．」』）！？!?ーっゃゅょぁぃぅぇぉッャュョァィゥェォ…‥")
PARTICLES = set("はがをにでとのもへやね")
CLOSE = "、。！？!?」』）…‥"
OPEN = "「『（"
HELPER_VERBS = {"いただく", "いただける", "くださる", "もらう", "もらえる", "くれる", "あげる", "おく", "しまう",
                "いる", "おる", "みる", "くる", "いく", "ある", "ほしい", "まいる", "いらっしゃる", "おる"}

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from janome.tokenizer import Tokenizer
        _tok = Tokenizer()
    return _tok


def _is_hira(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ"


def _heuristic(text: str) -> dict[int, float]:
    out = {}
    for i in range(1, len(text)):
        a, b = text[i - 1], text[i]
        if b in NO_START or a in OPEN or (b in PARTICLES and a not in CLOSE):
            continue
        if a in "」』" and _is_hira(b):
            out[i] = 30.0
        elif a in CLOSE:
            out[i] = 0.0
        elif b in OPEN:
            out[i] = 4.0
        elif a in PARTICLES:
            out[i] = 8.0 if not _is_hira(b) else 18.0
        else:
            out[i] = 40.0
    return out


def _attached(pos: str, sub: str, surface: str) -> bool:
    """Tokens that belong to the previous word's phrase: a line must not start with them."""
    if pos in ("助詞", "助動詞"):
        return True
    if pos == "記号":
        return surface not in OPEN
    return sub in ("接尾",)


@lru_cache(maxsize=4096)
def break_costs(text: str) -> dict[int, float]:
    """{char index i: cost of breaking between text[i-1] and text[i]}; missing index = not allowed."""
    try:
        toks = list(_tokenizer().tokenize(text))
    except Exception:  # janome missing or failed: keep the old heuristic
        return _heuristic(text)
    info, pos_i, base = [], 0, []
    for t in toks:
        p = t.part_of_speech.split(",")
        info.append((pos_i, t.surface, p[0], p[1]))
        base.append(t.base_form)
        pos_i += len(t.surface)
    if pos_i != len(text):
        return _heuristic(text)
    out = {}
    for j in range(1, len(info)):
        i, s, pos, sub = info[j]
        _, ps, ppos, psub = info[j - 1]
        if s[0] in NO_START or ps[-1] in OPEN or ppos == "接頭詞":
            continue
        if _attached(pos, sub, s):
            continue
        if ps[-1] in "」』" and pos in ("助詞", "助動詞"):
            continue
        if ps[-1] in "。！？!?":
            out[i] = -4.0                     # after a full sentence: the best place
        elif ppos == "記号" or ps[-1] in CLOSE:
            out[i] = 0.0                      # after 、」
        elif s in OPEN:
            out[i] = 4.0                      # before 「
        elif sub == "非自立" or (pos == "動詞" and sub == "非自立"):
            out[i] = 30.0                     # 〜して|まいる, 間の|こと: allowed, discouraged
        elif ppos == "助詞" and ps in ("て", "で") and base[j] in HELPER_VERBS:
            out[i] = 30.0                     # 言って|いただける: keep the helper verb with its verb
        elif ppos == "助詞":
            if psub == "接続助詞":
                out[i] = 4.0                  # 〜ば|, 〜て|, 〜が|
            elif psub == "連体化":
                out[i] = 14.0                 # お店の|品ぞろえ
            else:
                out[i] = 6.0                  # 話を|広げる
        elif ppos == "連体詞":
            out[i] = 30.0                     # その|方法
        elif ppos == "名詞" and pos == "名詞":
            out[i] = 35.0                     # compound noun
        elif ppos == "副詞":
            out[i] = 16.0                     # そう|言って, ひときわ|目を: the adverb belongs to its verb
        elif ppos in ("接続詞", "感動詞"):
            out[i] = 8.0
        else:
            out[i] = 20.0                     # 慣れた|段取り, 言って|いただける: modifier -> next phrase
    return out
