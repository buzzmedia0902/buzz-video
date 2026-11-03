"""The nearest tone of a reference (weighted matches against what each tone does, references/tones.md) and the
edit.json "look" overrides that bring the chosen tone closer to the reference."""
from __future__ import annotations

import os

import numpy as np


def near(x, lo: float, hi: float, soft: float):
    """1 inside [lo, hi], falling to 0 `soft` outside; None when x is unknown."""
    if x is None:
        return None
    d = max(lo - x, x - hi, 0.0)
    return max(0.0, 1.0 - d / soft)


def metrics_for_tone(m: dict) -> dict:
    cam, aud, tel = m.get("camera") or {}, m["audio"], m["telops"]
    pauses = aud.get("pauses") or {}
    colors = tel.get("colors", [])
    vivid = [c for c in colors if c["name"] not in ("白", "黒", "グレー")]
    music = aud.get("music")
    return {
        "cpm": m["shots"]["cuts_per_min"], "pause": pauses.get("p97_s"), "jet": pauses.get("jet_cut"),
        "zoom_sw": cam.get("zoom_switch_per_min"), "push": cam.get("push_per_min"),
        "bgm": (music["under_voice_db"] if music["present"] == "yes" else -60.0 if music["present"] == "no" else None)
        if music else None,
        "sfx": (aud.get("sfx") or {}).get("per_min_in_pauses"),
        "outline": (tel.get("outline_pct") or 0) >= 50 if "outline_pct" in tel else None,
        "plate": (tel.get("plate_pct") or 0) >= 40 if "plate_pct" in tel else None,
        "white": bool(colors) and colors[0]["name"] == "白", "vivid": len(vivid),
        "warm": any(c["name"] in ("赤", "オレンジ", "黄") for c in colors),
        "size": (tel.get("subtitle") or {}).get("line_height_pct"), "sat": m["look"]["saturation"],
        "bright": m["look"]["brightness"],
    }


def tone_scores(m: dict) -> tuple[dict, dict]:
    """Score of every tone: weighted matches of the measured numbers against what the tone does (references/tones.md)."""
    f = metrics_for_tone(m)
    b = lambda v: None if v is None else (1.0 if v else 0.0)
    size = None if f["size"] is None else f["size"] / 100
    table = {
        "trust": [(2, "cpm", near(f["cpm"], 8, 26, 8)), (3, "pause", near(f["pause"], 0.26, 0.40, 0.12)),
                  (2, "zoom_sw", near(f["zoom_sw"], 0, 5, 4)), (2, "bgm", near(f["bgm"], -20, -9, 8)),
                  (1, "sfx", near(f["sfx"], 0, 6, 6)), (4, "outline", b(f["outline"])), (3, "white", b(f["white"])),
                  (2, "vivid", near(f["vivid"], 1, 2, 2)),
                  (1.5, "size", near(size, 0.04, 0.075, 0.015))],
        "friendly": [(2, "cpm", near(f["cpm"], 5, 16, 7)), (3, "pause", near(f["pause"], 0.40, 0.65, 0.15)),
                     (2, "zoom_sw", near(f["zoom_sw"], 0, 4, 4)), (2, "bgm", near(f["bgm"], -22, -10, 8)),
                     (1, "sfx", near(f["sfx"], 0, 5, 5)), (4, "plate", b(f["plate"])),
                     (1, "no outline", None if f["outline"] is None else 1.0 - b(f["outline"])),
                     (2, "bright", near(f["bright"], 0.62, 1.0, 0.2))],
        "variety": [(2, "cpm", near(f["cpm"], 18, 70, 10)), (2, "pause", near(f["pause"], 0.14, 0.30, 0.10)),
                    (3, "zoom_sw", near(f["zoom_sw"], 5, 30, 4)), (1, "push", near(f["push"], 1.5, 10, 2)),
                    (2, "bgm", near(f["bgm"], -14, -4, 6)), (3, "sfx", near(f["sfx"], 6, 60, 6)),
                    (3, "vivid", near(f["vivid"], 2, 6, 2)), (2, "size", near(size, 0.05, 0.12, 0.02)),
                    (1, "outline", b(f["outline"])), (1, "sat", near(f["sat"], 0.4, 0.9, 0.12))],
        "wa": [(2, "cpm", near(f["cpm"], 3, 12, 5)), (3, "pause", near(f["pause"], 0.45, 0.85, 0.2)),
               (2, "zoom_sw", near(f["zoom_sw"], 0, 3, 3)), (2, "bgm", near(f["bgm"], -22, -10, 8)),
               (2, "outline", b(f["outline"])), (2, "bright", near(f["bright"], 0, 0.36, 0.14)),
               (2, "warm colours", b(f["warm"])), (1, "white", b(f["white"]))],
        "minimal": [(2, "cpm", near(f["cpm"], 0, 10, 6)), (1, "pause", near(f["pause"], 0.4, 1.2, 0.2)),
                    (3, "zoom_sw", near(f["zoom_sw"], 0, 0.5, 2)), (3, "bgm", near(f["bgm"], -60, -17, 8)),
                    (3, "sfx", near(f["sfx"], 0, 1, 3)),
                    (3, "no outline", None if f["outline"] is None else 1.0 - b(f["outline"])),
                    (2, "small text", near(size, 0.01, 0.038, 0.015)),
                    (1, "sat", near(f["sat"], 0, 0.3, 0.1))],
    }
    scores, parts = {}, {}
    for tone, terms in table.items():
        known = [(w, k, v) for w, k, v in terms if v is not None]
        scores[tone] = round(sum(w * v for w, k, v in known) / max(sum(w for w, _, _ in known), 1e-9), 2)
        parts[tone] = known
    return scores, parts


def why(m: dict, tone: str, parts: list) -> list[str]:
    f = metrics_for_tone(m)
    bits = {
        "pause": lambda: f"間の詰め方: 残っている間は {f['pause']:.2f} 秒まで（最長に近い値）",
        "cpm": lambda: f"カット数: 1分に {f['cpm']:.0f} 回",
        "outline": lambda: "テロップに黒フチあり" if f["outline"] else "テロップにフチなし",
        "white": lambda: "テロップの主な色は白" if f["white"] else "テロップの主な色は白ではない",
        "plate": lambda: "テロップの下に色板（吹き出し・帯）あり" if f["plate"] else "色板なし",
        "vivid": lambda: f"強調に使う色が {f['vivid']} 色",
        "bgm": lambda: (f"BGMが声の約 {f['bgm']:.0f} dB 下で鳴っている" if f["bgm"] and f["bgm"] > -45
                        else "BGMはほぼ聞こえない"),
        "sfx": lambda: f"効果音は間に1分あたり {f['sfx']:.1f} 回",
        "zoom_sw": lambda: f"カットで顔の大きさが変わるのは1分に {f['zoom_sw']:.1f} 回",
        "sat": lambda: f"彩度 {f['sat']:.2f}",
        "bright": lambda: f"画面の明るさ {f['bright']:.2f}", "warm colours": lambda: "強調に赤・金系の色",
        "small text": lambda: "文字が小さめ", "size": lambda: "字幕の大きさが標準的",
        "no outline": lambda: "フチなしの文字",
    }
    best = sorted((p for p in parts if p[2] >= 0.8 and p[1] in bits), key=lambda p: -p[0])[:4]
    return [bits[k]() for _, k, _ in best]


# ---------------------------------------------------------------- suggestions (edit.json "look")


def tone_defaults(tone: str, layout: str):
    """The tone's own CUT / CAMERA / AUDIO / STYLE (base look when the tone is not written yet)."""
    os.environ.setdefault("CV_ASPECT", "9:16" if layout == "tall" else "16:9")
    from cv import tones
    try:
        return tones.get(tone)
    except SystemExit:
        from cv.tones.base import Look
        return Look


def rnd(v: float, step: float) -> float:
    return round(round(v / step) * step, 3)


def suggest(m: dict, tone: str, layout: str) -> dict:
    cls = tone_defaults(tone, layout)
    look: dict = {}
    aud, cam, tel = m["audio"], m.get("camera") or {}, m["telops"]
    pauses = aud.get("pauses")
    cut: dict = {}
    if pauses and pauses["jet_cut"]:
        pause = rnd(pauses["p97_s"] + 0.02, 0.05)       # +-0.05 s is the accuracy of the measure
        if abs(pause - cls.CUT["pause"]) >= 0.10:
            cut["pause"] = pause
        if abs(pauses["peak_s"] - cls.CUT["tail"] - cls.CUT["lead"]) >= 0.08:
            cut.update(tail=rnd(0.58 * pauses["peak_s"], 0.01), lead=rnd(0.42 * pauses["peak_s"], 0.01))
    elif pauses and pauses["p90_s"] >= 0.5 and cls.CUT["pause"] > 0:
        cut["pause"] = 0.0
    cps = (m.get("speech") or {}).get("chars_per_sec")
    if cps and cps >= 8.6:
        cut["speed"] = 1.1
    if cut:
        look["cut"] = cut
    camera: dict = {}
    if m.get("camera"):         # face-size jumps at cuts include the speaker leaning: only clear cases become overrides
        sw, push = cam["zoom_switch_per_min"], cam["push_per_min"]
        if sw < 0.3:
            camera["med"] = 1.0
        elif sw >= 10:
            camera["hold"] = rnd(min(max(0.6 * 60 / sw, 3.0), 8.0), 0.5)
        if sw >= 3 and cam.get("med_ratio") and abs(cam["med_ratio"] - cls.CAMERA["med"]) >= 0.12 and cam["med_ratio"] < 1.3:
            camera["med"] = rnd(min(max(cam["med_ratio"], 1.05), 1.3), 0.01)
        if push >= 3.0:
            camera["punch_every"] = rnd(min(max(60 / push, 10.0), 90.0), 5)
        elif push == 0 and m["analysed_s"] >= 60 and cls.CAMERA["punch_every"] < 900:
            camera["punch_every"] = 999.0
    if camera:
        look["camera"] = camera
    audio: dict = {}
    music, sfx = aud.get("music"), aud.get("sfx") or {}
    if music:
        if music["present"] == "yes":
            db = rnd(min(max(music["under_voice_db"], -26.0), -6.0), 1.0)
            if abs(db - cls.AUDIO["music_db"]) >= 3.0:
                audio["music_db"] = db
        elif music["present"] == "no":
            audio["bgm"] = "none"
    if sfx.get("level") == "heavy" and cls.AUDIO["sfx_db"] < -1.5:
        audio["sfx_db"] = -1.0
    if audio:
        look["audio"] = audio
    style: dict = {}
    sub = tel.get("subtitle")
    if sub:
        sub_style: dict = {}
        base_y = cls.STYLE["sub"]["y"]
        base_y = base_y[layout] if isinstance(base_y, dict) else base_y
        if abs(sub["y"] - base_y) >= 0.03:
            sub_style["y"] = {layout: round(sub["y"], 2)}
        canvas_h = 1920 if layout == "tall" else 1080
        size = int(round(sub["line_height_pct"] / 100 * canvas_h / 1.12 / 2) * 2)
        base_size = cls.STYLE["sub"]["size"]
        base_size = base_size[layout] if isinstance(base_size, dict) else base_size
        if abs(size - base_size) >= 0.12 * base_size:
            sub_style["size"] = {layout: size}
        colors = tel.get("colors", [])
        if colors and colors[0]["name"] not in ("白", "黒", "グレー"):
            sub_style["fill"] = colors[0]["hex"]
        elif colors and colors[0]["name"] == "黒":
            sub_style["fill"] = "#101010"
        vivid = [c for c in colors if c["name"] not in ("白", "黒", "グレー") and c["share"] >= 0.08]
        pure = lambda c: (max(c["rgb"]) - min(c["rgb"])) / max(max(c["rgb"]), 1) * max(c["rgb"]) / 255
        best = max(vivid, key=pure, default=None)
        base_accent = np.array(cls.STYLE["accent"], float)
        if best and pure(best) >= 0.55 and np.linalg.norm(np.array(best["rgb"], float) - base_accent) > 60:
            style["accent"] = best["hex"]
        if sub_style:
            style["sub"] = sub_style
    if style:
        look["style"] = style
    lk = m["look"]
    if lk["saturation"] >= 0.46:
        look["grade"] = "eq=contrast=1.06:saturation=1.15"
    return {"tone": tone, "look": look,
            "note": "look goes into edit.json (project or version level); fonts are not measured: choose look.fonts by eye"}


# ---------------------------------------------------------------- main
