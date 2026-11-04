#!/usr/bin/env python3
"""Finish one version: voice cut like the picture, BGM, SFX, mix at -14 LUFS, mux, light copy, subtitles, chapters.

  python finish.py edit.json --version V [--no-light] [--master FILE] [--seed N]

Needs the silent master (render.py --video, full size or --half) and work/<V>/cues_out.json (every render.py call
writes it). The BGM style, the music level under the voice, the SFX set and an optional exact tempo come from
cues_out.json "audio" (tone defaults + the version's "bgm" overrides); "bgm": "none" means no music.
Outputs in out_dir (<project>_<format>_<tone>_<9x16>_<length> = stem):
  <stem>.mp4            master picture (stream copy) + AAC 256k / 48 kHz, -14 LUFS, true peak <= -1 dBTP measured on the AAC
  <stem>_軽量版.mp4      phone / chat copy (short, clip, vlog < 30 MB; youtube 720p about 2.5 Mbps)
  <stem>_字幕.srt        subtitles of the version
  <stem>_概要欄.txt      youtube only: chapter list for the description (checked against YouTube's rules)
  <stem>_カバー.png      short, clip, vlog: frame at 1 s
  work/<V>/report_<tag>.json   every number of the run (audio levels, limiter, QA, warnings)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BT709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
LIGHT_MAX_MB = 28.0
TP_LIMIT = -1.02             # dBTP the encoded AAC must stay under (spec: -1.0, a hair of margin for other meters)
FIRST_CEILING = -1.5         # limiter ceiling of the first try; AAC usually adds nothing at this headroom
INTRO_LABEL = "オープニング"


def log(msg: str) -> None:
    print(f"[finish] {msg}", file=sys.stderr, flush=True)


def length_ja(seconds: float) -> str:
    """58秒, 1分42秒, 9分24秒, 10分, 1時間2分3秒."""
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    if h:
        return f"{h}時間{m}分{sec}秒" if sec else f"{h}時間{m}分"
    if m:
        return f"{m}分{sec}秒" if sec else f"{m}分"
    return f"{sec}秒"


def find_master(work_render: str, version: str, tag: str, explicit: str | None) -> str:
    """The picture to finish: --master, else the newest of <v>_<tag>.mp4 / <v>_<tag>_half.mp4."""
    if explicit:
        return explicit
    cands = [p for p in (os.path.join(work_render, f"{version}_{tag}.mp4"),
                         os.path.join(work_render, f"{version}_{tag}_half.mp4")) if os.path.exists(p)]
    if not cands:
        raise SystemExit(f"missing work/render/{version}_{tag}.mp4: run render.py --version {version} --video first")
    return max(cands, key=os.path.getmtime)


def unique_stem(stem: str, work_dir: str, version: str) -> str:
    """Two versions with the same format, tone, ratio and length would overwrite each other: the later one gets
    `_<version>` appended."""
    for p in glob.glob(os.path.join(work_dir, "*", "report_*.json")):
        try:
            rep = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if os.path.splitext(rep.get("outputs", {}).get("video", ""))[0] == stem and rep.get("version") != version:
            return f"{stem}_{version}"
    return stem


# =================================================================== audio
def make_tracks(cues: dict, work_dir: str, seed: int, warns: list[str]):
    """Voice, music and SFX tracks of the version (music / sfx are None when the tone or version turns them off)."""
    from cv.audio import music, sfx, voice
    audio = cues.get("audio", {})
    vo = voice.build(cues, work_dir)
    log(f"voice: {vo.info['dialogue_shots']} dialogue / {vo.info['ambience_shots']} ambience shots, "
        f"{vo.info['stretched']} stretched, noise {vo.info['noise']}")
    mus, minfo, style = None, {}, audio.get("bgm") or "none"
    if style != "none":
        if style not in music.STYLES:
            raise SystemExit(f"bgm style {style!r} is not one of {', '.join(music.STYLES)} (or none)")
        mus, minfo = music.compose(style, cues["duration"], cues.get("sections", []), bpm=audio.get("bpm"), seed=seed,
                                   mood=audio.get("mood"))
        log(f"music: {style} {minfo['bpm']:.1f} BPM, key {minfo['key']}, {len(minfo['beats'])} beats")
    sf = None
    if cues.get("sfx") and (audio.get("sfx") or "none") != "none":
        if audio["sfx"] not in sfx.SETS:
            raise SystemExit(f"sfx set {audio['sfx']!r} is not one of {', '.join(sfx.SETS)}")
        if sfx.unknown_names(cues["sfx"]):
            warns.append(f"sfx names skipped (unknown): {', '.join(sfx.unknown_names(cues['sfx']))}")
        sf = sfx.render(audio["sfx"], cues["sfx"], cues["duration"], float(audio.get("sfx_db", 0.0)), seed)
        log(f"sfx: {len(cues['sfx'])} events ({audio['sfx']})")
    return vo, mus, minfo, sf


def encode_aac(wav: str, out: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", out],
                   check=True)


def master_and_encode(pre, work_dir: str) -> tuple[str, dict, dict, list[dict]]:
    """Normalise + limit, encode, measure the AAC; tighten the limiter until the encoded true peak is <= -1 dBTP
    (AAC can overshoot the PCM peak). Returns (m4a path, ffmpeg measurement, mix info, attempts)."""
    from cv.audio import dsp, mix
    wav, aac = os.path.join(work_dir, "mix.wav"), os.path.join(work_dir, "mix.m4a")
    ceiling, attempts = FIRST_CEILING, []
    for _ in range(5):
        out, info = mix.master(pre, ceiling)
        dsp.write_wav(wav, out)
        encode_aac(wav, aac)
        meas = dsp.measure_file(aac)
        attempts.append({"ceiling": round(ceiling, 2), "lufs": meas["I"], "true_peak": meas["TP"],
                         "limiter_gr_db": info["limiter_gr_db"]})
        if meas["TP"] <= TP_LIMIT:
            break
        ceiling -= (meas["TP"] - TP_LIMIT) + 0.1
    return aac, meas, info, attempts


# =================================================================== files
def mux(master: str, aac: str, out: str) -> None:
    """Master picture + the AAC exactly as measured (both copied, so what was measured is what ships)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", master, "-i", aac, "-map", "0:v:0", "-map", "1:a:0", "-c:v",
                    "copy", "-c:a", "copy", "-movflags", "+faststart", "-shortest", out], check=True)


def light_copy(final: str, out: str, fmt: str, dur: float, passlog: str) -> dict:
    """Two-pass x264: youtube 720p at about 2.5 Mbps; short / clip / vlog sized to LIGHT_MAX_MB (720 px short side when
    the budget is thin)."""
    import qa
    v = next(s for s in qa.probe_streams(final)["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    if fmt == "youtube":
        kbps, short = 2500, 720
    else:
        kbps = int(min(5500, LIGHT_MAX_MB * 8e3 / dur - 140))
        short = 720 if kbps < 2800 else min(w, h)
    scale = ""
    if min(w, h) > short:
        scale = f"scale=-2:{short}," if w >= h else f"scale={short}:-2,"
    preset = "slow" if dur <= 300 else "medium"                     # long videos: about twice as fast for a little less quality
    common = ["-vf", scale + BT709, "-c:v", "libx264", "-preset", preset, "-b:v", f"{kbps}k", "-maxrate",
              f"{int(kbps * 1.45)}k", "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, *common, "-pass", "1", "-passlogfile", passlog, "-an",
                    "-f", "null", "-"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final, *common, "-pass", "2", "-passlogfile", passlog,
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out], check=True)
    for f in os.listdir(os.path.dirname(passlog)):
        if f.startswith(os.path.basename(passlog)):
            os.remove(os.path.join(os.path.dirname(passlog), f))
    return {"kbps": kbps, "size_mb": round(os.path.getsize(out) / 1e6, 1)}


def ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def chapter_list(cues: dict) -> tuple[list[str], list[str]]:
    """Description-box chapter lines and warnings. YouTube needs the first chapter at 0:00, at least three
    chapters, each at least 10 s; the video's opening gets a label when the author's first chapter comes later."""
    ch = [(float(t), str(label)) for t, label in cues.get("chapters", [])]
    if not ch or ch[0][0] > 1.0:
        ch.insert(0, (0.0, INTRO_LABEL))
    ch[0] = (0.0, ch[0][1])
    warns = []
    if len(ch) < 3:
        warns.append(f"chapters: {len(ch)} (YouTube needs at least 3 to show them; add chapter overlays)")
    ends = [t for t, _ in ch[1:]] + [cues["duration"]]
    for (t, label), end in zip(ch, ends):
        if end - t < 10.0:
            warns.append(f"chapter {ts(t)} 「{label}」 is {end - t:.1f} s long (YouTube needs at least 10 s)")
    return [f"{ts(t)} {label}" for t, label in ch], warns


def write_deliverables(cues: dict, master: str, aac: str, stem: str, no_light: bool, wd: str, warns: list[str]) -> dict:
    """Video, light copy, subtitles and chapter list / cover; returns {kind: path}."""
    from cv import cues as C
    fmt = cues["format"]
    out = {"video": stem + ".mp4"}
    mux(master, aac, out["video"])
    if not no_light:
        out["light"] = stem + "_軽量版.mp4"
        light = light_copy(out["video"], out["light"], fmt, cues["duration"], os.path.join(wd, "2pass"))
        if fmt != "youtube" and light["size_mb"] >= 30:
            warns.append(f"light copy is {light['size_mb']} MB (limit 30 MB)")
    out["srt"] = stem + "_字幕.srt"
    C.to_srt([C.Sub(**s) for s in cues.get("subs", [])], out["srt"])
    if fmt == "youtube":
        lines, cw = chapter_list(cues)
        out["description"] = stem + "_概要欄.txt"
        with open(out["description"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        warns += cw
    else:
        out["cover"] = stem + "_カバー.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.0", "-i", out["video"], "-frames:v", "1", out["cover"]],
                       check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--version", required=True)
    ap.add_argument("--no-light", action="store_true", help="skip the light copy")
    ap.add_argument("--master", help="picture file (default: newest work/render/<v>_<tag>[_half].mp4)")
    ap.add_argument("--seed", type=int, help="BGM seed (default: stable per project + version)")
    a = ap.parse_args()
    t_start = time.time()
    from cv import media, spec
    from cv.audio import mix
    import qa

    project = spec.load(a.project)
    project.version(a.version)                       # raises a readable SpecError for an unknown version
    cues_path = project.work(a.version, "cues_out.json")
    if not os.path.exists(cues_path):
        raise SystemExit(f"missing {cues_path}: run render.py --version {a.version} --info first")
    cues = json.load(open(cues_path, encoding="utf-8"))
    tag, wd = cues["aspect"].replace(":", "x"), os.path.dirname(cues_path)
    master = find_master(os.path.join(project.work_dir, "render"), a.version, tag, a.master)
    if abs(media.duration(master) - cues["duration"]) > 0.1:
        raise SystemExit(f"{master} is {media.duration(master):.2f} s but cues_out.json says {cues['duration']:.2f} s: "
                         f"re-render the master after changing edit.json")
    audio = cues.get("audio", {})
    seed = a.seed if a.seed is not None else int(audio.get("seed", zlib.crc32(f"{project.name}/{a.version}".encode()) % 100000))
    log(f"{a.version}: {cues['duration']:.1f} s, picture {os.path.basename(master)}, bgm {audio.get('bgm')}, "
        f"sfx {audio.get('sfx')}, seed {seed}")

    warns: list[str] = []
    vo, mus, minfo, sf = make_tracks(cues, project.work_dir, seed, warns)
    mixed = mix.build(cues, vo.dialogue, vo.ambience, mus, sf)
    aac, meas, minf, attempts = master_and_encode(mixed.pre, wd)
    log(f"audio: {meas['I']:.2f} LUFS, {meas['TP']:.2f} dBTP on the AAC (limiter {minf['limiter_gr_db']:.2f} dB, "
        f"{len(attempts)} encode(s))")
    if meas["TP"] > -1.0 or abs(meas["I"] + 14.0) > 0.5:
        warns.append(f"audio off target: {meas['I']:.2f} LUFS, {meas['TP']:.2f} dBTP (wanted -14 +-0.5 LUFS, <= -1 dBTP)")
    if mixed.info.get("voice_peak_cap_db", 0.0) > 6.0:
        warns.append(f"one loud peak held the voice {mixed.info['voice_peak_cap_db']:.1f} dB below its level (a shout "
                     f"in otherwise quiet speech): listen to it; lower that clip or keep its audio ducked")
    if float(minf.get("limiter_gr_db", 0.0)) < -6.0:
        warns.append(f"limiter took {minf['limiter_gr_db']:.1f} dB off the loudest moment: check for distortion")

    os.makedirs(project.out_dir, exist_ok=True)
    stem = os.path.join(project.out_dir, f"{project.name.replace('/', '_')}_{spec.FORMAT_JA[cues['format']]}_"
                                         f"{spec.TONE_JA[cues['tone']]}_{tag}_{length_ja(cues['duration'])}")
    outputs = write_deliverables(cues, master, aac, unique_stem(stem, project.work_dir, a.version), a.no_light, wd, warns)
    for w in warns:
        log(f"warning: {w}")
    rep = qa.report(outputs["video"], cues, loud=(meas["I"], meas["TP"]), project=a.project, version=a.version)
    if rep.get("judder_frames"):
        warns.append(f"{rep['judder_frames']} repeated frames inside shots (a 24 / 25 fps or Auto-FPS clip in a "
                     f"{cues['fps']:.2f} fps edit?): add \"smooth\": true to those parts and re-render")
    rep.update(version=a.version, outputs=outputs, warnings=warns, seed=seed, voice=vo.info,
               bgm={k: v for k, v in minfo.items() if k != "beats"}, mix={**mixed.info, **minf, "attempts": attempts},
               seconds=round(time.time() - t_start, 1))
    with open(os.path.join(wd, f"report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("bgm", "voice", "mix")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
