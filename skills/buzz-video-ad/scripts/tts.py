#!/usr/bin/env python3
"""Narration for one version: synthesize every line, check it with Whisper, trim, tempo, normalize.

  python tts.py project.json --version vlog                # all lines
  python tts.py project.json --version vlog --line 3 9     # redo some lines (after fixing a reading)
  python tts.py --list-speakers --engine aivis             # voices available for the "narration voice" question

Version spec:
  "voice": {"engine": "aivis", "speaker": 497929760, "speed": 1.0, "tempo": 1.15, "pitch": 0.0,
            "intonation": 1.0, "takes": 3, "aliases": {"見本": "ミホン"}}
  aliases: how Whisper may spell a correctly spoken name (kanji homophones, romaji) -> expected spelling.
  "lines": [{"speech": "読み上げる文（難読語はかなで）", "sub": ["字幕の行"]}, ...]
Engines
  aivis       AivisSpeech, local (starts ~/.local/opt/aivisspeech_engine-*/run if needed, stops it afterwards).
              497929760 = morioki: calm, natural adult female; ACML 1.0 models: commercial use OK, credit optional.
  voicevox    VOICEVOX, local. Character voices; the post must credit "VOICEVOX:<character>".
  higgsfield  ElevenLabs v4 through the Higgsfield CLI (cloud, uses credits; `higgsfield auth login` first).
              speaker = preset voice name or id. Spell loanwords and numbers in kana.
  files       recordings supplied by the client: "dir": folder with line_01.(wav|m4a|mp3|aif) ... one per line,
              in line order; they are checked, trimmed and normalized like synthesized lines.
Writes work/<version>/voice/line_XX.wav (48 kHz mono, -16 LUFS) and timings.json for the template.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import media  # noqa: E402

LOCAL = {"aivis": ("~/.local/opt/aivisspeech_engine-*", 10101, 497929760),
         "voicevox": ("~/.local/opt/voicevox_engine-*", 50021, 8)}
# where the Windows installers put the engines (the app being open also works: its engine answers on the port)
LOCAL_WIN = {"aivis": "~/AppData/Local/Programs/AivisSpeech/AivisSpeech-Engine",
             "voicevox": "~/AppData/Local/Programs/VOICEVOX/vv-engine"}
WHISPER_MODELS = ("~/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin", "~/.local/share/whisper/ggml-small.bin")
PASS = 0.8  # transcript similarity needed to accept a take


# ---------------------------------------------------------------- engines
class LocalEngine:
    """AivisSpeech / VOICEVOX HTTP engine; started on demand and stopped again if we started it."""

    def __init__(self, name: str):
        pattern, self.port, self.default = LOCAL[name]
        self.name, self.url, self.proc = name, f"http://127.0.0.1:{self.port}", None
        dirs = sorted(glob.glob(os.path.expanduser(pattern)))
        if os.name == "nt":
            dirs += [d for d in [os.path.expanduser(LOCAL_WIN[name])] if os.path.isdir(d)]
        self.dir = dirs[-1] if dirs else None

    def alive(self) -> bool:
        try:
            urllib.request.urlopen(self.url + "/version", timeout=2).read()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not up yet"
            return False

    def __enter__(self) -> "LocalEngine":
        if self.alive():
            return self
        if not self.dir:
            raise SystemExit(f"{self.name} engine not found ({LOCAL[self.name][0]})")
        log = open(os.path.join(os.path.dirname(self.dir), f"{self.name}_engine.log"), "a")
        self.proc = subprocess.Popen([os.path.join(self.dir, "run.exe" if os.name == "nt" else "run"), "--host", "127.0.0.1", "--port", str(self.port)],
                                     cwd=self.dir, stdout=log, stderr=log)
        for _ in range(120):
            if self.alive():
                print(f"[tts] started {self.name} engine", file=sys.stderr)
                return self
            time.sleep(1)
        raise SystemExit(f"{self.name} engine did not start (see its log)")

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=20)
            print(f"[tts] stopped {self.name} engine", file=sys.stderr)

    def _post(self, path: str, data: bytes | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=120).read()

    def speakers(self) -> list[tuple[int, str]]:
        data = json.loads(urllib.request.urlopen(self.url + "/speakers", timeout=10).read())
        return [(s["id"], f"{sp['name']} / {s['name']}") for sp in data for s in sp["styles"]]

    def synth(self, text: str, voice: dict, out_wav: str, index: int = 0) -> None:
        spk = int(voice.get("speaker", self.default))
        q = json.loads(self._post(f"/audio_query?text={urllib.parse.quote(text)}&speaker={spk}"))
        q.update(speedScale=float(voice.get("speed", 1.0)), pitchScale=float(voice.get("pitch", 0.0)),
                 intonationScale=float(voice.get("intonation", 1.0)), prePhonemeLength=0.05, postPhonemeLength=0.12)
        with open(out_wav, "wb") as f:
            f.write(self._post(f"/synthesis?speaker={spk}", json.dumps(q).encode()))


class HiggsfieldEngine:
    """ElevenLabs v4 via the Higgsfield CLI: submit without --wait, poll, download (the CLI's --wait hangs)."""

    def __enter__(self) -> "HiggsfieldEngine":
        if not shutil.which("higgsfield"):
            raise SystemExit("higgsfield CLI not found (npm i -g @higgsfield/cli; higgsfield auth login)")
        return self

    def __exit__(self, *exc) -> None:
        pass

    def _cli(self, *args: str) -> dict | list:
        r = subprocess.run(["higgsfield", *args, "--json"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=180)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        return json.loads(r.stdout)

    def speakers(self) -> list[tuple[str, str]]:
        data = self._cli("voices", "list", "--size", "100")
        items = data.get("items", data) if isinstance(data, dict) else data
        return [(v["id"], f"{v['name']} ({v.get('gender', '?')}, {v.get('age', '?')})") for v in items]

    def synth(self, text: str, voice: dict, out_wav: str, index: int = 0) -> None:
        vid = str(voice.get("speaker", ""))
        if not vid:
            raise SystemExit("voice.speaker: set a Higgsfield preset voice name or id (tts.py --list-speakers)")
        if not re.fullmatch(r"[0-9a-f-]{20,}", vid):
            vid = next((i for i, n in self.speakers() if n.split(" (")[0] == vid), vid)
        dialogue = json.dumps([{"text": text, "voice_id": vid, "voice_type": "preset"}], ensure_ascii=False)
        for attempt in range(10):
            try:
                out = self._cli("generate", "create", "elevenlabs_v4", "--dialogue", dialogue)
                break
            except RuntimeError as e:
                if "rate_limit" not in str(e):
                    raise
                time.sleep(4 + 3 * attempt)
        job_id = out[0] if isinstance(out, list) else out["id"]
        for _ in range(150):
            job = self._cli("generate", "get", job_id)
            job = job[0] if isinstance(job, list) else job
            if job.get("status") == "completed" and job.get("result_url"):
                break
            if job.get("status") in ("failed", "canceled", "nsfw"):
                raise RuntimeError(f"higgsfield job {job_id}: {job.get('status')}")
            time.sleep(2)
        mp3 = out_wav + ".mp3"
        with urllib.request.urlopen(job["result_url"], timeout=60) as r, open(mp3, "wb") as f:
            f.write(r.read())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ar", str(media.SR), "-ac", "1", out_wav], check=True)
        os.remove(mp3)


class FilesEngine:
    """Pre-recorded lines (client's own voice): line_01.* ... in voice["dir"]."""

    def __enter__(self) -> "FilesEngine":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def synth(self, text: str, voice: dict, out_wav: str, index: int = 0) -> None:
        folder = os.path.expanduser(voice.get("dir", ""))
        hits = sorted(glob.glob(os.path.join(folder, f"line_{index:02d}.*")))
        if not hits:
            raise SystemExit(f"voice.dir: no line_{index:02d}.* in {folder or '(voice.dir not set)'}")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", hits[0], "-ar", str(media.SR), "-ac", "1", out_wav],
                       check=True)


def engine_for(name: str):
    if name == "files":
        return FilesEngine()
    if name in LOCAL:
        return LocalEngine(name)
    if name == "higgsfield":
        return HiggsfieldEngine()
    raise SystemExit(f"voice.engine must be aivis | voicevox | higgsfield | files, got {name!r}")


# ---------------------------------------------------------------- checking
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)  # katakana -> hiragana
    return re.sub(r"[\s、。，．,.!！?？…「」『』（）()・ー〜~]", "", s)


def transcribe(wav: str) -> str | None:
    model = next((os.path.expanduser(m) for m in WHISPER_MODELS if os.path.exists(os.path.expanduser(m))), None)
    if not model or not shutil.which("whisper-cli"):
        return None
    with tempfile.TemporaryDirectory() as d:
        w16 = os.path.join(d, "a.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-ar", "16000", "-ac", "1", w16], check=True)
        r = subprocess.run(["whisper-cli", "-m", model, "-l", "ja", "-nt", "-np", "-f", w16],
                           capture_output=True, text=True)
    return r.stdout.strip().replace(" ", "")


def score(transcript: str | None, *expected: str, aliases: dict | None = None) -> float:
    """Similarity of the transcript to the expected text. aliases rewrite known mis-hearings first
    (Whisper writes brand names as kanji homophones: {"見本": "ミホン"})."""
    if transcript is None:
        return 1.0  # no Whisper available: accept, but the report says unchecked
    for wrong, right in (aliases or {}).items():
        transcript = transcript.replace(wrong, right)
    return max(difflib.SequenceMatcher(None, _norm(transcript), _norm(e)).ratio() for e in expected if e)


# ---------------------------------------------------------------- post-processing
def finish_line(raw: str, out: str, tempo: float) -> float:
    """Trim silence, optional pitch-preserving tempo, -16 LUFS mono 48 kHz. Returns duration."""
    with tempfile.TemporaryDirectory() as d:
        trimmed = os.path.join(d, "trim.wav")
        trim = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", f"{trim},areverse,{trim},areverse",
                        "-ar", str(media.SR), "-ac", "1", trimmed], check=True)
        src = trimmed
        if abs(tempo - 1.0) > 1e-3:
            fast = os.path.join(d, "fast.wav")
            if shutil.which("rubberband"):
                subprocess.run(["rubberband", "--fine", "--tempo", f"{tempo}", "-q", trimmed, fast], check=True)
            else:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", trimmed, "-af", f"atempo={tempo}", fast], check=True)
            src = fast
        media.normalize(src, out, target=-16.0, limit=0.84)
    return media.duration(out)


def run(project_path: str, version: str, only: list[int] | None) -> None:
    from core import spec
    project = spec.load(project_path)
    v = project.version(version)
    lines, voice = v.get("lines"), v.get("voice", {})
    if not lines:
        raise SystemExit(f"versions.{version}.lines is empty")
    out_dir = os.path.join(project.work_dir, version, "voice")
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, "timings.json")
    old = {x["id"]: x for x in json.load(open(tpath, encoding="utf-8"))} if os.path.exists(tpath) else {}
    tempo, takes = float(voice.get("tempo", 1.0)), int(voice.get("takes", 3))
    if voice.get("engine") == "files":
        takes = 1  # a recording is what it is; a low score means: listen and ask for a retake
    rows = []
    with engine_for(voice.get("engine", "aivis")) as eng:
        for i, ln in enumerate(lines, 1):
            lid = f"{i:02d}"
            display = "".join(ln.get("sub", []))
            speech = ln.get("speech", display)
            if only and i not in only and lid in old:
                rows.append(old[lid])
                continue
            best = None
            for k in range(takes):
                raw = os.path.join(out_dir, f"raw_{lid}_{k}.wav")
                eng.synth(speech, voice, raw, index=i)
                text = transcribe(raw)
                sc = score(text, display, speech, aliases=voice.get("aliases"))
                print(f"line {lid} take {k}: {sc:.2f}  {text}", file=sys.stderr)
                if best is None or sc > best[0]:
                    best = (sc, raw, text)
                if sc >= PASS:
                    break
            dur = finish_line(best[1], os.path.join(out_dir, f"line_{lid}.wav"), tempo)
            for f in glob.glob(os.path.join(out_dir, f"raw_{lid}_*.wav")):
                os.remove(f)
            rows.append({"id": lid, "file": f"line_{lid}.wav", "display": display, "speech": speech,
                         "duration": round(dur, 4), "transcript": best[2], "score": round(best[0], 3),
                         "checked": best[2] is not None})
            if best[0] < PASS:
                print(f"!! line {lid}: best take {best[0]:.2f} heard as {best[2]!r}. If only a name is spelled "
                      f"differently, add it to voice.aliases; otherwise fix the reading in 'speech' (kana)", file=sys.stderr)
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    total = sum(r["duration"] for r in rows)
    print(json.dumps({"lines": len(rows), "speech_seconds": round(total, 2), "timings": tpath,
                      "low_scores": [r["id"] for r in rows if r["score"] < PASS]}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?")
    ap.add_argument("--version")
    ap.add_argument("--line", nargs="*", type=int, help="only (re)make these line numbers (1-based)")
    ap.add_argument("--list-speakers", action="store_true")
    ap.add_argument("--engine", default="aivis")
    a = ap.parse_args()
    if a.list_speakers:
        with engine_for(a.engine) as eng:
            for sid, name in eng.speakers():
                print(sid, name)
        return
    if not (a.project and a.version):
        raise SystemExit("usage: tts.py project.json --version NAME [--line N ...]")
    run(a.project, a.version, a.line)


if __name__ == "__main__":
    main()
