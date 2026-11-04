#!/bin/bash
# End-to-end smoke test of both skills on synthetic footage (macOS and Windows/Git Bash; CI runs it).
#   bash tests/smoke.sh [work_dir]
# Needs the setup done first (setup.sh / setup.ps1). Exits non-zero on the first failing step.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-${TMPDIR:-/tmp}/buzz-video-smoke}"
. "$ROOT/skills/buzz-video-content/scripts/platform.sh"   # the tools setup.sh installed (Homebrew / ~/.local/opt)
rm -rf "$W"; mkdir -p "$W/content/footage" "$W/ad/footage" "$W/ad/brand"
step() { echo; echo "=== $*"; }

step "footage"
curl -fsSL --retry 3 -o "$W/portrait.jpg" https://storage.googleapis.com/mediapipe-assets/portrait.jpg   # MediaPipe test asset (Apache-2.0)
# 12 s talking-head stand-in: the portrait, slowly zoomed, with a tone as the voice
ffmpeg -v error -y -loop 1 -i "$W/portrait.jpg" -f lavfi -i "sine=frequency=220:duration=12" \
  -vf "scale=1080:1350,pad=1080:1920:0:285,zoompan=z='1+0.0005*on':d=1:s=1080x1920:fps=30,format=yuv420p" \
  -t 12 -c:v libx264 -preset veryfast -c:a aac "$W/content/footage/talk.mp4"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=1080x1920:rate=30" -t 6 -c:v libx264 -preset veryfast -pix_fmt yuv420p "$W/ad/footage/shop1.mp4"
cp "$W/content/footage/talk.mp4" "$W/ad/footage/shop2.mp4"
ffmpeg -v error -y -f lavfi -i "color=c=0x221E1B:s=600x600,drawbox=x=150:y=150:w=300:h=300:color=0xD2BF5B:t=fill" -frames:v 1 "$W/ad/brand/logo.png"

step "content: env"
S="$ROOT/skills/buzz-video-content/scripts"
PY=$(bash "$S/env.sh" --check)
cd "$W/content"
cat > cues.json <<'J'
[{"src": "talk", "t0": 0.5, "t1": 3.0, "text": "AIで{動画編集}が変わる", "speaker": null},
 {"src": "talk", "t0": 3.4, "t1": 6.5, "text": "テロップも自動で入ります", "speaker": null},
 {"src": "talk", "t0": 7.0, "t1": 11.0, "text": "Windowsでも動くか確認", "speaker": null}]
J
cat > edit.json <<'J'
{"project": "smoke", "tone": "trust", "brand": {"accent": "#FEDB2E", "channel": "smoke test"},
 "sources": {"talk": "footage/talk.mp4"}, "cues": "cues.json",
 "versions": {"short": {"format": "short", "aspect": "9:16", "title": "【テスト】{AIの動画編集}が｜ここまで来た",
   "parts": [{"kind": "talk", "src": "talk", "from": 0.0, "to": 12.0}],
   "overlays": [{"type": "chapter", "src": "talk", "at": 3.4, "text": "見出しのテスト"},
                {"type": "card", "src": "talk", "at": 7.0, "dur": 4, "title": "3つのポイント", "items": ["速い", "安い", "うまい"]}]}}}
J
step "content: probe";  "$PY" "$S/probe.py" footage --out work/footage
step "content: faces";  "$PY" "$S/faces.py" edit.json
step "content: stills"; "$PY" "$S/render.py" edit.json --version short --stills 1.0 8.0
step "content: video";  "$PY" "$S/render.py" edit.json --version short --video
step "content: finish"; "$PY" "$S/finish.py" edit.json --version short
step "content: thumb";  "$PY" "$S/thumb.py" edit.json --version short --copy "テスト"
ls -R "$W/content/納品" 2>/dev/null || ls -R "$W/content"

step "ad: env"
S="$ROOT/skills/buzz-video-ad/scripts"
PY=$(bash "$S/env.sh" --check)
cd "$W/ad"
cat > project.json <<'J'
{"project": "SMOKE", "footage_dir": "footage", "out_dir": "納品", "work_dir": "work", "platform": "meta",
 "cta": "ご予約はリンクから",
 "brand": {"name": "SMOKE SALON", "logo": "brand/logo.png", "label": "SMOKE TEST", "tagline": "テストの動画です。",
           "colors": {"accent": "#D2BF5B", "ink": "#221E1B", "paper": "#F7F3EA", "night": "#1E1915"}},
 "clips": {},
 "versions": {"standard": {"taste": "standard", "duration": 10, "bgm": {"style": "piano"}, "shots": [
   {"clip": "shop2.mp4", "dur": 2.5, "in": 1.0, "zoom": [1.1, 1.2], "center": [0.5, 0.4],
    "caption": {"rows": [{"text": "顔に、テスト。", "size": 140, "font": "title"}], "label": "SMOKE", "pos": 0.3}},
   {"clip": "shop1.mp4", "dur": 2.5, "in": 0.5, "transition": "dissolve",
    "caption": {"rows": ["一本一本、", "ていねいに。"], "pos": "top"}}]}}}
J
step "ad: probe";    "$PY" "$S/probe.py" footage --out work/footage
step "ad: logo";     "$PY" "$S/logo_prep.py" brand/logo.png
step "ad: stills";   "$PY" "$S/render.py" project.json --version standard --aspect 9:16 --stills 1.0 4.0
step "ad: video";    "$PY" "$S/render.py" project.json --version standard --aspect 4:5 --video
step "ad: finish";   "$PY" "$S/finish.py" project.json --version standard --aspect 4:5
ls -R "$W/ad/納品"
step "loudness"
"$PY" "$ROOT/tests/check_outputs.py" "$W/content/納品" "$W/ad/納品"
echo; echo "SMOKE OK"
