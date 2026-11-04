#!/bin/bash
# One-time setup of this Mac (or Windows PC, in Git Bash) for buzz-video-content. Safe to run again: it only adds what is missing.
#   bash setup.sh           install what is missing (Homebrew formulas, free fonts, the ~550 MB Whisper model)
#   bash setup.sh --check   report only
# Needs macOS (Apple Vision, VideoToolbox) and Homebrew (https://brew.sh).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HOME/.local/share/content-video"
MODEL="${CV_WHISPER_MODEL:-$HOME/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin}"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin"
MODEL_SHA256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
FAILED=0
ok()  { echo "ok   $*"; }
ng()  { echo "NG   $*"; FAILED=1; }
opt() { echo "--   $*"; }

. "$HERE/platform.sh"
if [ $WIN = 1 ]; then
  . "$HERE/wintools.sh"
else
  command -v brew >/dev/null || echo "!!   Homebrew がありません。https://brew.sh の手順で入れてから、もう一度実行してください"
fi
sha256_of() { if command -v shasum >/dev/null; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }

tool() {  # formula command required(1/0) purpose
  if command -v "$2" >/dev/null; then ok "$2（$4）"; return; fi
  if [ $CHECK = 0 ]; then
    if [ $WIN = 1 ]; then win_tool "$2" && { ok "$2（$4。いま入れました）"; return; }
    elif command -v brew >/dev/null && brew install "$1"; then ok "$2（$4。いま入れました）"; return; fi
  fi
  local how="brew install $1"; [ $WIN = 1 ] && how="bash setup.sh で入れる"
  if [ "$3" = 1 ]; then ng "$2（$4）: $how"; else opt "$2 なし（$4）: $how"; fi
}
tool ffmpeg ffmpeg 1 "書き出し・音声"
tool uv uv 1 "Python環境"
tool whisper-cpp whisper-cli 1 "文字起こし"
tool rubberband rubberband 0 "1.1倍速の声"
tool yt-dlp yt-dlp 0 "参考動画・切り抜き元をURLから取得"

PY=""
VENV="${CONTENT_VIDEO_VENV:-$ROOT/venv}"
if [ $CHECK = 1 ]; then  # report only: do not create the env
  if "$(venv_python "$VENV")" -c "\1" 2>/dev/null; then PY="$(venv_python "$VENV")"; ok "Python環境（$PY）"
  else ng "Python環境がない（bash setup.sh で作る）"; fi
elif ! command -v uv >/dev/null; then ng "Python環境（uv が必要）"
elif PY=$(bash "$HERE/env.sh"); then ok "Python環境（$PY）"
else PY=""; ng "Python環境を作れなかった（上のエラーを確認）"; fi

if [ -f "$MODEL" ]; then
  ok "Whisperモデル（$MODEL）"
elif [ $CHECK = 0 ]; then
  echo "..   Whisperモデルをダウンロード（約550MB）"
  mkdir -p "$(dirname "$MODEL")"
  if curl -L --fail --retry 3 -o "$MODEL.part" "$MODEL_URL" &&
     [ "$(sha256_of "$MODEL.part")" = "$MODEL_SHA256" ]; then
    mv "$MODEL.part" "$MODEL" && ok "Whisperモデル（いま入れました）"
  else
    rm -f "$MODEL.part"; ng "Whisperモデルのダウンロードか照合に失敗（もう一度実行してください）"
  fi
else
  ng "Whisperモデルがない（$MODEL）"
fi

if [ $WIN = 1 ]; then  # no Apple Vision: the MediaPipe / RapidOCR helpers (tools/vision.py) in their own env
  if [ $CHECK = 0 ] && bash "$HERE/env.sh" --vision >/dev/null; then ok "顔検出・サムネの切り抜き（MediaPipe 版）"
  elif [ -x "$(venv_python "$ROOT/venv_vision")" ]; then ok "顔検出・サムネの切り抜き（MediaPipe 版）"
  else opt "顔検出の環境なし（初めて使うときに自動で作られます）"; fi
elif command -v swiftc >/dev/null && xcode-select -p >/dev/null 2>&1; then  # /usr/bin/swiftc exists even without the tools
  [ $CHECK = 0 ] && mkdir -p "$ROOT/bin"
  for src in "$HERE"/tools/*.swift; do
    out="$ROOT/bin/$(basename "$src" .swift)"
    if [ $CHECK = 0 ] && { [ ! -x "$out" ] || [ "$src" -nt "$out" ]; }; then
      swiftc -O "$src" -o "$out" || ng "$(basename "$src") をコンパイルできない"
    fi
  done
  ok "Apple Vision の補助ツール（顔の位置・サムネの切り抜き。足りなければ初回に自動でコンパイル）"
else
  opt "swiftc なし（xcode-select --install で入れると、顔を避けた配置とサムネの切り抜きが使える）"
fi

if [ -f "$ROOT/hlg2sdr_65.cube" ] || [ -f "$HOME/.local/share/sns-ad-video/hlg2sdr_65.cube" ]; then
  ok "HDR変換表（iPhoneのHDR素材用）"
elif [ $CHECK = 0 ] && [ -n "$PY" ] && "$PY" "$HERE/prep_hdr.py"; then
  ok "HDR変換表（いま作りました）"
else
  opt "HDR変換表なし（iPhoneのHDR素材を使う前に prep_hdr.py を実行）"
fi

if [ $WIN = 1 ] && [ $CHECK = 0 ]; then win_fonts || opt "M PLUS Rounded 1c を入れられなかった（游ゴシックで代用）"; fi
if [ -n "$PY" ]; then
  if [ $CHECK = 0 ] && [ $WIN = 0 ] && command -v brew >/dev/null; then "$PY" "$HERE/fonts.py" --install; else "$PY" "$HERE/fonts.py"; fi
fi
opt "話者の判定（speakers.py）を初めて使うときは、別のPython環境（約2GB）が自動で作られます"

if [ $FAILED = 0 ]; then echo "buzz-video-content: 準備できました"; else echo "buzz-video-content: NG の項目を入れてから、もう一度実行してください"; exit 1; fi
