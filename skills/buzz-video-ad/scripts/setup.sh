#!/bin/bash
# One-time setup of this Mac (or Windows PC, in Git Bash) for buzz-video-ad. Safe to run again: it only adds what is missing.
#   bash setup.sh           install what is missing (Homebrew formulas, the ~550 MB Whisper model for the
#                           narration listen-check)
#   bash setup.sh --check   report only
# Needs macOS and Homebrew (https://brew.sh). Narration engines are optional (see the README).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]//\\//}")" && pwd)"   # backslashes: called from Windows Python
MODEL="$HOME/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin"
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
tool rubberband rubberband 0 "ナレーションの速度調整"
tool whisper-cpp whisper-cli 0 "ナレーションの聞き取り確認"

PY=""
VENV="${SNS_AD_VENV:-$HOME/.local/share/sns-ad-video/venv}"
if [ $CHECK = 1 ]; then  # report only: do not create the env
  if "$(venv_python "$VENV")" -c "import numpy, scipy, PIL" 2>/dev/null; then PY="$(venv_python "$VENV")"; ok "Python環境（${PY}）"
  else ng "Python環境がない（bash setup.sh で作る）"; fi
elif ! command -v uv >/dev/null; then ng "Python環境（uv が必要）"
elif PY=$(bash "$HERE/env.sh"); then ok "Python環境（${PY}）"
else PY=""; ng "Python環境を作れなかった（上のエラーを確認）"; fi

if [ -f "$MODEL" ]; then
  ok "Whisperモデル（${MODEL}）"
elif [ $CHECK = 0 ] && command -v whisper-cli >/dev/null; then
  echo "..   Whisperモデルをダウンロード（約550MB）"
  mkdir -p "$(dirname "$MODEL")"
  if curl -L --fail --retry 3 -o "$MODEL.part" "$MODEL_URL" &&
     [ "$(sha256_of "$MODEL.part")" = "$MODEL_SHA256" ]; then
    mv "$MODEL.part" "$MODEL" && ok "Whisperモデル（いま入れました）"
  else
    rm -f "$MODEL.part"; opt "Whisperモデルのダウンロードか照合に失敗（聞き取り確認なしでも動く）"
  fi
else
  opt "Whisperモデルなし（ナレーションの聞き取り確認をしない）"
fi

if [ -f "${SNS_HLG_LUT:-$HOME/.local/share/sns-ad-video/hlg2sdr_65.cube}" ]; then
  ok "HDR変換表（iPhoneのHDR素材用）"
elif [ $CHECK = 0 ] && [ -n "$PY" ] && "$PY" "$HERE/prep_hdr.py"; then
  ok "HDR変換表（いま作りました）"
else
  opt "HDR変換表なし（iPhoneのHDR素材を使う前に prep_hdr.py を実行）"
fi

if [ $WIN = 1 ] && [ $CHECK = 0 ]; then win_fonts || opt "M PLUS Rounded 1c を入れられなかった（メイリオで代用）"; fi
[ -n "$PY" ] && "$PY" "$HERE/fonts.py"

ENGINE=$(ls -d "$HOME"/.local/opt/aivisspeech_engine-*/run "$HOME"/.local/opt/voicevox_engine-*/run \
  "$HOME"/AppData/Local/Programs/AivisSpeech/AivisSpeech-Engine/run.exe "$HOME"/AppData/Local/Programs/VOICEVOX/vv-engine/run.exe \
  2>/dev/null | head -1)
if [ -n "$ENGINE" ]; then
  ok "ナレーションの音声エンジン（$(basename "$(dirname "$ENGINE")")）"
else
  opt "ナレーションの音声エンジンなし（Vlogテイストのナレーションを使うときだけ必要。README の「ナレーション」を参照）"
fi

if [ $FAILED = 0 ]; then echo "buzz-video-ad: 準備できました"; else echo "buzz-video-ad: NG の項目を入れてから、もう一度実行してください"; exit 1; fi
