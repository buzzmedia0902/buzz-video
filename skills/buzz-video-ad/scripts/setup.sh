#!/bin/bash
# One-time setup of this Mac for buzz-video-ad. Safe to run again: it only adds what is missing.
#   bash setup.sh           install what is missing (Homebrew formulas, the ~550 MB Whisper model for the
#                           narration listen-check)
#   bash setup.sh --check   report only
# Needs macOS and Homebrew (https://brew.sh). Narration engines are optional (see the README).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HOME/.local/share/whisper/ggml-large-v3-turbo-q5_0.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin"
MODEL_SHA256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
FAILED=0
ok()  { echo "ok   $*"; }
ng()  { echo "NG   $*"; FAILED=1; }
opt() { echo "--   $*"; }

[ "$(uname -s)" = "Darwin" ] || { echo "buzz-video-ad は macOS 専用です"; exit 1; }
command -v brew >/dev/null || for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  [ -x "$b" ] && eval "$("$b" shellenv)" && break  # Homebrew is often missing from PATH in non-login shells
done
command -v brew >/dev/null || echo "!!   Homebrew がありません。https://brew.sh の手順で入れてから、もう一度実行してください"

tool() {  # formula command required(1/0) purpose
  if command -v "$2" >/dev/null; then ok "$2（$4）"; return; fi
  if [ $CHECK = 0 ] && command -v brew >/dev/null && brew install "$1"; then ok "$2（$4。いま入れました）"; return; fi
  if [ "$3" = 1 ]; then ng "$2（$4）: brew install $1"; else opt "$2 なし（$4）: brew install $1"; fi
}
tool ffmpeg ffmpeg 1 "書き出し・音声"
tool uv uv 1 "Python環境"
tool rubberband rubberband 0 "ナレーションの速度調整"
tool whisper-cpp whisper-cli 0 "ナレーションの聞き取り確認"

PY=""
VENV="${SNS_AD_VENV:-$HOME/.local/share/sns-ad-video/venv}"
if [ $CHECK = 1 ]; then  # report only: do not create the env
  if "$VENV/bin/python" -c "import numpy, scipy, PIL" 2>/dev/null; then PY="$VENV/bin/python"; ok "Python環境（$PY）"
  else ng "Python環境がない（bash setup.sh で作る）"; fi
elif ! command -v uv >/dev/null; then ng "Python環境（uv が必要）"
elif PY=$(bash "$HERE/env.sh"); then ok "Python環境（$PY）"
else PY=""; ng "Python環境を作れなかった（上のエラーを確認）"; fi

if [ -f "$MODEL" ]; then
  ok "Whisperモデル（$MODEL）"
elif [ $CHECK = 0 ] && command -v whisper-cli >/dev/null; then
  echo "..   Whisperモデルをダウンロード（約550MB）"
  mkdir -p "$(dirname "$MODEL")"
  if curl -L --fail --retry 3 -o "$MODEL.part" "$MODEL_URL" &&
     [ "$(shasum -a 256 "$MODEL.part" | cut -d' ' -f1)" = "$MODEL_SHA256" ]; then
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

[ -n "$PY" ] && "$PY" "$HERE/fonts.py"

ENGINE=$(ls -d "$HOME"/.local/opt/aivisspeech_engine-*/run "$HOME"/.local/opt/voicevox_engine-*/run 2>/dev/null | head -1)
if [ -n "$ENGINE" ]; then
  ok "ナレーションの音声エンジン（$(basename "$(dirname "$ENGINE")")）"
else
  opt "ナレーションの音声エンジンなし（Vlogテイストのナレーションを使うときだけ必要。README の「ナレーション」を参照）"
fi

if [ $FAILED = 0 ]; then echo "buzz-video-ad: 準備できました"; else echo "buzz-video-ad: NG の項目を入れてから、もう一度実行してください"; exit 1; fi
