#!/bin/bash
# Ensure the Python environment for sns-ad-video exists and print its python path.
#   PY=$(bash <skill>/scripts/env.sh)          -> python path
#   bash <skill>/scripts/env.sh --check         -> also report external tools (setup.sh installs them)
set -euo pipefail
VENV="${SNS_AD_VENV:-$HOME/.local/share/sns-ad-video/venv}"
PKGS="numpy scipy pillow"
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || { echo "uv が見つかりません（brew install uv）" >&2; exit 1; }
  mkdir -p "$(dirname "$VENV")"
  uv venv --python 3.12 "$VENV" >&2
fi
if ! "$VENV/bin/python" -c "import numpy, scipy, PIL" 2>/dev/null; then
  uv pip install --python "$VENV/bin/python" $PKGS >&2
fi
if [ "${1:-}" = "--check" ]; then
  for t in ffmpeg ffprobe; do command -v $t >/dev/null && echo "ok   $t" >&2 || echo "NG   $t（必須: brew install ffmpeg）" >&2; done
  command -v rubberband >/dev/null && echo "ok   rubberband（ナレーションの速度調整）" >&2 || echo "--   rubberband なし（ナレーション速度は TTS 側で調整）" >&2
  command -v whisper-cli >/dev/null && echo "ok   whisper-cli（ナレーションの聞き取り確認）" >&2 || echo "--   whisper-cli なし（聞き取り確認をスキップ）" >&2
  ls ~/.local/share/whisper/ggml-*.bin >/dev/null 2>&1 && echo "ok   whisper model: $(ls ~/.local/share/whisper/ggml-*.bin | head -1)" >&2 || true
  for e in ~/.local/opt/aivisspeech_engine-* ~/.local/opt/voicevox_engine-*; do [ -d "$e" ] && echo "ok   TTS engine: $e" >&2; done
fi
echo "$VENV/bin/python"
