#!/bin/bash
# Ensure the Python environment for buzz-video-ad exists and print its python path (macOS, or Git Bash on Windows).
#   PY=$(bash <skill>/scripts/env.sh)          -> python path
#   bash <skill>/scripts/env.sh --check         -> also report external tools (setup.sh installs them)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]//\\//}")" && pwd)"   # backslashes: called from Windows Python
. "$HERE/platform.sh"
VENV="${SNS_AD_VENV:-$HOME/.local/share/sns-ad-video/venv}"
PKGS="numpy scipy pillow"
[ $WIN = 1 ] && PKGS="$PKGS pillow-heif"
if [ ! -x "$(venv_python "$VENV")" ]; then
  command -v uv >/dev/null || { echo "uv が見つかりません（$(install_hint uv)）" >&2; exit 1; }
  mkdir -p "$(dirname "$VENV")"
  uv venv --python 3.12 "$VENV" >&2
fi
if ! "$(venv_python "$VENV")" -c "import numpy, scipy, PIL" 2>/dev/null ||
   { [ $WIN = 1 ] && ! "$(venv_python "$VENV")" -c "import pillow_heif" 2>/dev/null; }; then
  uv pip install --python "$(venv_python "$VENV")" $PKGS >&2
fi
write_sitecustomize "$(venv_python "$VENV")"
if [ "${1:-}" = "--check" ]; then
  for t in ffmpeg ffprobe; do command -v $t >/dev/null && echo "ok   $t" >&2 || echo "NG   ${t}（必須: $(install_hint ffmpeg)）" >&2; done
  command -v rubberband >/dev/null && echo "ok   rubberband（ナレーションの速度調整）" >&2 || echo "--   rubberband なし（ナレーション速度は TTS 側で調整）" >&2
  command -v whisper-cli >/dev/null && echo "ok   whisper-cli（ナレーションの聞き取り確認）" >&2 || echo "--   whisper-cli なし（聞き取り確認をスキップ）" >&2
  ls ~/.local/share/whisper/ggml-*.bin >/dev/null 2>&1 && echo "ok   whisper model: $(ls ~/.local/share/whisper/ggml-*.bin | head -1)" >&2 || true
  for e in ~/.local/opt/aivisspeech_engine-* ~/.local/opt/voicevox_engine-* \
           ~/AppData/Local/Programs/AivisSpeech/AivisSpeech-Engine ~/AppData/Local/Programs/VOICEVOX/vv-engine; do
    [ -d "$e" ] && echo "ok   TTS engine: $e" >&2
  done
fi
python_for_shell "$VENV"
