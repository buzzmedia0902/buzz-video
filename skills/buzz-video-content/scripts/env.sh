#!/bin/bash
# Ensure the Python environment for buzz-video-content exists and print its python path (macOS, or Git Bash on Windows).
#   PY=$(bash <skill>/scripts/env.sh)            -> python path
#   bash <skill>/scripts/env.sh --check           -> also report external tools (setup.sh installs them)
#   SPK=$(bash <skill>/scripts/env.sh --speakers) -> python of the speaker-ID env (torch, ~2 GB)
#   VIS=$(bash <skill>/scripts/env.sh --vision)   -> python of the MediaPipe/RapidOCR env (used where Apple Vision is
#                                                    missing; a native path, for subprocess calls from Python)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/platform.sh"
ROOT="$HOME/.local/share/content-video"
VENV="${CONTENT_VIDEO_VENV:-$ROOT/venv}"
PKGS="numpy scipy pillow skia-python==144.0.post2 janome"
[ $WIN = 1 ] && PKGS="$PKGS pillow-heif"
need_uv() { command -v uv >/dev/null || { echo "uv が見つかりません（$(install_hint uv)）" >&2; exit 1; }; }
if [ "${1:-}" = "--speakers" ]; then
  SV="$ROOT/venv_spk"
  if [ ! -x "$(venv_python "$SV")" ]; then need_uv; uv venv --python 3.11 "$SV" >&2; fi
  if ! "$(venv_python "$SV")" -c "import speechbrain, torch, torchaudio" 2>/dev/null; then
    uv pip install --python "$(venv_python "$SV")" numpy scipy "speechbrain==1.1.2" torch torchaudio >&2
  fi
  python_for_shell "$SV"
  exit 0
fi
if [ "${1:-}" = "--vision" ]; then
  VV="$ROOT/venv_vision"
  if [ ! -x "$(venv_python "$VV")" ]; then need_uv; uv venv --python 3.12 "$VV" >&2; fi
  if ! "$(venv_python "$VV")" -c "import mediapipe, rapidocr_onnxruntime, PIL" 2>/dev/null; then
    uv pip install --python "$(venv_python "$VV")" mediapipe rapidocr_onnxruntime pillow >&2
  fi
  native_path "$(venv_python "$VV")"
  exit 0
fi
if [ ! -x "$(venv_python "$VENV")" ]; then
  need_uv
  mkdir -p "$ROOT"
  uv venv --python 3.12 "$VENV" >&2
fi
if ! "$(venv_python "$VENV")" -c "import numpy, scipy, PIL, skia, janome" 2>/dev/null ||
   { [ $WIN = 1 ] && ! "$(venv_python "$VENV")" -c "import pillow_heif" 2>/dev/null; }; then
  need_uv
  uv pip install --python "$(venv_python "$VENV")" $PKGS >&2
fi
write_sitecustomize "$(venv_python "$VENV")"
if [ "${1:-}" = "--check" ]; then
  for t in ffmpeg ffprobe; do command -v $t >/dev/null && echo "ok   $t" >&2 || echo "NG   $t（必須: $(install_hint ffmpeg)）" >&2; done
  command -v whisper-cli >/dev/null && echo "ok   whisper-cli（文字起こし）" >&2 || echo "NG   whisper-cli（必須: $(install_hint whisper-cpp)）" >&2
  ls ~/.local/share/whisper/ggml-*.bin >/dev/null 2>&1 && echo "ok   whisper model: $(ls ~/.local/share/whisper/ggml-*.bin | head -1)" >&2 || echo "NG   whisper のモデルがない（~/.local/share/whisper/ に ggml-large-v3-turbo-q5_0.bin）" >&2
  command -v rubberband >/dev/null && echo "ok   rubberband（1.1倍速の声）" >&2 || echo "--   rubberband なし（速度は1.0倍のみ）" >&2
  command -v yt-dlp >/dev/null && echo "ok   yt-dlp（参考動画・切り抜き元の取得）" >&2 || echo "--   yt-dlp なし（URLからの取得は不可）" >&2
  command -v swiftc >/dev/null && echo "ok   swiftc（顔の位置・サムネの切り抜き）" >&2 || echo "--   swiftc なし（顔検出は MediaPipe 版を使います）" >&2
fi
python_for_shell "$VENV"
