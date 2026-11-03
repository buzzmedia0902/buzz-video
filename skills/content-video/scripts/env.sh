#!/bin/bash
# Ensure the Python environment for content-video exists and print its python path.
#   PY=$(bash <skill>/scripts/env.sh)            -> python path
#   bash <skill>/scripts/env.sh --check           -> also report external tools (setup.sh installs them)
#   SPK=$(bash <skill>/scripts/env.sh --speakers) -> python of the speaker-ID env (torch, ~2 GB)
set -euo pipefail
command -v brew >/dev/null || for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  [ -x "$b" ] && eval "$("$b" shellenv)" && break  # Homebrew is often missing from PATH in non-login shells (e.g. the desktop app)
done
ROOT="$HOME/.local/share/content-video"
VENV="${CONTENT_VIDEO_VENV:-$ROOT/venv}"
PKGS="numpy scipy pillow skia-python==144.0.post2 janome"
need_uv() { command -v uv >/dev/null || { echo "uv が見つかりません（brew install uv）" >&2; exit 1; }; }
if [ "${1:-}" = "--speakers" ]; then
  SV="$ROOT/venv_spk"
  if [ ! -x "$SV/bin/python" ]; then need_uv; uv venv --python 3.11 "$SV" >&2; fi
  if ! "$SV/bin/python" -c "import speechbrain, torch, torchaudio" 2>/dev/null; then
    uv pip install --python "$SV/bin/python" numpy scipy "speechbrain==1.1.2" torch torchaudio >&2
  fi
  echo "$SV/bin/python"
  exit 0
fi
if [ ! -x "$VENV/bin/python" ]; then
  need_uv
  mkdir -p "$ROOT"
  uv venv --python 3.12 "$VENV" >&2
fi
if ! "$VENV/bin/python" -c "import numpy, scipy, PIL, skia, janome" 2>/dev/null; then
  need_uv
  uv pip install --python "$VENV/bin/python" $PKGS >&2
fi
# Let scripts find Homebrew tools (ffmpeg, whisper-cli, ...) even when the calling shell lacks them on PATH
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sitecustomize.py"
[ -f "$SITE" ] || cat > "$SITE" <<'PYEOF'
import os
_p = os.environ.get("PATH", "").split(os.pathsep)
for _d in ("/usr/local/bin", "/opt/homebrew/bin"):
    if os.path.isdir(_d) and _d not in _p:
        _p.insert(0, _d)
os.environ["PATH"] = os.pathsep.join(_p)
PYEOF
if [ "${1:-}" = "--check" ]; then
  for t in ffmpeg ffprobe; do command -v $t >/dev/null && echo "ok   $t" >&2 || echo "NG   $t（必須: brew install ffmpeg）" >&2; done
  command -v whisper-cli >/dev/null && echo "ok   whisper-cli（文字起こし）" >&2 || echo "NG   whisper-cli（必須: brew install whisper-cpp）" >&2
  ls ~/.local/share/whisper/ggml-*.bin >/dev/null 2>&1 && echo "ok   whisper model: $(ls ~/.local/share/whisper/ggml-*.bin | head -1)" >&2 || echo "NG   whisper のモデルがない（~/.local/share/whisper/ に ggml-large-v3-turbo-q5_0.bin）" >&2
  command -v rubberband >/dev/null && echo "ok   rubberband（1.1倍速の声）" >&2 || echo "--   rubberband なし（速度は1.0倍のみ）" >&2
  command -v yt-dlp >/dev/null && echo "ok   yt-dlp（参考動画・切り抜き元の取得）" >&2 || echo "--   yt-dlp なし（URLからの取得は不可）" >&2
  command -v swiftc >/dev/null && echo "ok   swiftc（顔の位置・サムネの切り抜き）" >&2 || echo "--   swiftc なし（顔検出なしで中央固定）" >&2
fi
echo "$VENV/bin/python"
