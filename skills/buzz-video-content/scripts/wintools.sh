# Sourced by setup.sh on Windows (Git Bash): unpack the tools Homebrew installs on a Mac under ~/.local/opt and list
# their folders in $TOOL_PATHS (platform.sh and the venvs' sitecustomize put those on PATH). Needs platform.sh first.
OPT="$HOME/.local/opt"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
YTDLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
WHISPER_URL="https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.2/whisper-bin-x64.zip"
WHISPER_SHA256="49dcc16de826f20bd53d44f947a1ae49dfa81f86cad67a64d80820cb192d674a"
RUBBERBAND_URL="https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
RUBBERBAND_SHA256="f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644"
FONTS_URL="https://raw.githubusercontent.com/google/fonts/main/ofl/mplusrounded1c"

add_tool_path() {
  local w
  w="$(cygpath -w "$1")"
  mkdir -p "$(dirname "$TOOL_PATHS")"
  touch "$TOOL_PATHS"
  grep -qxF "$w" "$TOOL_PATHS" || printf '%s\n' "$w" >> "$TOOL_PATHS"
  PATH="$1:$PATH"; export PATH
}

download() {  # url sha256-or-empty dest
  curl -L --fail --retry 3 -sS -o "$3.part" "$1" || { rm -f "$3.part"; return 1; }
  if [ -n "$2" ] && [ "$(sha256sum "$3.part" | cut -d' ' -f1)" != "$2" ]; then
    rm -f "$3.part"; echo "checksum mismatch: $1" >&2; return 1
  fi
  mv "$3.part" "$3"
}

unzip_to() {  # zip folder
  powershell.exe -NoProfile -NonInteractive -Command \
    "Expand-Archive -Force -LiteralPath '$(cygpath -w "$1")' -DestinationPath '$(cygpath -w "$2")'"
}

fetch_zip() {  # name url sha256-or-empty -> unpacked into $OPT/name (replaced)
  local z="$OPT/$1.zip"
  mkdir -p "$OPT"
  download "$2" "$3" "$z" && rm -rf "${OPT:?}/$1" && unzip_to "$z" "$OPT/$1" && rm -f "$z"
}

win_tool() {  # command -> install it; 0 when it runs afterwards
  case "$1" in
    ffmpeg|ffprobe) fetch_zip ffmpeg "$FFMPEG_URL" "" &&
                    add_tool_path "$(dirname "$(ls "$OPT"/ffmpeg/*/bin/ffmpeg.exe | head -1)")" ;;
    uv) fetch_zip uv "$UV_URL" "" && add_tool_path "$OPT/uv" ;;
    whisper-cli) fetch_zip whisper "$WHISPER_URL" "$WHISPER_SHA256" && add_tool_path "$OPT/whisper/Release" ;;
    rubberband) fetch_zip rubberband "$RUBBERBAND_URL" "$RUBBERBAND_SHA256" &&
                add_tool_path "$(dirname "$(ls "$OPT"/rubberband/*/rubberband.exe | head -1)")" ;;
    yt-dlp) mkdir -p "$OPT/yt-dlp" && download "$YTDLP_URL" "" "$OPT/yt-dlp/yt-dlp.exe" && add_tool_path "$OPT/yt-dlp" ;;
    *) return 1 ;;
  esac && command -v "$1" >/dev/null
}

# M PLUS Rounded 1c into the user's fonts (Windows has neither Hiragino Maru nor the Font Book rounded faces)
win_fonts() {
  local dir w f reg="HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Fonts"
  dir="$(cygpath -u "$LOCALAPPDATA")/Microsoft/Windows/Fonts"
  mkdir -p "$dir"
  for w in Regular Medium Bold ExtraBold Black; do
    f="MPLUSRounded1c-$w.ttf"
    [ -f "$dir/$f" ] && continue
    download "$FONTS_URL/$f" "" "$dir/$f" || return 1
    MSYS_NO_PATHCONV=1 reg add "$reg" /v "M PLUS Rounded 1c $w (TrueType)" /t REG_SZ /d "$(cygpath -w "$dir/$f")" /f \
      >/dev/null || return 1
  done
}
