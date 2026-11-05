# Sourced by env.sh and setup.sh: what differs between macOS and Windows (Git Bash), in one place.
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WIN=1 ;; *) WIN=0 ;; esac
# Windows: folders of the tools setup.sh unpacked under ~/.local/opt (one Windows path per line)
TOOL_PATHS="$HOME/.local/share/buzz-video/paths.txt"
if [ $WIN = 1 ]; then
  if [ -f "$TOOL_PATHS" ]; then
    while IFS= read -r d; do d=${d%$'\r'}; [ -n "$d" ] && PATH="$(cygpath -u "$d"):$PATH"; done < "$TOOL_PATHS"
  fi
else
  command -v brew >/dev/null || for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$b" ] && eval "$("$b" shellenv)" && break  # Homebrew is often missing from PATH in non-login shells (e.g. the desktop app)
  done
fi
export PATH

venv_python() { if [ $WIN = 1 ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi; }
native_path() { if [ $WIN = 1 ]; then cygpath -w "$1"; else echo "$1"; fi; }
# how to get a missing tool: brew on a Mac, the setup on Windows
install_hint() { if [ $WIN = 1 ]; then echo "「buzz-video のセットアップをして」で入ります"; else echo "brew install $1"; fi; }

# Let the venv's scripts find the tools (Homebrew, or the folders of paths.txt) even when the calling shell lacks them
write_sitecustomize() {
  local site
  site="$("$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' | tr -d '\r')"   # Windows prints CRLF
  [ $WIN = 1 ] && site="$(cygpath -u "$site")"
  cat > "$site/sitecustomize.py" <<'PYEOF'
import os
_p = os.environ.get("PATH", "").split(os.pathsep)
_extra = ["/usr/local/bin", "/opt/homebrew/bin"]
_f = os.path.expanduser("~/.local/share/buzz-video/paths.txt")
if os.path.exists(_f):
    _extra += [_l.strip() for _l in open(_f, encoding="utf-8") if _l.strip()]
for _d in _extra:
    if os.path.isdir(_d) and _d not in _p:
        _p.insert(0, _d)
os.environ["PATH"] = os.pathsep.join(_p)
PYEOF
}

# The python to print for "$PY" script.py: on Windows a launcher next to the venv that turns on UTF-8 mode
# (otherwise Python reads the Japanese JSON and file names as cp932)
python_for_shell() {
  local py launcher
  py="$(venv_python "$1")"
  if [ $WIN = 0 ]; then echo "$py"; return; fi
  launcher="$1/../$(basename "$1")-python"
  printf '#!/bin/bash\nexport PYTHONUTF8=1\nexec "%s" "$@"\n' "$py" > "$launcher"
  chmod +x "$launcher"
  echo "$(cd "$(dirname "$launcher")" && pwd)/$(basename "$launcher")"
}
