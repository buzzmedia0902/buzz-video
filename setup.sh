#!/bin/bash
# One-time setup of this Mac for both skills. Safe to run again: it only adds what is missing.
#   bash setup.sh           install what is missing
#   bash setup.sh --check   report only
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
rc=0
# The skills were renamed (content-video -> buzz-video-content, sns-ad-video -> buzz-video-ad).
# Re-point links in ~/.claude/skills that were made for the old names (only symlinks; plugin installs are untouched).
SK="$HOME/.claude/skills"
for pair in content-video:buzz-video-content sns-ad-video:buzz-video-ad; do
  old="${pair%%:*}"; new="${pair##*:}"
  [ -L "$SK/$old" ] || continue
  case "$(readlink "$SK/$old")" in */skills/$old|*/skills/$old/) ;; *) continue ;; esac
  if [ "${1:-}" = "--check" ]; then echo "--   古いリンク $SK/$old（bash setup.sh で $new に張り替えます）"; continue; fi
  rm "$SK/$old" && ln -sfn "$HERE/skills/$new" "$SK/$new" && echo "ok   リンクを張り替えました: $old -> $new"
done
for s in buzz-video-content buzz-video-ad; do
  echo "== $s"
  bash "$HERE/skills/$s/scripts/setup.sh" "$@" || rc=1
done
exit $rc
