#!/bin/bash
# One-time setup of this Mac for both skills. Safe to run again: it only adds what is missing.
#   bash setup.sh           install what is missing
#   bash setup.sh --check   report only
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
rc=0
for s in content-video sns-ad-video; do
  echo "== $s"
  bash "$HERE/skills/$s/scripts/setup.sh" "$@" || rc=1
done
exit $rc
