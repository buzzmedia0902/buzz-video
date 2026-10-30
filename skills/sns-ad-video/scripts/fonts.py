#!/usr/bin/env python3
"""Which of the tastes' Japanese fonts this Mac has (setup.sh runs this).

The tastes prefer fonts that macOS downloads on request in Font Book; each one has a Hiragino fallback, so a
missing font is not fatal, the taste only looks plainer.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core.text import _asset_font  # noqa: E402

FONT_BOOK = {  # asset file -> name shown in Font Book
    "ToppanBunkyuMinchoPr6N-Regular.otf": "凸版文久明朝",
    "ToppanBunkyuMidashiMinchoStdN-ExtraBold.otf": "凸版文久見出し明朝",
    "ToppanBunkyuGothicPr6N.ttc": "凸版文久ゴシック",
    "ToppanBunkyuMidashiGothicStdN-ExtraBold.otf": "凸版文久見出しゴシック",
    "TsukushiAMaruGothic.ttc": "筑紫A丸ゴシック",
    "YuMincho.ttc": "游明朝体",
    "Klee.ttc": "クレー",
    "Hannotate.ttc": "手札体-简（Hannotate SC）",
}


def main() -> None:
    with open(os.path.join(HERE, "core", "text.py"), encoding="utf-8") as f:
        wanted = sorted(set(re.findall(r'_asset_font\("([^"]+)"', f.read())))
    missing = [FONT_BOOK.get(n, n) for n in wanted if not _asset_font(n)]
    if missing:
        print("--   Font Book（アプリ）で検索して「ダウンロード」すると、テイストの見た目がそろう: " + "、".join(missing))
    else:
        print("ok   フォント")


if __name__ == "__main__":
    main()
