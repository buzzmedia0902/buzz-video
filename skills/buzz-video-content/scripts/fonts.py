#!/usr/bin/env python3
"""Which of the tones' fonts this Mac has, and how to get the missing ones (setup.sh runs this).

  python fonts.py             report
  python fonts.py --install   also install the free Google fonts with Homebrew

A missing font is not fatal: the engine falls back to Hiragino Sans, but that tone then looks plainer.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import skia

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv.gfx import FAMILIES, downloaded  # noqa: E402

# macOS fonts that Font Book downloads on request (name shown in Font Book)
FONT_BOOK = {
    "Tsukushi A Round Gothic": "筑紫A丸ゴシック",
    "Tsukushi B Round Gothic": "筑紫B丸ゴシック",
    "Toppan Bunkyu Midashi Gothic": "凸版文久見出しゴシック",
    "Toppan Bunkyu Midashi Mincho": "凸版文久見出し明朝",
    "Kaiti SC": "楷体-简（Kaiti SC）",
    "Xingkai SC": "行楷-简（Xingkai SC）",
    "YuGothic": "游ゴシック体",
}
# free fonts (SIL Open Font License) from Homebrew
CASKS = {"Rounded Mplus 1c": "font-m-plus-rounded-1c"}
# keys no tone uses by default (only when edit.json "look": {"fonts": ...} picks them)
OPTIONAL = {"Noto Sans JP": "brew install --cask font-noto-sans-jp"}


def has(family: str) -> bool:
    return downloaded(family) and skia.FontMgr().matchFamilyStyle(family, skia.FontStyle()) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="brew install --cask the missing Google fonts")
    a = ap.parse_args()
    if os.name == "nt":
        missing = [f for f in ("Yu Gothic", "Yu Mincho", "Rounded Mplus 1c") if not has(f)]
        print("--   フォントなし（游ゴシックかメイリオで代用）: " + "、".join(missing) if missing else "ok   フォント")
        return
    missing_book = []
    for family in sorted(set(FAMILIES.values())):
        if has(family):
            continue
        if family in OPTIONAL:
            print(f"--   フォント {family} なし（look.fonts で使うときだけ必要: {OPTIONAL[family]}）")
            continue
        if family in CASKS and a.install:
            print(f"..   brew install --cask {CASKS[family]}", flush=True)
            if subprocess.run(["brew", "install", "--cask", CASKS[family]]).returncode == 0:
                continue
        if family in CASKS:
            print(f"--   フォント {family} なし: brew install --cask {CASKS[family]}")
        elif family in FONT_BOOK:
            missing_book.append(FONT_BOOK[family])
        else:
            print(f"--   フォント {family} なし（macOS標準のはずのフォント。ヒラギノで代用）")
    if missing_book:
        print("--   Font Book（アプリ）で検索して「ダウンロード」すると、トンマナの見た目がそろう: " + "、".join(missing_book))
    else:
        print("ok   フォント")


if __name__ == "__main__":
    main()
