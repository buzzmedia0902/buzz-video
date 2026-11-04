"""What differs between macOS and Windows, kept in one place so the rest of the engine does not branch."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

MAC = sys.platform == "darwin"
WIN = os.name == "nt"


def hwaccel() -> list[str]:
    """ffmpeg input options for hardware decoding: VideoToolbox on a Mac, software decoding elsewhere."""
    return ["-hwaccel", "videotoolbox"] if MAC else []


def ffpath(path: str) -> str:
    """A file path to put inside an ffmpeg filter argument (Windows drive colons and backslashes escaped)."""
    return path.replace("\\", "/").replace(":", r"\:") if WIN else path


def convert_still(src: str, dst: str, fmt: str = "jpeg") -> None:
    """Convert a HEIC/HEIF photo to JPEG or PNG: sips on a Mac, pillow-heif elsewhere."""
    if MAC:
        subprocess.run(["sips", "-s", "format", fmt, src, "--out", dst], check=True, capture_output=True)
        return
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
    im = Image.open(src)
    im.save(dst, "JPEG" if fmt == "jpeg" else "PNG", **({"quality": 95} if fmt == "jpeg" else {}))


def link_or_copy(src: str, dst: str) -> None:
    """Symlink when the OS allows it (Windows needs Developer Mode), else copy."""
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


# folders searched for font files by name (macOS system fonts, Windows system and per-user fonts)
FONT_DIRS = ([os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
              os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")] if WIN else
             ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
              os.path.expanduser("~/Library/Fonts")])


def font_file(*names: str) -> str | None:
    """First font file found among names (file names like "meiryo.ttc", case-insensitive on Windows)."""
    for name in names:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def ui_font() -> str:
    """A Latin UI font file for contact sheets and labels."""
    return font_file("Helvetica.ttc", "segoeui.ttf", "arial.ttf") or "DejaVuSans.ttf"


def jp_bold_font() -> str:
    """A bold Japanese font file for measuring or drawing labels with PIL."""
    return font_file("ヒラギノ角ゴシック W6.ttc", "YuGothB.ttc", "meiryob.ttc", "meiryo.ttc") or ui_font()
