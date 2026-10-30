"""Tone (トンマナ) registry: cv/tones/<key>.py defines LOOK, a cv.tones.base.Look subclass."""
from __future__ import annotations

import importlib


def get(key: str):
    try:
        return importlib.import_module(f"cv.tones.{key}").LOOK
    except ModuleNotFoundError as e:
        raise SystemExit(f"tone {key!r} is not implemented (cv/tones/{key}.py)") from e
