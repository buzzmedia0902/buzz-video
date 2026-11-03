#!/usr/bin/env python3
"""Bake the iPhone HLG -> SDR (BT.709) 3D LUT used by the engine for HDR clips. Run once per machine.

  python prep_hdr.py            -> ~/.local/share/sns-ad-video/hlg2sdr_65.cube

iPhone video is often HLG (BT.2020, arib-std-b67). Without conversion it looks washed out and grey.
This ffmpeg build has no zscale/libplacebo, so the conversion is done in numpy once (HLG inverse OETF ->
OOTF gamma 1.2 -> BT.2020->709 primaries -> x1000/203 -> extended-Reinhard roll-off -> BT.709 OETF) and
baked into a 65^3 cube that ffmpeg's lut3d applies while decoding (mean error ~0.03 of an 8-bit level).
"""
from __future__ import annotations

import os

import numpy as np

OUT = os.path.expanduser("~/.local/share/sns-ad-video/hlg2sdr_65.cube")
_A, _B, _C = 0.17883277, 0.28466892, 0.55991073
_Y2020 = np.array([0.2627, 0.6780, 0.0593], np.float32)
_Y709 = np.array([0.2126, 0.7152, 0.0722], np.float32)
_M2020_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]],
                      np.float32).T


def hlg_to_sdr(x: np.ndarray, exposure: float = 1.0, white: float = 4.0) -> np.ndarray:
    """float32 HLG BT.2020 RGB in [0, 1] -> float32 BT.709 display RGB in [0, 1]."""
    lin = np.where(x <= 0.5, x * x / 3.0, (np.exp((x - _C) / _A) + _B) / 12.0).astype(np.float32)
    y = lin @ _Y2020
    lin *= (np.maximum(y, 1e-6) ** 0.2)[..., None]
    lin = lin @ _M2020_709
    np.maximum(lin, 0, out=lin)
    lin *= 1000.0 / 203.0 * exposure
    lum = lin @ _Y709
    lin *= ((1 + lum / (white * white)) / (1 + lum))[..., None]
    np.clip(lin, 0, 1, out=lin)
    return np.where(lin <= 0.018, 4.5 * lin, 1.099 * np.power(lin, 0.45) - 0.099).astype(np.float32)


def bake(n: int = 65) -> None:
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # red varies fastest in .cube order
    rgb = np.stack([r, gg, b], axis=-1).reshape(-1, 3)
    out = hlg_to_sdr(rgb)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"TITLE \"iPhone HLG to SDR BT.709\"\nLUT_3D_SIZE {n}\n")
        np.savetxt(f, out, fmt="%.6f")
    print(OUT)


if __name__ == "__main__":
    bake()
