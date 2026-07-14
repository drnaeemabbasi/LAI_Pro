"""
Per-pixel feature stack for learned segmentation.

Deterministic and versioned: the exact feature list is recorded in a trained
model so prediction always uses the identical representation it was trained on
(a reproducibility requirement). Features combine colour (RGB + chromatic
coordinates), standard vegetation indices, perceptual colour (HSV), and a little
local context (neighbourhood mean/std of ExG) so the classifier can use texture
and shading cues a single-pixel threshold cannot.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2hsv

FEATURE_VERSION = 1
FEATURE_NAMES = [
    "R", "G", "B", "r_chroma", "g_chroma", "b_chroma",
    "ExG", "ExR", "ExGR", "H", "S", "V",
    "ExG_localmean", "ExG_localstd",
]


def extract_features(rgb, context_window: int = 9):
    """Return (features (H, W, F) float32, names). rgb is float [0,1] (H, W, 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = r + g + b + 1e-6
    rc, gc, bc = r / s, g / s, b / s
    exg = 2.0 * gc - rc - bc
    exr = 1.4 * rc - gc
    exgr = exg - exr
    hsv = rgb2hsv(np.clip(rgb, 0, 1))

    lm = uniform_filter(exg, size=context_window, mode="nearest")
    lsq = uniform_filter(exg * exg, size=context_window, mode="nearest")
    lstd = np.sqrt(np.maximum(lsq - lm * lm, 0.0))

    feats = np.stack([
        r, g, b, rc, gc, bc, exg, exr, exgr,
        hsv[..., 0], hsv[..., 1], hsv[..., 2], lm, lstd,
    ], axis=-1).astype(np.float32)
    return feats, list(FEATURE_NAMES)
