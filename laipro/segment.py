"""
Vegetation segmentation (deterministic index-based; the reproducible default).

Phase 2 adds an optional learned GPU segmentation model that trains on the
user's manual corrections; it will plug in here behind the same interface,
returning a boolean vegetation mask. Keeping this contract stable is what lets
the classical and learned paths be swapped without touching the geometry or
gap-fraction code.
"""

from __future__ import annotations
import numpy as np
from skimage.filters import threshold_otsu


def _norm_rgb(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = r + g + b + 1e-6
    return r / s, g / s, b / s


def excess_green(rgb):
    r, g, b = _norm_rgb(rgb)
    return 2.0 * g - r - b


def excess_green_red(rgb):
    """ExGR (Neto 2004): suppresses reddish soil/shadow that plain ExG can grab."""
    r, g, b = _norm_rgb(rgb)
    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    return exg - exr


VEG_INDEX_FUNCS = {"exg": excess_green, "exgr": excess_green_red}


def segment_vegetation(rgb, index_name="exg", manual_thresh=None, valid=None):
    """Return (veg_bool, threshold, index_array).

    Otsu is computed over VALID pixels only (inside the circle of interest, not
    masked), so the dark out-of-COI corners don't skew the threshold.
    """
    idx = VEG_INDEX_FUNCS[index_name](rgb)
    if manual_thresh is not None:
        t = float(manual_thresh)
    else:
        sample = idx[valid] if valid is not None else idx.ravel()
        t = float(threshold_otsu(sample))
    return idx > t, t, idx
