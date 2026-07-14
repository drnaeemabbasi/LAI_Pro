"""Image loading (incl. RAW) and RAW-preferring file discovery."""

from __future__ import annotations
import os
import glob
import numpy as np
from PIL import Image

try:
    import rawpy
    HAVE_RAWPY = True
except ImportError:
    HAVE_RAWPY = False

RAW_EXTS = {"nef", "cr2", "cr3", "arw", "dng", "orf", "rw2", "raf", "pef", "srw"}
IMAGE_EXTS = ("jpg", "jpeg", "tif", "tiff", "png") + tuple(sorted(RAW_EXTS))


def load_rgb(path: str, max_side: int | None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in RAW_EXTS:
        if not HAVE_RAWPY:
            raise SystemExit(f"Cannot read RAW '{path}': pip install rawpy")
        with rawpy.imread(path) as raw:
            rgb8 = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        im = Image.fromarray(rgb8)
    else:
        im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def find_images(folder: str, exts=IMAGE_EXTS) -> list[str]:
    """RAW-preferring discovery: for a shared filename stem, keep the RAW copy."""
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, f"*.{e}"))
        files += glob.glob(os.path.join(folder, f"*.{e.upper()}"))
    files = sorted(set(files))
    by_stem: dict[str, str] = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        is_raw = os.path.splitext(f)[1].lower().lstrip(".") in RAW_EXTS
        cur = by_stem.get(stem)
        cur_raw = cur is not None and os.path.splitext(cur)[1].lower().lstrip(".") in RAW_EXTS
        if cur is None or (is_raw and not cur_raw):
            by_stem[stem] = f
    return sorted(by_stem.values())
